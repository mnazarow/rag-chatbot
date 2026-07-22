"""Гибридный поиск: плотный (Qdrant) + лексический (BM25) -> реранк bge-reranker.

Модели грузятся один раз при импорте (синглтоны).
"""
from __future__ import annotations
import hashlib
import time
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from rank_bm25 import BM25Okapi

import settings
import query_filters
import vectorstore

# Векторная база (Qdrant или Milvus) — только через фасад vectorstore.
# Модели грузятся один раз при старте процесса из текущих настроек.


# Модели кэшируются ПО ПАРАМЕТРАМ (модель/устройство), а не в единственном слоте —
# иначе при смене модели в настройках старый синглтон продолжал бы обслуживать запросы
# («отравление» кэша). Публичные обёртки _embedder()/_reranker() читают текущие настройки,
# поэтому смена модели автоматически даёт новый инстанс; reset_models() сбрасывает кэш явно.
@lru_cache(maxsize=4)
def _embedder_for(model: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model, device=device)


def _embedder() -> SentenceTransformer:
    return _embedder_for(settings.get("EMBED_MODEL"), settings.device())


@lru_cache(maxsize=4)
def _reranker_for(model: str) -> FlagReranker:
    return FlagReranker(model, use_fp16=True)


def _reranker() -> FlagReranker:
    return _reranker_for(settings.get("RERANK_MODEL"))


def reset_models() -> None:
    """Сбросить кэш моделей эмбеддера/реранкера (после смены EMBED_MODEL/RERANK_MODEL/DEVICE).
    settings.update может вызывать retriever.reset_models(), чтобы не обслуживать запросы
    старой моделью. Также чистит внутрипроцессный LRU векторов запросов."""
    _embedder_for.cache_clear()
    _reranker_for.cache_clear()
    try:
        with _QEMB_LOCK:
            _QEMB_LRU.clear()
    except Exception:
        pass


def _rerank(pairs):
    """Реранк пар [вопрос, текст] с замером для дашборда."""
    import metrics
    with metrics.timer("rerank"):
        return _reranker().compute_score(pairs, normalize=True)


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


def _apply_hybrid(cands: list) -> None:
    """Гибрид «плотный+лексический»: подмешать НОРМИРОВАННЫЙ BM25 к оценке кросс-энкодера
    прямо в c['score']. Требует уже заполненных c['ce'] (кросс-энкодер, 0..1) и c['bm25'].

    Коэффициенты из настроек: HYBRID_BM25_WEIGHT (по умолчанию 0.0 — гибрид выключен,
    поведение НЕ меняется), HYBRID_CE_WEIGHT (по умолчанию 1.0). При включении:
        score = w_ce*ce + w_bm*bm25_norm,  где bm25_norm = min-max нормировка по пулу.
    Так BM25 реально влияет на ранжирование, а не только вычисляется впустую."""
    try:
        w_bm = float(settings.get("HYBRID_BM25_WEIGHT") or 0.0)
    except Exception:
        w_bm = 0.0
    if w_bm <= 0 or not cands:
        return
    try:
        w_ce = float(settings.get("HYBRID_CE_WEIGHT") or 1.0)
    except Exception:
        w_ce = 1.0
    bvals = [float(c.get("bm25", 0.0)) for c in cands]
    lo, hi = min(bvals), max(bvals)
    rng = (hi - lo) or 1.0
    for c in cands:
        bm_norm = (float(c.get("bm25", 0.0)) - lo) / rng
        c["bm25_norm"] = bm_norm
        c["score"] = w_ce * float(c.get("ce", c.get("score", 0.0))) + w_bm * bm_norm


def _expand(question: str) -> str:
    """Расширить запрос синонимами (если функция включена). Безопасно при сбое."""
    try:
        import synonyms
        return synonyms.expand_query(question)
    except Exception:
        return question


def _rewrite_query(question: str) -> str | None:
    """Улучшить запрос перед ВЕКТОРНЫМ поиском (QUERY_REWRITE):
      rewrite — LLM переформулирует вопрос в чистый поисковый запрос;
      hyde    — LLM пишет гипотетический абзац-ответ (ищем по его эмбеддингу).
    Результат кэшируется (Redis, ns=index). Безопасно при сбое — вернёт None."""
    mode = (settings.get("QUERY_REWRITE") or "off").strip().lower()
    if mode not in ("rewrite", "hyde"):
        return None

    def _gen():
        import llm_backend
        if mode == "hyde":
            sys = ("Напиши краткий (2-3 предложения) правдоподобный ответ на вопрос так, "
                   "как если бы он был в корпоративной базе знаний. Только текст ответа, "
                   "без пояснений и оговорок.")
        else:
            sys = ("Переформулируй вопрос в короткий поисковый запрос: оставь ключевые "
                   "термины, раскрой сокращения, убери лишние слова. Ответь ТОЛЬКО запросом.")
        out = llm_backend.chat(
            [{"role": "system", "content": sys},
             {"role": "user", "content": question}],
            temperature=0, model=settings.active_model())
        return (out or "").strip()[:1000]

    try:
        import cache
        key = "qr:" + mode + ":" + hashlib.sha1(cache.norm_q(question).encode("utf-8")).hexdigest()
        val = cache.get_or_set(key, 86400, _gen, ns="index")
    except Exception:
        try:
            val = _gen()
        except Exception:
            val = None
    return (val or "").strip() or None


def standalone_question(question: str, history: list | None) -> str:
    """Разрешение контекста диалога (DIALOG_REWRITE): переписать уточняющий вопрос в
    самостоятельный с учётом истории — для ПОИСКА. Ответ генерируется по оригиналу."""
    if not history or not settings.get("DIALOG_REWRITE"):
        return question
    turns = []
    for h in history[-6:]:
        role = h.get("role") if isinstance(h, dict) else None
        content = (h.get("content") if isinstance(h, dict) else "") or ""
        if role and content.strip():
            turns.append(f"{role}: {content.strip()[:500]}")
    if not turns:
        return question
    dialog = "\n".join(turns)

    def _gen():
        import llm_backend
        sys = ("Переформулируй ПОСЛЕДНИЙ вопрос пользователя в самостоятельный, понятный без "
               "истории: раскрой отсылки («это», «он», «а гарантия?»), подставь недостающие "
               "сущности из диалога. Верни ТОЛЬКО переформулированный вопрос, без пояснений.")
        out = llm_backend.chat(
            [{"role": "system", "content": sys},
             {"role": "user", "content": f"ДИАЛОГ:\n{dialog}\n\nПОСЛЕДНИЙ ВОПРОС: {question}"}],
            temperature=0, model=settings.active_model())
        return (out or "").strip()[:500]

    try:
        import cache
        key = "dlg:" + hashlib.sha1((dialog + "|" + cache.norm_q(question)).encode("utf-8")).hexdigest()
        val = cache.get_or_set(key, 3600, _gen, ns="index")
    except Exception:
        try:
            val = _gen()
        except Exception:
            val = None
    return (val or "").strip() or question


# слова в вопросе -> категория документа (мягкий интент-роутер)
_INTENT = {
    "price": ["цена", "цены", "стоит", "стоимость", "прайс", "тариф", "сколько стоит",
              "почём", "почем", "расценк"],
    "training": ["обучен", "тренинг", "вебинар", "курс", "урок", "онбординг",
                 "как научиться", "инструктаж"],
    "presentation": ["презентац", "слайд", "питч"],
}


def infer_category(question: str) -> str | None:
    """Угадать категорию по вопросу. None — фильтр не применять."""
    q = question.lower()
    for cat, kws in _INTENT.items():
        if any(kw in q for kw in kws):
            return cat
    return None


def _build_filter(filters: dict | None) -> dict | None:
    """Нейтральный фильтр (поле→значение) для vectorstore; пустые значения отброшены."""
    if not filters:
        return None
    flt = {k: v for k, v in filters.items() if v}
    return flt or None


# Внутрипроцессный LRU-кэш вектора запроса: горячие/повторные вопросы не гоняют ни
# эмбеддер, ни Redis (работает и при выключенном Redis). Ключ — (модель, норм. вопрос).
import threading as _threading
from collections import OrderedDict as _OrderedDict
_QEMB_LRU: "_OrderedDict[tuple, list]" = _OrderedDict()
_QEMB_MAX = 512
_QEMB_LOCK = _threading.Lock()


def _embed_query(question: str):
    """Вектор запроса с двухуровневым кэшем: внутрипроцессный LRU → Redis → расчёт.
    Ключ привязан к модели эмбеддингов и нормализованному вопросу. Возвращает list[float]."""
    model = settings.get("EMBED_MODEL")
    try:
        import cache
        nq = cache.norm_q(question)
    except Exception:
        nq = (question or "").strip().lower()
    lkey = (model, nq)
    with _QEMB_LOCK:
        v = _QEMB_LRU.get(lkey)
        if v is not None:
            _QEMB_LRU.move_to_end(lkey)
            return v

    def _enc():
        import metrics
        with metrics.timer("embed"):
            return _embedder().encode([question], normalize_embeddings=True)[0].tolist()

    try:
        import cache
        key = "emb:" + hashlib.sha1(f"{model}|{nq}".encode("utf-8")).hexdigest()
        vec = cache.get_or_set(key, 86400, _enc, ns="embed")
    except Exception:
        vec = _enc()
    with _QEMB_LOCK:
        _QEMB_LRU[lkey] = vec
        _QEMB_LRU.move_to_end(lkey)
        while len(_QEMB_LRU) > _QEMB_MAX:
            _QEMB_LRU.popitem(last=False)
    return vec


def _dense_search(qvec, qfilter):
    import metrics
    qv = qvec.tolist() if hasattr(qvec, "tolist") else qvec
    with metrics.timer("qdrant"):
        res = vectorstore.search(qv, settings.get("TOP_K_RETRIEVE"), flt=qfilter,
                                 with_payload=True)
    out = []
    for p in res:
        pl = p.get("payload") or {}
        if not pl.get("text"):
            continue
        out.append({"text": pl.get("text"), "source": pl.get("source"),
                    "page": pl.get("page"), "doc_category": pl.get("doc_category"),
                    "date": pl.get("date"), "t_start": pl.get("t_start"),
                    "t_end": pl.get("t_end"), "parent": pl.get("parent"),
                    "dense": p.get("score")})
    return out


def search(question: str, filters: dict | None = None,
           auto_filter: bool | None = None, trace: list | None = None) -> list[dict]:
    """Поиск с кэшированием результата в Redis. Ключ включает вопрос, явные фильтры и
    влияющие настройки; кэш в пространстве 'index' (сбрасывается при переиндексации).
    При выключенном/недоступном Redis считается напрямую. trace (если передан список) —
    наполняется этапами конвейера {key, ms, info} для анимации в интерфейсе."""
    if auto_filter is None:
        auto_filter = settings.get("AUTO_FILTER")
    try:
        import synonyms
        syn_sig = synonyms.signature()
    except Exception:
        syn_sig = ""
    import cache
    keyparts = "|".join(str(x) for x in [
        cache.norm_q(question), filters, auto_filter,
        settings.get("EMBED_MODEL"), settings.get("RERANK_MODEL"),
        settings.get("TOP_K_RETRIEVE"), settings.get("TOP_K_RERANK"),
        settings.get("MIN_SCORE"), settings.get("SMART_FILTER"),
        settings.get("QUERY_REWRITE"), syn_sig])
    ckey = "search:" + hashlib.sha1(keyparts.encode("utf-8")).hexdigest()
    # try сужён ТОЛЬКО вокруг операций кэша: при сбое чтения/записи кэша не должен
    # выполняться _search_raw дважды (ранее ошибка внутри _search_raw ловилась общим
    # except и запускала поиск повторно).
    try:
        hit = cache.get_json(ckey, ns="index")
    except Exception:
        hit = None
    if hit is not None:
        if trace is not None:
            trace.append({"key": "cache", "ms": 0, "info": {"hit": True}})
        return hit
    res = _search_raw(question, filters, auto_filter, trace)
    try:
        cache.set_json(ckey, int(settings.get("CACHE_SEARCH_TTL") or 21600), res, ns="index")
    except Exception:
        pass
    return res


def _search_raw(question: str, filters: dict | None = None,
                auto_filter: bool | None = None, trace: list | None = None) -> list[dict]:
    def rec(key, t0, info=None):
        if trace is not None:
            trace.append({"key": key, "ms": int((time.time() - t0) * 1000),
                          "info": info or {}})

    if auto_filter is None:
        auto_filter = settings.get("AUTO_FILTER")

    qx = _expand(question)        # запрос с синонимами (для BM25); ответ — по оригиналу
    # улучшение запроса для ВЕКТОРНОГО поиска (rewrite/hyde); BM25/реранк — по оригиналу
    q_rewrite = _rewrite_query(question)
    q_for_embed = q_rewrite or qx

    t = time.time()
    qvec = _embed_query(q_for_embed)
    rec("embed", t, {"model": settings.get("EMBED_MODEL"), "device": settings.device(),
                     "synonyms": qx != question,
                     "query_rewrite": (settings.get("QUERY_REWRITE") if q_rewrite else "off")})

    # 1) определяем фильтр: явный > умный (LLM) > авто-угаданная категория (правила)
    t = time.time()
    ftype = "явный" if filters is not None else "нет"
    if filters is None:
        if settings.get("SMART_FILTER"):
            filters = query_filters.extract(question) or None
            ftype = "умный (LLM)"
        elif auto_filter:
            cat = infer_category(question)
            filters = {"doc_category": cat} if cat else None
            ftype = ("авто: " + cat) if cat else "авто: нет"
    rec("filter", t, {"type": ftype, "filters": filters or {}})

    # 2) плотный поиск с фильтром; если фильтр дал мало — фолбэк без фильтра
    t = time.time()
    cands = _dense_search(qvec, _build_filter(filters))
    fb = False
    if len(cands) < 3 and filters:
        cands = _dense_search(qvec, None)
        fb = True
    rec("dense", t, {"top_k": settings.get("TOP_K_RETRIEVE"),
                     "candidates": len(cands), "fallback": fb})
    if not cands:
        return []

    # 2) лексический реранж по BM25 внутри кандидатов (дешёвый гибрид)
    t = time.time()
    bm25 = BM25Okapi([_tokenize(c["text"]) for c in cands])
    bm_scores = bm25.get_scores(_tokenize(qx))
    for c, s in zip(cands, bm_scores):
        c["bm25"] = float(s)
    rec("bm25", t, {"candidates": len(cands)})

    # 3) кросс-энкодер реранк — финальная релевантность 0..1
    t = time.time()
    pairs = [[question, c["text"]] for c in cands]
    scores = _rerank(pairs)
    if not isinstance(scores, list):
        scores = [scores]
    for c, s in zip(cands, scores):
        c["ce"] = float(s)        # чистая оценка кросс-энкодера
        c["score"] = float(s)     # по умолчанию итоговая == ce
    # гибрид: если включён (HYBRID_BM25_WEIGHT>0), подмешать нормированный BM25 в score
    _apply_hybrid(cands)

    cands.sort(key=lambda c: c["score"], reverse=True)
    min_score = settings.get("MIN_SCORE")
    top_k = settings.get("TOP_K_RERANK")
    top = [c for c in cands if c["score"] >= min_score][:top_k]
    rec("rerank", t, {"model": settings.get("RERANK_MODEL"), "top_k": top_k,
                      "min_score": min_score, "kept": len(top),
                      "candidates": len(cands),
                      "top": [{"source": c.get("source", ""),
                               "score": round(c.get("score", 0), 3)}
                              for c in cands[:5]]})
    return top


# ====================== Фолбэк: расширенный поиск, когда ответа нет ======================

_FB_STOP = set("и в во не на по с со о об а но что как так это для из у к до за от же бы "
               "ли про о про the a of to is про расскажи покажи дай что-то".split())


def _keywords(question: str) -> list:
    out = []
    for w in (question or "").lower().split():
        w = w.strip("?.,!:;()\"'«»—-")
        if len(w) >= 3 and w not in _FB_STOP:
            out.append(w)
    return out[:12]


def _hit_from_payload(p: dict) -> dict:
    return {"text": p.get("text", ""), "source": p.get("source"),
            "page": p.get("page"), "doc_category": p.get("doc_category"),
            "date": p.get("date"), "t_start": p.get("t_start"),
            "t_end": p.get("t_end"), "parent": p.get("parent")}


def _all_sources() -> list:
    """Список всех источников (имён файлов) из векторной базы через facet."""
    try:
        return [v for v in vectorstore.list_values("source", 100000) if v]
    except Exception:
        return []


def _chunks_of_source(src: str, limit: int = 50) -> list:
    try:
        pts, _ = vectorstore.scroll(flt={"source": src}, limit=limit,
                                    with_payload=True, with_vectors=False)
        return [_hit_from_payload(p.get("payload") or {}) for p in pts]
    except Exception:
        return []


def _rerank_keep(question: str, cands: list, relaxed: bool = True) -> list:
    """BM25 + кросс-энкодер по произвольному набору кандидатов; порог можно ослабить."""
    if not cands:
        return []
    # дедуп
    seen, uniq = set(), []
    for c in cands:
        k = (c.get("source"), c.get("page"), (c.get("text") or "")[:60])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    cands = uniq[:300]
    bm = BM25Okapi([_tokenize(c["text"]) for c in cands])
    for c, s in zip(cands, bm.get_scores(_tokenize(_expand(question)))):
        c["bm25"] = float(s)
    scores = _rerank([[question, c["text"]] for c in cands])
    if not isinstance(scores, list):
        scores = [scores]
    for c, s in zip(cands, scores):
        c["ce"] = float(s)
        c["score"] = float(s)
    _apply_hybrid(cands)          # тот же гибрид, что и в основном поиске (по умолчанию off)
    cands.sort(key=lambda c: c["score"], reverse=True)
    # в режиме фолбэка порог не применяем: возвращаем лучшие фрагменты найденных
    # файлов и отдаём их LLM (он сам ответит по контексту или честно скажет «нет»).
    if relaxed:
        return cands[:settings.get("TOP_K_RERANK")]
    return [c for c in cands if c["score"] >= settings.get("MIN_SCORE")][:settings.get("TOP_K_RERANK")]


def lexical_search(question: str) -> list:
    """Лексический фолбэк: широкий пул по эмбеддингу + файлы, чьи имена содержат слова
    запроса; затем реранк с ослабленным порогом."""
    kws = _keywords(question)
    cands = []
    try:
        qv = _embed_query(question)
        import metrics
        with metrics.timer("qdrant"):
            pts = vectorstore.search(qv, 200, with_payload=True)
        cands += [_hit_from_payload(p.get("payload") or {}) for p in pts]
    except Exception:
        pass
    if kws:
        try:
            matched = [s for s in _all_sources()
                       if any(kw in s.lower() for kw in kws)][:8]
            for src in matched:
                cands += _chunks_of_source(src, 40)
        except Exception:
            pass
    return _rerank_keep(question, cands, relaxed=True)


def deep_search(question: str):
    """Глубокий фолбэк: LLM по списку имён файлов выбирает кандидатов, ищем в них.
    Возвращает (hits, picked_files)."""
    sources = _all_sources()
    if not sources:
        return [], []
    # ограничиваем список имён для LLM — иначе на больших каталогах вызов очень долгий
    sources = sources[:120]
    listing = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sources))
    sys_p = ("Помоги найти документы. Ниже нумерованный список файлов. Назови номера "
             "не более 3 файлов, которые вероятнее всего содержат ответ на вопрос. "
             "Ответь ТОЛЬКО номерами через запятую, без пояснений.")
    try:
        import llm_backend
        out = llm_backend.chat(
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": f"Вопрос: {question}\n\nФайлы:\n{listing}"}],
            temperature=0, model=settings.active_model())
    except Exception:
        return [], []
    import re
    idxs = [int(x) - 1 for x in re.findall(r"\d+", out or "")][:3]
    picked = [sources[i] for i in idxs if 0 <= i < len(sources)]
    if not picked:
        return [], []
    cands = []
    for src in picked:
        cands += _chunks_of_source(src, 60)
    return _rerank_keep(question, cands, relaxed=True), picked


def no_answer_fallback(question: str, trace: list | None = None) -> list:
    """Связка фолбэков (лексический → глубокий) для случая «ответ не найден»."""
    t = time.time()
    hits = lexical_search(question)
    if trace is not None:
        trace.append({"key": "fb_lexical", "ms": int((time.time() - t) * 1000),
                      "info": {"found": len(hits)}})
    if hits:
        return hits
    t = time.time()
    hits, picked = deep_search(question)
    if trace is not None:
        trace.append({"key": "fb_deep", "ms": int((time.time() - t) * 1000),
                      "info": {"found": len(hits), "files": picked}})
    return hits


def rerank_texts(question: str, items: list, top_k: int | None = None) -> list:
    """Отобрать самые релевантные фрагменты из готового списка (без Qdrant).
    Используется для «подложенного» к вопросу документа. items: [{text,source,page}].
    Для длинных файлов сначала отсев BM25 до 120 кандидатов, затем кросс-энкодер."""
    if not items:
        return []
    if len(items) > 120:
        bm = BM25Okapi([_tokenize(i["text"]) for i in items])
        sc = bm.get_scores(_tokenize(question))
        idx = sorted(range(len(items)), key=lambda k: sc[k], reverse=True)[:120]
        items = [items[k] for k in idx]
    pairs = [[question, i["text"]] for i in items]
    scores = _rerank(pairs)
    if not isinstance(scores, list):
        scores = [scores]
    for i, s in zip(items, scores):
        i["score"] = float(s)
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[: (top_k or settings.get("TOP_K_RERANK"))]
