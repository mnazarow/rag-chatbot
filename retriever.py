"""Гибридный поиск: плотный (Qdrant) + лексический (BM25) -> реранк bge-reranker.

Модели грузятся один раз при импорте (синглтоны).
"""
from __future__ import annotations
import hashlib
import time
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from rank_bm25 import BM25Okapi

import settings
import query_filters

# клиенты/модели создаются при старте процесса из текущих настроек
# (поля scope=restart применяются после перезапуска сервиса)
_client = QdrantClient(url=settings.get("QDRANT_URL"), timeout=settings.get("QDRANT_TIMEOUT"))
_COLLECTION = settings.get("QDRANT_COLLECTION")


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.get("EMBED_MODEL"), device=settings.device())


@lru_cache(maxsize=1)
def _reranker() -> FlagReranker:
    return FlagReranker(settings.get("RERANK_MODEL"), use_fp16=True)


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


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


def _build_filter(filters: dict | None) -> qm.Filter | None:
    if not filters:
        return None
    must = [qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
            for k, v in filters.items() if v]
    return qm.Filter(must=must) if must else None


def _embed_query(question: str):
    """Вектор запроса с кэшированием в Redis (ключ привязан к модели эмбеддингов;
    при выключенном/недоступном Redis считается напрямую). Возвращает list[float]."""
    model = settings.get("EMBED_MODEL")

    def _enc():
        return _embedder().encode([question], normalize_embeddings=True)[0].tolist()

    try:
        import cache
        key = "emb:" + hashlib.sha1(f"{model}|{question}".encode("utf-8")).hexdigest()
        return cache.get_or_set(key, 86400, _enc, ns="embed")
    except Exception:
        return _enc()


def _dense_search(qvec, qfilter):
    qv = qvec.tolist() if hasattr(qvec, "tolist") else qvec
    res = _client.query_points(
        _COLLECTION,
        query=qv,
        query_filter=qfilter,
        limit=settings.get("TOP_K_RETRIEVE"),
        with_payload=True,
    ).points
    return [
        {"text": p.payload["text"], "source": p.payload["source"],
         "page": p.payload.get("page"), "doc_category": p.payload.get("doc_category"),
         "date": p.payload.get("date"),
         "t_start": p.payload.get("t_start"), "t_end": p.payload.get("t_end"),
         "dense": p.score}
        for p in res
    ]


def search(question: str, filters: dict | None = None,
           auto_filter: bool | None = None, trace: list | None = None) -> list[dict]:
    """Поиск с кэшированием результата в Redis. Ключ включает вопрос, явные фильтры и
    влияющие настройки; кэш в пространстве 'index' (сбрасывается при переиндексации).
    При выключенном/недоступном Redis считается напрямую. trace (если передан список) —
    наполняется этапами конвейера {key, ms, info} для анимации в интерфейсе."""
    if auto_filter is None:
        auto_filter = settings.get("AUTO_FILTER")
    keyparts = "|".join(str(x) for x in [
        question, filters, auto_filter,
        settings.get("EMBED_MODEL"), settings.get("RERANK_MODEL"),
        settings.get("TOP_K_RETRIEVE"), settings.get("TOP_K_RERANK"),
        settings.get("MIN_SCORE"), settings.get("SMART_FILTER")])
    ckey = "search:" + hashlib.sha1(keyparts.encode("utf-8")).hexdigest()
    try:
        import cache
        hit = cache.get_json(ckey, ns="index")
        if hit is not None:
            if trace is not None:
                trace.append({"key": "cache", "ms": 0, "info": {"hit": True}})
            return hit
        res = _search_raw(question, filters, auto_filter, trace)
        cache.set_json(ckey, 300, res, ns="index")
        return res
    except Exception:
        return _search_raw(question, filters, auto_filter, trace)


def _search_raw(question: str, filters: dict | None = None,
                auto_filter: bool | None = None, trace: list | None = None) -> list[dict]:
    def rec(key, t0, info=None):
        if trace is not None:
            trace.append({"key": key, "ms": int((time.time() - t0) * 1000),
                          "info": info or {}})

    if auto_filter is None:
        auto_filter = settings.get("AUTO_FILTER")

    t = time.time()
    qvec = _embed_query(question)
    rec("embed", t, {"model": settings.get("EMBED_MODEL"), "device": settings.device()})

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
    bm_scores = bm25.get_scores(_tokenize(question))
    for c, s in zip(cands, bm_scores):
        c["bm25"] = float(s)
    rec("bm25", t, {"candidates": len(cands)})

    # 3) кросс-энкодер реранк — финальная релевантность 0..1
    t = time.time()
    pairs = [[question, c["text"]] for c in cands]
    scores = _reranker().compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    for c, s in zip(cands, scores):
        c["score"] = float(s)

    cands.sort(key=lambda c: c["score"], reverse=True)
    min_score = settings.get("MIN_SCORE")
    top_k = settings.get("TOP_K_RERANK")
    top = [c for c in cands if c["score"] >= min_score][:top_k]
    rec("rerank", t, {"model": settings.get("RERANK_MODEL"), "top_k": top_k,
                      "min_score": min_score, "kept": len(top),
                      "candidates": len(cands)})
    return top


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
    scores = _reranker().compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    for i, s in zip(items, scores):
        i["score"] = float(s)
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[: (top_k or settings.get("TOP_K_RERANK"))]
