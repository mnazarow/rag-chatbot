"""Автокалибровка параметров поиска на эталонном тестовом наборе.

Тестовый набор хранится в БД (kv_store, ключ ``calib_testset``) и переносится
вместе с настройками при копировании/миграции между СУБД. Каждый элемент:

    {"q": "вопрос",
     "sources": ["часть имени ожидаемого файла", ...],   # опционально
     "answer":  ["ключевое слово/фраза из ответа", ...]}   # опционально

Элемент без ``sources`` и ``answer`` считается НЕГАТИВНЫМ: правильный результат —
честное «не знаю» (после порога не остаётся ни одного фрагмента). Это позволяет
балансировать полноту (находить, когда ответ есть) и точность (молчать, когда
ответа нет).

Идея калибровки: дорогой этап (эмбеддинг + плотный поиск + кросс-энкодер реранк)
выполняется ОДИН раз на вопрос по широкому пулу кандидатов. Затем перебор сетки
параметров (``MIN_SCORE``, ``TOP_K_RERANK``, ``TOP_K_RETRIEVE``) — это лишь
нарезка уже посчитанных кандидатов, поэтому быстрый. Реранк-оценки не зависят от
размера пула, так что результат в точности воспроизводит конвейер ``_search_raw``.

Опционально (use_llm) для рекомендованной и текущей комбинаций прогоняется
генерация ответа LLM и проверяется наличие ожидаемого текста — это валидация
качества именно ответа, а не только поиска.
"""
from __future__ import annotations
import json
import threading
import time
from pathlib import Path

import settings
import db
import retriever

_KV_KEY = "calib_testset"

# Сетка по умолчанию (можно переопределить из запроса)
DEFAULT_GRID = {
    "min_score": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "k_rerank": [3, 5, 8, 10],
    "k_retrieve": [20, 30, 40, 60],
}

# Состояние фонового задания
_job: dict = {"running": False, "phase": "", "done": 0, "total": 0,
              "result": None, "error": None, "started": 0.0, "finished": 0.0,
              "live": None}
_lock = threading.Lock()


# ----------------------------- тестовый набор -----------------------------
def load_testset() -> list:
    try:
        raw = db.kv_get(_KV_KEY)
    except Exception:
        raw = None
    if not raw:
        return []
    try:
        items = json.loads(raw)
        return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []
    except Exception:
        return []


def save_testset(items: list) -> dict:
    clean = []
    for it in (items or []):
        if not isinstance(it, dict):
            continue
        q = (it.get("q") or "").strip()
        if not q:
            continue
        clean.append({
            "q": q,
            "sources": [str(s).strip() for s in (it.get("sources") or []) if str(s).strip()],
            "answer": [str(s).strip() for s in (it.get("answer") or []) if str(s).strip()],
        })
    try:
        db.kv_set(_KV_KEY, json.dumps(clean, ensure_ascii=False))
    except Exception as e:
        return {"ok": False, "msg": f"не удалось сохранить: {e}"}
    return {"ok": True, "count": len(clean)}


def example_testset() -> list:
    """Шаблон-пример для заполнения (для кнопки «Вставить пример»)."""
    return [
        {"q": "Какие у компании корпоративные ценности?",
         "sources": [], "answer": ["партнерство", "ответственность"]},
        {"q": "Расскажи про дорожную карту проекта",
         "sources": ["дорожная карта"], "answer": []},
        {"q": "Сколько стоит лицензия на 100 пользователей?",
         "sources": ["прайс"], "answer": []},
        {"q": "Какой у нас адрес офиса на Марсе?",
         "sources": [], "answer": []},
    ]


# ----------------------------- ядро калибровки -----------------------------
def _mode_filter(question: str, flags: dict | None):
    """Фильтр для плотного поиска по флагам режима (умный LLM / авто по категории / нет)."""
    flags = flags or {}
    filters = None
    if flags.get("SMART_FILTER"):
        try:
            import query_filters
            filters = query_filters.extract(question) or None
        except Exception:
            filters = None
    elif flags.get("AUTO_FILTER"):
        cat = retriever.infer_category(question)
        filters = {"doc_category": cat} if cat else None
    return filters


def _candidates(question: str, pool: int, flags: dict | None = None) -> list:
    """Широкий пул кандидатов с реранк-оценкой и плотным рангом (один раз на вопрос).
    Фильтр плотного поиска воспроизводит поведение выбранного режима работы."""
    qvec = retriever._embed_query(question)
    qv = qvec.tolist() if hasattr(qvec, "tolist") else qvec
    qfilter = retriever._build_filter(_mode_filter(question, flags))
    pts = retriever._client.query_points(
        retriever._COLLECTION, query=qv, query_filter=qfilter, limit=pool,
        with_payload=True).points
    if len(pts) < 3 and qfilter is not None:   # фолбэк без фильтра, как в конвейере
        pts = retriever._client.query_points(
            retriever._COLLECTION, query=qv, query_filter=None, limit=pool,
            with_payload=True).points
    cands = []
    for rank, p in enumerate(pts):
        pl = p.payload or {}
        cands.append({"source": pl.get("source"), "page": pl.get("page"),
                      "text": pl.get("text", ""), "dense_rank": rank,
                      "dense": p.score})
    if not cands:
        return []
    scores = retriever._reranker().compute_score(
        [[question, c["text"]] for c in cands], normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    for c, s in zip(cands, scores):
        c["score"] = float(s)
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands


def _select(cands: list, min_score: float, k_rerank: int, k_retrieve: int) -> list:
    """Воспроизводит выбор фрагментов конвейером при заданных параметрах."""
    pool = [c for c in cands if c["dense_rank"] < k_retrieve]
    pool.sort(key=lambda c: c["score"], reverse=True)
    return [c for c in pool if c["score"] >= min_score][:k_rerank]


def _is_negative(item: dict) -> bool:
    return not ((item.get("sources") or []) or (item.get("answer") or []))


def _covered(item: dict, sel: list) -> bool:
    srcs = [s.lower() for s in (item.get("sources") or [])]
    ans = [s.lower() for s in (item.get("answer") or [])]
    for h in sel:
        hs = (h.get("source") or "").lower()
        ht = (h.get("text") or "").lower()
        if srcs and any(s in hs for s in srcs):
            return True
        if ans and any(a in ht for a in ans):
            return True
    return False


def _evaluate_combo(items, all_cands, ms, krr, krt) -> dict:
    correct = pos = pos_ok = neg = neg_ok = 0
    for item, cands in zip(items, all_cands):
        sel = _select(cands, ms, krr, krt)
        if _is_negative(item):
            neg += 1
            ok = (len(sel) == 0)
            neg_ok += ok
        else:
            pos += 1
            ok = _covered(item, sel)
            pos_ok += ok
        correct += ok
    n = len(items) or 1
    return {"min_score": ms, "k_rerank": krr, "k_retrieve": krt,
            "accuracy": round(correct / n, 4), "correct": correct, "total": len(items),
            "pos_recall": round(pos_ok / pos, 4) if pos else None,
            "neg_spec": round(neg_ok / neg, 4) if neg else None}


def _answer_accuracy(items, all_cands, ms, krr, krt) -> dict:
    """LLM-проверка: для элементов с ожидаемым текстом генерируем ответ и ищем его."""
    import prompts
    import llm_backend
    model = settings.active_model()
    tot = ok = 0
    details = []
    for item, cands in zip(items, all_cands):
        ans = item.get("answer") or []
        if not ans:
            continue
        sel = _select(cands, ms, krr, krt)
        ctx = prompts.build_context(sel) if sel else ""
        msgs = [{"role": "system", "content": settings.get("SYSTEM_PROMPT")},
                {"role": "user", "content": prompts.build_user_message(item["q"], ctx)}]
        try:
            out = (llm_backend.chat(msgs, temperature=0, model=model) or "").lower()
        except Exception as e:
            out = ""
            details.append({"q": item["q"], "ok": False, "err": str(e)[:120]})
            tot += 1
            continue
        hit = any(a.lower() in out for a in ans)
        tot += 1
        ok += hit
        details.append({"q": item["q"], "ok": bool(hit)})
    return {"total": tot, "ok": ok,
            "acc": round(ok / tot, 4) if tot else None, "details": details}


def _fmt_hits(sel: list) -> str:
    if not sel:
        return "      (ничего не выбрано — система ответит «не знаю»)"
    out = []
    for h in sel:
        loc = (h.get("source") or "?")
        if h.get("page"):
            loc += f", стр. {h['page']}"
        out.append(f"      • score={h.get('score', 0):.3f}  [{loc}]")
    return "\n".join(out)


def _mode_filter_desc(flags: dict) -> str:
    if flags.get("SMART_FILTER"):
        return "умные фильтры из вопроса (LLM)"
    if flags.get("AUTO_FILTER"):
        return "авто-фильтр по категории (правила)"
    return "без фильтров"


def _report_header(items, g, n_pos, n_neg, mode_labels) -> str:
    import datetime
    L = []
    L.append("=" * 78)
    L.append("ОТЧЁТ АВТОКАЛИБРОВКИ ПАРАМЕТРОВ ПОИСКА")
    L.append(datetime.datetime.now().strftime("Дата: %Y-%m-%d %H:%M:%S"))
    L.append("=" * 78)
    L.append("")
    L.append("ЧТО ЭТО. Подбор параметров MIN_SCORE / TOP_K_RERANK / TOP_K_RETRIEVE на")
    L.append("эталонном наборе «вопрос → ожидаемый результат». Для каждого вопроса поиск и")
    L.append("кросс-энкодер реранк выполняются один раз по широкому пулу кандидатов, далее")
    L.append("перебирается сетка параметров (это лишь нарезка уже посчитанных кандидатов —")
    L.append("реранк-оценки не зависят от размера пула, поэтому результат в точности")
    L.append("воспроизводит рабочий конвейер).")
    L.append("")
    L.append("РЕЖИМЫ РАБОТЫ (варианты), для которых выполнена калибровка:")
    for ml in mode_labels:
        L.append(f"  • {ml}")
    L.append("  Режимы различаются фильтрацией кандидатов перед реранком; параметры порога/")
    L.append("  выборки подбираются под каждый режим отдельно. Граф-RAG (Hybrid+) для сводных")
    L.append("  вопросов не параметризуется этими настройками — калибруется его векторная часть.")
    L.append("")
    L.append("МЕТРИКИ:")
    L.append("  • Точность      — доля корректно обработанных вопросов (нашёл нужное ИЛИ")
    L.append("                    корректно промолчал на негативном вопросе).")
    L.append("  • Полнота       — доля позитивных вопросов, где найден ожидаемый результат.")
    L.append("  • Специфичность — доля негативных вопросов, где система корректно промолчала.")
    L.append("  Засчитывается попадание, если среди выбранных фрагментов есть ожидаемый файл")
    L.append("  (по части имени) ИЛИ их текст содержит ожидаемое ключевое слово.")
    L.append("  Негативный вопрос — без ожиданий: правильно ответить «не знаю».")
    L.append("")
    L.append(f"ВОПРОСОВ: {len(items)}  (позитивных: {n_pos}, негативных: {n_neg})")
    L.append(f"СЕТКА: MIN_SCORE={g['min_score']}  TOP_K_RERANK={g['k_rerank']}  "
             f"TOP_K_RETRIEVE={g['k_retrieve']}")
    return "\n".join(L)


def _report_variant(label, flags, items, all_cands, combos, best, cur_eval, llm,
                    n_pos, n_neg) -> str:
    """Секция отчёта для одного режима работы."""
    L = []
    L.append("")
    L.append("#" * 78)
    L.append(f"РЕЖИМ: {label}   (фильтрация: {_mode_filter_desc(flags)})")
    L.append("#" * 78)
    L.append("")
    L.append("-" * 78)
    L.append("ИТОГ")
    L.append("-" * 78)
    L.append(f"Текущие параметры:      MIN_SCORE={cur_eval['min_score']}  "
             f"TOP_K_RERANK={cur_eval['k_rerank']}  TOP_K_RETRIEVE={cur_eval['k_retrieve']}")
    L.append(f"  точность={cur_eval['accuracy'] * 100:.1f}%  "
             f"полнота={_p(cur_eval['pos_recall'])}  специфичность={_p(cur_eval['neg_spec'])}")
    L.append(f"Рекомендовано:          MIN_SCORE={best['min_score']}  "
             f"TOP_K_RERANK={best['k_rerank']}  TOP_K_RETRIEVE={best['k_retrieve']}")
    L.append(f"  точность={best['accuracy'] * 100:.1f}%  "
             f"полнота={_p(best['pos_recall'])}  специфичность={_p(best['neg_spec'])}")
    L.append("  (при равной точности предпочтён более высокий порог и меньшая выборка —")
    L.append("   это честнее в части «не знаю» и дешевле по ресурсам.)")
    L.append("")
    L.append("-" * 78)
    L.append("ВСЕ КОМБИНАЦИИ (по убыванию качества)")
    L.append("-" * 78)
    L.append(f"{'MIN_SCORE':>10} {'RERANK':>7} {'RETRIEVE':>9} {'точность':>9} "
             f"{'полнота':>8} {'специф.':>8}")
    for r in combos:
        mark = "  <-- рекомендовано" if (r["min_score"] == best["min_score"]
            and r["k_rerank"] == best["k_rerank"]
            and r["k_retrieve"] == best["k_retrieve"]) else ""
        L.append(f"{r['min_score']:>10} {r['k_rerank']:>7} {r['k_retrieve']:>9} "
                 f"{r['accuracy'] * 100:>8.1f}% {_p(r['pos_recall']):>8} "
                 f"{_p(r['neg_spec']):>8}{mark}")
    L.append("")
    L.append("-" * 78)
    L.append("ПОВОПРОСНЫЙ РАЗБОР ПРИ РЕКОМЕНДОВАННЫХ ПАРАМЕТРАХ")
    L.append("-" * 78)
    bms, bkr, bkt = best["min_score"], best["k_rerank"], best["k_retrieve"]
    for i, (item, cands) in enumerate(zip(items, all_cands), 1):
        sel = _select(cands, bms, bkr, bkt)
        neg = _is_negative(item)
        ok = (len(sel) == 0) if neg else _covered(item, sel)
        L.append(f"[{i}] {item['q']}")
        L.append(f"    тип: {'негативный (ожидается «не знаю»)' if neg else 'позитивный'}")
        if not neg:
            if item.get("sources"):
                L.append(f"    ожидаемые файлы: {', '.join(item['sources'])}")
            if item.get("answer"):
                L.append(f"    ожидаемые слова: {', '.join(item['answer'])}")
        L.append(f"    выбрано фрагментов: {len(sel)}")
        L.append(_fmt_hits(sel))
        if neg:
            verdict = ("OK — корректно промолчал" if ok
                       else f"ОШИБКА — выбрано {len(sel)} фрагм., ожидалось молчание")
        else:
            if ok:
                verdict = "OK — найден ожидаемый источник/текст"
            else:
                verdict = ("ОШИБКА — среди выбранных нет ожидаемых файлов и нет ключевых "
                           "слов (тема не в индексе, либо нужен ниже порог / шире выборка)")
        L.append(f"    вердикт: {verdict}")
        L.append("")
    if llm:
        L.append("-" * 78)
        L.append("LLM-ПРОВЕРКА ТЕКСТА ОТВЕТА")
        L.append("-" * 78)
        for tag, key in (("рекомендованные", "recommended"), ("текущие", "current")):
            d = llm.get(key)
            if not d:
                continue
            L.append(f"{tag}: {d['ok']}/{d['total']} ответов содержат ожидаемый текст "
                     f"({_p(d['acc'])})")
            for it in (d.get("details") or []):
                mk = "OK " if it.get("ok") else "нет"
                extra = f"  ({it['err']})" if it.get("err") else ""
                L.append(f"    [{mk}] {it['q']}{extra}")
            L.append("")
    return "\n".join(L)


def _p(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _grid_lists(grid: dict | None) -> dict:
    g = dict(DEFAULT_GRID)
    if grid:
        for k in ("min_score", "k_rerank", "k_retrieve"):
            v = grid.get(k)
            if isinstance(v, list) and v:
                g[k] = v
    return g


def available_modes() -> list:
    """Список режимов работы (вариантов) для выбора в калибровке."""
    out = []
    for k, m in getattr(settings, "MODES", {}).items():
        out.append({"key": k, "label": m.get("label", k),
                    "desc": m.get("desc", ""),
                    "filter": _mode_filter_desc(m.get("flags", {}))})
    return out


def _norm_modes(modes) -> list:
    valid = list(getattr(settings, "MODES", {}).keys())
    if modes is None:   # не задано вовсе — режим по умолчанию (текущий)
        try:
            cur = settings.current_mode()
        except Exception:
            cur = None
        return [cur] if cur in valid else (valid[:1] or ["basic"])
    return [m for m in modes if m in valid]   # явный список (может быть пустым)


def _agg(combos, field):
    out = {}
    for r in combos:
        v = r[field]
        if v not in out or r["accuracy"] > out[v]:
            out[v] = r["accuracy"]
    return [{"v": k, "acc": out[k]} for k in sorted(out)]


def _calibrate_mode(items, g, pool, n_pos, n_neg, mode, use_llm) -> tuple:
    """Калибровка одного режима. Возвращает (variant_dict, report_section_text)."""
    flags = getattr(settings, "MODES", {}).get(mode, {}).get("flags", {})
    label = getattr(settings, "MODES", {}).get(mode, {}).get("label", mode)
    lv = _job["live"]
    lv.update(stage="search", q_done=0, last_q="", history=[],
              best_acc=0.0, best_params=None, mode=mode, mode_label=label)
    n_combos = lv["combos_total"]

    _job.update(phase=f"[{label}] Поиск и реранк", total=len(items), done=0)
    all_cands = []
    for it in items:
        lv["last_q"] = it["q"][:80]
        all_cands.append(_candidates(it["q"], pool, flags))
        _job["done"] += 1
        lv["q_done"] = _job["done"]

    _job.update(phase=f"[{label}] Перебор сетки", total=n_combos, done=0)
    lv["stage"] = "grid"
    combos = []
    best_acc = -1.0
    for ms in g["min_score"]:
        for krr in g["k_rerank"]:
            for krt in g["k_retrieve"]:
                r = _evaluate_combo(items, all_cands, ms, krr, krt)
                combos.append(r)
                _job["done"] += 1
                lv["combos_done"] = _job["done"]
                if r["accuracy"] > best_acc:
                    best_acc = r["accuracy"]
                    lv["best_acc"] = best_acc
                    lv["best_params"] = {"min_score": ms, "k_rerank": krr, "k_retrieve": krt}
                if (_job["done"] % max(1, n_combos // 60) == 0 or _job["done"] == n_combos):
                    lv["history"].append({"n": _job["done"], "acc": round(best_acc, 4)})

    combos.sort(key=lambda r: (r["accuracy"], r["min_score"],
                               -r["k_rerank"], -r["k_retrieve"]), reverse=True)
    best = combos[0]
    cur = {"min_score": settings.get("MIN_SCORE"),
           "k_rerank": settings.get("TOP_K_RERANK"),
           "k_retrieve": settings.get("TOP_K_RETRIEVE")}
    cur_eval = _evaluate_combo(items, all_cands, cur["min_score"],
                               cur["k_rerank"], cur["k_retrieve"])

    llm = None
    if use_llm:
        lv["stage"] = "llm"
        _job.update(phase=f"[{label}] LLM-проверка (рекомендованные)", total=0, done=0)
        llm = {"recommended": _answer_accuracy(
            items, all_cands, best["min_score"], best["k_rerank"], best["k_retrieve"])}
        same = (best["min_score"] == cur_eval["min_score"]
                and best["k_rerank"] == cur_eval["k_rerank"]
                and best["k_retrieve"] == cur_eval["k_retrieve"])
        if not same:
            _job.update(phase=f"[{label}] LLM-проверка (текущие)")
            llm["current"] = _answer_accuracy(
                items, all_cands, cur["min_score"], cur["k_rerank"], cur["k_retrieve"])

    section = _report_variant(label, flags, items, all_cands, combos, best,
                              cur_eval, llm, n_pos, n_neg)
    variant = {
        "mode": mode, "label": label, "filter": _mode_filter_desc(flags),
        "best": best, "current": cur_eval, "combos": combos[:15], "llm": llm,
        "by_min_score": _agg(combos, "min_score"),
        "by_k_rerank": _agg(combos, "k_rerank"),
        "by_k_retrieve": _agg(combos, "k_retrieve"),
        "history": list(lv.get("history", [])),
    }
    return variant, section


_ENGINES = {
    "vector": {"label": "Векторный", "desc": "Поиск + реранк по документам."},
    "lightrag": {"label": "LightRAG (граф)", "desc": "Ответ из графа знаний (нужен граф)."},
    "kag": {"label": "KAG", "desc": "Декомпозиция → мультихоп → знания графа → ответ."},
}


def available_engines() -> list:
    return [{"key": k, "label": v["label"], "desc": v["desc"]} for k, v in _ENGINES.items()]


def _norm_engines(engines) -> list:
    return [e for e in (engines or []) if e in _ENGINES]


def _engine_answer(engine, q):
    """Прогнать вопрос через движок end-to-end. Возврат (hits, text, answered)."""
    if engine == "vector":
        hits = retriever.search(q) or []
        return hits, "", bool(hits)
    if engine == "kag":
        import asyncio
        import kag
        res = asyncio.run(kag.answer(q))
        return res.get("hits", []), res.get("text", "") or "", bool(res.get("answered", True))
    if engine == "lightrag":
        import asyncio
        import graph_rag
        text = asyncio.run(graph_rag.answer(q)) or ""
        ans = bool(text.strip()) and "нет точного ответа" not in text.lower()
        return [], text, ans
    return [], "", False


def _eval_engine(items, engine) -> dict:
    """End-to-end оценка движка на тестовом наборе (на ТЕКУЩИХ настройках)."""
    label = _ENGINES.get(engine, {}).get("label", engine)
    lv = _job["live"]
    lv.update(stage="llm", q_done=0, mode_label=label, last_q="")
    _job.update(phase=f"[Движок: {label}] оценка", total=len(items), done=0)
    correct = pos = pos_ok = neg = neg_ok = 0
    rows = []
    for it in items:
        q = it["q"]
        neg_q = _is_negative(it)
        srcs = [s.lower() for s in (it.get("sources") or [])]
        ans = [s.lower() for s in (it.get("answer") or [])]
        lv["last_q"] = q[:80]
        try:
            hits, text, answered = _engine_answer(engine, q)
        except Exception as e:
            hits, text, answered = [], "", False
            rows.append({"q": q, "ok": False, "err": str(e)[:120]})
            _job["done"] += 1
            lv["q_done"] = _job["done"]
            if neg_q:
                neg += 1
            else:
                pos += 1
            continue
        tl = (text or "").lower()
        cov = False
        for h in hits:
            hs = (h.get("source") or "").lower()
            ht = (h.get("text") or "").lower()
            if srcs and any(s in hs for s in srcs):
                cov = True
            if ans and any(a in ht for a in ans):
                cov = True
        if ans and any(a in tl for a in ans):
            cov = True
        if srcs and any(s in tl for s in srcs):
            cov = True
        if neg_q:
            neg += 1
            ok = (not answered)
            neg_ok += ok
        else:
            pos += 1
            ok = cov
            pos_ok += ok
        correct += ok
        rows.append({"q": q, "ok": bool(ok), "neg": neg_q})
        _job["done"] += 1
        lv["q_done"] = _job["done"]
    n = len(items) or 1
    return {"engine": engine, "label": label,
            "accuracy": round(correct / n, 4),
            "pos_recall": round(pos_ok / pos, 4) if pos else None,
            "neg_spec": round(neg_ok / neg, 4) if neg else None,
            "rows": rows}


def _report_engines(evals) -> str:
    if not evals:
        return ""
    L = ["", "#" * 78, "ОЦЕНКА ДВИЖКОВ (end-to-end на текущих настройках)", "#" * 78, ""]
    L.append("Прогон каждого вопроса через движок целиком и проверка попадания (ожидаемый")
    L.append("источник/слово найдены) или корректного «не знаю» на негативных вопросах.")
    L.append("Параметры KAG/графа настраиваются вручную в их группах настроек.")
    L.append("")
    for e in evals:
        L.append(f"— {e['label']}: точность {e['accuracy'] * 100:.1f}%  "
                 f"полнота {_p(e['pos_recall'])}  специфичность {_p(e['neg_spec'])}")
        for r in e.get("rows", []):
            mk = "OK " if r.get("ok") else "нет"
            extra = f"  ({r['err']})" if r.get("err") else ""
            L.append(f"    [{mk}] {r['q']}{extra}")
        L.append("")
    return "\n".join(L)


def _run(use_llm: bool, grid: dict | None, modes, engines=None) -> None:
    try:
        items = load_testset()
        if not items:
            raise RuntimeError("тестовый набор пуст — добавьте вопросы")
        g = _grid_lists(grid)
        pool = max(g["k_retrieve"])
        n_pos = sum(0 if _is_negative(i) else 1 for i in items)
        n_neg = len(items) - n_pos
        modes = _norm_modes(modes)
        engines = _norm_engines(engines)
        if not modes and not engines:
            raise RuntimeError("не выбрано ни одного режима или движка")
        n_combos = len(g["min_score"]) * len(g["k_rerank"]) * len(g["k_retrieve"])
        _job["live"].update(q_total=len(items), combos_total=n_combos,
                            mode_total=len(modes), mode_idx=0)

        mode_labels = [getattr(settings, "MODES", {}).get(m, {}).get("label", m)
                       for m in modes]
        variants, sections = [], []
        for mi, mode in enumerate(modes):
            _job["live"]["mode_idx"] = mi + 1
            v, sec = _calibrate_mode(items, g, pool, n_pos, n_neg, mode, use_llm)
            variants.append(v)
            sections.append(sec)

        # end-to-end оценка движков (vector/lightrag/kag) на текущих настройках
        engine_evals = []
        for engine in engines:
            try:
                engine_evals.append(_eval_engine(items, engine))
            except Exception as e:
                print(f"[calib] движок {engine} не оценён: {e}")

        report = (_report_header(items, g, n_pos, n_neg, mode_labels) + "\n"
                  + "\n".join(sections) + "\n" + _report_engines(engine_evals)
                  + "\n" + "=" * 78 + "\nКонец отчёта.")
        log_id = None
        try:
            parts = "; ".join(f"{v['label']}: {v['best']['accuracy'] * 100:.0f}% "
                              f"({v['best']['min_score']}/{v['best']['k_rerank']}/"
                              f"{v['best']['k_retrieve']})" for v in variants)
            ep = "; ".join(f"{e['label']}: {e['accuracy'] * 100:.0f}%" for e in engine_evals)
            summary = (f"режимов: {len(variants)} — {parts}"
                       + (f" | движки — {ep}" if ep else "")
                       + f" (вопросов: {len(items)})")
            log_id = db.ingest_log_save("Автокалибровка", summary, report)
        except Exception as e:
            print(f"[calib] не удалось сохранить лог в БД: {e}")

        _job["live"]["stage"] = "report"
        _job["result"] = {
            "n_items": len(items), "n_pos": n_pos, "n_neg": n_neg,
            "grid": g, "report": report, "log_id": log_id,
            "multi": len(variants) > 1, "variants": variants,
            "engine_evals": engine_evals,
        }
    except Exception as e:
        _job["error"] = str(e)
    finally:
        _job["running"] = False
        _job["finished"] = time.time()
        _job["phase"] = "Готово" if not _job.get("error") else "Ошибка"
        if isinstance(_job.get("live"), dict):
            _job["live"]["stage"] = "done" if not _job.get("error") else "error"


def start(use_llm: bool = False, grid: dict | None = None, modes=None,
          engines=None) -> dict:
    with _lock:
        if _job["running"]:
            return {"ok": False, "msg": "калибровка уже выполняется"}
        if not load_testset():
            return {"ok": False, "msg": "тестовый набор пуст — добавьте вопросы"}
        _job.update(running=True, phase="Запуск", done=0, total=0,
                    result=None, error=None, started=time.time(), finished=0.0,
                    live={"stage": "load", "q_done": 0, "q_total": 0, "last_q": "",
                          "combos_done": 0, "combos_total": 0, "best_acc": 0.0,
                          "best_params": None, "history": [], "mode": "",
                          "mode_label": "", "mode_idx": 0, "mode_total": 0})
    threading.Thread(target=_run, args=(bool(use_llm), grid, modes, engines),
                     daemon=True).start()
    return {"ok": True, "msg": "калибровка запущена"}


def status() -> dict:
    return {k: _job.get(k) for k in
            ("running", "phase", "done", "total", "result", "error", "started",
             "finished", "live")}


def apply_params(min_score=None, k_rerank=None, k_retrieve=None, mode=None) -> dict:
    ch = {}
    if min_score is not None:
        ch["MIN_SCORE"] = float(min_score)
    if k_rerank is not None:
        ch["TOP_K_RERANK"] = int(k_rerank)
    if k_retrieve is not None:
        ch["TOP_K_RETRIEVE"] = int(k_retrieve)
    if not ch and not mode:
        return {"ok": False, "msg": "нет параметров для применения"}
    if ch:
        settings.update(ch)
    switched = None
    if mode and mode in getattr(settings, "MODES", {}):
        try:
            settings.set_mode(mode)
            switched = settings.MODES[mode].get("label", mode)
        except Exception as e:
            return {"ok": False, "msg": f"параметры применены, но режим не переключён: {e}"}
    try:
        import cache
        cache.bump("index")
    except Exception:
        pass
    msg = "параметры применены" + (f"; режим: {switched}" if switched else "")
    return {"ok": True, "applied": ch, "mode": switched, "msg": msg}


# ===================================================================================
# Альтернативный набор: авто Q&A из папки test + оценка по сходству ответа с эталоном
# ===================================================================================
_AUTO_KV = "calib_testset_auto"
_auto: dict = {"running": False, "phase": "", "done": 0, "total": 0,
               "result": None, "error": None, "started": 0.0, "finished": 0.0}
_auto_lock = threading.Lock()


def auto_load() -> list:
    try:
        raw = db.kv_get(_AUTO_KV)
        items = json.loads(raw) if raw else []
        return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []
    except Exception:
        return []


def auto_save(items: list) -> None:
    try:
        db.kv_set(_AUTO_KV, json.dumps(items, ensure_ascii=False))
    except Exception as e:
        print(f"[calib-auto] не удалось сохранить набор: {e}")


def auto_status() -> dict:
    out = {k: _auto.get(k) for k in
           ("running", "phase", "done", "total", "result", "error", "started", "finished")}
    out["count"] = len(auto_load())
    return out


def _passages(folder: str, max_passages: int = 400) -> list:
    """Нарезать текст файлов из DOCS_DIR/<folder> на пассажи (text, source)."""
    import fsutil
    from loaders import load_file
    root = Path(settings.get("DOCS_DIR")).expanduser()
    base = root / folder
    if not base.exists():
        raise RuntimeError(f"папка не найдена: {base} — создайте подпапку «{folder}» "
                           "в каталоге документов и положите туда файлы")
    try:
        from ingest import SUPPORTED
    except Exception:
        SUPPORTED = {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".csv", ".txt",
                     ".md", ".html", ".htm"}
    out = []
    for path in fsutil.iter_doc_files(base, SUPPORTED):
        try:
            parts = load_file(path)
        except Exception:
            continue
        try:
            rel = str(path.relative_to(root))
        except Exception:
            rel = path.name
        buf = ""
        for part in parts:
            t = (part.get("text") or "").strip()
            if not t:
                continue
            buf += ("\n" if buf else "") + t
            while len(buf) >= 1400:
                out.append({"text": buf[:1400], "source": rel})
                buf = buf[1400:]
        if len(buf.strip()) >= 40:
            out.append({"text": buf.strip(), "source": rel})
        if len(out) >= max_passages:
            break
    return out


def _gen_pair(passage: dict) -> dict | None:
    import llm_backend
    import re
    sys_p = ("На основе ТОЛЬКО приведённого фрагмента придумай один осмысленный вопрос и "
             "краткий точный ответ строго по фрагменту (1–3 предложения, без воды). Ответь "
             "СТРОГО одним JSON-объектом: {\"q\": \"вопрос\", \"a\": \"ответ\"} — без пояснений.")
    try:
        out = llm_backend.chat([{"role": "system", "content": sys_p},
                                {"role": "user", "content": passage["text"][:2500]}], 0.2)
    except Exception:
        return None
    m = re.search(r"\{.*\}", out or "", re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    q = str(d.get("q") or "").strip()
    a = str(d.get("a") or "").strip()
    if len(q) < 5 or len(a) < 2:
        return None
    return {"q": q, "a": a, "source": passage.get("source", "")}


def _auto_gen_run(n: int, folder: str) -> None:
    try:
        ps = _passages(folder)
        if not ps:
            raise RuntimeError("в папке нет распознаваемых документов с текстом")
        import random
        random.shuffle(ps)
        _auto.update(phase=f"Генерация пар из «{folder}»", total=n, done=0)
        items = []
        for p in ps:
            if len(items) >= n:
                break
            pair = _gen_pair(p)
            if pair:
                items.append(pair)
                _auto["done"] = len(items)
        if not items:
            raise RuntimeError("LLM не вернула ни одной валидной пары — проверьте модель")
        auto_save(items)
        _auto["result"] = {"kind": "generate", "count": len(items), "items": items}
    except Exception as e:
        _auto["error"] = str(e)
    finally:
        _auto["running"] = False
        _auto["finished"] = time.time()
        _auto["phase"] = "Готово" if not _auto.get("error") else "Ошибка"


def _answer_text(q: str, engine: str) -> str:
    """Полный ответ системы на вопрос выбранным движком (для оценки по сходству)."""
    if engine == "kag":
        import asyncio
        import kag
        return (asyncio.run(kag.answer(q)) or {}).get("text", "") or ""
    if engine == "lightrag":
        import asyncio
        import graph_rag
        return asyncio.run(graph_rag.answer(q)) or ""
    # vector / по умолчанию — полный конвейер: поиск → контекст → генерация
    import prompts
    import llm_backend
    hits = retriever.search(q) or []
    if not hits:
        return "В доступных документах нет точного ответа на этот вопрос."
    ctx = prompts.build_context(hits)
    msgs = [{"role": "system", "content": settings.get("SYSTEM_PROMPT")},
            {"role": "user", "content": prompts.build_user_message(q, ctx)}]
    try:
        return llm_backend.chat(msgs, 0.1, settings.active_model()) or ""
    except Exception:
        return ""


def _similarity(a: str, b: str) -> float:
    """Косинусная близость ответов через эмбеддер (0..1)."""
    if not (a or "").strip() or not (b or "").strip():
        return 0.0
    try:
        import numpy as np
        v = retriever._embedder().encode([a, b], normalize_embeddings=True)
        return max(0.0, min(1.0, float(np.dot(v[0], v[1]))))
    except Exception:
        return 0.0


def _auto_report(engine, thr, acc, ok, total, rows) -> str:
    import datetime
    L = ["=" * 78, "ОТЧЁТ ОЦЕНКИ ПО АВТО-НАБОРУ (папка test)",
         datetime.datetime.now().strftime("Дата: %Y-%m-%d %H:%M:%S"), "=" * 78, "",
         "Пары «вопрос → эталонный ответ» сгенерированы LLM из файлов папки test.",
         "Система отвечает на каждый вопрос текущим движком, ответ сравнивается с эталоном",
         "по косинусной близости эмбеддингов. «Отклонение» = (1 − близость)·100%.",
         f"Зачёт, если отклонение ≤ {thr:.0f}%.", "",
         f"ДВИЖОК: {engine}   ПАР: {total}   ТОЧНОСТЬ: {acc * 100:.1f}% ({ok}/{total})",
         "-" * 78]
    for i, r in enumerate(rows, 1):
        verdict = "OK " if r["ok"] else "нет"
        L.append(f"[{i}] [{verdict}] откл. {r['dev']}% (сходство {r['sim']}%)  «{r['q']}»")
        L.append(f"      эталон:  {r['ref']}")
        L.append(f"      система: {(r['sys'] or '')[:300]}")
        if r.get("source"):
            L.append(f"      источник: {r['source']}")
        L.append("")
    L.append("=" * 78)
    L.append("Конец отчёта.")
    return "\n".join(L)


def _auto_eval_run(deviation, engine) -> None:
    try:
        items = auto_load()
        if not items:
            raise RuntimeError("сначала сгенерируйте набор из папки test")
        engine = engine if engine in _ENGINES else settings.get("ENGINE")
        thr = max(0.0, min(100.0, float(deviation)))
        _auto.update(phase=f"Оценка ({engine}), отклонение ≤{thr:.0f}%",
                     total=len(items), done=0)
        rows, ok = [], 0
        for it in items:
            sys_ans = _answer_text(it["q"], engine)
            sim = _similarity(sys_ans, it.get("a", ""))
            dev = round((1.0 - sim) * 100, 1)
            passed = dev <= thr
            ok += passed
            rows.append({"q": it["q"], "ref": it.get("a", ""), "sys": sys_ans,
                         "sim": round(sim * 100, 1), "dev": dev, "ok": bool(passed),
                         "source": it.get("source", "")})
            _auto["done"] = len(rows)
        acc = round(ok / len(items), 4)
        report = _auto_report(engine, thr, acc, ok, len(items), rows)
        log_id = None
        try:
            log_id = db.ingest_log_save(
                "Автокалибровка (папка test)",
                f"движок {engine}, отклонение ≤{thr:.0f}% — точность {acc * 100:.1f}% "
                f"({ok}/{len(items)})", report)
        except Exception as e:
            print(f"[calib-auto] лог не сохранён: {e}")
        _auto["result"] = {"kind": "eval", "engine": engine, "deviation": thr,
                           "accuracy": acc, "ok": ok, "total": len(items),
                           "rows": rows, "report": report, "log_id": log_id}
    except Exception as e:
        _auto["error"] = str(e)
    finally:
        _auto["running"] = False
        _auto["finished"] = time.time()
        _auto["phase"] = "Готово" if not _auto.get("error") else "Ошибка"


def auto_generate(n: int = 50, folder: str = "test") -> dict:
    with _auto_lock:
        if _auto["running"]:
            return {"ok": False, "msg": "операция уже выполняется"}
        try:
            n = max(1, min(200, int(n)))
        except Exception:
            n = 50
        _auto.update(running=True, phase="Запуск", done=0, total=n, result=None,
                     error=None, started=time.time(), finished=0.0)
    threading.Thread(target=_auto_gen_run, args=(n, folder or "test"), daemon=True).start()
    return {"ok": True, "msg": f"генерация {n} пар запущена"}


def auto_run(deviation=30, engine=None) -> dict:
    with _auto_lock:
        if _auto["running"]:
            return {"ok": False, "msg": "операция уже выполняется"}
        if not auto_load():
            return {"ok": False, "msg": "набор пуст — сначала сгенерируйте пары из папки test"}
        _auto.update(running=True, phase="Запуск", done=0, total=0, result=None,
                     error=None, started=time.time(), finished=0.0)
    threading.Thread(target=_auto_eval_run, args=(deviation, engine), daemon=True).start()
    return {"ok": True, "msg": "оценка запущена"}
