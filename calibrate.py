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
              "result": None, "error": None, "started": 0.0, "finished": 0.0}
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
def _candidates(question: str, pool: int) -> list:
    """Широкий пул кандидатов с реранк-оценкой и плотным рангом (один раз на вопрос)."""
    qvec = retriever._embed_query(question)
    qv = qvec.tolist() if hasattr(qvec, "tolist") else qvec
    pts = retriever._client.query_points(
        retriever._COLLECTION, query=qv, limit=pool, with_payload=True).points
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


def _grid_lists(grid: dict | None) -> dict:
    g = dict(DEFAULT_GRID)
    if grid:
        for k in ("min_score", "k_rerank", "k_retrieve"):
            v = grid.get(k)
            if isinstance(v, list) and v:
                g[k] = v
    return g


def _run(use_llm: bool, grid: dict | None) -> None:
    try:
        items = load_testset()
        if not items:
            raise RuntimeError("тестовый набор пуст — добавьте вопросы")
        g = _grid_lists(grid)
        pool = max(g["k_retrieve"])
        n_pos = sum(0 if _is_negative(i) else 1 for i in items)

        _job.update(phase="Поиск и реранк по вопросам", total=len(items), done=0)
        all_cands = []
        for it in items:
            all_cands.append(_candidates(it["q"], pool))
            _job["done"] += 1

        _job.update(phase="Перебор параметров",
                    total=len(g["min_score"]) * len(g["k_rerank"]) * len(g["k_retrieve"]),
                    done=0)
        combos = []
        for ms in g["min_score"]:
            for krr in g["k_rerank"]:
                for krt in g["k_retrieve"]:
                    combos.append(_evaluate_combo(items, all_cands, ms, krr, krt))
                    _job["done"] += 1

        # сортировка: точность ↓, затем выше порог (честнее), меньше k_rerank, меньше k_retrieve
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
            _job.update(phase="LLM-проверка ответов (рекомендованные)", total=0, done=0)
            llm = {"recommended": _answer_accuracy(
                items, all_cands, best["min_score"], best["k_rerank"], best["k_retrieve"])}
            same = (best["min_score"] == cur_eval["min_score"]
                    and best["k_rerank"] == cur_eval["k_rerank"]
                    and best["k_retrieve"] == cur_eval["k_retrieve"])
            if not same:
                _job.update(phase="LLM-проверка ответов (текущие)")
                llm["current"] = _answer_accuracy(
                    items, all_cands, cur["min_score"], cur["k_rerank"], cur["k_retrieve"])

        _job["result"] = {
            "n_items": len(items), "n_pos": n_pos, "n_neg": len(items) - n_pos,
            "best": best, "current": cur_eval,
            "combos": combos[:15], "llm": llm,
            "grid": g,
        }
    except Exception as e:
        _job["error"] = str(e)
    finally:
        _job["running"] = False
        _job["finished"] = time.time()
        _job["phase"] = "Готово" if not _job.get("error") else "Ошибка"


def start(use_llm: bool = False, grid: dict | None = None) -> dict:
    with _lock:
        if _job["running"]:
            return {"ok": False, "msg": "калибровка уже выполняется"}
        if not load_testset():
            return {"ok": False, "msg": "тестовый набор пуст — добавьте вопросы"}
        _job.update(running=True, phase="Запуск", done=0, total=0,
                    result=None, error=None, started=time.time(), finished=0.0)
    threading.Thread(target=_run, args=(bool(use_llm), grid), daemon=True).start()
    return {"ok": True, "msg": "калибровка запущена"}


def status() -> dict:
    return {k: _job.get(k) for k in
            ("running", "phase", "done", "total", "result", "error", "started", "finished")}


def apply_params(min_score=None, k_rerank=None, k_retrieve=None) -> dict:
    ch = {}
    if min_score is not None:
        ch["MIN_SCORE"] = float(min_score)
    if k_rerank is not None:
        ch["TOP_K_RERANK"] = int(k_rerank)
    if k_retrieve is not None:
        ch["TOP_K_RETRIEVE"] = int(k_retrieve)
    if not ch:
        return {"ok": False, "msg": "нет параметров для применения"}
    settings.update(ch)
    try:
        import cache
        cache.bump("index")
    except Exception:
        pass
    return {"ok": True, "applied": ch, "msg": "параметры применены"}
