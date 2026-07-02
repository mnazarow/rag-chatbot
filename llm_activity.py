"""Реестр запросов к LLM в реальном времени.

Каждый вызов модели (генерация ответа в чате/Телеграме, описание изображения
vision-моделью, служебные вызовы — фильтр запроса, калибровка, обогащение и т. п.)
регистрируется здесь: что за вызов, модель, бэкенд, статус, объём вывода и время.
Дашборд опрашивает снимок и показывает «живые» и недавно завершённые запросы.

ВАЖНО — общий реестр между процессами. Индексация запускается отдельным процессом
(`ingest.py`), и описание картинок vision-моделью идёт именно там. Чтобы такие
вызовы были видны на дашборде (его обслуживает процесс веб-приложения), реестр
хранится в Redis, если он включён. Без Redis — в памяти текущего процесса (тогда
видны только вызовы самого веб-приложения).
"""
from __future__ import annotations
import itertools
import os
import threading
import time

FINISHED_TTL = 30.0     # сек показывать завершённые
RUNNING_TTL = 1800      # сек авто-уборки «зависших» выполняющихся (Redis safety)
KEEP_RECENT = 25        # минимум последних завершённых в снимке (память)
MAX_ITEMS = 400

_PREFIX = "rag:llmact:"   # ключи Redis: rag:llmact:i:<id>, rag:llmact:calls/errors

_lock = threading.Lock()
_items: dict[str, dict] = {}
_counter = itertools.count(1)
_totals = {"calls": 0, "errors": 0, "ptok": 0, "ctok": 0, "genms": 0}
_pid = os.getpid()


def _redis():
    try:
        import cache
        return cache.client()
    except Exception:
        return None


def _new_id() -> str:
    return f"{_pid}-{next(_counter)}"


# --------------------------------------------------------------------------- #
#  Запись                                                                      #
# --------------------------------------------------------------------------- #
def begin(kind: str, model: str = "", backend: str = "", label: str = "",
          prompt: str = "") -> str:
    cid = _new_id()
    now = time.time()
    rec = {"id": cid, "kind": kind or "llm", "model": model or "",
           "backend": backend or "", "label": (label or "")[:160],
           "prompt": (prompt or "")[:16000],   # полный запрос (для раскрытия строки)
           "started": now, "updated": now, "finished": None,
           "done": False, "ok": None, "chars": 0, "error": None,
           "ptok": 0, "ctok": 0, "gen_ms": 0}   # токены запрос/ответ + время генерации
    c = _redis()
    if c is not None:
        try:
            import json
            c.setex(_PREFIX + "i:" + cid, RUNNING_TTL,
                    json.dumps(rec, ensure_ascii=False))
            c.incr(_PREFIX + "calls")
            return cid
        except Exception:
            pass
    with _lock:
        _items[cid] = rec
        _totals["calls"] += 1
        _gc_locked()
    return cid


def tokens(cid, chars: int) -> None:
    c = _redis()
    if c is not None:
        try:
            import json
            k = _PREFIX + "i:" + str(cid)
            raw = c.get(k)
            if raw:
                rec = json.loads(raw)
                if not rec.get("done"):
                    rec["chars"] = int(chars)
                    rec["updated"] = time.time()
                    c.setex(k, RUNNING_TTL, json.dumps(rec, ensure_ascii=False))
            return
        except Exception:
            pass
    with _lock:
        it = _items.get(cid)
        if it and not it.get("done"):
            it["chars"] = int(chars)
            it["updated"] = time.time()


def end(cid, ok: bool = True, chars: int | None = None,
        error: str | None = None, ptok: int = 0, ctok: int = 0,
        gen_ms: int = 0) -> None:
    ptok, ctok, gen_ms = int(ptok or 0), int(ctok or 0), int(gen_ms or 0)
    now = time.time()
    c = _redis()
    if c is not None:
        try:
            import json
            k = _PREFIX + "i:" + str(cid)
            raw = c.get(k)
            rec = json.loads(raw) if raw else {"id": str(cid), "kind": "llm",
                                               "model": "", "backend": "", "label": "",
                                               "started": now, "chars": 0}
            rec["done"] = True
            rec["ok"] = bool(ok)
            rec["finished"] = now
            rec["updated"] = now
            if chars is not None:
                rec["chars"] = int(chars)
            rec["ptok"], rec["ctok"], rec["gen_ms"] = ptok, ctok, gen_ms
            if error:
                rec["error"] = str(error)[:200]
            c.setex(k, int(FINISHED_TTL), json.dumps(rec, ensure_ascii=False))
            if not ok:
                c.incr(_PREFIX + "errors")
            if ptok:
                c.incrby(_PREFIX + "ptok", ptok)
            if ctok:
                c.incrby(_PREFIX + "ctok", ctok)
                # для средней скорости: эффективное время генерации
                eff = gen_ms if gen_ms > 0 else int((now - rec.get("started", now)) * 1000)
                if eff > 0:
                    c.incrby(_PREFIX + "genms", eff)
            return
        except Exception:
            pass
    with _lock:
        it = _items.get(cid)
        if not it:
            return
        it["done"] = True
        it["ok"] = bool(ok)
        it["finished"] = now
        it["updated"] = now
        if chars is not None:
            it["chars"] = int(chars)
        it["ptok"], it["ctok"], it["gen_ms"] = ptok, ctok, gen_ms
        if error:
            it["error"] = str(error)[:200]
        if not ok:
            _totals["errors"] += 1
        _totals["ptok"] += ptok
        _totals["ctok"] += ctok
        if ctok:
            eff = gen_ms if gen_ms > 0 else int((now - it.get("started", now)) * 1000)
            if eff > 0:
                _totals["genms"] += eff


# --------------------------------------------------------------------------- #
#  Чтение                                                                      #
# --------------------------------------------------------------------------- #
def _gc_locked() -> None:
    now = time.time()
    done = [v for v in _items.values() if v.get("done")]
    done.sort(key=lambda x: x.get("finished", 0))
    keep_done = set(id(v) for v in done[-KEEP_RECENT:])
    drop = []
    for k, v in _items.items():
        if v.get("done") and id(v) not in keep_done \
                and now - v.get("finished", now) > FINISHED_TTL:
            drop.append(k)
    for k in drop:
        _items.pop(k, None)
    if len(_items) > MAX_ITEMS:
        order = sorted(_items.values(),
                       key=lambda x: (not x.get("done"), x.get("updated", 0)))
        for v in order[: len(_items) - MAX_ITEMS]:
            _items.pop(v["id"], None)


def _view(it: dict) -> dict:
    end_t = it.get("finished") if it.get("done") else time.time()
    elapsed_ms = int((end_t - it.get("started", end_t)) * 1000)
    ctok = int(it.get("ctok", 0) or 0)
    gen_ms = int(it.get("gen_ms", 0) or 0) or elapsed_ms
    tps = round(ctok / (gen_ms / 1000.0), 1) if (ctok > 0 and gen_ms > 0) else None
    return {
        "id": it["id"], "kind": it.get("kind", "llm"), "model": it.get("model", ""),
        "backend": it.get("backend", ""), "label": it.get("label", ""),
        "done": it.get("done", False), "ok": it.get("ok"), "error": it.get("error"),
        "chars": it.get("chars", 0),
        "ptok": it.get("ptok", 0), "ctok": ctok, "tps": tps,
        "elapsed_ms": elapsed_ms,
    }


def snapshot(limit: int = 60) -> dict:
    c = _redis()
    if c is not None:
        try:
            import json
            items = []
            for k in c.scan_iter(_PREFIX + "i:*", count=200):
                raw = c.get(k)
                if raw:
                    try:
                        items.append(json.loads(raw))
                    except Exception:
                        pass
            calls = int(c.get(_PREFIX + "calls") or 0)
            errors = int(c.get(_PREFIX + "errors") or 0)
            ptok = int(c.get(_PREFIX + "ptok") or 0)
            ctok = int(c.get(_PREFIX + "ctok") or 0)
            genms = int(c.get(_PREFIX + "genms") or 0)
            return _assemble(items, limit, calls, errors, ptok, ctok, genms)
        except Exception:
            pass
    with _lock:
        _gc_locked()
        items = list(_items.values())
        totals = dict(_totals)
    return _assemble(items, limit, totals["calls"], totals["errors"],
                     totals["ptok"], totals["ctok"], totals["genms"])


def get(cid) -> dict:
    """Полная запись вызова (с полным текстом запроса) по id — для раскрытия строки."""
    cid = str(cid)
    c = _redis()
    if c is not None:
        try:
            import json
            raw = c.get(_PREFIX + "i:" + cid)
            if raw:
                r = json.loads(raw)
                v = _view(r)
                v["prompt"] = r.get("prompt", "")
                return v
        except Exception:
            pass
    with _lock:
        it = _items.get(cid)
        if it:
            v = _view(it)
            v["prompt"] = it.get("prompt", "")
            return v
    return {}


def _assemble(items: list[dict], limit: int, calls: int, errors: int,
              ptok: int = 0, ctok: int = 0, genms: int = 0) -> dict:
    running = [v for v in items if not v.get("done")]
    finished = [v for v in items if v.get("done")]
    running.sort(key=lambda x: x.get("started", 0))
    finished.sort(key=lambda x: x.get("finished", 0), reverse=True)
    views = [_view(v) for v in running] + [_view(v) for v in finished[:limit]]
    avg_tps = round(ctok / (genms / 1000.0), 1) if (ctok > 0 and genms > 0) else None
    return {"items": views, "running": len(running),
            "total_calls": calls, "total_errors": errors,
            "total_prompt_tokens": ptok, "total_completion_tokens": ctok,
            "total_tokens": ptok + ctok, "avg_tps": avg_tps}
