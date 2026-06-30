"""Реестр запросов к LLM в реальном времени.

Каждый вызов модели (генерация ответа в чате/Телеграме, описание изображения
vision-моделью, служебные вызовы — фильтр запроса, калибровка, обогащение и т. п.)
регистрируется здесь: что за вызов, модель, бэкенд, статус, объём вывода и время.
Дашборд опрашивает снимок и показывает «живые» и недавно завершённые запросы.

Хранилище — в памяти процесса, потокобезопасное. Выполняющиеся показываются всегда,
завершённые — ещё `FINISHED_TTL` секунд (но не меньше `KEEP_RECENT` последних).
"""
from __future__ import annotations
import itertools
import threading
import time

FINISHED_TTL = 30.0     # сек показывать завершённые
KEEP_RECENT = 25        # минимум последних завершённых в снимке
MAX_ITEMS = 400

_lock = threading.Lock()
_items: dict[int, dict] = {}
_counter = itertools.count(1)
_totals = {"calls": 0, "errors": 0}


def _gc_locked() -> None:
    now = time.time()
    done = [v for v in _items.values() if v.get("done")]
    # сортируем завершённые по времени окончания (новые в конце)
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


def begin(kind: str, model: str = "", backend: str = "", label: str = "") -> int:
    """Зарегистрировать начало вызова LLM. Возвращает id для tokens()/end()."""
    with _lock:
        cid = next(_counter)
        now = time.time()
        _items[cid] = {
            "id": cid, "kind": kind or "llm", "model": model or "",
            "backend": backend or "", "label": (label or "")[:160],
            "started": now, "updated": now, "finished": None,
            "done": False, "ok": None, "chars": 0, "error": None,
        }
        _totals["calls"] += 1
        _gc_locked()
        return cid


def tokens(cid: int, chars: int) -> None:
    """Обновить объём уже сгенерированного вывода (символы)."""
    with _lock:
        it = _items.get(cid)
        if it and not it.get("done"):
            it["chars"] = int(chars)
            it["updated"] = time.time()


def end(cid: int, ok: bool = True, chars: int | None = None,
        error: str | None = None) -> None:
    with _lock:
        it = _items.get(cid)
        if not it:
            return
        it["done"] = True
        it["ok"] = bool(ok)
        it["finished"] = time.time()
        it["updated"] = it["finished"]
        if chars is not None:
            it["chars"] = int(chars)
        if error:
            it["error"] = str(error)[:200]
        if not ok:
            _totals["errors"] += 1


def _view(it: dict) -> dict:
    end_t = it.get("finished") if it.get("done") else time.time()
    return {
        "id": it["id"], "kind": it["kind"], "model": it["model"],
        "backend": it["backend"], "label": it["label"],
        "done": it["done"], "ok": it["ok"], "error": it.get("error"),
        "chars": it.get("chars", 0),
        "elapsed_ms": int((end_t - it["started"]) * 1000),
    }


def snapshot(limit: int = 60) -> dict:
    with _lock:
        _gc_locked()
        items = list(_items.values())
        totals = dict(_totals)
    running = [v for v in items if not v.get("done")]
    finished = [v for v in items if v.get("done")]
    running.sort(key=lambda x: x.get("started", 0))                 # старые сверху
    finished.sort(key=lambda x: x.get("finished", 0), reverse=True)  # свежие сверху
    views = [_view(v) for v in running] + [_view(v) for v in finished[:limit]]
    return {"items": views, "running": len(running),
            "total_calls": totals["calls"], "total_errors": totals["errors"]}
