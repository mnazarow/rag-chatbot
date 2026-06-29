"""Реестр «живой» активности чат-системы в реальном времени.

Сюда регистрируются короткоживущие процессы: обработка запросов веб-чата и
Телеграм-бота (поиск → генерация ответа), парсинг справочника и приложенных
файлов и любые другие операции, которые хочется показать на дашборде в разделе
«Текущие запросы». Долгие фоновые задачи (индексация, граф, бенчмарк и т. п.) уже
учитываются в admin_ops (см. `active_jobs`) и добавляются к этому списку в API.

Хранилище — в памяти процесса, потокобезопасное. Завершённые элементы показываются
ещё `FINISHED_TTL` секунд (чтобы было видно, что задача только что закончилась), а
«зависшие» без обновлений дольше `STALE_TTL` секунд автоматически убираются.
"""
from __future__ import annotations
import itertools
import threading
import time

FINISHED_TTL = 8.0     # сек показывать завершённые
STALE_TTL = 120.0      # сек до авто-уборки «зависших» активных
MAX_ITEMS = 300

_lock = threading.Lock()
_items: dict[int, dict] = {}
_counter = itertools.count(1)


def _gc_locked() -> None:
    now = time.time()
    drop = []
    for k, v in _items.items():
        if v.get("done"):
            if now - v.get("finished", now) > FINISHED_TTL:
                drop.append(k)
        elif now - v.get("updated", now) > STALE_TTL:
            drop.append(k)
    for k in drop:
        _items.pop(k, None)
    if len(_items) > MAX_ITEMS:
        # убираем самые старые завершённые, затем самые старые по обновлению
        order = sorted(_items.values(),
                       key=lambda x: (not x.get("done"), x.get("updated", 0)))
        for v in order[: len(_items) - MAX_ITEMS]:
            _items.pop(v["id"], None)


def start(kind: str, label: str, stage: str = "", detail: str = "",
          total: int | None = None) -> int:
    """Зарегистрировать новый процесс. Возвращает id для update/finish."""
    with _lock:
        tid = next(_counter)
        now = time.time()
        _items[tid] = {
            "id": tid, "kind": kind, "label": label or "", "stage": stage or "",
            "detail": detail or "", "started": now, "updated": now,
            "done": False, "ok": None, "finished": None,
            "current": 0, "total": total,
        }
        _gc_locked()
        return tid


def update(tid: int, stage: str | None = None, detail: str | None = None,
           current: int | None = None, total: int | None = None) -> None:
    with _lock:
        it = _items.get(tid)
        if not it:
            return
        if stage is not None:
            it["stage"] = stage
        if detail is not None:
            it["detail"] = detail
        if current is not None:
            it["current"] = current
        if total is not None:
            it["total"] = total
        it["updated"] = time.time()


def finish(tid: int, ok: bool = True, stage: str | None = None,
           detail: str | None = None) -> None:
    with _lock:
        it = _items.get(tid)
        if not it:
            return
        it["done"] = True
        it["ok"] = ok
        it["finished"] = time.time()
        it["updated"] = it["finished"]
        if stage is not None:
            it["stage"] = stage
        if detail is not None:
            it["detail"] = detail


def _view(it: dict) -> dict:
    now = time.time()
    end = it.get("finished") if it.get("done") else now
    pct = None
    if it.get("total"):
        try:
            pct = round(min(100.0, it["current"] * 100.0 / it["total"]), 1)
        except Exception:
            pct = None
    return {
        "id": it["id"], "kind": it["kind"], "label": it["label"],
        "stage": it["stage"], "detail": it["detail"],
        "done": it["done"], "ok": it["ok"],
        "elapsed_ms": int((end - it["started"]) * 1000),
        "current": it.get("current"), "total": it.get("total"), "progress": pct,
    }


def snapshot() -> dict:
    with _lock:
        _gc_locked()
        items = sorted(_items.values(),
                       key=lambda x: (x.get("done"), -x.get("started", 0)))
        views = [_view(i) for i in items]
    active = [v for v in views if not v["done"]]
    by_kind: dict[str, int] = {}
    for v in active:
        by_kind[v["kind"]] = by_kind.get(v["kind"], 0) + 1
    return {"items": views, "active": len(active), "by_kind": by_kind}


class track:
    """Контекстный менеджер: with activity.track('chat', preview) as t: t.update(...)."""

    def __init__(self, kind: str, label: str, stage: str = "", **kw):
        self.kind, self.label, self.stage, self.total = kind, label, stage, kw.get("total")
        self.id = None

    def __enter__(self):
        self.id = start(self.kind, self.label, self.stage, total=self.total)
        return self

    def update(self, **kw):
        update(self.id, **kw)

    def __exit__(self, exc_type, exc, tb):
        finish(self.id, ok=(exc_type is None))
        return False
