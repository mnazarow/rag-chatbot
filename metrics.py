"""Лёгкий реестр обращений к компонентам конвейера (Qdrant, эмбеддер, реранкер,
БД, LightRAG, KAG). Потокобезопасно, в памяти процесса приложения.

Хранит кумулятивные счётчики: число вызовов, ошибок, суммарное время. Из них
дашборд считает скорость (запросов/с) и среднюю задержку в реальном времени —
разностью между двумя опросами. Компоненты, работающие в отдельном процессе
индексации (ingest.py), здесь не учитываются — это метрики «горячего» пути
запросов веб-чата/Телеграма/VoIP.
"""
from __future__ import annotations
import threading
import time

_lock = threading.Lock()
_data: dict = {}   # name -> {"calls", "errors", "total_ms", "last_ts"}


def _slot(name: str) -> dict:
    d = _data.get(name)
    if d is None:
        d = {"calls": 0, "errors": 0, "total_ms": 0.0, "last_ts": 0.0}
        _data[name] = d
    return d


def record(name: str, ms: float = 0.0, ok: bool = True) -> None:
    """Зафиксировать один вызов компонента name с длительностью ms (мс)."""
    with _lock:
        d = _slot(name)
        d["calls"] += 1
        if not ok:
            d["errors"] += 1
        d["total_ms"] += float(ms or 0.0)
        d["last_ts"] = time.time()


class timer:
    """Контекст-менеджер замера: `with metrics.timer("qdrant"): ...`."""

    __slots__ = ("name", "_t")

    def __init__(self, name: str):
        self.name = name
        self._t = 0.0

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        record(self.name, (time.perf_counter() - self._t) * 1000.0, exc_type is None)
        return False


def snapshot() -> dict:
    """Копия кумулятивных счётчиков по всем компонентам + метка времени сервера."""
    with _lock:
        comps = {k: dict(v) for k, v in _data.items()}
    for v in comps.values():
        v["avg_ms"] = round(v["total_ms"] / v["calls"], 1) if v["calls"] else 0.0
    return {"ts": time.time(), "components": comps}


def reset() -> None:
    with _lock:
        _data.clear()
