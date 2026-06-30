"""Очередь к LLM: ограничение числа одновременных запросов к модели.

Когда модель/GPU перегружены, параллельные запросы только замедляют друг друга и
рискуют упасть по таймауту. Этот модуль пропускает к модели не более
`LLM_MAX_CONCURRENCY` запросов одновременно; остальные ждут в очереди (не дольше
`LLM_QUEUE_TIMEOUT` секунд). 0 — без ограничения (очередь выключена).

Гейт общий для синхронного (`chat`) и асинхронного (`chat_stream`) путей и для
описания изображений vision-моделью. Реализован на Condition, чтобы лимит можно
было менять «на лету» из админки. Состояние (running/waiting/max) показывается на
дашборде в блоке «Запросы к LLM».
"""
from __future__ import annotations
import threading
import time

import settings

_cond = threading.Condition()
_running = 0
_waiting = 0
_peak_wait = 0


def _limit() -> int:
    try:
        return int(settings.get("LLM_MAX_CONCURRENCY") or 0)
    except Exception:
        return 0


def _timeout() -> float:
    try:
        return float(settings.get("LLM_QUEUE_TIMEOUT") or 0)
    except Exception:
        return 0.0


def acquire() -> None:
    """Занять слот к LLM (блокирующе). Если лимит 0 — проходит сразу."""
    global _running, _waiting
    m = _limit()
    if m <= 0:
        with _cond:
            _running += 1
        return
    deadline = None
    to = _timeout()
    if to and to > 0:
        deadline = time.time() + to
    with _cond:
        _waiting += 1
        try:
            while _running >= max(1, _limit()):
                if _limit() <= 0:      # лимит сняли «на лету» — выходим
                    break
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break          # вышло время ожидания — пропускаем (не висим)
                _cond.wait(timeout=min(0.5, remaining) if remaining else 0.5)
        finally:
            _waiting -= 1
            _running += 1


def release() -> None:
    global _running
    with _cond:
        _running = max(0, _running - 1)
        _cond.notify()


class slot:
    """Контекстный менеджер: with llm_queue.slot(): <вызов LLM>."""

    def __enter__(self):
        acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        release()
        return False


def stats() -> dict:
    with _cond:
        return {"max": _limit(), "running": _running, "waiting": _waiting,
                "enabled": _limit() > 0, "timeout": _timeout()}
