"""Очередь к LLM: ограничение числа одновременных запросов к модели.

Когда модель/GPU перегружены, параллельные запросы только замедляют друг друга и
рискуют упасть по таймауту. Этот модуль пропускает к модели не более
`LLM_MAX_CONCURRENCY` запросов одновременно; остальные ждут в очереди (не дольше
`LLM_QUEUE_TIMEOUT` секунд). 0 — без ограничения (очередь выключена).

Гейт ОБЩИЙ между процессами: индексация идёт отдельным процессом (`ingest.py`), и
описание картинок vision-моделью должно учитываться в общей очереди вместе с чатом
и Телеграмом. Для этого активные/ждущие слоты хранятся в Redis (sorted set с TTL —
самоочищается, если процесс упал). Без Redis — счётчики в памяти текущего процесса.

acquire() возвращает токен, который нужно передать в release().
"""
from __future__ import annotations
import os
import threading
import time
import uuid

import settings

_ACTIVE = "rag:llmq:active"     # zset: member=token, score=срок годности (ts)
_WAIT = "rag:llmq:waiting"      # zset: member=token, score=срок годности (ts)
_HOLD_TTL = 1800                # сек: макс. удержание слота (safety от утечки)
_WAIT_TTL = 300

_lock = threading.Lock()
_local_active: dict[str, float] = {}
_local_wait: dict[str, float] = {}


def _redis():
    try:
        import cache
        return cache.client()
    except Exception:
        return None


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


def _prune_local(d: dict) -> None:
    now = time.time()
    for k in [k for k, v in d.items() if v <= now]:
        d.pop(k, None)


def _active_count(c) -> int:
    now = time.time()
    if c is not None:
        try:
            c.zremrangebyscore(_ACTIVE, "-inf", now)
            return int(c.zcard(_ACTIVE) or 0)
        except Exception:
            pass
    with _lock:
        _prune_local(_local_active)
        return len(_local_active)


def _waiting_count(c) -> int:
    now = time.time()
    if c is not None:
        try:
            c.zremrangebyscore(_WAIT, "-inf", now)
            return int(c.zcard(_WAIT) or 0)
        except Exception:
            pass
    with _lock:
        _prune_local(_local_wait)
        return len(_local_wait)


def _add_active(c, tok: str) -> None:
    if c is not None:
        try:
            c.zadd(_ACTIVE, {tok: time.time() + _HOLD_TTL})
            return
        except Exception:
            pass
    with _lock:
        _local_active[tok] = time.time() + _HOLD_TTL


def _rem(c, key: str, local: dict, tok: str) -> None:
    if c is not None:
        try:
            c.zrem(key, tok)
            return
        except Exception:
            pass
    with _lock:
        local.pop(tok, None)


def acquire() -> str:
    """Занять слот к LLM (блокирующе). Возвращает токен для release()."""
    tok = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
    c = _redis()
    m = _limit()
    if m <= 0:
        _add_active(c, tok)              # учитываем для отображения/счётчика
        return tok
    deadline = None
    to = _timeout()
    if to and to > 0:
        deadline = time.time() + to
    # отметимся как ожидающие
    if c is not None:
        try:
            c.zadd(_WAIT, {tok: time.time() + _WAIT_TTL})
        except Exception:
            with _lock:
                _local_wait[tok] = time.time() + _WAIT_TTL
    else:
        with _lock:
            _local_wait[tok] = time.time() + _WAIT_TTL
    try:
        while True:
            m = _limit()
            if m <= 0:
                break
            if _active_count(c) < m:
                break
            if deadline is not None and time.time() > deadline:
                break               # вышло время ожидания — проходим всё равно
            time.sleep(0.1)
    finally:
        _rem(c, _WAIT, _local_wait, tok)
    _add_active(c, tok)
    return tok


def release(tok: str | None) -> None:
    if not tok:
        return
    c = _redis()
    _rem(c, _ACTIVE, _local_active, tok)


class slot:
    """Контекстный менеджер: with llm_queue.slot(): <вызов LLM>."""

    def __enter__(self):
        self.tok = acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        release(self.tok)
        return False


def stats() -> dict:
    c = _redis()
    m = _limit()
    return {"max": m, "running": _active_count(c), "waiting": _waiting_count(c),
            "enabled": m > 0, "timeout": _timeout(), "shared": bool(c)}
