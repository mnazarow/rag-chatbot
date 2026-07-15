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


# --- SQLite-фолбэк (общая rag_logs.db) — чтобы очередь была общей без Redis --- #
# vision-вызовы из процесса индексации учитываются в общей очереди через тот же файл
# rag_logs.db (см. procshare.py). Kind: 'active' (занятые слоты) / 'waiting' (ожидающие).
def _sql():
    try:
        import procshare
        return procshare.conn()
    except Exception:
        return None


def _kind_for(key: str) -> str:
    return "active" if key == _ACTIVE else "waiting"


def _sql_add(conn, kind: str, tok: str, ttl: float) -> None:
    conn.execute("INSERT OR REPLACE INTO llm_queue(kind,tok,expire) VALUES(?,?,?)",
                 (kind, tok, time.time() + ttl))


def _sql_rem(conn, kind: str, tok: str) -> None:
    conn.execute("DELETE FROM llm_queue WHERE kind=? AND tok=?", (kind, tok))


def _sql_count(conn, kind: str) -> int:
    conn.execute("DELETE FROM llm_queue WHERE expire < ?", (time.time(),))   # авто-уборка
    r = conn.execute("SELECT COUNT(*) FROM llm_queue WHERE kind=?", (kind,)).fetchone()
    return int(r[0]) if r else 0


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


def _delay() -> float:
    try:
        return float(settings.get("LLM_REQUEST_DELAY") or 0)
    except Exception:
        return 0.0


_pace_lock = threading.Lock()
_pace_next = [0.0]       # локальный «момент, когда можно начать следующий запрос»
# Lua-скрипт для Redis: атомарно резервирует следующий слот старта, spaced by delay.
_PACE_LUA = ("local n=tonumber(redis.call('get',KEYS[1]) or '0') "
             "local now=tonumber(ARGV[1]) local d=tonumber(ARGV[2]) "
             "local start=math.max(now,n) redis.call('set',KEYS[1],start+d) "
             "redis.call('pexpire',KEYS[1],math.ceil((d+1)*1000)) return tostring(start)")


def _pace() -> None:
    """Выдержать минимальную паузу между началами запросов к LLM (общую для всех
    процессов через Redis; иначе — в пределах процесса)."""
    d = _delay()
    if d <= 0:
        return
    now = time.time()
    start = now
    c = _redis()
    if c is not None:
        try:
            start = float(c.eval(_PACE_LUA, 1, "rag:llmq:next", now, d))
        except Exception:
            with _pace_lock:
                start = max(now, _pace_next[0])
                _pace_next[0] = start + d
    else:
        conn = _sql()
        done = False
        if conn is not None:
            try:
                # атомарно резервируем общий «момент старта» (BEGIN IMMEDIATE — блокировка записи)
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("SELECT next FROM llm_queue_pace WHERE k='next'").fetchone()
                n = float(row[0]) if row and row[0] is not None else 0.0
                start = max(now, n)
                conn.execute("INSERT OR REPLACE INTO llm_queue_pace(k,next) VALUES('next',?)",
                             (start + d,))
                conn.execute("COMMIT")
                done = True
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
        if not done:
            with _pace_lock:
                start = max(now, _pace_next[0])
                _pace_next[0] = start + d
    wait = start - time.time()
    if wait > 0:
        time.sleep(min(wait, 60.0))


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
    else:
        conn = _sql()
        if conn is not None:
            try:
                return _sql_count(conn, "active")
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
    else:
        conn = _sql()
        if conn is not None:
            try:
                return _sql_count(conn, "waiting")
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
    else:
        conn = _sql()
        if conn is not None:
            try:
                _sql_add(conn, "active", tok, _HOLD_TTL)
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
    else:
        conn = _sql()
        if conn is not None:
            try:
                _sql_rem(conn, _kind_for(key), tok)
                return
            except Exception:
                pass
    with _lock:
        local.pop(tok, None)


# Атомарный «занять слот, если активных < m». Проверка счётчика и добавление ДОЛЖНЫ быть
# атомарны, иначе всплеск параллельных вызовов (напр. 8 vision-запросов ingest) проскакивает
# лимит (виден эффект «выполняется 3 / 1»). Для Redis — Lua (atomic), для SQLite — транзакция
# BEGIN IMMEDIATE, для памяти процесса — под _lock.
_ACQ_LUA = ("redis.call('zremrangebyscore', KEYS[1], '-inf', ARGV[1]) "
            "local n = redis.call('zcard', KEYS[1]) "
            "if tonumber(n) < tonumber(ARGV[2]) then "
            "redis.call('zadd', KEYS[1], ARGV[3], ARGV[4]) return 1 end return 0")


def _try_acquire(c, tok: str, m: int) -> bool:
    """Атомарно занять слот, если активных < m. True — занято, False — мест нет."""
    now = time.time()
    if c is not None:
        try:
            return int(c.eval(_ACQ_LUA, 1, _ACTIVE, now, m, now + _HOLD_TTL, tok)) == 1
        except Exception:
            pass
    else:
        conn = _sql()
        if conn is not None:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM llm_queue WHERE expire < ?", (now,))
                r = conn.execute("SELECT COUNT(*) FROM llm_queue WHERE kind='active'").fetchone()
                if (int(r[0]) if r else 0) < m:
                    conn.execute("INSERT OR REPLACE INTO llm_queue(kind,tok,expire) VALUES('active',?,?)",
                                 (tok, now + _HOLD_TTL))
                    conn.execute("COMMIT")
                    return True
                conn.execute("COMMIT")
                return False
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
    with _lock:                                 # локально в пределах процесса — атомарно
        _prune_local(_local_active)
        if len(_local_active) < m:
            _local_active[tok] = now + _HOLD_TTL
            return True
    return False


def acquire() -> str:
    """Занять слот к LLM (блокирующе). Возвращает токен для release()."""
    tok = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
    c = _redis()
    m = _limit()
    if m <= 0:
        _add_active(c, tok)              # учитываем для отображения/счётчика
        _pace()                          # пауза между запросами (если задана)
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
        conn = _sql()
        if conn is not None:
            try:
                _sql_add(conn, "waiting", tok, _WAIT_TTL)
            except Exception:
                with _lock:
                    _local_wait[tok] = time.time() + _WAIT_TTL
        else:
            with _lock:
                _local_wait[tok] = time.time() + _WAIT_TTL
    got = False
    try:
        while True:
            m = _limit()
            if m <= 0:
                break                        # лимит сняли на лету — проходим
            if _try_acquire(c, tok, m):      # атомарно заняли слот, если было место
                got = True
                break
            if deadline is not None and time.time() > deadline:
                break               # вышло время ожидания — проходим всё равно
            time.sleep(0.1)
    finally:
        _rem(c, _WAIT, _local_wait, tok)
    if not got:
        _add_active(c, tok)          # лимит снят или таймаут ожидания — учитываем слот принудительно
    _pace()                          # пауза между запросами (если задана)
    return tok


def release(tok: str | None) -> None:
    if not tok:
        return
    # Снимаем слот со ВСЕХ уровней (Redis + общий SQLite + локальный), а не только с
    # «текущего»: acquire мог занять слот на одном уровне (напр. локальном при сбое SQLite),
    # а release — искать на другом, из-за чего токен «подвисал» бы до TTL. Best-effort.
    c = _redis()
    if c is not None:
        try:
            c.zrem(_ACTIVE, tok)
        except Exception:
            pass
    conn = _sql()
    if conn is not None:
        try:
            _sql_rem(conn, "active", tok)
        except Exception:
            pass
    with _lock:
        _local_active.pop(tok, None)


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
            "enabled": m > 0, "timeout": _timeout(), "delay": _delay(),
            # общая очередь есть либо через Redis, либо через общую rag_logs.db (procshare)
            "shared": bool(c) or (_sql() is not None)}
