"""Меж-процессный обмен состоянием через общую rag_logs.db (когда Redis выключен).

Индексация идёт отдельным процессом (`ingest.py`), и описание картинок vision-моделью
происходит именно там. Чтобы такие вызовы были ВИДНЫ на дашборде и УЧИТЫВАЛИСЬ в общей
очереди к LLM без Redis, состояние можно разделять через уже общий файл `rag_logs.db`
(тот же, что и журнал запросов — он смонтирован/доступен обоим процессам).

Модуль отдаёт кэшированное на процесс SQLite-соединение (WAL + busy_timeout) и создаёт
нужные таблицы. Сами операции чтения/записи делают `llm_activity` и `llm_queue`.

Соединение — по одному на поток (sqlite3-объект нельзя делить между потоками). Ошибки
глушатся: при любой проблеме модули откатываются на состояние в памяти процесса.
"""
from __future__ import annotations
import sqlite3
import threading
from pathlib import Path

# Тот же путь, что и у журнала запросов (db.py DB_PATH) — общий для веб-процесса и ingest.py
DB_PATH = Path(__file__).resolve().parent / "rag_logs.db"

_local = threading.local()
_disabled = False   # ставится в True, если фолбэк явно выключен конфигом


def enabled() -> bool:
    """Разрешён ли SQLite-фолбэк (config.PROC_SHARE_SQLITE, по умолчанию да)."""
    if _disabled:
        return False
    try:
        import config
        return bool(getattr(config, "PROC_SHARE_SQLITE", True))
    except Exception:
        return True


def _ensure_schema(c: sqlite3.Connection) -> None:
    # Список вызовов LLM (реестр реального времени) — см. llm_activity.py
    c.execute("""CREATE TABLE IF NOT EXISTS llm_activity(
        id TEXT PRIMARY KEY, data TEXT NOT NULL,
        started REAL, finished REAL, done INTEGER DEFAULT 0, updated REAL)""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_llmact_done ON llm_activity(done, finished)")
    c.execute("""CREATE TABLE IF NOT EXISTS llm_activity_totals(
        name TEXT PRIMARY KEY, val INTEGER DEFAULT 0)""")
    # Слоты очереди к LLM (active/waiting) с истечением — см. llm_queue.py
    c.execute("""CREATE TABLE IF NOT EXISTS llm_queue(
        kind TEXT, tok TEXT, expire REAL, PRIMARY KEY(kind, tok))""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_llmq_kind ON llm_queue(kind, expire)")
    # Пейсинг (минимальный интервал между началами запросов) — единый «момент старта»
    c.execute("""CREATE TABLE IF NOT EXISTS llm_queue_pace(k TEXT PRIMARY KEY, next REAL)""")


def conn() -> sqlite3.Connection | None:
    """Кэшированное на поток соединение к rag_logs.db или None, если фолбэк недоступен."""
    if not enabled():
        return None
    c = getattr(_local, "conn", None)
    if c is not None:
        return c
    try:
        # isolation_level=None → автокоммит (нам нужны немедленные, атомарные операции)
        c = sqlite3.connect(DB_PATH, timeout=5.0, isolation_level=None,
                            check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL")     # конкурентные читатели + один писатель
        c.execute("PRAGMA busy_timeout=5000")    # ждать снятия блокировки, а не падать
        c.execute("PRAGMA synchronous=NORMAL")
        _ensure_schema(c)
        _local.conn = c
        return c
    except Exception:
        return None


def available() -> bool:
    """Доступен ли общий SQLite-канал (для флага «shared» на дашборде)."""
    return conn() is not None
