"""Юнит-тесты журнала на SQLite in-file (db).

sqlite реально доступен в среде — заглушка не нужна. Каждый тест работает с
собственной свежей временной БД: подменяем db.DB_PATH и сбрасываем thread-local
кэш соединения перед init(). Диалект форсируем в 'sqlite' (не зависим от настроек
DB_BACKEND).
"""

import tempfile
import threading
from pathlib import Path

import db


def _fresh_db():
    """Новая пустая SQLite-база + инициализированная схема; возвращает модуль db."""
    tmp = Path(tempfile.mkdtemp()) / "t.db"
    db.DB_PATH = tmp
    db._local = threading.local()  # сбросить закэшованное соединение потока
    db.init("sqlite")
    return db


def test_init_creates_tables_kv_and_requests():
    d = _fresh_db()
    # таблицы существуют: kv_set/kv_get и запись в requests не падают
    d.kv_set("k", "v")
    assert d.kv_get("k") == "v"
    rid = d.log_request("q", None, 0, 0.0, 1, 0, False, [])
    assert isinstance(rid, int) and rid > 0


def test_kv_get_missing_returns_none():
    d = _fresh_db()
    assert d.kv_get("no_such_key") is None


def test_kv_set_overwrites():
    d = _fresh_db()
    d.kv_set("x", "1")
    d.kv_set("x", "2")
    assert d.kv_get("x") == "2"


def test_log_request_appears_in_recent():
    d = _fresh_db()
    rid = d.log_request(
        "Сколько стоит?",
        "price",
        3,
        0.87,
        120,
        42,
        True,
        [{"source": "prices.xlsx", "page": 1}],
        session_id="s1",
        answer="Три года.",
    )
    rows = d.recent(10)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == rid
    assert row["question"] == "Сколько стоит?"
    assert row["category"] == "price"
    assert row["answered"] == 1
    # sources сериализуются/десериализуются как JSON-список
    assert row["sources"] == [{"source": "prices.xlsx", "page": 1}]


def test_recent_orders_newest_first_and_limits():
    d = _fresh_db()
    for i in range(5):
        d.log_request(f"q{i}", None, 0, 0.0, 1, 0, True, [])
    rows = d.recent(3)
    assert len(rows) == 3
    # DESC по id: последний вставленный — первым
    assert rows[0]["question"] == "q4"


def test_init_idempotent_preserves_data():
    d = _fresh_db()
    d.kv_set("keep", "yes")
    d.init("sqlite")  # повторный init не должен ронять/чистить данные
    d.init("sqlite")  # и повторные миграции колонок идемпотентны
    assert d.kv_get("keep") == "yes"
