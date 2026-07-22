"""Юнит-тесты разбора умных фильтров (query_filters) и расширения синонимами.

query_filters.extract вызывает LLM — подменяем llm_backend.chat детерминированной
строкой и проверяем ТОЛЬКО парсинг/валидацию (без сети).

synonyms в conftest подменён облегчённой заглушкой (чтобы не тянуть БД в тестах
промптов), поэтому реальный модуль загружаем отдельной копией из файла с фейковым
`db` — проверяем чистую логику expand_query/matched_groups.
"""

import importlib.util
import os
import sys
import types

import llm_backend
import query_filters

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------- query_filters -------------------------------
def _with_llm(reply, fn):
    orig = llm_backend.chat
    llm_backend.chat = lambda *a, **k: reply
    try:
        return fn()
    finally:
        llm_backend.chat = orig


def test_extract_valid_fields():
    out = _with_llm(
        '{"doc_category":"price","product":"  Pro ","date":"2024-05-x"}',
        lambda: query_filters.extract("сколько стоит Pro в мае 2024"),
    )
    assert out == {"doc_category": "price", "product": "Pro", "date": "2024-05"}


def test_extract_invalid_category_dropped():
    out = _with_llm('{"doc_category":"nonsense"}', lambda: query_filters.extract("вопрос"))
    assert "doc_category" not in out


def test_extract_bad_date_dropped():
    out = _with_llm('{"date":"вчера"}', lambda: query_filters.extract("вопрос"))
    assert "date" not in out


def test_extract_no_json_returns_empty():
    assert _with_llm("тут нет json", lambda: query_filters.extract("q")) == {}


def test_extract_empty_object():
    assert _with_llm("{}", lambda: query_filters.extract("q")) == {}


def test_extract_llm_failure_safe():
    orig = llm_backend.chat

    def boom(*a, **k):
        raise RuntimeError("llm down")

    llm_backend.chat = boom
    try:
        assert query_filters.extract("q") == {}
    finally:
        llm_backend.chat = orig


# --------------------------------- synonyms ----------------------------------
def _load_real_synonyms(rows, enabled="1"):
    """Загрузить настоящий модуль synonyms из файла с фейковым db (обход заглушки)."""
    fake_db = types.ModuleType("db")
    fake_db.kv_get = lambda k: enabled
    fake_db.kv_set = lambda k, v: None
    fake_db.syn_list = lambda: rows
    old = sys.modules.get("db")
    sys.modules["db"] = fake_db
    try:
        spec = importlib.util.spec_from_file_location("synonyms_real", ROOT + "/synonyms.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if old is not None:
            sys.modules["db"] = old
        else:
            sys.modules.pop("db", None)
    return mod


_ROWS = [
    {"id": 1, "term": "авто", "syns": ["машина", "автомобиль"]},
    {"id": 2, "term": "один", "syns": []},
]  # группа <2 членов — игнорируется


def test_synonyms_groups_need_two_members():
    syn = _load_real_synonyms(_ROWS)
    groups = syn._groups()
    assert groups == [["авто", "машина", "автомобиль"]]


def test_synonyms_expand_adds_missing_members():
    syn = _load_real_synonyms(_ROWS)
    out = syn.expand_query("нужна машина срочно")
    assert "авто" in out and "автомобиль" in out and out.startswith("нужна машина")


def test_synonyms_expand_idempotent_when_all_present():
    syn = _load_real_synonyms(_ROWS)
    text = "машина автомобиль авто"
    assert syn.expand_query(text) == text


def test_synonyms_expand_noop_without_match():
    syn = _load_real_synonyms(_ROWS)
    assert syn.expand_query("посторонний текст") == "посторонний текст"


def test_synonyms_disabled_returns_input():
    syn = _load_real_synonyms(_ROWS, enabled="0")
    assert syn.expand_query("нужна машина") == "нужна машина"
    assert syn.matched_groups("нужна машина") == []


def test_synonyms_matched_groups():
    syn = _load_real_synonyms(_ROWS)
    assert syn.matched_groups("где моё авто") == [["авто", "машина", "автомобиль"]]
