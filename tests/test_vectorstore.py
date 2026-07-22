"""Юнит-тесты сборки булева выражения Milvus (vectorstore._m_expr).

Проверяем экранирование строк и allow-list имён полей (защита от инъекции в
выражение). Клиент Milvus не требуется — функция чистая. httpx подменён заглушкой
в conftest на уровне импорта модуля.
"""

import vectorstore as V


def test_m_expr_empty():
    assert V._m_expr(None) == ""
    assert V._m_expr({}) == ""


def test_m_expr_string_equality():
    assert V._m_expr({"doc_category": "price"}) == 'doc_category == "price"'


def test_m_expr_bool_and_number():
    assert V._m_expr({"flag": True}) == "flag == true"
    assert V._m_expr({"flag": False}) == "flag == false"
    assert V._m_expr({"n": 5}) == "n == 5"
    assert V._m_expr({"x": 1.5}) == "x == 1.5"


def test_m_expr_multiple_fields_joined_with_and():
    expr = V._m_expr({"doc_category": "price", "year": 2024})
    assert " and " in expr
    assert 'doc_category == "price"' in expr
    assert "year == 2024" in expr


def test_m_expr_escapes_backslash_then_quote():
    # порядок экранирования: сначала обратный слэш, потом кавычка
    assert V._m_expr({"t": 'a"b\\c'}) == 't == "a\\"b\\\\c"'


def test_m_expr_rejects_unsafe_field_names():
    # имя поля не по allow-list (_FIELD_RE) — условие пропускается (анти-инъекция)
    assert V._m_expr({"a; DROP TABLE": "x"}) == ""
    assert V._m_expr({"1bad": "x"}) == ""
    assert V._m_expr({"ok_field": "v", "bad-name": "y"}) == 'ok_field == "v"'


def test_m_expr_skips_none_values():
    assert V._m_expr({"a": None, "b": "v"}) == 'b == "v"'
