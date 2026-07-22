"""Юнит-тесты извлечения метаданных (metadata): даты и категории.

Логика валидности месяца проверяется на чистом _fmt_ym (детерминированно, без
файловой системы). Для _date используем реальные временные файлы, чтобы фолбэк на
mtime был предсказуем, а не зависел от текущей даты.
"""

import tempfile
from pathlib import Path

import metadata as M


def _f(name: str) -> Path:
    p = Path(tempfile.mkdtemp()) / name
    p.write_text("x", encoding="utf-8")
    return p


# ---- валидность месяца (чистая функция) ----
def test_fmt_ym_valid_month():
    assert M._fmt_ym("2024", 5) == "2024-05"
    assert M._fmt_ym("2024", 12) == "2024-12"
    assert M._fmt_ym("2024", 1) == "2024-01"


def test_fmt_ym_invalid_month_empty():
    assert M._fmt_ym("2024", 13) == ""
    assert M._fmt_ym("2024", 0) == ""


# ---- регэкспы дат ----
def test_date_pattern_year_month():
    pat, fmt = M._DATE_PATTERNS[0]
    assert fmt(pat.search("2024-05")) == "2024-05"
    # невалидный месяц не даёт значения (совпадение пропускается)
    assert fmt(pat.search("2024-13")) == ""


def test_version_not_treated_as_date():
    # «v2023.2» — версия, не дата: перед годом буква → отрицательный lookbehind
    pat = M._DATE_PATTERNS[0][0]
    assert pat.search("v2023.2") is None
    assert pat.search("2023.2") is not None


def test_date_from_filename_ym():
    assert M._date(_f("отчёт-2024-05.txt")) == "2024-05"


def test_date_from_filename_dmy():
    assert M._date(_f("акт-12.03.2024.txt")) == "2024-03"


def test_date_invalid_month_not_that_value():
    # «2024-13» не должно превратиться в «2024-13»
    assert M._date(_f("f-2024-13.txt")) != "2024-13"


# ---- категории ----
def test_category_by_keyword():
    assert M._category(Path("Прайс-лист 2024.pdf")) == "price"
    assert M._category(Path("Презентация продукта.pdf")) == "presentation"
    assert M._category(Path("Вебинар по продажам.pdf")) == "training"


def test_category_by_extension_fallback():
    assert M._category(Path("data.xlsx")) == "price"
    assert M._category(Path("deck.pptx")) == "presentation"
    assert M._category(Path("record.mp4")) == "training"
    assert M._category(Path("readme.txt")) == "document"


def test_extract_shape():
    md = M.extract(_f("Прайс-2024-05.xlsx"))
    assert md["doc_category"] == "price"
    assert md["title"] == "Прайс-2024-05"
    assert md["date"] == "2024-05"
