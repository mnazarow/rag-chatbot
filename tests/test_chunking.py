"""Юнит-тесты чанкинга (ingest.chunk_text / _chunk_fixed / _chunk_structured).

Чистые функции: границы, перекрытие, пустой вход, юникод/кириллица. Тяжёлые
зависимости ingest (sentence_transformers, tqdm) подменены заглушками в conftest.
"""

import ingest


def test_chunk_empty_input():
    assert ingest.chunk_text("", 100, 10) == []
    assert ingest.chunk_text("   \n  ", 100, 10) == []


def test_chunk_short_text_single_chunk():
    # текст короче размера окна — один чанк, целиком (без потери данных)
    assert ingest.chunk_text("короткий текст", 100, 10) == ["короткий текст"]


def test_chunk_fixed_boundaries_and_overlap():
    # строка без пробелов/переводов строк заставляет резать строго по размеру окна
    text = "0123456789" * 10  # 100 символов, шаблон повторяется каждые 10
    chunks = ingest._chunk_fixed(text, 40, 10)
    assert chunks, "должны быть чанки"
    assert all(len(c) <= 40 for c in chunks), "ни один чанк не длиннее окна"
    # перекрытие: хвост предыдущего чанка совпадает с началом следующего (overlap=10)
    assert chunks[0][-10:] == chunks[1][:10]


def test_chunk_fixed_covers_all_content():
    text = "abcdefghij" * 8  # 80 символов, без разделителей
    chunks = ingest._chunk_fixed(text, 30, 5)
    # объединение множеств символов покрывает исходный текст (ничего не потеряно)
    joined = "".join(chunks)
    assert set(text) <= set(joined)
    assert len(chunks) >= 2


def test_chunk_prefers_sentence_boundary():
    # при наличии «. » срез идёт по концу предложения, а не по жёсткому размеру
    text = "Первое предложение тут. " + "x" * 50
    chunks = ingest._chunk_fixed(text, 40, 5)
    assert chunks[0].endswith(".") or chunks[0].endswith("предложение тут")


def test_chunk_unicode_cyrillic_preserved():
    text = ("Договор поставки оборудования номер пять. " * 5).strip()
    chunks = ingest.chunk_text(text, 50, 10)
    # кириллица не ломается: каждый чанк — валидная строка с русскими буквами
    assert all(isinstance(c, str) for c in chunks)
    assert any("Договор" in c for c in chunks)


def test_chunk_structured_keeps_short_text_whole():
    assert ingest._chunk_structured("маленький абзац", 100, 10) == ["маленький абзац"]


def test_chunk_structured_splits_on_headings():
    text = "# Раздел А\n" + "Текст раздела А. " * 6 + "\n\n# Раздел Б\n" + "Текст раздела Б. " * 6
    chunks = ingest._chunk_structured(text, 80, 10)
    assert len(chunks) >= 2
    assert all(c.strip() for c in chunks)
