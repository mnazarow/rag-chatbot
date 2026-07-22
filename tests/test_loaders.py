"""Юнит-тесты определения кодировки в loaders._read_text_any.

Порядок декодирования: charset_normalizer → utf-8-sig → cp1251 → latin-1 →
utf-8(replace). На коротких байтовых фикстурах детектор charset_normalizer
ненадёжен, поэтому для проверки детерминированной фолбэк-цепочки его временно
нейтрализуем (from_bytes().best() → None). Проверяем, что кириллица не «молча»
ломается на cp1251 (ради чего хелпер и существует).
"""

import sys
import types
from pathlib import Path

import loaders


def _write(tmp, name, data: bytes) -> Path:
    p = Path(tmp) / name
    p.write_bytes(data)
    return p


class _NoDetect:
    def best(self):
        return None


def _neutralize_detector():
    """Подменить charset_normalizer так, чтобы сработала внутренняя фолбэк-цепочка."""
    fake = types.ModuleType("charset_normalizer")
    fake.from_bytes = lambda raw: _NoDetect()
    old = sys.modules.get("charset_normalizer")
    sys.modules["charset_normalizer"] = fake
    return old


def _restore_detector(old):
    if old is not None:
        sys.modules["charset_normalizer"] = old
    else:
        sys.modules.pop("charset_normalizer", None)


def test_read_utf8(tmp_path=None):
    import tempfile

    tmp = tmp_path or tempfile.mkdtemp()
    p = _write(tmp, "u.txt", "Привет, мир".encode())
    assert loaders._read_text_any(p) == "Привет, мир"


def test_read_utf8_sig_bom_stripped_or_kept():
    import tempfile

    tmp = tempfile.mkdtemp()
    # BOM + текст: utf-8-sig декодирует без ошибки; кириллица целая
    p = _write(tmp, "b.txt", b"\xef\xbb\xbf" + "Текст с BOM".encode())
    out = loaders._read_text_any(p)
    assert "Текст с BOM" in out


def test_read_cp1251_fallback():
    import tempfile

    tmp = tempfile.mkdtemp()
    p = _write(tmp, "c.txt", "Договор №5 от мая".encode("cp1251"))
    old = _neutralize_detector()
    try:
        out = loaders._read_text_any(p)
    finally:
        _restore_detector(old)
    # ключевое: cp1251-кириллица восстановлена корректно, а не превращена в мусор
    assert out == "Договор №5 от мая"


def test_read_never_raises_on_garbage():
    import tempfile

    tmp = tempfile.mkdtemp()
    p = _write(tmp, "g.bin", bytes(range(0, 256)))
    # финальный фолбэк utf-8(errors='replace') — функция обязана вернуть строку
    out = loaders._read_text_any(p)
    assert isinstance(out, str)
