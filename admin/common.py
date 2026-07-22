"""Общие хелперы административной подсистемы (вынесено из admin_ops.py).

Здесь только чистые функции и данные без тяжёлых зависимостей, чтобы устранить
дублирование (напр. ``_jlen`` был скопирован дословно в двух местах) и дать
единую точку для наборов расширений файлов.
"""
from __future__ import annotations

import hashlib
import json as _json
import threading
import time
from pathlib import Path

# --- наборы расширений (тип файла → способ извлечения текста) ---
_AV_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".mp4", ".mov", ".mkv", ".webm"}
_UNSUPPORTED_FIX = {
    ".doc": "Сконвертируйте в .docx (Word → «Сохранить как») или установите LibreOffice.",
    ".rtf": "Сконвертируйте в .docx или .txt.",
    ".odt": "Сконвертируйте в .docx.",
    ".pages": "Apple Pages не читается — экспортируйте в PDF/DOCX.",
    ".numbers": "Apple Numbers не читается — экспортируйте в XLSX/CSV.",
    ".key": "Apple Keynote не читается — экспортируйте в PDF/PPTX.",
    ".xlsb": "Бинарный Excel — сохраните как .xlsx или установите pyxlsb.",
    ".epub": "Сконвертируйте в PDF/TXT.",
    ".fb2": "Сконвертируйте в TXT/PDF.",
    ".djvu": "Сконвертируйте в PDF.",
}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".jfif"}
# OCR-форматы (картинки и RAW-фото): индексируются, но при проверке каталога
# их не парсим поштучно — OCR слишком долгий для тысяч файлов
_OCR_EXTS = _IMG_EXTS | {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf",
                         ".rw2", ".orf", ".sr2"}
# Архивы: индексируются (распаковкой), но при проверке не распаковываем — долго
_ARCHIVE_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2"}
# Спец-инструменты извлечения текста (CAD, 3D-обмен, старый .doc, письма, архивы)
_CAD_EXTS = {".dxf", ".dwg", ".stp", ".step", ".igs", ".iges"}
_TOOL_EXTS = _CAD_EXTS | _ARCHIVE_EXTS | {".doc", ".msg"}

# поддерживаемые типы — для подсказки «сколько документов в папке»
_SUPPORTED = {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".xlsm", ".xls", ".csv",
              ".txt", ".md", ".html", ".htm", ".mhtml", ".mht",
              ".xml", ".json", ".url", ".msg", ".svg",
              ".dxf", ".dwg", ".stp", ".step", ".igs", ".iges",
              ".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2",
              ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".jfif",
              ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".rw2", ".orf", ".sr2",
              ".mp3", ".wav", ".m4a", ".aac", ".mp4", ".mov", ".mkv", ".webm"}


def _fmt_bytes(n: int) -> str:
    n = float(n or 0)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024 or unit == "ГБ":
            return f"{n:.0f} {unit}" if unit == "Б" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} ГБ"


def _num(s):
    try:
        s = str(s).strip()
        return float(s) if "." in s else int(s)
    except Exception:
        return None


def _dir_size_mb(path: Path) -> float:
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / 1e6, 1)
    except Exception:
        return 0.0


def _sha256_file(p) -> str | None:
    """SHA-256 файла потоково (без загрузки целиком в память). None при ошибке чтения —
    вызывающий код (дедуп, каталог) корректно обрабатывает отсутствие хэша."""
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _jlen(gdir: Path, name: str, key=None):
    """Длина коллекции внутри JSON-файла графа (vdb_*/kv_store_*). None — нет файла/ошибка.

    Ранее эта функция дословно дублировалась внутри двух функций admin_ops; вынесена сюда.
    ``gdir`` — каталог graph_storage."""
    f = gdir / name
    if not f.exists():
        return None
    try:
        d = _json.loads(f.read_text(encoding="utf-8"))
        v = d.get(key) if key else d
        return len(v) if hasattr(v, "__len__") else None
    except Exception:
        return None


# --- процесс-локальный кэш (обязательный фолбэк, когда Redis выключен/недоступен) ---
_MEM_CACHE: dict = {}
_MEM_LOCK = threading.Lock()


def mem_get_or_set(name: str, ttl: float, producer):
    """Мини-кэш в памяти процесса с TTL. Используется как обязательный кэш там, где
    cache.get_or_set (Redis) при выключенном Redis считает producer() каждый раз."""
    now = time.time()
    with _MEM_LOCK:
        ent = _MEM_CACHE.get(name)
        if ent and ent[0] > now:
            return ent[1]
    val = producer()
    with _MEM_LOCK:
        _MEM_CACHE[name] = (now + ttl, val)
    return val


def mem_invalidate(name: str | None = None) -> None:
    with _MEM_LOCK:
        if name is None:
            _MEM_CACHE.clear()
        else:
            _MEM_CACHE.pop(name, None)


# --- классификация файлов и извлечение сводок (чистые функции, разделяемые доменами) ---
# Раньше жили в admin_ops.py; вынесены сюда, т.к. их используют сразу несколько
# вынесенных доменов (web_crawl, каталог документов) — единая точка без обратной
# зависимости на admin_ops исключает циклический импорт.

def _file_method(ext: str) -> str:
    """Как из файла извлекается текст: транскрибация / OCR / спец-инструмент / прямой."""
    ext = (ext or "").lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext in _AV_EXTS:
        return "transcribed"   # аудио/видео → Whisper
    if ext in _OCR_EXTS:
        return "ocr"           # изображения/RAW-фото → OCR
    if ext in _TOOL_EXTS:
        return "tool"          # DWG/STEP/IGES/.doc/.msg/архивы → спец-парсеры
    return "text"              # PDF/DOCX/XLSX/… → прямое извлечение текста


def _extract_summary(text: str) -> str:
    """Достаёт машиночитаемую строку 'SUMMARY ...' из вывода задачи."""
    for line in reversed((text or "").splitlines()):
        if line.startswith("SUMMARY "):
            return line[len("SUMMARY "):].strip()
        if line.startswith("FATAL:"):
            return line.strip()
    return ""


# путь к статистике времени индексации (rag/ingest_stats.json)
_INGEST_STATS = Path(__file__).resolve().parent.parent / "ingest_stats.json"


def _ingest_stats() -> dict:
    """Время обработки файлов из последней индексации (ingest_stats.json)."""
    if _INGEST_STATS.exists():
        try:
            return _json.loads(_INGEST_STATS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}
