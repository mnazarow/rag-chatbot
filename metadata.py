"""Извлечение метаданных из файла — дёшево, без LLM (rule-based).

Метаданные кладутся в payload Qdrant и используются для фильтрации:
  - doc_category: price | presentation | training | document
  - title: имя файла без расширения
  - date: YYYY-MM (из имени файла, иначе из mtime)
Опционально можно включить LLM-обогащение (продукт/тема) — см. enrich_with_llm.
"""
from __future__ import annotations
import re
import time
from pathlib import Path

# ключевые слова -> категория (по имени файла и пути)
_CATEGORY_KEYWORDS = {
    "price": ["прайс", "price", "прейскурант", "стоимост", "тариф", "ценник"],
    "presentation": ["презентац", "presentation", "slide", "слайд", "deck", "питч"],
    "training": ["обучен", "тренинг", "training", "вебинар", "webinar",
                 "курс", "урок", "lesson", "онбординг", "onboarding", "инструктаж"],
}
_AV_EXT = {".mp3", ".wav", ".m4a", ".aac", ".mp4", ".mov", ".mkv", ".webm"}
_TABLE_EXT = {".xlsx", ".xls", ".csv"}

# даты в имени файла: 2024, 2024-05, 05.2024, 12.03.2024
_DATE_PATTERNS = [
    (re.compile(r"(20\d{2})[-_.](\d{1,2})"), lambda m: f"{m.group(1)}-{int(m.group(2)):02d}"),
    (re.compile(r"(\d{1,2})[.\-_](\d{1,2})[.\-_](20\d{2})"), lambda m: f"{m.group(3)}-{int(m.group(2)):02d}"),
    (re.compile(r"\b(20\d{2})\b"), lambda m: f"{m.group(1)}-01"),
]


def _category(path: Path) -> str:
    ext = path.suffix.lower()
    name = (str(path)).lower()
    for cat, kws in _CATEGORY_KEYWORDS.items():
        if any(kw in name for kw in kws):
            return cat
    # фолбэк по типу файла
    if ext in _TABLE_EXT:
        return "price"
    if ext == ".pptx":
        return "presentation"
    if ext in _AV_EXT:
        return "training"
    return "document"


def _date(path: Path) -> str:
    name = path.name
    for pat, fmt in _DATE_PATTERNS:
        m = pat.search(name)
        if m:
            return fmt(m)
    # из времени модификации файла (если файла нет на диске — текущий месяц)
    try:
        return time.strftime("%Y-%m", time.localtime(path.stat().st_mtime))
    except Exception:
        return time.strftime("%Y-%m")


def extract(path: Path) -> dict:
    """Базовые метаданные файла."""
    return {
        "doc_category": _category(path),
        "title": path.stem,
        "date": _date(path),
    }


# ----- опционально: LLM-обогащение (продукт/тема). Дороже, по флагу. -----
def enrich_with_llm(text_sample: str, ollama_url: str, model: str) -> dict:
    """Вернуть {'topic': ...} по фрагменту текста. Используйте выборочно."""
    import httpx
    prompt = (
        "Определи краткую тему документа (2-4 слова, по-русски) по фрагменту. "
        "Ответь только темой.\n\nФрагмент:\n" + text_sample[:1500]
    )
    try:
        r = httpx.post(f"{ollama_url}/api/generate", timeout=60,
                       json={"model": model, "prompt": prompt, "stream": False,
                             "options": {"temperature": 0}})
        return {"topic": r.json().get("response", "").strip()[:60]}
    except Exception:
        return {"topic": ""}
