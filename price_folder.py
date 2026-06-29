"""Прайс-папка: контекст для «ценовых» вопросов напрямую из указанной папки.

Идея: если вопрос про цену/стоимость/прайс, ответ строится по файлам из отдельной
папки с прайс-листами — БЕЗ индексации в Qdrant. Файлы читаются по требованию
(с кэшем по mtime), нарезаются на фрагменты и реранкуются под конкретный вопрос
кросс-энкодером; лучшие фрагменты добавляются в контекст ответа. Так цены всегда
актуальны (не нужно переиндексировать при смене прайса) и точны для табличных
справок, где векторный поиск слабее.

Включается настройкой PRICE_FOLDER, папка — PRICE_DIR, число фрагментов — PRICE_TOP_K.
"""
from __future__ import annotations
import os
import threading
from pathlib import Path

import settings

_SUPPORTED = {".xlsx", ".xls", ".csv", ".tsv", ".pdf", ".docx", ".doc",
              ".txt", ".md", ".rtf", ".ods"}
_MAX_CHUNKS = 8000          # страховка от слишком больших папок

_lock = threading.Lock()
_cache = {"sig": None, "items": []}


def enabled() -> bool:
    return bool(settings.get("PRICE_FOLDER")) and bool((settings.get("PRICE_DIR") or "").strip())


def folder() -> Path | None:
    p = (settings.get("PRICE_DIR") or "").strip()
    return Path(p).expanduser() if p else None


def _files() -> list[Path]:
    d = folder()
    if not d or not d.is_dir():
        return []
    out = []
    for f in sorted(d.rglob("*")):
        if f.is_file() and f.suffix.lower() in _SUPPORTED:
            out.append(f)
    return out


def _signature(files: list[Path]) -> str:
    parts = []
    for f in files:
        try:
            st = f.stat()
            parts.append(f"{f}:{int(st.st_mtime)}:{st.st_size}")
        except Exception:
            continue
    return "|".join(parts)


def _build_items(files: list[Path]) -> list[dict]:
    import loaders
    from ingest import chunk_text
    size = int(settings.get("CHUNK_SIZE") or 900)
    overlap = int(settings.get("CHUNK_OVERLAP") or 150)
    items: list[dict] = []
    for f in files:
        try:
            for part in loaders.load_file(f):
                for ch in chunk_text(part.get("text") or "", size, overlap):
                    if ch.strip():
                        items.append({"text": ch, "source": f.name,
                                      "page": part.get("page")})
                        if len(items) >= _MAX_CHUNKS:
                            return items
        except Exception as e:
            print(f"[price] разбор {f.name}: {e}")
    return items


def _items() -> list[dict]:
    files = _files()
    sig = _signature(files)
    with _lock:
        if _cache["sig"] == sig:
            return _cache["items"]
    items = _build_items(files)        # парсинг вне блокировки
    with _lock:
        _cache["sig"] = sig
        _cache["items"] = items
    return items


def is_price_query(question: str) -> bool:
    """Похоже ли на ценовой вопрос (интент прайса)."""
    if not question:
        return False
    try:
        import retriever
        if retriever.infer_category(question) == "price":
            return True
    except Exception:
        pass
    ql = question.lower()
    kws = ("цена", "цены", "ценам", "стоит", "стоимост", "прайс", "тариф",
           "почем", "почём", "сколько стоит", "расценк", "по чем")
    return any(k in ql for k in kws)


def hits(question: str, top_k: int | None = None) -> list[dict]:
    """Лучшие фрагменты прайса под вопрос (реранк). Пустой список, если выключено
    или ничего не нашлось."""
    if not enabled():
        return []
    items = _items()
    if not items:
        return []
    try:
        import retriever
        k = int(top_k or settings.get("PRICE_TOP_K") or 6)
        return retriever.rerank_texts(question, items, k)
    except Exception as e:
        print(f"[price] реранк: {e}")
        return []


def status() -> dict:
    """Состояние прайс-папки для интерфейса."""
    d = folder()
    files = _files() if d else []
    return {"enabled": enabled(), "dir": str(d) if d else "",
            "exists": bool(d and d.is_dir()), "files": len(files),
            "cached_chunks": len(_cache.get("items") or [])}
