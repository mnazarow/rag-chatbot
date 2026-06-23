"""Индексация документов в LightRAG (построение графа знаний).

ВНИМАНИЕ: индексация прогоняет каждый документ через LLM для извлечения
сущностей и связей — это медленно и ресурсоёмко (часы на ~1000 файлов
локально). Запускать осознанно.

Запуск:  python ingest_lightrag.py
Использует те же загрузчики, что и основной вариант (через ../loaders.py).
"""
from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# переиспользуем загрузчики основного проекта
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loaders import load_file  # noqa: E402

from rag_lightrag import build_rag  # noqa: E402

load_dotenv()

DOCS_DIR = Path(os.getenv("DOCS_DIR", "/opt/db")).expanduser()
SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv",
             ".txt", ".md", ".html", ".htm",
             ".mp3", ".wav", ".m4a", ".aac", ".mp4", ".mov", ".mkv", ".webm"}


async def main():
    if not DOCS_DIR.exists():
        raise SystemExit(f"DOCS_DIR не найдена: {DOCS_DIR}")
    rag = await build_rag()

    files = [p for p in DOCS_DIR.rglob("*")
             if p.is_file() and p.suffix.lower() in SUPPORTED]
    print(f"Файлов к индексации: {len(files)}")

    for path in tqdm(files, desc="LightRAG ingest"):
        source = str(path.relative_to(DOCS_DIR))
        text = "\n\n".join(part["text"] for part in load_file(path) if part["text"].strip())
        if not text.strip():
            continue
        # ids/file_paths помогают LightRAG ссылаться на источники
        await rag.ainsert(text, ids=[source], file_paths=[source])

    print("Готово. Граф сохранён в", os.getenv("LIGHTRAG_DIR", "./lightrag_storage"))


if __name__ == "__main__":
    asyncio.run(main())
