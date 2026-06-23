"""Индексация документов в Qdrant.

- инкрементально: пропускает неизменённые файлы (по хешу + mtime);
- чанкинг с перекрытием;
- эмбеддинги bge-m3 на Apple Metal (MPS);
- метаданные (источник, страница, тип, дата) для цитирования и фильтрации.

Запуск:  python ingest.py            # индексировать всю DOCS_DIR
         python ingest.py --reset   # пересоздать коллекцию с нуля
"""
from __future__ import annotations
import argparse
import hashlib
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
import settings
import metadata as meta
import enrich
from loaders import load_file

# параметры индексации берутся из рантайм-настроек (правятся в админке),
# процесс ingest запускается заново на каждую переиндексацию и читает свежие значения
COLLECTION = settings.get("QDRANT_COLLECTION")
DOCS_DIR = Path(settings.get("DOCS_DIR")).expanduser()

SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv",
             ".txt", ".md", ".html", ".htm",
             ".mp3", ".wav", ".m4a", ".aac", ".mp4", ".mov", ".mkv", ".webm"}


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        # стараемся резать по концу предложения/строки
        cut = text.rfind("\n", start, end)
        if cut == -1 or cut <= start + size // 2:
            cut = text.rfind(". ", start, end)
        if cut == -1 or cut <= start + size // 2:
            cut = end
        chunks.append(text[start:cut].strip())
        start = max(cut - overlap, start + 1)
    return [c for c in chunks if c]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(str(path.stat().st_mtime_ns).encode())
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:16]


def ensure_collection(client: QdrantClient, reset: bool):
    exists = client.collection_exists(COLLECTION)
    if reset and exists:
        client.delete_collection(COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qm.VectorParams(
                size=config.EMBED_DIM, distance=qm.Distance.COSINE
            ),
        )
        # индексы payload — для инкрементального обновления и фильтрации
        for field in ("source", "fhash", "doc_category", "date", "ftype",
                      "product", "topic", "doc_type"):
            client.create_payload_index(
                COLLECTION, field, qm.PayloadSchemaType.KEYWORD
            )


def already_indexed(client: QdrantClient, source: str, fhash: str) -> bool:
    res = client.scroll(
        COLLECTION,
        scroll_filter=qm.Filter(must=[
            qm.FieldCondition(key="source", match=qm.MatchValue(value=source)),
            qm.FieldCondition(key="fhash", match=qm.MatchValue(value=fhash)),
        ]),
        limit=1,
    )
    return len(res[0]) > 0


def delete_old_versions(client: QdrantClient, source: str):
    client.delete(
        COLLECTION,
        points_selector=qm.FilterSelector(filter=qm.Filter(must=[
            qm.FieldCondition(key="source", match=qm.MatchValue(value=source)),
        ])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="пересоздать коллекцию")
    args = ap.parse_args()

    if not DOCS_DIR.exists():
        raise SystemExit(f"DOCS_DIR не найдена: {DOCS_DIR} (укажите в админке)")

    embed_model = settings.get("EMBED_MODEL")
    device = settings.get("DEVICE")
    chunk_size = settings.get("CHUNK_SIZE")
    chunk_overlap = settings.get("CHUNK_OVERLAP")

    print(f"Документы: {DOCS_DIR}")
    client = QdrantClient(url=settings.get("QDRANT_URL"))
    ensure_collection(client, args.reset)

    print(f"Загружаю эмбеддер {embed_model} на {device} ...")
    embedder = SentenceTransformer(embed_model, device=device)

    files = [p for p in DOCS_DIR.rglob("*")
             if p.is_file() and p.suffix.lower() in SUPPORTED]
    print(f"Найдено файлов: {len(files)}")

    n_new = n_chunks = 0
    for path in tqdm(files, desc="Индексация"):
        source = str(path.relative_to(DOCS_DIR))
        fhash = file_hash(path)
        if not args.reset and already_indexed(client, source, fhash):
            continue
        delete_old_versions(client, source)  # файл изменился — чистим старое

        points = []
        for part in load_file(path):
            for chunk in chunk_text(part["text"], chunk_size, chunk_overlap):
                points.append({"chunk": chunk, "page": part["page"]})

        if not points:
            continue

        md = meta.extract(path)  # rule-based метаданные (категория, дата, заголовок)
        if settings.get("LLM_METADATA"):
            # LLM-обогащение: продукт/тема/тип (по первому чанку, один вызов на файл)
            e = enrich.extract_structured(points[0]["chunk"])
            for k in ("product", "topic", "doc_type"):
                if e.get(k):
                    md[k] = e[k]
            # категорию уточняем, только если правило дало общий "document"
            if md.get("doc_category") == "document" and e.get("category"):
                md["doc_category"] = e["category"]

        vectors = embedder.encode(
            [p["chunk"] for p in points],
            normalize_embeddings=True, batch_size=32, show_progress_bar=False,
        )
        client.upsert(
            COLLECTION,
            points=[
                qm.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec.tolist(),
                    payload={
                        "text": p["chunk"],
                        "source": source,
                        "page": p["page"],
                        "ftype": path.suffix.lower().lstrip("."),
                        "fhash": fhash,
                        "indexed_at": time.strftime("%Y-%m-%d"),
                        **md,  # doc_category, title, date
                    },
                )
                for p, vec in zip(points, vectors)
            ],
        )
        n_new += 1
        n_chunks += len(points)

    print(f"Готово. Обновлено файлов: {n_new}, чанков добавлено: {n_chunks}")


if __name__ == "__main__":
    main()
