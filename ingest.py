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
import json
import os
import shutil
import signal
import tempfile
import time
import uuid
import warnings
from pathlib import Path

# до импорта моделей: отключаем параллелизм HF-токенайзеров (fork при распаковке
# архивов/конвертации даёт предупреждение и риск зависаний)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# шумные предупреждения библиотек парсинга — не засоряем лог индексации
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", message=".*Data Validation extension.*")

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
import settings
import db
import metadata as meta
import enrich
import fsutil
from loaders import load_file


class _Timeout(Exception):
    """Превышен лимит времени обработки одного файла."""


def _timeout_handler(signum, frame):
    raise _Timeout()

# параметры индексации берутся из рантайм-настроек (правятся в админке),
# процесс ingest запускается заново на каждую переиндексацию и читает свежие значения
COLLECTION = settings.get("QDRANT_COLLECTION")
DOCS_DIR = Path(settings.get("DOCS_DIR")).expanduser()

# Время обработки по файлам (записывается при индексации, читается админкой/каталогом)
INGEST_STATS = Path(__file__).resolve().parent / "ingest_stats.json"

SUPPORTED = {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".xlsm", ".xls", ".csv",
             ".txt", ".md", ".html", ".htm", ".mhtml", ".mht",
             ".xml", ".json", ".url", ".msg", ".svg",
             ".dxf", ".dwg", ".stp", ".step", ".igs", ".iges",
             ".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2",
             ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".jfif",
             ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".rw2", ".orf", ".sr2",
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
                size=settings.get("EMBED_DIM"), distance=qm.Distance.COSINE
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

    # Источник документов: папка (по умолчанию) или PostgreSQL (файлы, ранее
    # загруженные в doc_catalog). PG-источник позволяет индексировать без папки.
    from_pg = (settings.get("CATALOG_SOURCE") == "postgresql"
               and db._dialect() == "postgresql")
    if not from_pg and not DOCS_DIR.exists():
        raise SystemExit(f"FATAL: DOCS_DIR не найдена: {DOCS_DIR} (укажите в админке)")

    embed_model = settings.get("EMBED_MODEL")
    device = settings.device()
    chunk_size = settings.get("CHUNK_SIZE")
    chunk_overlap = settings.get("CHUNK_OVERLAP")

    # --- фатальные ошибки инициализации: понятное сообщение и выход ---
    print(f"Документы: {DOCS_DIR}")
    try:
        client = QdrantClient(url=settings.get("QDRANT_URL"),
                              timeout=settings.get("QDRANT_INGEST_TIMEOUT"))
        ensure_collection(client, args.reset)
    except Exception as e:
        raise SystemExit(f"FATAL: не удалось подключиться к Qdrant ({settings.get('QDRANT_URL')}): {e}")

    print(f"Загружаю эмбеддер {embed_model} на {device} ...")
    try:
        embedder = SentenceTransformer(embed_model, device=device)
    except Exception as e:
        raise SystemExit(f"FATAL: не удалось загрузить модель эмбеддингов '{embed_model}' на {device}: {e}")

    # Устойчивый обход: одна недоступная папка (Errno 5 на сетевой/битой шаре и т.п.)
    # не должна срывать всю индексацию — такие каталоги пропускаются с пометкой в логе.
    _skipped_dirs = []

    def _walk_err(e):
        path = getattr(e, "filename", "") or str(e)
        _skipped_dirs.append(path)
        print(f"  ! пропущен недоступный путь: {path} ({e})")

    if from_pg:
        work = db.catalog_index_list()   # [{rel_path, fname, ext, sha256}]
        print(f"Источник: PostgreSQL (doc_catalog). Файлов с содержимым: {len(work)}")
        if not work:
            print("В PostgreSQL нет файлов с содержимым — сначала «Загрузить каталог "
                  "данных в PostgreSQL».")
    else:
        work = list(fsutil.iter_doc_files(DOCS_DIR, SUPPORTED, onerror=_walk_err))
        print(f"Найдено файлов: {len(work)}")
        if _skipped_dirs:
            print(f"Пропущено недоступных папок: {len(_skipped_dirs)} "
                  f"(см. строки выше — проверьте носитель/доступ к этим путям)")

    # время обработки по файлам: сохраняем прошлые значения (для пропущенных
    # неизменённых файлов), при --reset считаем заново
    prev_stats = {}
    if INGEST_STATS.exists() and not args.reset:
        try:
            prev_stats = json.loads(INGEST_STATS.read_text(encoding="utf-8"))
        except Exception:
            prev_stats = {}
    file_times = dict(prev_stats.get("files", {})) if not args.reset else {}

    # лимит времени на обработку одного файла (0 = без лимита). Защищает от
    # «зависания» на тяжёлом DWG/видео и т.п. Работает на Unix (SIGALRM).
    file_timeout = int(settings.get("FILE_PARSE_TIMEOUT") or 0)
    use_alarm = file_timeout > 0 and hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _timeout_handler)
        print(f"Лимит на файл: {file_timeout} c (превышение — пропуск)")

    run_start = time.time()
    run_proc_ms = 0
    n_new = n_chunks = n_skip = n_timeout = 0
    errors = []  # (файл, причина)
    tmpdir = tempfile.mkdtemp(prefix="rag_pg_") if from_pg else None
    for item in tqdm(work, desc="Индексация"):
        tmp_path = None
        source = ""
        t_file = time.time()
        try:
            if from_pg:
                # документ берётся из PostgreSQL: пишем содержимое во временный файл
                # (с исходным именем — чтобы загрузчик выбрал парсер по расширению)
                source = item["rel_path"]
                fhash = item.get("sha256") or ""
                if not args.reset and fhash and already_indexed(client, source, fhash):
                    continue
                base = Path(item.get("fname") or Path(source).name).name or "file"
                tmp_path = Path(tmpdir) / base
                # потоковая выгрузка содержимого (bytea или Large Object) во временный файл
                if not db.catalog_export_to(source, tmp_path):
                    n_skip += 1
                    continue
                path = tmp_path
                meta_path = Path(source)   # метаданные — по исходному пути (с папками)
            else:
                path = item
                source = str(path.relative_to(DOCS_DIR))
                fhash = file_hash(path)
                if not args.reset and already_indexed(client, source, fhash):
                    continue
                meta_path = path
            delete_old_versions(client, source)  # файл новый/изменился — чистим старое

            if use_alarm:
                signal.alarm(file_timeout)
            points = []
            for part in load_file(path):
                for chunk in chunk_text(part["text"], chunk_size, chunk_overlap):
                    points.append({"chunk": chunk, "page": part["page"],
                                   "t_start": part.get("t_start"),
                                   "t_end": part.get("t_end")})
            if use_alarm:
                signal.alarm(0)

            if not points:
                n_skip += 1  # пустой/нечитаемый файл — пропускаем, не ошибка
                continue

            md = meta.extract(meta_path)  # rule-based метаданные (категория, дата, заголовок)
            if settings.get("LLM_METADATA"):
                try:
                    e = enrich.extract_structured(points[0]["chunk"])
                    for k in ("product", "topic", "doc_type"):
                        if e.get(k):
                            md[k] = e[k]
                    if md.get("doc_category") == "document" and e.get("category"):
                        md["doc_category"] = e["category"]
                except Exception as me:
                    # обогащение метаданными не критично — продолжаем без него
                    print(f"  ~ метаданные LLM пропущены для {source}: {me}")

            # пишем пачками — большие файлы не упираются в таймаут и не съедают память
            BATCH = 256
            for i in range(0, len(points), BATCH):
                batch = points[i:i + BATCH]
                vectors = embedder.encode(
                    [p["chunk"] for p in batch],
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
                                # тайминги для аудио/видео (кадры/фрагменты в выдаче)
                                **({"t_start": p["t_start"], "t_end": p["t_end"]}
                                   if p.get("t_start") is not None else {}),
                                **md,  # doc_category, title, date
                            },
                        )
                        for p, vec in zip(batch, vectors)
                    ],
                )
            n_new += 1
            n_chunks += len(points)
            proc_ms = int((time.time() - t_file) * 1000)
            run_proc_ms += proc_ms
            file_times[source] = {"ms": proc_ms, "chunks": len(points),
                                  "ts": time.time()}
            # индексация из БД: сохраняем извлечённый текст обратно в каталог, чтобы
            # предпросмотр работал без папки с файлами
            if from_pg:
                try:
                    full = "\n\n".join(p["chunk"] for p in points)[:500_000]
                    db.catalog_update_text(source, full)
                except Exception:
                    pass
        except _Timeout:
            if use_alarm:
                signal.alarm(0)
            n_timeout += 1
            errors.append((source, f"превышен лимит {file_timeout} c — пропущен"))
            print(f"  ⏱ таймаут {file_timeout} c, пропуск: {source}")
            continue
        except KeyboardInterrupt:
            if use_alarm:
                signal.alarm(0)
            raise
        except Exception as e:
            if use_alarm:
                signal.alarm(0)
            # один битый файл не должен ронять всю индексацию
            errors.append((source, str(e)[:200]))
            print(f"  ! ошибка обработки {source}: {e}")
            continue
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"Готово. Обновлено файлов: {n_new}, чанков добавлено: {n_chunks}, "
          f"пропущено пустых: {n_skip}, по таймауту: {n_timeout}, ошибок: {len(errors)}")
    if errors:
        print("Файлы с ошибками:")
        for s, e in errors[:50]:
            print(f"  - {s}: {e}")
        if len(errors) > 50:
            print(f"  … и ещё {len(errors) - 50}")
    # время обработки: чистим записи об удалённых файлах и сохраняем сводку
    if from_pg:
        cur_sources = {it["rel_path"] for it in work}
    else:
        cur_sources = {str(p.relative_to(DOCS_DIR)) for p in work}
    file_times = {k: v for k, v in file_times.items() if k in cur_sources}
    run_end = time.time()
    stats_out = {
        "files": file_times,
        "last_run": {
            "started": run_start, "finished": run_end,
            "duration_sec": round(run_end - run_start, 1),
            "files_processed": n_new, "chunks": n_chunks,
            "skipped": n_skip, "errors": len(errors),
            "processed_ms": run_proc_ms,
            "avg_ms": round(run_proc_ms / n_new) if n_new else 0,
        },
        "total_ms": sum(v.get("ms", 0) for v in file_times.values()),
        "total_files_timed": len(file_times),
        "updated": run_end,
    }
    try:
        INGEST_STATS.write_text(json.dumps(stats_out, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    except Exception as e:
        print(f"  ~ не удалось записать {INGEST_STATS.name}: {e}")

    # индекс изменился — сбрасываем кэш поиска/ответов (Redis, пространство index)
    try:
        import cache
        cache.bump("index")
    except Exception:
        pass

    # машиночитаемая сводка (последняя строка) — её разбирает админка
    print(f"SUMMARY files_ok={n_new} chunks={n_chunks} skipped={n_skip} errors={len(errors)}")


if __name__ == "__main__":
    main()
