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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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
                      "product", "topic", "doc_type", "vision_desc"):
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
    run_parse_ms = run_embed_ms = 0   # для диагностики узкого места
    n_new = n_chunks = n_skip = n_timeout = 0
    errors = []  # (файл, причина)
    tmpdir = tempfile.mkdtemp(prefix="rag_pg_") if from_pg else None
    total_work = len(work)

    # число потоков извлечения: 0 = авто (по ядрам, ≤8). Таймаут на файл (SIGALRM)
    # работает только однопоточно — при нём принудительно 1 поток.
    workers = int(settings.get("INGEST_WORKERS") or 0)
    if workers <= 0:
        workers = min(8, os.cpu_count() or 4)
    if use_alarm:
        workers = 1
    print(f"Потоков извлечения: {workers}")

    # --- эмбеддинг + запись в Qdrant (всегда в основном потоке) ---
    def _embed_upsert(source, fhash, points, ftype, meta_path, parse_ms=0):
        nonlocal n_new, n_chunks, run_proc_ms, run_parse_ms, run_embed_ms
        t_embed = time.time()
        md = meta.extract(meta_path)
        if settings.get("LLM_METADATA"):
            try:
                e = enrich.extract_structured(points[0]["chunk"])
                for k in ("product", "topic", "doc_type"):
                    if e.get(k):
                        md[k] = e[k]
                if md.get("doc_category") == "document" and e.get("category"):
                    md["doc_category"] = e["category"]
            except Exception as me:
                print(f"  ~ метаданные LLM пропущены для {source}: {me}")
        try:
            enc_batch = int(settings.get("EMBED_BATCH") or 32)
        except Exception:
            enc_batch = 32
        enc_batch = max(1, enc_batch)
        BATCH = max(256, enc_batch)   # размер группы upsert не меньше batch эмбеддера
        for i in range(0, len(points), BATCH):
            batch = points[i:i + BATCH]
            vectors = embedder.encode(
                [p["chunk"] for p in batch],
                normalize_embeddings=True, batch_size=enc_batch, show_progress_bar=False,
            )
            if len(points) > BATCH:
                done = min(i + BATCH, len(points))
                print(f"    {source}: {int(done * 100 / len(points))}% "
                      f"({done}/{len(points)} чанков)", flush=True)
            client.upsert(
                COLLECTION, wait=False,
                points=[
                    qm.PointStruct(
                        id=str(uuid.uuid4()), vector=vec.tolist(),
                        payload={
                            "text": p["chunk"], "source": source, "page": p["page"],
                            "ftype": ftype, "fhash": fhash,
                            "indexed_at": time.strftime("%Y-%m-%d"),
                            **({"t_start": p["t_start"], "t_end": p["t_end"]}
                               if p.get("t_start") is not None else {}),
                            **({"vision_desc": True} if p.get("vision_desc") else {}),
                            **md,
                        },
                    )
                    for p, vec in zip(batch, vectors)
                ],
            )
        embed_ms = int((time.time() - t_embed) * 1000)
        n_new += 1
        n_chunks += len(points)
        run_parse_ms += parse_ms
        run_embed_ms += embed_ms
        proc_ms = parse_ms + embed_ms
        run_proc_ms += proc_ms
        file_times[source] = {"ms": proc_ms, "parse_ms": parse_ms, "embed_ms": embed_ms,
                              "chunks": len(points), "ftype": ftype,
                              "category": md.get("doc_category") or "document",
                              "ts": time.time()}
        print(f"    · {source}: парсинг {parse_ms} мс · эмбеддинг+Qdrant {embed_ms} мс "
              f"· чанков {len(points)}", flush=True)
        if from_pg:
            try:
                full = "\n\n".join(p["chunk"] for p in points)[:500_000]
                db.catalog_update_text(source, full)
            except Exception:
                pass

    # --- извлечение одного файла (в рабочем потоке; без записей в Qdrant) ---
    def _parse(item):
        tmp_path = None
        source = ""
        t_parse = time.time()
        try:
            if from_pg:
                source = item["rel_path"]
                fhash = item.get("sha256") or ""
                if not args.reset and fhash and already_indexed(client, source, fhash):
                    return {"status": "skip"}
                base = Path(item.get("fname") or Path(source).name).name or "file"
                tmp_path = Path(tmpdir) / f"{uuid.uuid4().hex}_{base}"
                if not db.catalog_export_to(source, tmp_path):
                    return {"status": "empty", "tmp": tmp_path}
                path = tmp_path
                meta_path = Path(source)
            else:
                path = item
                source = str(path.relative_to(DOCS_DIR))
                fhash = file_hash(path)
                if not args.reset and already_indexed(client, source, fhash):
                    return {"status": "skip"}
                meta_path = path
            points = []
            for part in load_file(path):
                for chunk in chunk_text(part["text"], chunk_size, chunk_overlap):
                    points.append({"chunk": chunk, "page": part["page"],
                                   "t_start": part.get("t_start"),
                                   "t_end": part.get("t_end"),
                                   "vision_desc": part.get("vision_desc")})
            if not points:
                return {"status": "empty", "tmp": tmp_path}
            return {"status": "ok", "source": source, "fhash": fhash, "points": points,
                    "ftype": path.suffix.lower().lstrip("."), "meta_path": meta_path,
                    "tmp": tmp_path, "parse_ms": int((time.time() - t_parse) * 1000)}
        except Exception as e:
            return {"status": "error", "source": source, "msg": str(e)[:200],
                    "tmp": tmp_path}

    # --- обработка результата извлечения (основной поток) ---
    def _consume(res, idx):
        nonlocal n_skip
        tmp = res.get("tmp")
        try:
            st = res.get("status")
            if st == "skip":
                return
            if st == "empty":
                n_skip += 1
                return
            if st == "error":
                errors.append((res.get("source", ""), res.get("msg", "")))
                print(f"  ! ошибка обработки {res.get('source','')}: {res.get('msg','')}")
                return
            source = res["source"]
            print(f"[{idx}/{total_work}] {int(idx * 100 / total_work)}% "
                  f"индексирую: {source}", flush=True)
            delete_old_versions(client, source)
            _embed_upsert(source, res["fhash"], res["points"], res["ftype"],
                          res["meta_path"], res.get("parse_ms", 0))
        finally:
            if tmp is not None:
                try:
                    tmp.unlink()
                except Exception:
                    pass

    if workers <= 1:
        # последовательный путь (поддерживает лимит времени на файл через SIGALRM)
        for idx, item in enumerate(work, 1):
            t_file = time.time()
            tmp_path = None
            source = ""
            try:
                if from_pg:
                    source = item["rel_path"]
                    fhash = item.get("sha256") or ""
                    if not args.reset and fhash and already_indexed(client, source, fhash):
                        continue
                    base = Path(item.get("fname") or Path(source).name).name or "file"
                    tmp_path = Path(tmpdir) / base
                    if not db.catalog_export_to(source, tmp_path):
                        n_skip += 1
                        continue
                    path = tmp_path
                    meta_path = Path(source)
                else:
                    path = item
                    source = str(path.relative_to(DOCS_DIR))
                    fhash = file_hash(path)
                    if not args.reset and already_indexed(client, source, fhash):
                        continue
                    meta_path = path
                delete_old_versions(client, source)
                print(f"[{idx}/{total_work}] {int(idx * 100 / total_work)}% "
                      f"индексирую: {source}", flush=True)
                if use_alarm:
                    signal.alarm(file_timeout)
                t_parse = time.time()
                points = []
                for part in load_file(path):
                    for chunk in chunk_text(part["text"], chunk_size, chunk_overlap):
                        points.append({"chunk": chunk, "page": part["page"],
                                       "t_start": part.get("t_start"),
                                       "t_end": part.get("t_end"),
                                       "vision_desc": part.get("vision_desc")})
                if use_alarm:
                    signal.alarm(0)
                parse_ms = int((time.time() - t_parse) * 1000)
                if not points:
                    n_skip += 1
                    continue
                _embed_upsert(source, fhash, points, path.suffix.lower().lstrip("."),
                              meta_path, parse_ms)
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
                errors.append((source, str(e)[:200]))
                print(f"  ! ошибка обработки {source}: {e}")
                continue
            finally:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
    else:
        # параллельный путь: до workers извлечений одновременно, запись — в осн. потоке
        with ThreadPoolExecutor(max_workers=workers) as exio:
            work_it = iter(work)
            pend = deque()

            def _submit():
                try:
                    it = next(work_it)
                except StopIteration:
                    return False
                pend.append(exio.submit(_parse, it))
                return True

            for _ in range(workers * 2):
                if not _submit():
                    break
            idx = 0
            while pend:
                res = pend.popleft().result()
                idx += 1
                try:
                    _consume(res, idx)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    src = res.get("source", "") if isinstance(res, dict) else ""
                    errors.append((src, str(e)[:200]))
                    print(f"  ! ошибка записи {src}: {e}")
                _submit()

    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    wall = max(1, int((time.time() - run_start) * 1000))
    print(f"Готово. Обновлено файлов: {n_new}, чанков добавлено: {n_chunks}, "
          f"пропущено пустых: {n_skip}, по таймауту: {n_timeout}, ошибок: {len(errors)}")
    print(f"Тайминги (сумма по файлам): извлечение {run_parse_ms} мс, эмбеддинг+Qdrant "
          f"{run_embed_ms} мс; общее время {wall} мс, потоков {workers}. "
          f"Если 'извлечение' >> общего времени — параллельность работает; если "
          f"'эмбеддинг+Qdrant' доминирует — узкое место в эмбеддере/Qdrant (потоки не помогут).")
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
