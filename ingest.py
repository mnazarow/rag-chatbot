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

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
import settings
import vectorstore
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
# Прогресс индексации (файлов обработано/всего) — читается админкой для прогресс-бара
INGEST_PROGRESS = Path(__file__).resolve().parent / "ingest_progress.json"


def _atomic_write_text(path: Path, text: str) -> None:
    """Атомарная запись текста: во временный файл рядом, затем os.replace. Иначе
    читатель (админка) может увидеть наполовину записанный/усечённый JSON."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_progress(done: int, total: int, current: str = "", phase: str = "index") -> None:
    """Записать прогресс индексации в файл (для графического прогресс-бара в панели)."""
    try:
        pct = int(done * 100 / total) if total else (100 if phase == "done" else 0)
        _atomic_write_text(INGEST_PROGRESS, json.dumps(
            {"done": int(done), "total": int(total), "pct": max(0, min(100, pct)),
             "current": (current or "")[:200], "phase": phase, "ts": time.time()},
            ensure_ascii=False))
    except Exception:
        pass

SUPPORTED = {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".xlsm", ".xls", ".csv",
             ".txt", ".md", ".html", ".htm", ".mhtml", ".mht",
             ".xml", ".json", ".url", ".msg", ".svg",
             ".dxf", ".dwg", ".stp", ".step", ".igs", ".iges",
             ".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2",
             ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".jfif",
             ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".rw2", ".orf", ".sr2",
             ".mp3", ".wav", ".m4a", ".aac", ".mp4", ".mov", ".mkv", ".webm"}


def _chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
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


_HEAD_RE = None


def _is_heading(line: str) -> bool:
    """Похоже ли на заголовок раздела (граница чанка)."""
    global _HEAD_RE
    import re
    if _HEAD_RE is None:
        _HEAD_RE = re.compile(r"^\s*\d+(\.\d+)*[.)]\s+\S")
    l = line.strip()
    if not l:
        return False
    if l.startswith("#"):                       # markdown-заголовок
        return True
    if _HEAD_RE.match(l) and len(l) < 100:       # нумерованный раздел «1.2 …»
        return True
    if len(l) <= 80 and l == l.upper() and re.search(r"[A-ZА-Я]", l):  # ЗАГОЛОВОК КАПСОМ
        return True
    if len(l) <= 60 and l.endswith(":"):         # короткая строка-заголовок с двоеточием
        return True
    return False


def _looks_tabular(b: str) -> bool:
    lines = [x for x in b.split("\n") if x.strip()]
    if not lines:
        return False
    piped = sum(1 for x in lines if x.count("|") >= 2 or x.count("\t") >= 2)
    return piped >= max(2, len(lines) // 2)


def _chunk_structured(text: str, size: int, overlap: int) -> list[str]:
    """Резать по структуре: заголовки и абзацы — границы; таблицы/списки держим целиком;
    крупные блоки дробим по предложениям (фолбэк на размерное)."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    lines = text.split("\n")
    blocks, cur = [], []

    def _flush():
        if cur:
            b = "\n".join(cur).strip()
            if b:
                blocks.append(b)

    for ln in lines:
        if _is_heading(ln) and cur:
            _flush()
            cur = [ln]
        elif ln.strip() == "":
            _flush()
            cur = []
        else:
            cur.append(ln)
    _flush()

    chunks, buf = [], ""
    for b in blocks:
        if len(b) > size:
            if buf.strip():
                chunks.append(buf.strip())
                buf = ""
            if _looks_tabular(b):
                chunks.append(b[:size * 3])       # таблицу не рвём (с разумным потолком)
            else:
                chunks.extend(_chunk_fixed(b, size, overlap))
            continue
        if buf and len(buf) + len(b) + 2 > size:
            chunks.append(buf.strip())
            buf = b
        else:
            buf = (buf + "\n\n" + b) if buf else b
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Разбиение на чанки: структурное (по заголовкам/абзацам) при STRUCTURE_CHUNK,
    иначе — размерное с перекрытием."""
    try:
        if settings.get("STRUCTURE_CHUNK"):
            return _chunk_structured(text, size, overlap)
    except Exception:
        pass
    return _chunk_fixed(text, size, overlap)


def _append_llm_desc(points, source, file_text, chunk_size, chunk_overlap, capped):
    """Опция INDEX_LLM_DESCRIBE: добавить в индекс краткое LLM-описание файла."""
    if capped or not points or not file_text.strip():
        return
    try:
        if not settings.get("INDEX_LLM_DESCRIBE"):
            return
    except Exception:
        return
    try:
        import loaders
        desc = loaders.describe_file_llm(source, file_text)
    except Exception as e:
        print(f"  ~ LLM-описание файла не удалось ({source}): {e}", flush=True)
        return
    if not desc:
        return
    body = "Описание документа (LLM):\n" + desc
    for chunk in chunk_text(body, chunk_size, chunk_overlap):
        points.append({"chunk": chunk, "page": None, "t_start": None,
                       "t_end": None, "vision_desc": None})
    print(f"    ~ {source}: добавлено LLM-описание документа", flush=True)


def file_hash(path: Path) -> str:
    st = path.stat()
    # Опция INGEST_CONTENT_HASH: для небольших файлов хешируем содержимое — устойчиво
    # к изменению содержимого без смены mtime/size (rsync -a, git checkout, восстановление
    # из бэкапа). По умолчанию (ключа нет) — дёшево, по mtime+size.
    try:
        if settings.get("INGEST_CONTENT_HASH"):
            try:
                cap = int(settings.get("INGEST_CONTENT_HASH_MAX") or 1_000_000)
            except Exception:
                cap = 1_000_000
            if st.st_size <= cap:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for blk in iter(lambda: f.read(1 << 20), b""):
                        h.update(blk)
                return h.hexdigest()[:16]
    except Exception:
        pass
    h = hashlib.sha256()
    h.update(str(st.st_mtime_ns).encode())
    h.update(str(st.st_size).encode())
    return h.hexdigest()[:16]


def ensure_collection(reset: bool):
    """Создать/пересоздать коллекцию в активной векторной базе (Qdrant или Milvus)."""
    vectorstore.ensure_collection(int(settings.get("EMBED_DIM")), reset=reset)


def already_indexed(source: str, fhash: str) -> bool:
    pts, _ = vectorstore.scroll(flt={"source": source, "fhash": fhash}, limit=1,
                                with_payload=False, with_vectors=False)
    return len(pts) > 0


def delete_old_versions(source: str):
    vectorstore.delete({"source": source})


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
    try:
        max_chunks = int(settings.get("INGEST_MAX_CHUNKS") or 0)
    except Exception:
        max_chunks = 0
    try:
        llm_desc_on = bool(settings.get("INDEX_LLM_DESCRIBE"))
    except Exception:
        llm_desc_on = False

    # --- фатальные ошибки инициализации: понятное сообщение и выход ---
    print(f"Документы: {DOCS_DIR}")
    _vb = vectorstore.backend()
    try:
        ensure_collection(args.reset)
    except Exception as e:
        _tgt = vectorstore.milvus_uri() if _vb == "milvus" else settings.get("QDRANT_URL")
        raise SystemExit(f"FATAL: не удалось подключиться к векторной базе "
                         f"({_vb}: {_tgt}): {e}")

    # Режим массовой загрузки: отключаем HNSW-индексацию Qdrant на время ingest (иначе
    # построение индекса на пороге ~20000 конкурирует с записью → Qdrant 500 на больших
    # файлах). Включаем обратно в конце и через atexit (на случай прерывания). Отключить —
    # QDRANT_BULK_INDEXING=0. Ключ серверный (нет поля в админке), поэтому берём из config
    # (settings.get вернёт None для ключей без FIELDS); если вынесут в UI — уважим и его.
    _bulk_cfg = getattr(config, "QDRANT_BULK_INDEXING", True)
    _bulk_sv = settings.get("QDRANT_BULK_INDEXING")
    _bulk_idx = (_vb != "milvus") and (bool(_bulk_sv) if _bulk_sv is not None else bool(_bulk_cfg))
    if _bulk_idx:
        import atexit
        import signal
        import sys as _sys
        vectorstore.qdrant_bulk_indexing(False)
        atexit.register(vectorstore.qdrant_bulk_indexing, True)   # бэкстоп при прерывании
        # SIGTERM (супервизор/остановка сервиса) → штатный выход, чтобы atexit успел вернуть
        # индексацию (иначе indexing_threshold остался бы 0 до следующего прогона).
        try:
            signal.signal(signal.SIGTERM, lambda *_a: _sys.exit(0))
        except Exception:
            pass

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
        # telegram/ индексируется отдельно tg_train (payload tg=True/tg_chat_id);
        # исключаем из общего обхода, иначе переиндексация затрёт эти метки и
        # delete_user()/delete_all() перестанут находить чанки (M31).
        work = list(fsutil.iter_doc_files(DOCS_DIR, SUPPORTED, onerror=_walk_err,
                                          exclude_dirs={"telegram"}))
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
    n_dup = 0
    _dedup_on = bool(settings.get("INDEX_DEDUP"))
    _seen_hashes: dict = {}          # content-hash -> первый source (для дедупа)
    errors = []  # (файл, причина)
    tmpdir = tempfile.mkdtemp(prefix="rag_pg_") if from_pg else None
    total_work = len(work)
    _write_progress(0, total_work, phase="index")

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
        nonlocal n_new, n_chunks, run_proc_ms, run_parse_ms, run_embed_ms, n_dup
        # дедуп по содержимому: пропускаем файл, чей извлечённый текст уже проиндексирован
        if _dedup_on and points:
            chash = hashlib.sha256(
                "\n".join((p.get("chunk") or "") for p in points).encode("utf-8")).hexdigest()
            if chash in _seen_hashes:
                n_dup += 1
                print(f"  = дубликат содержимого: {source} (совпадает с {_seen_hashes[chash]}) — пропущен",
                      flush=True)
                return
            _seen_hashes[chash] = source
        t_embed = time.time()
        md = meta.extract(meta_path)
        if settings.get("LLM_METADATA"):
            try:
                e = enrich.extract_structured(points[0]["chunk"])
                for k in ("product", "topic", "doc_type"):
                    if e.get(k):
                        md[k] = e[k]
                if e.get("tags"):
                    md["tags"] = e["tags"]         # авто-теги документа (для поиска/фильтра)
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

        # контекстные чанки: префикс заголовка документа + темы для ЭМБЕДДИНГА
        _ctx_on = bool(settings.get("INDEX_CONTEXTUAL"))
        _ctx_prefix = ""
        if _ctx_on:
            import os as _os
            _title = _os.path.splitext(_os.path.basename(source))[0].replace("_", " ")
            _bits = [_title]
            for _k in ("product", "topic"):
                if md.get(_k):
                    _bits.append(str(md[_k]))
            _ctx_prefix = " · ".join(b for b in _bits if b).strip()

        def _embed_text(chunk):
            return (_ctx_prefix + "\n" + chunk) if (_ctx_on and _ctx_prefix) else chunk

        # small-to-big: «родительский» фрагмент = окно соседних чанков
        _parent_on = bool(settings.get("INDEX_PARENT_CONTEXT"))
        _pw = max(0, int(settings.get("PARENT_WINDOW") or 1)) if _parent_on else 0

        def _parent_text(idx):
            lo, hi = max(0, idx - _pw), min(len(points), idx + _pw + 1)
            return "\n\n".join((points[j].get("chunk") or "") for j in range(lo, hi))[:6000]

        for i in range(0, len(points), BATCH):
            batch = points[i:i + BATCH]
            vectors = embedder.encode(
                [_embed_text(p["chunk"]) for p in batch],
                normalize_embeddings=True, batch_size=enc_batch, show_progress_bar=False,
            )
            if len(points) > BATCH:
                done = min(i + BATCH, len(points))
                print(f"    {source}: {int(done * 100 / len(points))}% "
                      f"({done}/{len(points)} чанков)", flush=True)
            vectorstore.upsert([
                {"id": str(uuid.uuid4()), "vector": vec.tolist(),
                 "payload": {
                     "text": p["chunk"], "source": source, "page": p["page"],
                     "ftype": ftype, "fhash": fhash,
                     "indexed_at": time.strftime("%Y-%m-%d"),
                     **({"parent": _parent_text(i + j)} if _parent_on else {}),
                     **({"t_start": p["t_start"], "t_end": p["t_end"]}
                        if p.get("t_start") is not None else {}),
                     **({"vision_desc": True} if p.get("vision_desc") else {}),
                     **md,
                 }}
                for j, (p, vec) in enumerate(zip(batch, vectors))
            ], wait=False)
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
                if not args.reset and fhash and already_indexed(source, fhash):
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
                if not args.reset and already_indexed(source, fhash):
                    return {"status": "skip"}
                meta_path = path
            points = []
            _capped = False
            _ftext = []          # накопитель текста файла для LLM-описания
            for part in load_file(path):
                if llm_desc_on and sum(len(x) for x in _ftext) < 20000:
                    _ftext.append(part.get("text", "") or "")
                for chunk in chunk_text(part["text"], chunk_size, chunk_overlap):
                    points.append({"chunk": chunk, "page": part["page"],
                                   "t_start": part.get("t_start"),
                                   "t_end": part.get("t_end"),
                                   "vision_desc": part.get("vision_desc")})
                    if max_chunks and len(points) >= max_chunks:
                        _capped = True
                        break
                if _capped:
                    break
            if _capped:
                print(f"  ! {source}: превышен лимит чанков на файл "
                      f"({max_chunks}) — проиндексирована только часть "
                      f"(увеличьте INGEST_MAX_CHUNKS или уберите огромный архив)",
                      flush=True)
            _append_llm_desc(points, source, "\n".join(_ftext),
                             chunk_size, chunk_overlap, _capped)
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
            _write_progress(idx, total_work, source)
            delete_old_versions(source)
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
        # FIXME(review): логика парсинга файла продублирована с _parse/_consume
        # (параллельный путь). Объединять рискованно из-за SIGALRM-таймаута, который
        # работает только в основном потоке; при рефакторинге вынести общий парсер файла.
        for idx, item in enumerate(work, 1):
            t_file = time.time()
            tmp_path = None
            source = ""
            try:
                if from_pg:
                    source = item["rel_path"]
                    fhash = item.get("sha256") or ""
                    if not args.reset and fhash and already_indexed(source, fhash):
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
                    if not args.reset and already_indexed(source, fhash):
                        continue
                    meta_path = path
                print(f"[{idx}/{total_work}] {int(idx * 100 / total_work)}% "
                      f"индексирую: {source}", flush=True)
                _write_progress(idx, total_work, source)
                if use_alarm:
                    signal.alarm(file_timeout)
                t_parse = time.time()
                points = []
                _capped = False
                _ftext = []
                for part in load_file(path):
                    if llm_desc_on and sum(len(x) for x in _ftext) < 20000:
                        _ftext.append(part.get("text", "") or "")
                    for chunk in chunk_text(part["text"], chunk_size, chunk_overlap):
                        points.append({"chunk": chunk, "page": part["page"],
                                       "t_start": part.get("t_start"),
                                       "t_end": part.get("t_end"),
                                       "vision_desc": part.get("vision_desc")})
                        if max_chunks and len(points) >= max_chunks:
                            _capped = True
                            break
                    if _capped:
                        break
                if _capped:
                    print(f"  ! {source}: превышен лимит чанков на файл "
                          f"({max_chunks}) — проиндексирована только часть "
                          f"(увеличьте INGEST_MAX_CHUNKS или уберите огромный архив)",
                          flush=True)
                _append_llm_desc(points, source, "\n".join(_ftext),
                                 chunk_size, chunk_overlap, _capped)
                if use_alarm:
                    signal.alarm(0)
                parse_ms = int((time.time() - t_parse) * 1000)
                if not points:
                    n_skip += 1
                    continue
                # Удаляем старые версии ТОЛЬКО после успешного парсинга (как в _consume),
                # иначе сбой парсинга стёр бы уже проиндексированный документ.
                # FIXME(review): точки пишутся со случайными uuid4 id (см. _embed_upsert),
                # поэтому нужен явный delete по source. Детерминированные id
                # (uuid5 от source+chunk_idx) убрали бы гонку delete→upsert, но при
                # изменении числа чанков оставляли бы «хвост» — нужен отдельный проход.
                delete_old_versions(source)
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
    # Включаем индексацию обратно — Qdrant построит HNSW один раз (после массовой загрузки)
    if _bulk_idx:
        vectorstore.qdrant_bulk_indexing(True)
    _write_progress(total_work, total_work, phase="done")

    wall = max(1, int((time.time() - run_start) * 1000))
    print(f"Готово. Обновлено файлов: {n_new}, чанков добавлено: {n_chunks}, "
          f"пропущено пустых: {n_skip}, дубликатов: {n_dup}, по таймауту: {n_timeout}, "
          f"ошибок: {len(errors)}")
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
        _atomic_write_text(INGEST_STATS,
                           json.dumps(stats_out, ensure_ascii=False, indent=2))
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
