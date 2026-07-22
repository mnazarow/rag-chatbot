"""Режим обучения Телеграм-бота: пользователи (с разрешением) присылают документы,
которые сохраняются в отдельную папку на каждого пользователя, распознаются и
добавляются в базу знаний (Qdrant). Поддерживается просмотр и удаление файлов
по пользователю, а также удаление всех документов, добавленных через Телеграм.

Файлы хранятся в DOCS_DIR/telegram/<chat_id>/<имя>. В Qdrant каждый чанк получает
служебные поля payload: tg=True и tg_chat_id=<chat_id> — по ним выполняется удаление.
"""
from __future__ import annotations
import re
import shutil
import time
from pathlib import Path

import settings


def _root() -> Path:
    return Path(settings.get("DOCS_DIR")).expanduser() / "telegram"


def user_dir(chat_id) -> Path:
    return _root() / str(int(chat_id))


def _source(chat_id, name) -> str:
    return f"telegram/{int(chat_id)}/{name}"


def _safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", (name or "файл")).strip()
    return name[:180] or "файл"


def _unique(p: Path) -> Path:
    if not p.exists():
        return p
    stem, suf, i = p.stem, p.suffix, 2
    while (p.parent / f"{stem}_{i}{suf}").exists():
        i += 1
    return p.parent / f"{stem}_{i}{suf}"


def _bump():
    try:
        import cache
        cache.bump("index")
        cache.bump("stats")
    except Exception:
        pass


# ----------------------------- индексация -----------------------------
def index_path(path: Path, source: str, chat_id) -> int:
    """Разобрать файл, нарезать на чанки, заэмбедить и добавить в векторную базу
    (через фасад vectorstore). Возвращает число добавленных чанков.

    Точки получают детерминированные id (uuid5 по source+номер чанка), поэтому
    повторная индексация того же документа заменяет те же точки, а не плодит дубли.
    В payload проставляются служебные поля tg=True и tg_chat_id — по ним работает
    удаление delete_user/delete_all (см. ниже)."""
    import retriever
    import uuid
    import loaders
    import vectorstore
    from ingest import chunk_text

    items = []
    for part in loaders.load_file(Path(path)):
        for ch in chunk_text(part.get("text") or "",
                             int(settings.get("CHUNK_SIZE")),
                             int(settings.get("CHUNK_OVERLAP"))):
            if ch.strip():
                items.append((ch, part.get("page")))
    if not items:
        return 0
    try:
        batch = max(1, int(settings.get("EMBED_BATCH") or 32))
    except Exception:
        batch = 32
    vecs = retriever._embedder().encode([t for t, _ in items],
                                        normalize_embeddings=True, batch_size=batch)
    ext = Path(path).suffix.lower().lstrip(".")
    day = time.strftime("%Y-%m-%d")
    pts = []
    for i, ((t, pg), v) in enumerate(zip(items, vecs)):
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{i}"))
        pts.append({
            "id": pid,
            "vector": (v.tolist() if hasattr(v, "tolist") else list(v)),
            "payload": {"text": t, "source": source, "page": pg, "ftype": ext,
                        "doc_category": "document", "indexed_at": day,
                        "tg": True, "tg_chat_id": int(chat_id)},
        })
    # Чистим прежние точки этого source (переиндексация): детерминированные id
    # перезапишут совпадающие, а этот вызов убирает «хвост», если чанков стало меньше.
    try:
        vectorstore.delete({"source": source})
    except Exception:
        pass
    vectorstore.upsert(pts)
    _bump()
    return len(items)


def save_and_index(chat_id, tmp_path: str, name: str) -> dict:
    """Сохранить присланный файл в папку пользователя и проиндексировать его.

    Различаем два исхода:
      - индексация прошла, но текста нет (chunks==0 без исключения) — файл бесполезен,
        удаляем его;
      - индексация упала с исключением (эмбеддер/векторная база/парсер) — файл НЕ теряем,
        оставляем на диске и возвращаем {chunks: 0, error: ...} для повторной попытки."""
    d = user_dir(chat_id)
    d.mkdir(parents=True, exist_ok=True)
    dest = _unique(d / _safe_name(name))
    shutil.copyfile(tmp_path, dest)
    try:
        n = index_path(dest, _source(chat_id, dest.name), chat_id)
    except Exception as e:
        print(f"[tg_train] индексация {dest.name}: {e}")
        return {"name": dest.name, "chunks": 0, "error": str(e)}
    if n == 0:
        try:
            dest.unlink()
        except Exception:
            pass
    return {"name": dest.name, "chunks": n}


# ----------------------------- удаление точек векторной базы -----------------------------
def _delete_by(flt: dict) -> None:
    """Удалить точки по нейтральному фильтру (поле→значение) через фасад vectorstore."""
    try:
        import vectorstore
        vectorstore.delete(flt)
        _bump()
    except Exception as e:
        print(f"[tg_train] удаление точек: {e}")


# ----------------------------- список и удаление файлов -----------------------------
def list_files(chat_id) -> list:
    d = user_dir(chat_id)
    out = []
    if d.exists():
        for f in sorted(d.iterdir(), key=lambda x: x.name.lower()):
            if f.is_file():
                st = f.stat()
                out.append({"name": f.name, "size": st.st_size, "mtime": st.st_mtime})
    return out


def user_file_counts() -> dict:
    """{chat_id: число файлов} по папкам пользователей."""
    out, r = {}, _root()
    if r.exists():
        for d in r.iterdir():
            if d.is_dir() and d.name.lstrip("-").isdigit():
                out[int(d.name)] = sum(1 for f in d.iterdir() if f.is_file())
    return out


def delete_file(chat_id, name: str) -> bool:
    f = user_dir(chat_id) / _safe_name(name)
    if f.exists() and f.is_file():
        try:
            f.unlink()
        except Exception as e:
            print(f"[tg_train] не удалён файл {f}: {e}")
    _delete_by({"source": _source(chat_id, _safe_name(name))})
    return True


def delete_user(chat_id) -> bool:
    d = user_dir(chat_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    _delete_by({"tg_chat_id": int(chat_id)})
    return True


def delete_all() -> bool:
    r = _root()
    if r.exists():
        shutil.rmtree(r, ignore_errors=True)
    _delete_by({"tg": True})
    return True
