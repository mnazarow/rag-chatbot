"""Резервное копирование и восстановление.

Три области (scope):
  - settings : только настройки (runtime_config.json, .env)
  - service  : настройки + служебные данные БЕЗ каталога документов
               (журнал/оценки, тайминги, SSH-хосты, граф знаний, LoRA-адаптер)
  - full     : всё вышеперечисленное + каталог документов (DOCS_DIR)

Каждый архив — tar.gz с manifest.json, где для каждого файла записан SHA-256.
Целостность проверяется пересчётом хешей; рядом кладётся <архив>.sha256 (хеш
всего файла архива). Восстановление выполняется только после проверки целостности
и с защитой путей (файлы пишутся строго внутрь ROOT/DOCS_DIR).

Векторный индекс Qdrant НЕ включается (он производный от документов) — после
восстановления служебных данных/документов выполните переиндексацию.
"""
from __future__ import annotations
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import settings

ROOT = Path(__file__).resolve().parent
BACKUP_DIR = ROOT / "backups"
MANIFEST_NAME = "manifest.json"
VERSION = 1

# Файлы и папки в каталоге проекта (kind="root")
_SETTINGS_FILES = ["runtime_config.json", ".env"]
_SERVICE_FILES = _SETTINGS_FILES + ["rag_logs.db", "ingest_stats.json",
                                    "remote_hosts.json"]
_SERVICE_DIRS = ["graph_storage", "finetune/adapter", "finetune/data"]

SCOPES = ("settings", "service", "full")
SCOPE_LABEL = {"settings": "настройки",
               "service": "служебные данные (без каталога)",
               "full": "настройки и данные с каталогом"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_stream(fileobj) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: fileobj.read(1 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


def _safe_rel(rel: str) -> str:
    parts = [s for s in str(rel).replace("\\", "/").split("/")
             if s not in ("", ".", "..")]
    return "/".join(parts)


def _iter_files(scope: str):
    """Перечислить (kind, relpath, abspath) для выбранной области."""
    root_files = _SETTINGS_FILES if scope == "settings" else _SERVICE_FILES
    for f in root_files:
        p = ROOT / f
        if p.is_file():
            yield ("root", f, p)
    if scope in ("service", "full"):
        for d in _SERVICE_DIRS:
            base = ROOT / d
            if base.is_dir():
                for p in sorted(base.rglob("*")):
                    if p.is_file():
                        yield ("root", str(p.relative_to(ROOT)), p)
    if scope == "full":
        docs = Path(settings.get("DOCS_DIR")).expanduser()
        if docs.is_dir():
            import fsutil
            for p in sorted(fsutil.walk_files(docs)):
                yield ("docs", str(p.relative_to(docs)), p)


def create(scope: str, progress=None) -> dict:
    """Создать архив резервной копии. progress(done,total,name) — опционально."""
    if scope not in SCOPES:
        return {"ok": False, "msg": f"неизвестная область: {scope}"}
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"backup-{scope}-{ts}.tar.gz"
    arc = BACKUP_DIR / name

    files = list(_iter_files(scope))
    total = len(files)
    manifest = {"version": VERSION, "scope": scope, "created": time.time(),
                "app": "rag-chatbot", "files": []}
    try:
        # dereference=True — символьные ссылки (.env может быть симлинком) кладём как файлы
        with tarfile.open(arc, "w:gz", dereference=True) as tar:
            for i, (kind, rel, p) in enumerate(files, 1):
                try:
                    sha = _sha256(p)
                    sz = p.stat().st_size
                except Exception as e:
                    manifest["files"].append({"kind": kind, "path": rel,
                                              "error": str(e)[:120]})
                    continue
                manifest["files"].append({"kind": kind, "path": rel,
                                          "sha256": sha, "size": sz})
                tar.add(str(p), arcname=f"{kind}/{rel}", recursive=False)
                if progress:
                    progress(i, total, rel)
            data = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size = len(data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))
    except Exception as e:
        try:
            arc.unlink(missing_ok=True)
        except Exception:
            pass
        return {"ok": False, "msg": f"ошибка создания архива: {e}"}

    arc_sha = _sha256(arc)
    try:
        (BACKUP_DIR / (name + ".sha256")).write_text(arc_sha, encoding="utf-8")
    except Exception:
        pass

    v = verify(arc)
    return {"ok": True, "name": name, "path": str(arc),
            "size": arc.stat().st_size, "sha256": arc_sha, "scope": scope,
            "files": sum(1 for f in manifest["files"] if "sha256" in f),
            "integrity_ok": v.get("integrity_ok"), "verify": v}


def verify(path) -> dict:
    """Проверить целостность архива по manifest.json (пересчёт SHA-256)."""
    path = Path(path)
    if not path.exists():
        return {"ok": False, "integrity_ok": False, "msg": "файл не найден"}
    try:
        with tarfile.open(path, "r:*") as tar:
            try:
                mf = tar.extractfile(MANIFEST_NAME)
                manifest = json.loads(mf.read().decode("utf-8"))
            except Exception:
                return {"ok": False, "integrity_ok": False,
                        "msg": "нет manifest.json — это не резервная копия RAG"}
            names = set(tar.getnames())
            entries = [e for e in manifest.get("files", []) if e.get("sha256")]
            issues, checked = [], 0
            for e in entries:
                arcname = f"{e['kind']}/{e['path']}"
                if arcname not in names:
                    issues.append(f"отсутствует: {arcname}")
                    continue
                f = tar.extractfile(arcname)
                if f is None or _sha256_stream(f) != e["sha256"]:
                    issues.append(f"повреждён (хеш): {arcname}")
                else:
                    checked += 1
    except Exception as e:
        return {"ok": False, "integrity_ok": False,
                "msg": f"не удалось открыть архив: {e}"}
    integrity = len(issues) == 0
    return {"ok": True, "integrity_ok": integrity,
            "scope": manifest.get("scope"), "created": manifest.get("created"),
            "version": manifest.get("version"), "files": len(entries),
            "checked": checked, "issues": issues[:50],
            "scope_label": SCOPE_LABEL.get(manifest.get("scope"), manifest.get("scope"))}


def restore(path, progress=None) -> dict:
    """Восстановить из архива — ТОЛЬКО после успешной проверки целостности.
    Перед перезаписью делает страховочную копию текущих служебных данных."""
    path = Path(path)
    v = verify(path)
    if not v.get("ok") or not v.get("integrity_ok"):
        return {"ok": False, "msg": "архив не прошёл проверку целостности",
                "verify": v}

    # страховочная копия текущего состояния (служебные данные, без каталога)
    safety = None
    try:
        s = create("service")
        if s.get("ok"):
            safety = s.get("name")
    except Exception:
        pass

    docs = Path(settings.get("DOCS_DIR")).expanduser().resolve()
    root = ROOT.resolve()
    restored, skipped = 0, []
    try:
        with tarfile.open(path, "r:*") as tar:
            manifest = json.loads(tar.extractfile(MANIFEST_NAME).read().decode("utf-8"))
            entries = [e for e in manifest.get("files", []) if e.get("sha256")]
            total = len(entries)
            for i, e in enumerate(entries, 1):
                rel = _safe_rel(e["path"])
                if not rel:
                    skipped.append(e.get("path"))
                    continue
                base = root if e["kind"] == "root" else docs
                dest = (base / Path(*rel.split("/"))).resolve()
                if base not in dest.parents and dest != base:
                    skipped.append(rel)
                    continue
                src = tar.extractfile(f"{e['kind']}/{e['path']}")
                if src is None:
                    skipped.append(rel)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as out:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                restored += 1
                if progress:
                    progress(i, total, rel)
    except Exception as e:
        return {"ok": False, "msg": f"ошибка восстановления: {e}",
                "restored": restored, "safety_backup": safety}

    return {"ok": True, "restored": restored, "skipped": skipped[:50],
            "scope": v.get("scope"), "safety_backup": safety,
            "note": "Перезапустите сервис; при восстановлении документов/служебных "
                    "данных выполните переиндексацию для обновления базы Qdrant."}


def list_backups() -> list:
    """Список существующих архивов с метаданными (без пересчёта хешей)."""
    if not BACKUP_DIR.exists():
        return []
    out = []
    for p in sorted(BACKUP_DIR.glob("backup-*.tar.gz"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        scope = None
        parts = p.name.split("-")
        if len(parts) >= 2 and parts[1] in SCOPES:
            scope = parts[1]
        sha = None
        side = BACKUP_DIR / (p.name + ".sha256")
        if side.exists():
            try:
                sha = side.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        out.append({"name": p.name, "scope": scope,
                    "scope_label": SCOPE_LABEL.get(scope, scope),
                    "size": p.stat().st_size, "created": p.stat().st_mtime,
                    "sha256": sha})
    return out


def delete(name: str) -> dict:
    """Удалить архив и его .sha256 (только из папки backups, без обхода путей)."""
    base = (name or "").strip()
    if not base or "/" in base or "\\" in base or not base.endswith(".tar.gz"):
        return {"ok": False, "msg": "недопустимое имя"}
    p = BACKUP_DIR / base
    if not p.exists():
        return {"ok": False, "msg": "не найдено"}
    try:
        p.unlink()
        (BACKUP_DIR / (base + ".sha256")).unlink(missing_ok=True)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


def path_for(name: str) -> Path | None:
    """Безопасный путь к архиву по имени (для скачивания)."""
    base = (name or "").strip()
    if not base or "/" in base or "\\" in base or not base.endswith(".tar.gz"):
        return None
    p = BACKUP_DIR / base
    return p if p.is_file() else None
