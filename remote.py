"""Удалённые хосты: деплой Qdrant/LightRAG по SSH, перенос данных, переключение.

Поддержка ОС: linux | macos | windows. Аутентификация — логин/пароль (paramiko).
Перенос: документы (SFTP) и снапшоты Qdrant (HTTP API). Переключение бэкенда —
через настройки (QDRANT_URL / LLM_BASE_URL), с возможностью вернуться на локальный.

ОГРАНИЧЕНИЯ (важно):
  - Linux: для установки Docker нужен root или passwordless sudo на удалённом хосте.
  - macOS/Windows: Docker должен быть предустановлен (Docker Desktop), SSH включён.
  - Пароли хранятся в remote_hosts.json (чувствительно) — держите файл закрытым,
    а хосты — в доверенной сети.
"""
from __future__ import annotations
import json
import threading
import time
from pathlib import Path

import httpx

import settings

ROOT = Path(__file__).resolve().parent
_HOSTS_FILE = ROOT / "remote_hosts.json"
_LOG = "/tmp/rag_remote.log"

_job = {"running": False, "started": None, "finished": None, "ok": None,
        "log": "", "label": "", "logfile": _LOG}

# --- команды установки по ОС ---
_QDRANT_CMDS = {
    "linux": [
        "command -v docker >/dev/null 2>&1 || (curl -fsSL https://get.docker.com | sudo -n sh)",
        "(sudo -n docker rm -f rag_qdrant 2>/dev/null; sudo -n docker run -d --restart unless-stopped "
        "-p 6333:6333 -v qdrant_storage:/qdrant/storage --name rag_qdrant qdrant/qdrant:v1.12.4) "
        "|| (docker rm -f rag_qdrant 2>/dev/null; docker run -d --restart unless-stopped "
        "-p 6333:6333 -v qdrant_storage:/qdrant/storage --name rag_qdrant qdrant/qdrant:v1.12.4)",
    ],
    "macos": [
        "docker rm -f rag_qdrant 2>/dev/null; docker run -d --restart unless-stopped "
        "-p 6333:6333 -v qdrant_storage:/qdrant/storage --name rag_qdrant qdrant/qdrant:v1.12.4",
    ],
    "windows": [
        "docker rm -f rag_qdrant; docker run -d --restart unless-stopped "
        "-p 6333:6333 -v qdrant_storage:/qdrant/storage --name rag_qdrant qdrant/qdrant:v1.12.4",
    ],
}
_LIGHTRAG_DEPS = "lightrag-hku==1.3.0 nano-vectordb==0.0.4.3 tiktoken==0.8.0 networkx==3.4.2"
_LIGHTRAG_CMDS = {
    "linux": [
        "python3 -m venv ~/rag_lightrag_env",
        f"~/rag_lightrag_env/bin/pip install -q {_LIGHTRAG_DEPS}",
    ],
    "macos": [
        "python3 -m venv ~/rag_lightrag_env",
        f"~/rag_lightrag_env/bin/pip install -q {_LIGHTRAG_DEPS}",
    ],
    "windows": [
        "py -m venv %USERPROFILE%\\rag_lightrag_env",
        f"%USERPROFILE%\\rag_lightrag_env\\Scripts\\pip install {_LIGHTRAG_DEPS}",
    ],
}


# ============================ хранилище хостов ============================
def _load() -> dict:
    if _HOSTS_FILE.exists():
        try:
            return json.loads(_HOSTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"hosts": [], "active": None, "local_backup": None}


def _save(d: dict) -> None:
    _HOSTS_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def _find(d: dict, name: str) -> dict | None:
    return next((h for h in d.get("hosts", []) if h.get("name") == name), None)


def list_hosts() -> dict:
    d = _load()
    hosts = []
    for h in d.get("hosts", []):
        hh = dict(h)
        hh["password"] = "••••" if h.get("password") else ""  # маскируем
        hosts.append(hh)
    return {"hosts": hosts, "active": d.get("active")}


def save_host(host: dict) -> dict:
    name = (host.get("name") or "").strip()
    if not name or not host.get("host") or not host.get("user"):
        return {"ok": False, "msg": "нужны как минимум: имя, host, user"}
    d = _load()
    existing = _find(d, name)
    rec = {
        "name": name,
        "os": host.get("os", "linux"),
        "host": host.get("host"),
        "port": int(host.get("port") or 22),
        "user": host.get("user"),
        # пустой пароль = не менять (если хост уже есть)
        "password": host.get("password") or (existing.get("password") if existing else ""),
        "qdrant_url": (host.get("qdrant_url") or "").strip(),
        "llm_url": (host.get("llm_url") or "").strip(),
        "docs_dir": (host.get("docs_dir") or "").strip(),
    }
    d["hosts"] = [rec if h.get("name") == name else h for h in d.get("hosts", [])]
    if not existing:
        d["hosts"].append(rec)
    _save(d)
    return {"ok": True, "msg": f"хост «{name}» сохранён"}


def delete_host(name: str) -> dict:
    d = _load()
    d["hosts"] = [h for h in d.get("hosts", []) if h.get("name") != name]
    if d.get("active") == name:
        d["active"] = None
    _save(d)
    return {"ok": True, "msg": "хост удалён"}


# ============================ SSH / фоновые задачи ============================
def _ssh(h: dict):
    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(h["host"], port=int(h.get("port", 22)), username=h["user"],
              password=h.get("password", ""), timeout=25)
    return c


def _bg(label: str, fn) -> dict:
    if _job["running"]:
        return {"ok": False, "msg": "удалённая операция уже идёт"}

    def run():
        _job.update(running=True, started=time.time(), finished=None, ok=None,
                    log="", label=label)
        with open(_LOG, "w", buffering=1, errors="ignore") as fp:
            def log(msg):
                fp.write(str(msg).rstrip() + "\n")
                fp.flush()
            try:
                ok = fn(log)
                _job["ok"] = bool(ok)
            except Exception as e:
                log(f"ОШИБКА: {e}")
                _job["ok"] = False
        _job["log"] = _tail(_LOG)
        _job["running"] = False
        _job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": f"{label}: запущено (лог — в «Состояние и операции»)"}


def _tail(path: str, n: int = 6000) -> str:
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read()[-n:]
    except Exception:
        return ""


def status() -> dict:
    d = dict(_job)
    if d.get("running"):
        d["log"] = _tail(_LOG)
    return d


# ============================ деплой ============================
def deploy(name: str, what: str) -> dict:
    d = _load()
    h = _find(d, name)
    if not h:
        return {"ok": False, "msg": "хост не найден"}
    cmds = (_QDRANT_CMDS if what == "qdrant" else _LIGHTRAG_CMDS).get(h.get("os", "linux"))
    if not cmds:
        return {"ok": False, "msg": "неизвестная ОС"}

    def fn(log):
        log(f"== Деплой {what} на {h['name']} ({h['host']}, {h['os']}) ==")
        try:
            c = _ssh(h)
        except Exception as e:
            log(f"SSH не подключился: {e}")
            return False
        ok = True
        try:
            for cmd in cmds:
                log(f"$ {cmd}")
                rc, out = _exec(c, cmd)
                log(out.strip())
                log(f"[rc={rc}]")
                if rc != 0:
                    ok = False
                    break
        finally:
            c.close()
        log("Готово." if ok else "Завершено с ошибкой.")
        return ok

    return _bg(f"Деплой {what} → {name}", fn)


def _exec(c, cmd, timeout=3600):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="ignore")
    err = stderr.read().decode(errors="ignore")
    rc = stdout.channel.recv_exit_status()
    return rc, (out + err)


# ============================ перенос документов (SFTP) ============================
def transfer_docs(name: str, direction: str) -> dict:
    d = _load()
    h = _find(d, name)
    if not h:
        return {"ok": False, "msg": "хост не найден"}
    local = Path(settings.get("DOCS_DIR")).expanduser()
    remote = h.get("docs_dir") or ("/opt/db" if h.get("os") != "windows" else "C:/rag/db")

    def fn(log):
        try:
            c = _ssh(h)
            sftp = c.open_sftp()
        except Exception as e:
            log(f"SSH/SFTP не подключился: {e}")
            return False
        n = 0
        try:
            if direction == "push":
                log(f"Документы: локально {local} → {h['host']}:{remote}")
                _sftp_mkdirs(sftp, remote)
                for p in local.rglob("*"):
                    if p.is_file():
                        rel = p.relative_to(local).as_posix()
                        dst = remote.rstrip("/") + "/" + rel
                        _sftp_mkdirs(sftp, dst.rsplit("/", 1)[0])
                        sftp.put(str(p), dst)
                        n += 1
                        if n % 50 == 0:
                            log(f"  отправлено {n} файлов...")
            else:  # pull
                log(f"Документы: {h['host']}:{remote} → локально {local}")
                local.mkdir(parents=True, exist_ok=True)
                for rpath in _sftp_walk(sftp, remote):
                    rel = rpath[len(remote):].lstrip("/")
                    dst = local / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(rpath, str(dst))
                    n += 1
                    if n % 50 == 0:
                        log(f"  получено {n} файлов...")
        except Exception as e:
            log(f"ОШИБКА переноса: {e}")
            return False
        finally:
            c.close()
        log(f"Готово. Файлов перенесено: {n}.")
        return True

    return _bg(f"Документы {'→' if direction == 'push' else '←'} {name}", fn)


def _sftp_mkdirs(sftp, path):
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    cur = "/" if path.startswith("/") else ""
    for part in parts:
        cur = (cur + part) if cur in ("", "/") else (cur + "/" + part)
        cur = cur if cur.startswith("/") or ":" in cur else "/" + cur
        try:
            sftp.stat(cur)
        except IOError:
            try:
                sftp.mkdir(cur)
            except Exception:
                pass


def _sftp_walk(sftp, root):
    import stat as _stat
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = sftp.listdir_attr(d)
        except Exception:
            continue
        for e in entries:
            full = d.rstrip("/") + "/" + e.filename
            if _stat.S_ISDIR(e.st_mode):
                stack.append(full)
            else:
                yield full


# ============================ перенос снапшота Qdrant (HTTP) ============================
def transfer_snapshot(name: str, direction: str) -> dict:
    d = _load()
    h = _find(d, name)
    if not h:
        return {"ok": False, "msg": "хост не найден"}
    remote_q = h.get("qdrant_url")
    if not remote_q:
        return {"ok": False, "msg": "у хоста не задан qdrant_url"}
    local_q = settings.get("QDRANT_URL")
    coll = settings.get("QDRANT_COLLECTION")
    src, dst = (local_q, remote_q) if direction == "push" else (remote_q, local_q)

    def fn(log):
        try:
            log(f"Снапшот коллекции {coll}: {src} → {dst}")
            r = httpx.post(f"{src}/collections/{coll}/snapshots", timeout=600)
            snap = (r.json().get("result", {}) or {}).get("name")
            if not snap:
                log(f"не удалось создать снапшот: {r.text[:300]}")
                return False
            log(f"снапшот: {snap}, скачиваю...")
            data = httpx.get(f"{src}/collections/{coll}/snapshots/{snap}", timeout=1800).content
            log(f"скачано {len(data)} байт, загружаю на приёмник...")
            up = httpx.post(f"{dst}/collections/{coll}/snapshots/upload?priority=snapshot",
                            files={"snapshot": (snap, data)}, timeout=1800)
            ok = up.status_code < 300
            log(f"[upload rc={up.status_code}] {up.text[:300]}")
            return ok
        except Exception as e:
            log(f"ОШИБКА: {e}")
            return False

    return _bg(f"Snapshot {'→' if direction == 'push' else '←'} {name}", fn)


# ============================ переключение бэкенда ============================
def switch(name: str) -> dict:
    d = _load()
    h = _find(d, name)
    if not h:
        return {"ok": False, "msg": "хост не найден"}
    if not h.get("qdrant_url") and not h.get("llm_url"):
        return {"ok": False, "msg": "у хоста не заданы qdrant_url/llm_url для переключения"}
    if not d.get("local_backup"):
        d["local_backup"] = {
            "QDRANT_URL": settings.get("QDRANT_URL"),
            "LLM_BACKEND": settings.get("LLM_BACKEND"),
            "LLM_BASE_URL": settings.get("LLM_BASE_URL"),
            "OLLAMA_URL": settings.get("OLLAMA_URL"),
        }
    changes = {}
    if h.get("qdrant_url"):
        changes["QDRANT_URL"] = h["qdrant_url"]
    if h.get("llm_url"):
        changes["LLM_BACKEND"] = "openai"
        changes["LLM_BASE_URL"] = h["llm_url"]
    settings.update(changes)
    d["active"] = name
    _save(d)
    return {"ok": True, "msg": f"переключено на «{name}». Перезапустите сервис, чтобы применить адрес Qdrant.",
            "restart": True}


def restore() -> dict:
    d = _load()
    bak = d.get("local_backup")
    if not bak:
        return {"ok": False, "msg": "нет сохранённой локальной конфигурации"}
    settings.update(bak)
    d["active"] = None
    d["local_backup"] = None
    _save(d)
    return {"ok": True, "msg": "возврат на локальный бэкенд. Перезапустите сервис.",
            "restart": True}
