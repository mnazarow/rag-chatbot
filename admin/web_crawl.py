"""Домен веб-парсинга административной подсистемы RAG (вынесено из admin_ops.py).

Полный конвейер работы с сайтами:
  - хранение списка сайтов, статистики, исключений, пер-сайтовых настроек,
    авторизации и отпечатков инкрементального парсинга (в БД kv_store);
  - загрузка страниц/файлов (``_Renderer`` для JS-рендера, HTTP-клиент, SSRF-защита);
  - обход сайта (``_web_crawl``: robots/sitemap/ссылки/глубина/лимиты);
  - извлечение текста, структура сайтов (``web_structure``) и запуск индексации
    (``ingest_web``), удаление (``delete_web``).

Модуль самодостаточен и НЕ импортирует admin_ops — общие чистые хелперы берутся из
``admin.common``/``admin.jobs``. ``admin_ops`` ре-экспортирует эти имена (в т.ч.
``_web_job``), поэтому обратная совместимость (``admin_ops.<имя>``) сохранена.
Фоновая задача парисинга хранит статус в общем словаре ``_web_job`` — тот же объект
читают ``admin_ops.status``/``active_jobs``/``server_load`` (импортируют его отсюда).
"""
from __future__ import annotations

import ipaddress as _ipaddress
import json as _json
import os
import socket as _socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

import settings
import db
import vectorstore

from admin.common import (
    _fmt_bytes, _sha256_file, _file_method, _extract_summary, _ingest_stats,
)
from admin.jobs import _tail, _read_full_log

ROOT = Path(__file__).resolve().parent.parent

# Статус фоновой задачи парсинга сайтов. Живёт здесь (домен-владелец), а
# admin_ops импортирует ТОТ ЖЕ объект — словарь только мутируется (.update/[]),
# никогда не переприсваивается, поэтому изменения видны во всех модулях.
_web_job = {"running": False, "started": None, "finished": None, "ok": None,
            "log": "", "summary": "", "logfile": ""}


_WEB_SOURCES = ROOT / "web_sources.txt"
_WEB_STATS = ROOT / "web_stats.json"    # результаты парсинга по каждому сайту (ошибки, лимиты)
_web_dl_lock = threading.Lock()         # защита выбора имени файла при параллельном скачивании


# Список сайтов и статистику парсинга храним в БД (таблица kv_store), чтобы они
# переживали пересоздание контейнера в Docker: файлы в /app не персистентны, а
# rag_logs.db смонтирован. Старые файлы web_sources.txt/web_stats.json один раз
# мигрируются в БД (для нативных установок, где они уже есть).

def _web_sources_load() -> list:
    """URL сохранённых сайтов из БД (с одноразовой миграцией из web_sources.txt)."""
    raw = db.kv_get("web_sources")
    if raw is None and _WEB_SOURCES.exists():
        try:
            raw = _WEB_SOURCES.read_text(encoding="utf-8")
            db.kv_set("web_sources", raw)
        except Exception:
            raw = ""
    return [u.strip() for u in (raw or "").splitlines() if u.strip()]


def _web_sources_save(urls: list) -> None:
    try:
        db.kv_set("web_sources", "\n".join(urls))
    except Exception as e:
        print(f"[web] не удалось сохранить список сайтов: {e}")


def _web_stats_load() -> dict:
    """Результаты парсинга по сайтам из БД (с миграцией из web_stats.json)."""
    raw = db.kv_get("web_stats")
    if raw is None and _WEB_STATS.exists():
        try:
            raw = _WEB_STATS.read_text(encoding="utf-8")
            db.kv_set("web_stats", raw)
        except Exception:
            raw = ""
    try:
        return _json.loads(raw) if raw else {}
    except Exception:
        return {}


def _web_stats_save(stats: dict) -> None:
    try:
        db.kv_set("web_stats", _json.dumps(stats, ensure_ascii=False))
    except Exception as e:
        print(f"[web] не удалось сохранить статистику парсинга: {e}")


def _web_excludes_load() -> dict:
    """Исключения по ключевым словам для каждого сайта: {url: [kw, ...]} из БД."""
    raw = db.kv_get("web_excludes")
    try:
        d = _json.loads(raw) if raw else {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _web_excludes_save(m: dict) -> None:
    try:
        db.kv_set("web_excludes", _json.dumps(m, ensure_ascii=False))
    except Exception as e:
        print(f"[web] не удалось сохранить исключения: {e}")


def _web_parse_keywords(val) -> list:
    """Строку/список ключевых слов → нормализованный список (нижний регистр, без дублей)."""
    import re
    if isinstance(val, str):
        parts = re.split(r"[\s,;\n]+", val)
    else:
        parts = list(val or [])
    out, seen = [], set()
    for p in parts:
        k = str(p).strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out[:50]


def _web_excluded(url: str, excludes) -> bool:
    """URL исключён, если содержит любое из ключевых слов (без учёта регистра)."""
    if not excludes:
        return False
    u = (url or "").lower()
    return any(kw in u for kw in excludes if kw)


def web_set_excludes(url: str, keywords) -> dict:
    """Задать/обновить список исключений по ключевым словам для сайта. Применяется при
    следующем парсинге этого сайта (ручном или по расписанию)."""
    url = (url or "").strip()
    if not url:
        return {"ok": False, "msg": "не указан URL сайта"}
    kws = _web_parse_keywords(keywords)
    m = _web_excludes_load()
    if kws:
        m[url] = kws
    else:
        m.pop(url, None)
    _web_excludes_save(m)
    return {"ok": True, "url": url, "exclude": kws,
            "msg": (f"исключения сохранены ({len(kws)} слов) — применятся при следующем "
                    "парсинге этого сайта") if kws else "исключения очищены"}


def _web_excludes_all_load() -> list:
    """Глобальные исключения по ключевым словам — действуют для ВСЕХ сайтов."""
    return _web_parse_keywords(db.kv_get("web_excludes_all") or "")


def _web_excludes_all_save(kws: list) -> None:
    try:
        db.kv_set("web_excludes_all", ", ".join(kws))
    except Exception as e:
        print(f"[web] не удалось сохранить глобальные исключения: {e}")


def web_set_excludes_all(keywords) -> dict:
    """Задать глобальные исключения (для всех сайтов). Объединяются с исключениями
    конкретного сайта при парсинге."""
    kws = _web_parse_keywords(keywords)
    _web_excludes_all_save(kws)
    return {"ok": True, "exclude_all": kws,
            "msg": (f"глобальные исключения сохранены ({len(kws)} слов) — применятся ко "
                    "всем сайтам при следующем парсинге") if kws else "глобальные исключения очищены"}


def _web_site_excludes(url: str, per_site_map: dict | None = None,
                       global_list: list | None = None) -> list:
    """Итоговые исключения для сайта: глобальные + персональные (объединение без дублей)."""
    per_site_map = _web_excludes_load() if per_site_map is None else per_site_map
    global_list = _web_excludes_all_load() if global_list is None else global_list
    merged = list(dict.fromkeys([*(global_list or []), *(per_site_map.get(url) or [])]))
    return merged


# ---------- пер-сайтовые настройки обхода ----------
# kv "web_site_cfg": {url: {depth,max_pages,max_files,concurrency,same_domain,js_render}}
# Пустые/None-поля означают «брать глобальную настройку».
_WEB_CFG_KEYS = ("depth", "max_pages", "max_files", "concurrency", "same_domain",
                 "js_render", "crawl_delay")


def _web_cfg_load() -> dict:
    raw = db.kv_get("web_site_cfg")
    try:
        d = _json.loads(raw) if raw else {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _web_cfg_save(m: dict) -> None:
    try:
        db.kv_set("web_site_cfg", _json.dumps(m, ensure_ascii=False))
    except Exception as e:
        print(f"[web] не удалось сохранить настройки сайта: {e}")


def web_set_site_cfg(url: str, cfg: dict) -> dict:
    """Задать пер-сайтовые настройки обхода (пустые поля = глобальные). Применяются при
    следующем парсинге этого сайта."""
    url = (url or "").strip()
    if not url:
        return {"ok": False, "msg": "не указан URL сайта"}
    cfg = cfg or {}
    clean = {}
    for k in ("depth", "max_pages", "max_files", "concurrency"):
        v = cfg.get(k)
        if v not in (None, "", "auto"):
            try:
                clean[k] = max(0, int(v))
            except (TypeError, ValueError):
                pass
    for k in ("same_domain", "js_render", "crawl_delay"):
        v = cfg.get(k)
        if v in ("", None, "global", "auto"):
            continue
        if isinstance(v, str):
            clean[k] = v.lower() in ("1", "true", "on", "yes", "да")
        else:
            clean[k] = bool(v)
    # расписание автопарсинга сайта (пусто/global = следовать глобальному)
    sch = (cfg.get("schedule") or "").strip().lower()
    if sch == "off" or (sch not in ("", "global", "auto") and _web_sched_interval(sch) is not None):
        clean["schedule"] = sch
    m = _web_cfg_load()
    if clean:
        m[url] = clean
    else:
        m.pop(url, None)
    _web_cfg_save(m)
    return {"ok": True, "url": url, "cfg": clean,
            "msg": "настройки сайта сохранены" if clean else "настройки сайта сброшены (глобальные)"}


def _web_sched_interval(sch) -> int | None:
    """Интервал автопарсинга сайта в секундах по значению schedule, или None
    (не по пер-сайтовому расписанию: пусто/global/off)."""
    s = (sch or "").strip().lower()
    if s in ("", "global", "auto", "off", "none"):
        return None
    if s == "hourly":
        return 3600
    if s == "daily":
        return 86400
    if s == "weekly":
        return 604800
    import re as _re
    mm = _re.match(r"^(\d+)h$", s)
    if mm:
        return max(1, int(mm.group(1))) * 3600
    return None


def _web_sched_last_load() -> dict:
    raw = db.kv_get("web_sched_last")
    try:
        d = _json.loads(raw) if raw else {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _web_sched_last_save(m: dict) -> None:
    try:
        db.kv_set("web_sched_last", _json.dumps(m, ensure_ascii=False))
    except Exception:
        pass


def web_sched_due() -> list:
    """Сайты, которым пора перепарситься по их пер-сайтовому расписанию (интервал истёк)."""
    cfg = _web_cfg_load()
    last = _web_sched_last_load()
    now = time.time()
    due = []
    for u in _web_sources_load():
        iv = _web_sched_interval((cfg.get(u) or {}).get("schedule"))
        if iv is None:
            continue
        if now - float(last.get(u, 0) or 0) >= iv:
            due.append(u)
    return due


def web_sched_mark(urls) -> None:
    last = _web_sched_last_load()
    now = time.time()
    for u in (urls or []):
        last[u] = now
    _web_sched_last_save(last)


def _web_eff(url: str, cfg_map: dict | None = None) -> dict:
    """Эффективные настройки обхода сайта: пер-сайтовые поверх глобальных."""
    cfg_map = _web_cfg_load() if cfg_map is None else cfg_map
    c = cfg_map.get(url) or {}

    def _pick(key, gkey, default_int=None):
        v = c.get(key)
        return v if v is not None else settings.get(gkey)

    eff = {
        "depth": max(0, int(c.get("depth") if c.get("depth") is not None
                             else (settings.get("WEB_CRAWL_DEPTH") or 0))),
        "max_pages": max(1, int(c.get("max_pages") if c.get("max_pages") is not None
                                else (settings.get("WEB_MAX_PAGES") or 1))),
        "max_files": max(0, int(c.get("max_files") if c.get("max_files") is not None
                                else (settings.get("WEB_MAX_FILES") or 0))),
        "concurrency": max(1, int(c.get("concurrency") if c.get("concurrency") is not None
                                  else (settings.get("WEB_CONCURRENCY") or 1))),
        "same_domain": bool(c.get("same_domain")) if "same_domain" in c
                       else bool(settings.get("WEB_SAME_DOMAIN")),
        "js_render": bool(c.get("js_render")) if "js_render" in c
                     else bool(settings.get("WEB_JS_RENDER")),
        "crawl_delay": bool(c.get("crawl_delay")) if "crawl_delay" in c
                       else bool(settings.get("WEB_RESPECT_CRAWL_DELAY")),
    }
    return eff


# ---------- авторизация на сайт (секреты) ----------
# kv "web_auth": {url: {type: basic|cookie|header, user, password, cookie, hname, hvalue}}
def _web_auth_load() -> dict:
    raw = db.kv_get("web_auth")
    try:
        d = _json.loads(raw) if raw else {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _web_auth_save(m: dict) -> None:
    try:
        db.kv_set("web_auth", _json.dumps(m, ensure_ascii=False))
    except Exception as e:
        print(f"[web] не удалось сохранить авторизацию сайта: {e}")


def web_set_auth(url: str, auth: dict) -> dict:
    """Задать авторизацию для сайта (Basic-логин, Cookie или произвольный заголовок).
    Секреты хранятся в БД и в API наружу не отдаются открытым текстом."""
    url = (url or "").strip()
    if not url:
        return {"ok": False, "msg": "не указан URL сайта"}
    auth = auth or {}
    typ = (auth.get("type") or "none").strip().lower()
    m = _web_auth_load()
    if typ in ("", "none"):
        m.pop(url, None)
        _web_auth_save(m)
        return {"ok": True, "url": url, "msg": "авторизация отключена"}
    entry = {"type": typ}
    if typ == "basic":
        entry["user"] = (auth.get("user") or "").strip()
        entry["password"] = auth.get("password") or ""
    elif typ == "cookie":
        entry["cookie"] = (auth.get("cookie") or "").strip()
    elif typ == "header":
        entry["hname"] = (auth.get("hname") or "").strip()
        entry["hvalue"] = auth.get("hvalue") or ""
    else:
        return {"ok": False, "msg": "тип авторизации: none|basic|cookie|header"}
    m[url] = entry
    _web_auth_save(m)
    return {"ok": True, "url": url, "type": typ, "msg": f"авторизация ({typ}) сохранена"}


def _web_auth_for(url: str, auth_map: dict | None = None) -> tuple[dict, dict]:
    """Заголовки и cookies для запросов к сайту по его авторизации. → (headers, cookies)."""
    import base64
    auth_map = _web_auth_load() if auth_map is None else auth_map
    a = auth_map.get(url) or {}
    headers, cookies = {}, {}
    typ = a.get("type")
    if typ == "basic" and a.get("user"):
        token = base64.b64encode(f"{a['user']}:{a.get('password','')}".encode()).decode()
        headers["Authorization"] = "Basic " + token
    elif typ == "cookie" and a.get("cookie"):
        for part in a["cookie"].split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                cookies[k.strip()] = v.strip()
    elif typ == "header" and a.get("hname"):
        headers[a["hname"]] = a.get("hvalue", "")
    return headers, cookies


# ---------- отпечатки для инкрементального парсинга ----------
# kv "web_fp": {url: {etag, lastmod, size, kind, path, text}} — обновляется при парсинге.
def _web_fp_load() -> dict:
    raw = db.kv_get("web_fp")
    try:
        d = _json.loads(raw) if raw else {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _web_fp_save(m: dict) -> None:
    try:
        db.kv_set("web_fp", _json.dumps(m, ensure_ascii=False))
    except Exception as e:
        print(f"[web] не удалось сохранить отпечатки: {e}")


def _web_filehash_load() -> dict:
    """Дедуп файлов между сайтами: {sha256: rel_path}."""
    raw = db.kv_get("web_filehash")
    try:
        d = _json.loads(raw) if raw else {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _web_filehash_save(m: dict) -> None:
    try:
        db.kv_set("web_filehash", _json.dumps(m, ensure_ascii=False))
    except Exception:
        pass




def _web_cond_headers(fp: dict | None) -> dict:
    """Условные заголовки If-None-Match/If-Modified-Since из отпечатка."""
    h = {}
    if fp:
        if fp.get("etag"):
            h["If-None-Match"] = fp["etag"]
        if fp.get("lastmod"):
            h["If-Modified-Since"] = fp["lastmod"]
    return h


def _web_slug(url: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9._-]", "_", url)[:120] or "page"


def _web_list() -> list:
    """Список сохранённых сайтов: URL + файл + признак наличия в папке."""
    urls = _web_sources_load()
    webdir = Path(settings.get("DOCS_DIR")).expanduser() / "web"
    stats = _web_stats_load()
    excl = _web_excludes_load()
    cfg = _web_cfg_load()
    auth = _web_auth_load()
    out = []
    for u in urls:
        rel = f"web/{_web_slug(u)}.html"
        st = stats.get(u, {})
        out.append({"url": u, "source": rel,
                    "indexed": (webdir / f"{_web_slug(u)}.html").exists(),
                    "ok": st.get("ok"), "pages": st.get("pages"), "files": st.get("files"),
                    "errors": st.get("errors", []), "limits": st.get("limits", {}),
                    "ts": st.get("ts"),
                    "exclude": excl.get(u, []),
                    "cfg": cfg.get(u, {}),
                    "auth_type": (auth.get(u, {}) or {}).get("type", "none"),
                    "depth_urls": st.get("depth_urls", []),
                    "progress": st.get("progress"),
                    "n_items": len(st.get("items") or [])})
    return out


def web_saved_urls() -> list:
    """Список сохранённых для парсинга сайтов (URL)."""
    return _web_sources_load()


def web_structure() -> dict:
    """Структура спарсенных сайтов в реальном времени: по каждому сайту — прогресс
    и дерево элементов (объединённый документ страниц + скачанные файлы) с признаком
    добавления в БД, числом чанков, способом распознавания, наличием LLM-описания,
    типом, размером, датой изменения и временем обработки. Тяжёлые данные (чанки,
    описания, время обработки) — из тех же источников, что каталог документов."""
    docs = Path(settings.get("DOCS_DIR")).expanduser()
    stats = _web_stats_load()
    urls = _web_sources_load()

    # число чанков и описания по источникам — одним фасет-запросом (кэш index, 15 c)
    def _facets():
        chunks, described = {}, set()
        try:
            for h in vectorstore.facet("source", 100000):
                chunks[h.get("value")] = h.get("count", 0)
            for h in vectorstore.facet("source", 100000, flt={"vision_desc": True}):
                if h.get("value"):
                    described.add(h.get("value"))
        except Exception:
            pass
        return {"chunks": chunks, "described": list(described)}
    try:
        import cache
        fac = cache.get_or_set("webstruct_facet:" + str(vectorstore.backend()) + ":"
                               + str(settings.get("QDRANT_COLLECTION")),
                               15, _facets, ns="index")
    except Exception:
        fac = _facets()
    chunk_map = fac.get("chunks", {})
    desc_set = set(fac.get("described", []))

    # время обработки по файлам из последней индексации
    try:
        proc_map = {k: (v.get("ms") if isinstance(v, dict) else None)
                    for k, v in (_ingest_stats().get("files") or {}).items()}
    except Exception:
        proc_map = {}

    sites = []
    for u in urls:
        st = stats.get(u, {})
        items = []
        for it in (st.get("items") or []):
            src = it.get("source")
            ext = Path(src).suffix if src else ""
            n_ch = chunk_map.get(src, 0) if src else 0
            fpath = (docs / src) if src else None
            mtime = None
            try:
                if fpath and fpath.exists():
                    mtime = fpath.stat().st_mtime
            except Exception:
                pass
            items.append({
                "kind": it.get("kind"), "name": it.get("name"), "url": it.get("url"),
                "source": src, "ok": it.get("ok", True),
                "size": it.get("size", 0), "type": ext.lstrip(".").upper() or "HTML",
                "method": _file_method(ext) if ext else "text",
                "chunks": n_ch, "in_db": n_ch > 0,
                "described": (src in desc_set), "mtime": mtime,
                "proc_ms": proc_map.get(src),
                "pages": it.get("pages"),
            })
        sites.append({"url": u, "ok": st.get("ok"), "ts": st.get("ts"),
                      "progress": st.get("progress") or {"phase": "—", "pct": 0},
                      "errors": st.get("errors", []), "limits": st.get("limits", {}),
                      "items": items})
    return {"sites": sites, "running": bool(_web_job.get("running"))}


def get_web_urls() -> dict:
    sites = _web_list()
    return {"urls": [s["url"] for s in sites], "sites": sites,
            "exclude_all": _web_excludes_all_load(),
            "log": _tail(_web_job["logfile"]) if _web_job.get("logfile") else _web_job.get("log", "")}


def delete_web(url: str) -> dict:
    """Удалить сайт из списка, его файл и чанки из базы знаний (Qdrant)."""
    url = (url or "").strip()
    if not url:
        return {"ok": False, "msg": "не указан URL"}
    # 1) убрать из списка
    sites = [s["url"] for s in _web_list() if s["url"] != url]
    _web_sources_save(sites)
    # убрать сохранённую статистику парсинга по этому сайту
    _st = _web_stats_load()
    if url in _st:
        _st.pop(url, None)
        _web_stats_save(_st)
    # убрать исключения по ключевым словам для этого сайта
    _ex = _web_excludes_load()
    if url in _ex:
        _ex.pop(url, None)
        _web_excludes_save(_ex)
    # убрать пер-сайтовые настройки и авторизацию
    _cfg = _web_cfg_load()
    if url in _cfg:
        _cfg.pop(url, None)
        _web_cfg_save(_cfg)
    _au = _web_auth_load()
    if url in _au:
        _au.pop(url, None)
        _web_auth_save(_au)
    # убрать отпечатки инкремента для страниц/файлов этого домена
    try:
        from urllib.parse import urlparse as _up
        _net = _up(url).netloc
        _fp = _web_fp_load()
        _rm = [k for k in _fp if _up(k).netloc == _net]
        if _rm:
            for k in _rm:
                _fp.pop(k, None)
            _web_fp_save(_fp)
    except Exception:
        pass
    # 2) удалить файл
    slug = _web_slug(url)
    f = Path(settings.get("DOCS_DIR")).expanduser() / "web" / f"{slug}.html"
    if f.exists():
        try:
            f.unlink()
        except Exception:
            pass
    # 3) удалить чанки из векторной базы (Qdrant/Milvus) по source
    src = f"web/{slug}.html"
    try:
        vectorstore.delete({"source": src})
    except Exception as e:
        return {"ok": True, "msg": f"удалён из списка и папки; из векторной базы не удалён: {e}"}
    return {"ok": True, "msg": "сайт удалён из базы знаний"}


def stop_web_parse() -> dict:
    """Запросить кооперативную остановку парсинга сайтов. Обход прекращается между
    страницами/сайтами (в пределах таймаута текущей страницы), уже скачанное — сохраняется."""
    if not _web_job.get("running"):
        return {"ok": False, "msg": "парсинг сайтов не запущен"}
    _web_job["stop"] = True
    return {"ok": True, "msg": "остановка запрошена — обход завершится после текущей страницы"}


def web_reparse_fresh(url: str, index: bool = True) -> dict:
    """Удалить скачанные файлы сайта и его чанки, сбросить отпечатки инкремента —
    и спарсить сайт заново «с нуля» (всё скачивается повторно). Сайт остаётся в списке."""
    from urllib.parse import urlparse as _up
    url = (url or "").strip()
    if not url:
        return {"ok": False, "msg": "не указан URL"}
    if _web_job.get("running"):
        return {"ok": False, "msg": "парсинг сайтов уже идёт — дождитесь завершения"}
    docroot = Path(settings.get("DOCS_DIR")).expanduser()
    slug = _web_slug(url)
    removed = 0
    # 1) страница-сводка сайта + её чанки
    page_rel = f"web/{slug}.html"
    pf = docroot / page_rel
    if pf.exists():
        try:
            pf.unlink(); removed += 1
        except Exception:
            pass
    try:
        vectorstore.delete({"source": page_rel})
    except Exception:
        pass
    # 2) скачанные файлы этого сайта (по его статистике) + их чанки.
    # ВАЖНО: из-за дедупа один и тот же файл может быть общим для нескольких сайтов —
    # такие файлы (на которые ссылаются ДРУГИЕ сайты) не удаляем, иначе осиротим их чанки.
    _all_stats = _web_stats_load()
    st = _all_stats.get(url) or {}
    other_sources = set()
    for _ou, _os in _all_stats.items():
        if _ou == url:
            continue
        for _oi in (_os.get("items") or []):
            if _oi.get("source"):
                other_sources.add(_oi["source"])
    skipped_shared = 0
    for it in (st.get("items") or []):
        if it.get("kind") != "file":
            continue
        rel = it.get("source")
        if not rel:
            continue
        if rel in other_sources:
            skipped_shared += 1
            continue   # файл общий с другим сайтом — не трогаем
        fpath = docroot / rel
        if fpath.exists():
            try:
                fpath.unlink(); removed += 1
            except Exception:
                pass
        try:
            vectorstore.delete({"source": rel})
        except Exception:
            pass
    # 3) сбросить отпечатки инкремента для домена (чтобы всё скачалось заново)
    try:
        net = _up(url).netloc
        fpm = _web_fp_load()
        rm = [k for k in fpm if _up(k).netloc == net]
        for k in rm:
            fpm.pop(k, None)
        if rm:
            _web_fp_save(fpm)
    except Exception:
        pass
    # 4) очистить сохранённую статистику сайта
    try:
        stall = _web_stats_load()
        if url in stall:
            stall.pop(url, None)
            _web_stats_save(stall)
    except Exception:
        pass
    # 5) спарсить заново (save=False — не трогаем список остальных сайтов)
    r = ingest_web([url], index=index, save=False)
    pref = f"удалено файлов: {removed}"
    pref += (f" (пропущено общих с др. сайтами: {skipped_shared}). " if skipped_shared else ". ")
    r["msg"] = pref + (r.get("msg") or "")
    return r


# расширения «страниц» (их обходим как HTML); всё остальное считаем файлами и скачиваем
_WEB_PAGE_EXT = {"", ".html", ".htm", ".php", ".asp", ".aspx", ".jsp", ".cfm", ".shtml"}


def _web_is_file(url: str) -> bool:
    from urllib.parse import urlparse
    return os.path.splitext(urlparse(url).path)[1].lower() not in _WEB_PAGE_EXT


class _Renderer:
    """Однократно запущенный headless-Chromium (Playwright) для JS-страниц.

    Оптимизации скорости: один переиспользуемый контекст (не создаём браузерный
    контекст на каждую страницу), блокировка картинок/стилей/шрифтов/медиа (на текст
    не влияют), быстрый критерий готовности (domcontentloaded вместо networkidle) и
    настраиваемый таймаут — раньше networkidle ждал до 45 с на каждой странице."""

    def __init__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._b = self._pw.chromium.launch(
            args=["--no-sandbox", "--disable-dev-shm-usage"])
        self._wait = (settings.get("WEB_JS_WAIT") or "domcontentloaded").strip()
        if self._wait not in ("domcontentloaded", "load", "networkidle"):
            self._wait = "domcontentloaded"
        try:
            self._to = max(5, int(settings.get("WEB_PAGE_TIMEOUT") or 25)) * 1000
        except Exception:
            self._to = 25000
        try:
            self._settle = max(0, int(settings.get("WEB_JS_WAIT_MS") or 0))
        except Exception:
            self._settle = 0
        self._block = bool(settings.get("WEB_JS_BLOCK_ASSETS"))
        # Периодически пересоздаём контекст браузера: на длинных обходах (тысячи
        # страниц) Chromium накапливает память даже при закрытых вкладках — это ведёт
        # к замедлению и «зависанию». 0 — не пересоздавать.
        try:
            self._recycle_every = max(0, int(settings.get("WEB_JS_RECYCLE") or 150))
        except Exception:
            self._recycle_every = 150
        self._auth = (None, None, None)   # (headers, cookies, base_url) — восстановить после пересоздания
        self._warmed = set()
        self._render_count = 0
        self._ctx = self._make_context()

    def _make_context(self):
        ctx = self._b.new_context(user_agent=_WEB_UA)
        # ОБЩИЙ таймаут на ВСЕ операции страницы (goto/content/close/evaluate). Без него
        # «застрявшая» страница (бесконечный JS, битый DOM) вешает весь обход навсегда —
        # особенно pg.content(), у которого своего таймаута нет.
        try:
            ctx.set_default_timeout(self._to)
            ctx.set_default_navigation_timeout(self._to)
        except Exception:
            pass
        if self._block:
            try:
                ctx.route("**/*", self._route)
            except Exception:
                pass
        h, c, b = self._auth
        if h or c:
            self._apply_auth(ctx, h, c, b)
        return ctx

    def _recycle(self):
        """Пересоздать контекст, освободив накопленную Chromium память (на длинных обходах)."""
        try:
            self._ctx.close()
        except Exception:
            pass
        self._warmed = set()
        self._ctx = self._make_context()

    def fetch_bytes(self, url):
        """Скачать файл через браузерный контекст (несёт cookies, в т.ч. cf_clearance
        после прохождения Cloudflare-челленджа). Один раз «прогреваем» домен переходом
        на корень. Возвращает bytes или None."""
        from urllib.parse import urlparse
        pr = urlparse(url)
        try:
            if pr.netloc not in self._warmed:
                try:
                    pg = self._ctx.new_page()
                    pg.goto(f"{pr.scheme}://{pr.netloc}/", wait_until="domcontentloaded",
                            timeout=self._to)
                    pg.wait_for_timeout(2500)
                    pg.close()
                except Exception:
                    pass
                self._warmed.add(pr.netloc)
            r = self._ctx.request.get(url, timeout=120000)
            if r.status == 200:
                return r.body()
        except Exception as e:
            print(f"[web] browser fetch {url}: {e}")
        return None

    @staticmethod
    def _route(route):
        try:
            if route.request.resource_type in ("image", "media", "font", "stylesheet"):
                route.abort()
            else:
                route.continue_()
        except Exception:
            try:
                route.continue_()
            except Exception:
                pass

    @staticmethod
    def _apply_auth(ctx, headers=None, cookies=None, base_url=None):
        try:
            ctx.set_extra_http_headers(headers or {})
        except Exception:
            pass
        try:
            ctx.clear_cookies()
        except Exception:
            pass
        if cookies and base_url:
            from urllib.parse import urlparse
            dom = urlparse(base_url).netloc.split(":")[0]
            arr = [{"name": k, "value": v, "domain": dom, "path": "/"}
                   for k, v in cookies.items()]
            try:
                ctx.add_cookies(arr)
            except Exception:
                pass

    def set_auth(self, headers=None, cookies=None, base_url=None):
        """Применить авторизацию сайта к контексту браузера (заголовки + cookies).
        Сохраняем её, чтобы восстановить после периодического пересоздания контекста."""
        self._auth = (headers, cookies, base_url)
        self._apply_auth(self._ctx, headers, cookies, base_url)

    def clear_auth(self):
        self._auth = (None, None, None)
        try:
            self._ctx.set_extra_http_headers({})
        except Exception:
            pass
        try:
            self._ctx.clear_cookies()
        except Exception:
            pass

    def render(self, url: str, patient: bool = False):
        """patient=True — «терпеливый» рендер для тяжёлых SPA: дожидаемся простоя сети
        (networkidle) и даём странице время дорисоваться. Используется как повтор, если
        быстрый рендер (domcontentloaded) вернул почти пустую страницу."""
        wait = "networkidle" if patient else self._wait
        to = max(self._to, 40000) if patient else self._to
        settle = max(self._settle, 2500) if patient else self._settle
        # периодически пересоздаём контекст — не даём Chromium разрастись на длинном обходе
        if self._recycle_every and self._render_count and \
                self._render_count % self._recycle_every == 0:
            self._recycle()
        self._render_count += 1
        pg = None
        try:
            pg = self._ctx.new_page()
            try:
                pg.goto(url, wait_until=wait, timeout=to)
            except Exception:
                try:
                    pg.goto(url, wait_until="domcontentloaded", timeout=to)
                except Exception:
                    pass
            if settle:
                try:
                    pg.wait_for_timeout(settle)
                except Exception:
                    pass
            return pg.content()   # ограничен default timeout контекста — не зависнет навсегда
        except Exception as e:
            print(f"[web] render {url}: {e}")
            return None
        finally:
            # ВСЕГДА закрываем вкладку — иначе при ошибке она утекает и Chromium пухнет
            if pg is not None:
                try:
                    pg.close()
                except Exception:
                    pass

    def close(self):
        for fn in (getattr(self._ctx, "close", None), getattr(self._b, "close", None),
                   getattr(self._pw, "stop", None)):
            try:
                if fn:
                    fn()
            except Exception:
                pass


_WEB_TAGSTRIP = None
_WEB_JS_MIN_WORDS = 40   # меньше стольких слов видимого текста → страница считается «JS-пустой»

# Реалистичный «браузерный» User-Agent + заголовки: многие сайты (в т.ч. на Bitrix,
# за WAF/Cloudflare) блокируют явно ботовые UA (403), особенно на скачивании файлов
# из /upload/. Притворяемся обычным Chrome — это резко снижает число отказов.
_WEB_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_WEB_HEADERS = {
    "User-Agent": _WEB_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
              "image/webp,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


# --- SSRF-защита: не ходим на приватные/loopback/link-local адреса (S4/M41) ---
_HOST_SAFE_CACHE: dict = {}


def _host_is_safe(host: str) -> bool:
    """True, если ВСЕ адреса, в которые резолвится host, публичные. Блокируем
    127.0.0.0/8, 10/8, 172.16/12, 192.168/16, 169.254/16, ::1 и прочие приватные/
    служебные диапазоны. Не резолвится — считаем небезопасным (не ходим)."""
    if not host:
        return False
    h = host.lower()
    if h in _HOST_SAFE_CACHE:
        return _HOST_SAFE_CACHE[h]
    safe = True
    try:
        infos = _socket.getaddrinfo(h, None)
        if not infos:
            safe = False
        for info in infos:
            ip = _ipaddress.ip_address(info[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
                safe = False
                break
    except Exception:
        safe = False
    _HOST_SAFE_CACHE[h] = safe
    return safe


def _url_is_safe(url: str) -> bool:
    """Проверка URL перед обходом: только http/https и публичный (не приватный) хост.
    file:// и прочие схемы уже отсекаются раньше по схеме."""
    from urllib.parse import urlparse as _up
    try:
        pr = _up(url)
        if pr.scheme not in ("http", "https"):
            return False
        return _host_is_safe(pr.hostname or "")
    except Exception:
        return False


def _web_client(workers: int = 6):
    """Общий httpx-клиент с пулом keep-alive соединений для всего прогона парсинга —
    убирает повторные TLS-рукопожатия (особенно на множестве файлов одного хоста).

    ВАЖНО: только HTTP/1.1. HTTP/2 мультиплексирует все запросы через ОДНО соединение,
    и если сервер его сбрасывает (частое поведение под нагрузкой), рушатся сразу все
    параллельные запросы («Server disconnected») и портится h2-стейт-машина («Invalid
    input … CLOSED»). На HTTP/1.1 сбой затрагивает лишь один запрос, а битое соединение
    просто выбрасывается из пула — обход устойчив."""
    conns = max(10, int(workers or 1) * 3)
    limits = httpx.Limits(max_connections=conns, max_keepalive_connections=conns)
    try:
        to = max(5, int(settings.get("WEB_PAGE_TIMEOUT") or 25))
    except Exception:
        to = 25
    return httpx.Client(follow_redirects=True, headers=dict(_WEB_HEADERS),
                        limits=limits, timeout=to)


def _visible_text_len(html: str) -> int:
    """Грубая оценка объёма видимого текста (для решения «нужен ли браузер»). Дёшево:
    вырезаем script/style и теги регуляркой — без полноценного парсинга."""
    global _WEB_TAGSTRIP
    if not html:
        return 0
    import re
    if _WEB_TAGSTRIP is None:
        _WEB_TAGSTRIP = (re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I),
                         re.compile(r"<[^>]+>"))
    s = _WEB_TAGSTRIP[0].sub(" ", html)
    s = _WEB_TAGSTRIP[1].sub(" ", s)
    return len(s.split())


def _web_fetch_http(url: str, client, log, retries: int = 2):
    """Быстрая обычная загрузка страницы (httpx, без браузера). Потокобезопасна при
    общем клиенте. Разрыв соединения сервером (частое под нагрузкой: «Server
    disconnected») повторяется несколько раз с короткой паузой — на новом соединении
    из пула обычно проходит."""
    if not _url_is_safe(url):                      # SSRF: приватный/loopback/link-local
        log(f"ERR {url}: адрес заблокирован (приватный/loopback/link-local)")
        return None
    last = None
    for attempt in range(retries + 1):
        try:
            r = (client.get(url) if client is not None
                 else httpx.get(url, timeout=25, follow_redirects=True, headers=dict(_WEB_HEADERS)))
            if r.status_code != 200:
                return None
            ct = r.headers.get("content-type", "")
            if ct and "html" not in ct.lower():
                return None
            return r.text
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError,
                httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as e:
            last = e
            if attempt < retries:
                time.sleep(0.4 * (attempt + 1))
                continue
        except Exception as e:
            last = e
            break
    log(f"ERR {url}: {last}")
    return None


def _web_fetch(url: str, renderer, log, client=None):
    """HTML страницы: сначала быстрая обычная загрузка; браузер — только если включён
    и (в умном режиме) обычной загрузкой получилось мало текста."""
    html = _web_fetch_http(url, client, log)
    if renderer is not None:
        js_auto = bool(settings.get("WEB_JS_AUTO"))
        need = (html is None) or (not js_auto) or (_visible_text_len(html) < _WEB_JS_MIN_WORDS)
        if need:
            rhtml = renderer.render(url)
            if rhtml:
                return rhtml
    return html


# _fmt_bytes вынесен в admin/common.py и импортирован выше.


def _web_save_bytes(dest_dir, name, data, log):
    """Сохранить байты в dest_dir с уникальным именем (потокобезопасно). → Path/None."""
    try:
        with _web_dl_lock:
            dest_dir.mkdir(parents=True, exist_ok=True)
            out = dest_dir / name
            i = 1
            while out.exists() or out.with_name(out.name + ".part").exists():
                out = dest_dir / f"{i}_{name}"
                i += 1
            tmp = out.with_name(out.name + ".part")
            tmp.touch()
        with open(tmp, "wb") as f:
            f.write(data)
        tmp.rename(out)
        log(f"    ФАЙЛ сохранён (браузер): {out.name} ({_fmt_bytes(len(data))})")
        return out
    except Exception as e:
        log(f"    ERR сохранение (браузер) {name}: {e}")
        return None


def _web_download(url: str, dest_dir, log, client=None,
                  extra_headers=None, cookies=None, cond=None, browser=None):
    """Скачать файл по ссылке потоково, с процентом загрузки. Возвращает
    (path, reason, fp, not_modified): path — Path при успехе или None; reason — причина
    отказа; fp — отпечаток {etag,lastmod,size} для инкремента; not_modified — сервер
    ответил 304 (условный запрос). Заголовки — «браузерные» (UA + Referer) + авторизация
    и условные заголовки (If-None-Match/If-Modified-Since), если переданы."""
    if not _url_is_safe(url):                      # SSRF: приватный/loopback/link-local
        log(f"    адрес заблокирован (приватный/loopback/link-local): {url}")
        return None, "адрес заблокирован (приватный/loopback)", None, False
    import re
    from urllib.parse import urlparse, unquote
    tmp = None
    try:
        pr = urlparse(url)
        referer = f"{pr.scheme}://{pr.netloc}/"
        name = unquote(os.path.basename(pr.path)) or _web_slug(url)
        name = re.sub(r"[^\w.\-]+", "_", name)[:150] or "file"
        _streamer = (client.stream if client is not None else httpx.stream)
        _hdrs = {"Referer": referer, "Accept": "*/*"}
        if client is None:
            _hdrs["User-Agent"] = _WEB_UA
        if extra_headers:
            _hdrs.update(extra_headers)
        if cond:
            _hdrs.update(cond)
        _skw = {"headers": _hdrs, "cookies": cookies or None} if client is not None else \
               {"follow_redirects": True, "headers": _hdrs, "cookies": cookies or None}
        retries, last_err = 2, None
        for attempt in range(retries + 1):
            try:
                with _streamer("GET", url, timeout=120, **_skw) as r:
                    fp = {"etag": r.headers.get("ETag"), "lastmod": r.headers.get("Last-Modified")}
                    if r.status_code == 304:
                        log(f"    = файл не изменился: {name}")
                        return None, "не изменился", fp, True
                    if r.status_code != 200:
                        # 403/429/503 — вероятно WAF/Cloudflare: пробуем через браузер
                        if r.status_code in (403, 429, 503) and browser is not None:
                            log(f"    HTTP {r.status_code} — пробую скачать через браузер: {url}")
                            data = browser.fetch_bytes(url)
                            if data:
                                p2 = _web_save_bytes(dest_dir, name, data, log)
                                if p2 is not None:
                                    return p2, None, {"size": len(data)}, False
                        log(f"    ERR файл {url}: HTTP {r.status_code}")
                        return None, f"HTTP {r.status_code}", None, False
                    # уникальное имя резервируем только теперь (когда точно качаем)
                    with _web_dl_lock:
                        out = dest_dir; dest_dir.mkdir(parents=True, exist_ok=True)
                        out = dest_dir / name
                        i = 1
                        while out.exists() or out.with_name(out.name + ".part").exists():
                            out = dest_dir / f"{i}_{name}"
                            i += 1
                        tmp = out.with_name(out.name + ".part")
                        tmp.touch()
                    total = int(r.headers.get("content-length") or 0)
                    got, last = 0, -20
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_bytes(256 * 1024):
                            f.write(chunk)
                            got += len(chunk)
                            if total:
                                pct = int(got * 100 / total)
                                if pct >= last + 20:
                                    last = pct
                                    log(f"        {name}: {pct}% "
                                        f"({_fmt_bytes(got)} из {_fmt_bytes(total)})")
                # Защита от «обрезанных» загрузок: если сервер обещал content-length,
                # но прислал меньше — файл неполный (частая причина битых архивов/PDF).
                # Бросаем как временную ошибку → сработает повтор, а не сохранение обрезка.
                if total and got < total:
                    raise httpx.ReadError(f"неполная загрузка: {got} из {total} байт")
                tmp.rename(out)
                log(f"    ФАЙЛ сохранён: {out.name} ({_fmt_bytes(got)})")
                fp["size"] = got
                return out, None, fp, False
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError,
                    httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as e:
                last_err = e
                try:
                    if tmp is not None:
                        tmp.unlink()
                    tmp = None
                except Exception:
                    pass
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break
        try:
            if tmp is not None:
                tmp.unlink()
        except Exception:
            pass
        log(f"    ERR файл {url}: {last_err}")
        return None, (str(last_err)[:120] or "обрыв соединения"), None, False
    except Exception as e:
        try:
            if tmp is not None:
                tmp.unlink()
        except Exception:
            pass
        log(f"    ERR файл {url}: {e}")
        return None, (str(e)[:120] or "ошибка сети"), None, False


# Предел размера HTML для тяжёлого разбора (trafilatura/BeautifulSoup). Отдельные
# «раздутые» страницы (мегабайты DOM) заставляют парсер работать очень долго и
# создают ощущение зависшего обхода — усекаем перед разбором.
_WEB_MAX_HTML = 4_000_000


def _web_extract(html: str) -> str:
    """Извлечь основной текст страницы. trafilatura (если установлена) даёт чистый
    контент с таблицами; иначе — BeautifulSoup с предпочтением <main>/<article>."""
    try:
        import trafilatura
        txt = trafilatura.extract(html, include_tables=True, include_comments=False,
                                  favor_recall=True)
        if txt and len(txt.strip()) > 40:
            return txt.strip()
    except Exception:
        pass
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "noscript", "header", "footer", "nav",
                       "aside", "form"]):
            t.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        lines = [ln.strip() for ln in main.get_text(separator="\n").splitlines()]
        return "\n".join(ln for ln in lines if ln)
    except Exception:
        return ""


def _web_title(html: str, url: str) -> str:
    try:
        from bs4 import BeautifulSoup
        s = BeautifulSoup(html, "html.parser")
        if s.title and s.title.string:
            return s.title.string.strip()
    except Exception:
        pass
    return url


def _web_links(html: str, base: str, seed_netloc: str, same_domain: bool) -> list:
    """Все ссылки страницы (http/https, без mailto/tel/якорей). Без фильтра по типу —
    классификация на «страницы»/«файлы» делается в обходе."""
    from urllib.parse import urljoin, urlparse
    out = []
    try:
        from bs4 import BeautifulSoup
        s = BeautifulSoup(html, "html.parser")
        for a in s.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            pr = urlparse(urljoin(base, href))
            if pr.scheme not in ("http", "https"):
                continue
            if same_domain and pr.netloc.replace("www.", "") != \
                    seed_netloc.replace("www.", ""):
                continue
            out.append(pr._replace(fragment="").geturl())
    except Exception:
        pass
    return out


def _web_robots(base_url: str, client, headers=None, cookies=None) -> dict:
    """Прочитать robots.txt: правила Disallow для «*», Crawl-delay и ссылки Sitemap.
    Возвращает {disallow:[...], crawl_delay:float, sitemaps:[...]}."""
    from urllib.parse import urlparse
    out = {"disallow": [], "crawl_delay": 0.0, "sitemaps": []}
    pr = urlparse(base_url)
    robots_url = f"{pr.scheme}://{pr.netloc}/robots.txt"
    try:
        r = (client.get(robots_url, headers=headers or None, cookies=cookies or None)
             if client is not None else
             httpx.get(robots_url, timeout=15, follow_redirects=True, headers=dict(_WEB_HEADERS)))
        if r.status_code != 200:
            return out
        agents, reading_rules = [], False
        for raw in r.text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip().lower(), v.strip()
            if k == "user-agent":
                if reading_rules:
                    agents, reading_rules = [], False
                agents.append(v.lower())
            elif k == "sitemap" and v:
                out["sitemaps"].append(v)
            else:
                reading_rules = True
                star = "*" in agents
                if k == "disallow" and star and v:
                    out["disallow"].append(v)
                elif k == "crawl-delay" and star:
                    try:
                        out["crawl_delay"] = max(out["crawl_delay"], float(v))
                    except Exception:
                        pass
    except Exception:
        pass
    return out


def _web_path_disallowed(url: str, disallow) -> bool:
    if not disallow:
        return False
    from urllib.parse import urlparse
    path = urlparse(url).path or "/"
    for d in disallow:
        if d == "/":
            return True
        if path.startswith(d.rstrip("*")):
            return True
    return False


def _web_sitemap_urls(sitemap_seeds, client, limit, headers=None, cookies=None) -> list:
    """Собрать URL страниц из sitemap(ов), рекурсивно раскрывая sitemapindex. Поддержка
    .gz. Ограничение по числу URL и по числу обойденных карт (защита от гигантских карт)."""
    import re as _re
    urls, seen_sm, queue = [], set(), list(sitemap_seeds or [])
    while queue and len(urls) < limit and len(seen_sm) < 300:
        sm = queue.pop(0)
        if sm in seen_sm:
            continue
        seen_sm.add(sm)
        try:
            r = (client.get(sm, headers=headers or None, cookies=cookies or None, timeout=30)
                 if client is not None else
                 httpx.get(sm, timeout=30, follow_redirects=True, headers=dict(_WEB_HEADERS)))
            if r.status_code != 200:
                continue
            data = r.content
            if sm.lower().endswith(".gz") or data[:2] == b"\x1f\x8b":
                import gzip
                data = gzip.decompress(data)
            text = data.decode("utf-8", "ignore")
        except Exception:
            continue
        locs = _re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", text, _re.I)
        if "<sitemapindex" in text.lower():
            for loc in locs:
                if loc not in seen_sm:
                    queue.append(loc)
        else:
            for loc in locs:
                urls.append(loc)
                if len(urls) >= limit:
                    break
    return urls


def _web_get_page(url, client, log, extra_headers=None, cookies=None, cond=None, retries=2):
    """Загрузить страницу с авторизацией и условными заголовками. Возвращает dict:
    {status, html, etag, lastmod, not_modified}. Разрыв соединения повторяется."""
    if not _url_is_safe(url):                      # SSRF: приватный/loopback/link-local
        log(f"ERR {url}: адрес заблокирован (приватный/loopback/link-local)")
        return {"status": 0, "html": None, "not_modified": False, "etag": None, "lastmod": None}
    hdrs = {}
    if extra_headers:
        hdrs.update(extra_headers)
    if cond:
        hdrs.update(cond)
    last = None
    for attempt in range(retries + 1):
        try:
            if client is not None:
                r = client.get(url, headers=hdrs or None, cookies=cookies or None)
            else:
                r = httpx.get(url, timeout=25, follow_redirects=True,
                              headers={**_WEB_HEADERS, **hdrs}, cookies=cookies or None)
            meta = {"etag": r.headers.get("ETag"), "lastmod": r.headers.get("Last-Modified")}
            if r.status_code == 304:
                return {"status": 304, "html": None, "not_modified": True, **meta}
            if r.status_code != 200:
                return {"status": r.status_code, "html": None, "not_modified": False, **meta}
            ct = r.headers.get("content-type", "")
            if ct and "html" not in ct.lower():
                return {"status": 200, "html": None, "not_modified": False, **meta}
            return {"status": 200, "html": r.text, "not_modified": False, **meta}
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError,
                httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as e:
            last = e
            if attempt < retries:
                time.sleep(0.4 * (attempt + 1))
                continue
        except Exception as e:
            last = e
            break
    log(f"ERR {url}: {last}")
    return {"status": 0, "html": None, "not_modified": False, "etag": None, "lastmod": None}


def _web_crawl(seed: str, renderer, log, ctx: dict):
    """Обойти сайт (BFS по уровням) с учётом ctx: глубина/лимиты/домен/параллелизм,
    исключения, robots (disallow+crawl-delay), sitemap-сидирование, авторизация
    (headers/cookies), инкремент (условные GET, переиспользование неизменённых страниц),
    пер-сайтовое включение браузера. Возвращает (pages, files, stat).

    Ускорение: обычная загрузка идёт параллельно (httpx), браузер (не потокобезопасен) —
    последовательно и только где реально нужен."""
    from urllib.parse import urlparse
    from concurrent.futures import ThreadPoolExecutor
    import faulthandler   # сторож: при зависании страницы дампит стек в stderr (docker logs)
    # Таймер faulthandler один на процесс — при параллельных сайтах потоки перезаписывают/
    # отменяют чужой таймер, поэтому сторож включаем только в последовательном режиме.
    _watch = bool(ctx.get("watchdog", True))
    def _wd_arm(sec):
        if _watch:
            try:
                faulthandler.dump_traceback_later(sec)
            except Exception:
                pass
    def _wd_cancel():
        if _watch:
            try:
                faulthandler.cancel_dump_traceback_later()
            except Exception:
                pass
    depth = ctx["depth"]; max_pages = ctx["max_pages"]; same_domain = ctx["same_domain"]
    excludes = ctx.get("excludes"); client = ctx.get("client")
    headers = ctx.get("headers") or {}; cookies = ctx.get("cookies") or {}
    disallow = ctx.get("disallow") or []; crawl_delay = float(ctx.get("crawl_delay") or 0.0)
    # Ограничиваем паузу robots.txt: некоторые сайты задают Crawl-delay 20+ секунд, что
    # делает обход практически бесконечным (1 стр. / 20 с) и выглядит как зависание.
    # Соблюдаем robots, но не медленнее заданного максимума. 0 — не ограничивать.
    try:
        _cd_max = float(settings.get("WEB_CRAWL_DELAY_MAX") or 0)
    except Exception:
        _cd_max = 0.0
    if _cd_max > 0 and crawl_delay > _cd_max:
        log(f"  crawl-delay {crawl_delay:g}c ограничен до {_cd_max:g}c (WEB_CRAWL_DELAY_MAX)")
        crawl_delay = _cd_max
    use_render = ctx.get("use_render", True)
    incremental = bool(ctx.get("incremental")); fp_map = ctx.get("fp_map") or {}
    fp_out = ctx.get("fp_out"); reused = ctx.get("reused")
    seed_netloc = urlparse(seed).netloc
    # sitemap-сидирование: все URL карты — на уровень 0 (плюс сама стартовая)
    level = [(seed, 0)] + [(u, 0) for u in (ctx.get("sitemap_urls") or [])]
    seen, pages, files = set(), [], set()
    errors, depth_limit_hit, excluded_n, robots_skipped = [], False, 0, 0
    depth_pages, depth_seen = [], set()
    # crawl-delay несовместим с параллелизмом — соблюдаем вежливую паузу последовательно
    par = 1 if crawl_delay > 0 else max(1, int(ctx.get("workers") or 1))
    js_auto = bool(settings.get("WEB_JS_AUTO"))
    base_wait = (settings.get("WEB_JS_WAIT") or "domcontentloaded").strip().lower()
    _stop = ctx.get("stop")   # кооперативная остановка: проверяем между страницами
    def _stopped():
        try:
            return bool(_stop and _stop())
        except Exception:
            return False

    def _skip(u):
        nonlocal excluded_n, robots_skipped
        if _web_excluded(u, excludes):
            excluded_n += 1
            return True
        if disallow and _web_path_disallowed(u, disallow):
            robots_skipped += 1
            return True
        return False

    while level and len(pages) < max_pages:
        if _stopped():
            errors.append("остановлено пользователем")
            break
        batch = []
        for url, d in level:
            if url in seen:
                continue
            seen.add(url)
            if _skip(url):
                continue
            if _web_is_file(url):
                files.add(url)
            else:
                batch.append((url, d))
        if not batch:
            break

        # фаза 1: обычная загрузка (httpx) с авторизацией и условными заголовками
        def _get(u):
            cond = _web_cond_headers(fp_map.get(u)) if incremental else None
            return u, _web_get_page(u, client, log, headers, cookies, cond)
        results = {}
        if par > 1 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=min(par, len(batch))) as ex:
                for u, res in ex.map(lambda ud: _get(ud[0]), batch):
                    results[u] = res
        else:
            for _i, (u, dd) in enumerate(batch, 1):
                # при crawl-delay фаза 1 идёт медленно и молча — логируем прогресс, чтобы
                # обход не выглядел зависшим (видно, что идёт загрузка уровня)
                if crawl_delay > 0:
                    log(f"  · загрузка {_i}/{len(batch)} (пауза {crawl_delay:g}c): {u}")
                _, res = _get(u)
                results[u] = res
                if crawl_delay > 0:
                    time.sleep(crawl_delay)

        # фаза 2: разбор + браузер (только где нужен)
        fetched = []
        for (u, dd) in batch:
            if _stopped():
                break
            res = results.get(u) or {}
            # инкремент: сервер сказал «не изменилось» — берём текст из кэша, без рендера
            if incremental and res.get("not_modified"):
                cached = (fp_map.get(u) or {}).get("text")
                if cached:
                    pages.append((u, (fp_map[u].get("title") or u), cached))
                    if reused is not None:
                        reused.append(u)
                    log(f"  = не изменилось (из кэша): {u}")
                    # ссылки со страницы при 304 не переобходим (структура та же)
                    continue
            html = res.get("html")
            if renderer is not None and use_render:
                need = (html is None) or (not js_auto) or (_visible_text_len(html) < _WEB_JS_MIN_WORDS)
                if need:
                    log(f"  · рендер (браузер): {u}")
                    _wd_arm(150)   # если рендер завис >150с — стек в stderr (только seq-режим)
                    try:
                        rhtml = renderer.render(u)
                        if base_wait != "networkidle" and _visible_text_len(rhtml or "") < _WEB_JS_MIN_WORDS:
                            rhtml2 = renderer.render(u, patient=True)
                            if _visible_text_len(rhtml2 or "") > _visible_text_len(rhtml or ""):
                                rhtml = rhtml2
                        if rhtml:
                            html = rhtml
                    finally:
                        _wd_cancel()
            fetched.append((u, dd, html, res))

        next_level = []
        for url, d, html, res in fetched:
            if len(pages) >= max_pages:
                break
            if html is None:
                errors.append(f"страница не загружена: {url}"
                              + (f" (HTTP {res.get('status')})" if res.get("status") else ""))
                continue
            if len(html) > _WEB_MAX_HTML:
                log(f"  ⚠ большой HTML {len(html)//1024} КБ — усечён для разбора: {url}")
                html = html[:_WEB_MAX_HTML]
            log(f"  · разбор: {url}")
            _wd_arm(120)   # если разбор завис >120с — стек в stderr (только seq-режим)
            try:
                text = _web_extract(html)
                title = _web_title(html, url)
            finally:
                _wd_cancel()
            if text:
                pages.append((url, title, text))
                log(f"  стр. {len(pages)}/{max_pages}: {url}  ({len(text)} симв.)")
                # обновляем отпечаток страницы для инкремента
                if fp_out is not None:
                    fp_out[url] = {"etag": res.get("etag"), "lastmod": res.get("lastmod"),
                                   "kind": "page", "title": title,
                                   "text": text[:200000]}
            else:
                log(f"  стр. (без текста): {url}")
                errors.append(f"страница без извлечённого текста (возможно, JS-сайт): {url}")
            for link in _web_links(html, url, seed_netloc, same_domain):
                if link in seen or _skip(link):
                    continue
                if _web_is_file(link):
                    files.add(link)
                elif d < depth:
                    next_level.append((link, d + 1))
                else:
                    depth_limit_hit = True
                    if url not in depth_seen and len(depth_pages) < 200:
                        depth_seen.add(url)
                        depth_pages.append(url)
        level = next_level

    pages_limit_hit = bool(level) and len(pages) >= max_pages
    return pages, files, {"errors": errors, "pages_limit_hit": pages_limit_hit,
                          "depth_limit_hit": depth_limit_hit, "excluded": excluded_n,
                          "robots_skipped": robots_skipped, "depth_pages": depth_pages}


def ingest_web(urls: list, index: bool = True, save: bool = True) -> dict:
    """Скачать до 50 сайтов (с обходом ссылок), извлечь текст в DOCS_DIR/web и
    (при index=True) переиндексировать. При index=False — только парсинг без
    индексации (быстро обновить содержимое; проиндексировать можно позже кнопкой
    «Переиндексировать»). Глубина/лимит/домен — настройки WEB_CRAWL_DEPTH/MAX_PAGES/SAME_DOMAIN.

    save=True — переписать список сохранённых сайтов на переданный (обычный запуск из
    очереди адресов). save=False — НЕ трогать список (перепарсинг одного уже
    сохранённого сайта: остальные сайты в списке сохраняются)."""
    import re
    urls = [u.strip() for u in (urls or []) if u.strip().startswith(("http://", "https://"))][:50]
    if not urls:
        return {"ok": False, "msg": "укажите хотя бы один URL (http/https), максимум 50"}
    # SSRF (S4/M41): резолвим хост и отсеиваем приватные/loopback/link-local адреса
    # ПЕРЕД обходом. FIXME(review): httpx follow_redirects=True — редиректы на внутренний
    # адрес заново не проверяются (и остаётся TOCTOU/DNS-rebinding до фактического connect).
    _blocked = [u for u in urls if not _url_is_safe(u)]
    urls = [u for u in urls if _url_is_safe(u)]
    if not urls:
        return {"ok": False, "msg": "все адреса указывают на приватную/локальную сеть — "
                                    "парсинг таких адресов запрещён"}
    if _web_job["running"]:
        return {"ok": False, "msg": "парсинг сайтов уже идёт"}
    if save:
        _web_sources_save(urls)
    excludes_map = _web_excludes_load()
    excludes_all = _web_excludes_all_load()
    webdir = Path(settings.get("DOCS_DIR")).expanduser() / "web"
    logfile = "/tmp/rag_web.log"
    _slug = _web_slug

    def run():
        _web_job.update(running=True, started=time.time(), finished=None, ok=None,
                        log="", summary="", logfile=logfile, stop=False)
        ok = err = 0
        rc = -1
        try:
            import html as _html
            webdir.mkdir(parents=True, exist_ok=True)
            web_paths = []
            # пер-сайтовые настройки, авторизация и отпечатки инкремента
            cfg_map = _web_cfg_load()
            auth_map = _web_auth_load()
            fp_map = _web_fp_load()
            fp_out = {}
            hash_map = _web_filehash_load()   # дедуп файлов между сайтами {sha: rel}
            hash_out = {}
            use_sitemap = bool(settings.get("WEB_USE_SITEMAP"))
            respect_robots = bool(settings.get("WEB_RESPECT_ROBOTS"))
            incremental = bool(settings.get("WEB_INCREMENTAL"))
            # глобальные значения — только для баннера; фактические считаются на сайт
            depth = max(0, int(settings.get("WEB_CRAWL_DEPTH") or 0))
            max_pages = max(1, int(settings.get("WEB_MAX_PAGES") or 1))
            max_files = max(0, int(settings.get("WEB_MAX_FILES") or 0))
            workers = max(1, int(settings.get("WEB_CONCURRENCY") or 1))
            filesdir = webdir / "files"
            with open(logfile, "w", buffering=1, errors="ignore") as fp:
                import threading as _th
                from urllib.parse import urlparse as _urlparse
                # RLock (B2/M37): позволяет держать замок при мутации stats_map и тут же
                # вызвать _save_stats() (который снова берёт замок для json.dumps) без
                # дедлока — сериализация и все мутации stats_map идут строго под ним.
                _wlock = _th.RLock()  # защита лога/статистики/счётчиков при параллельных сайтах
                def _rawlog(m):
                    with _wlock:
                        fp.write(m + "\n"); fp.flush()
                _log = _rawlog
                def _save_stats():
                    with _wlock:
                        _web_stats_save(stats_map)
                counters = {"ok": 0, "err": 0}
                def _inc(k):
                    with _wlock:
                        counters[k] += 1
                site_par = max(1, int(settings.get("WEB_SITE_CONCURRENCY") or 1))

                # общий httpx-клиент (keep-alive) — потокобезопасен для запросов
                client = _web_client(max(workers, site_par * workers))
                # доступность Playwright проверяем один раз; сам браузер — свой на каждый сайт
                _pw_ok = bool(settings.get("WEB_JS_RENDER"))
                if _pw_ok:
                    try:
                        import playwright  # noqa: F401
                    except Exception as e:
                        _pw_ok = False
                        _rawlog(f"Headless-браузер недоступен ({e}); обычная загрузка. "
                                "Установка: pip install playwright && playwright install chromium")
                fp.write(f"Парсинг: глубина {depth}, до {max_pages} стр. и {max_files} "
                         f"файлов/сайт, параллельно ×{workers}"
                         f"{f', сайтов ×{site_par}' if site_par > 1 else ''}"
                         f"{', sitemap' if use_sitemap else ''}"
                         f"{', robots' if respect_robots else ''}"
                         f"{', инкремент' if incremental else ''}\n")
                stats_map = {}
                try:
                    def _parse_site(u):
                        # свой логгер (с префиксом сайта в параллельном режиме) и свой браузер
                        _host = _urlparse(u).netloc or u
                        def _log(m):
                            _rawlog(f"[{_host}] {m}" if site_par > 1 else m)
                        renderer = None
                        if _pw_ok and _web_eff(u, cfg_map).get("js_render"):
                            try:
                                renderer = _Renderer()
                            except Exception as e:
                                _log(f"Headless-браузер недоступен ({e}); обычная загрузка")
                                renderer = None
                        site_errors, site_limits, site_items = [], {}, []
                        site_depth_urls = []
                        pages, dl, site_ok = [], 0, False
                        # добавление нового ключа верхнего уровня — под локом (иначе гонка с
                        # json.dumps в _save_stats из другого потока-сайта)
                        with _wlock:
                            stats_map[u] = {"url": u, "ts": time.time(), "items": [],
                                            "progress": {"phase": "crawl", "pct": 8}}
                        _save_stats()
                        try:
                            _log(f"САЙТ: {u}")
                            # эффективные настройки: пер-сайтовые поверх глобальных
                            eff = _web_eff(u, cfg_map)
                            s_depth = eff["depth"]; s_maxp = eff["max_pages"]
                            s_maxf = eff["max_files"]; s_workers = eff["concurrency"]
                            s_same = eff["same_domain"]; s_render = eff["js_render"]
                            s_delay = eff["crawl_delay"]
                            if cfg_map.get(u):
                                _log(f"  настройки сайта: глубина {s_depth}, до {s_maxp} стр./"
                                     f"{s_maxf} файлов, ×{s_workers}, "
                                     f"{'тот же домен' if s_same else 'любой домен'}, "
                                     f"браузер {'вкл' if s_render else 'выкл'}")
                            site_excludes = _web_site_excludes(u, excludes_map, excludes_all)
                            if site_excludes:
                                _log(f"  исключения по словам: {', '.join(site_excludes)}"
                                     + (f" (глобальных: {len(excludes_all)})" if excludes_all else ""))
                            # авторизация на сайт
                            a_headers, a_cookies = _web_auth_for(u, auth_map)
                            if a_headers or a_cookies:
                                _log(f"  авторизация: {(auth_map.get(u) or {}).get('type')}")
                            if renderer is not None:
                                renderer.set_auth(a_headers, a_cookies, u)
                            # robots.txt + sitemap
                            disallow, crawl_delay, sm_seeds = [], 0.0, []
                            if respect_robots:
                                rob = _web_robots(u, client, a_headers, a_cookies)
                                disallow = rob["disallow"]; crawl_delay = rob["crawl_delay"]
                                sm_seeds = rob["sitemaps"]
                                if crawl_delay and not s_delay:
                                    _log(f"  robots.txt: crawl-delay {crawl_delay}c ОТКЛЮЧЁН "
                                         "(задержка выключена для сайта/глобально)")
                                    crawl_delay = 0.0
                                if disallow or crawl_delay:
                                    _log(f"  robots.txt: запрещённых путей {len(disallow)}"
                                         + (f", crawl-delay {crawl_delay}c" if crawl_delay else ""))
                            sitemap_urls = []
                            if use_sitemap:
                                from urllib.parse import urlparse as _up
                                _pr = _up(u)
                                seeds = sm_seeds or [f"{_pr.scheme}://{_pr.netloc}/sitemap.xml"]
                                sitemap_urls = _web_sitemap_urls(seeds, client, s_maxp * 2,
                                                                 a_headers, a_cookies)
                                if sitemap_urls:
                                    _log(f"  sitemap: найдено URL {len(sitemap_urls)}")
                            _ctx = {"depth": s_depth, "max_pages": s_maxp, "same_domain": s_same,
                                    "workers": s_workers, "excludes": site_excludes, "client": client,
                                    "headers": a_headers, "cookies": a_cookies,
                                    "disallow": disallow, "crawl_delay": crawl_delay,
                                    "sitemap_urls": sitemap_urls, "use_render": s_render,
                                    "incremental": incremental, "fp_map": fp_map, "fp_out": fp_out,
                                    "reused": [], "watchdog": site_par <= 1,
                                    "stop": lambda: bool(_web_job.get("stop"))}
                            pages, file_urls, cstat = _web_crawl(u, renderer, _log, _ctx)
                            site_errors.extend(cstat.get("errors", []))
                            if cstat.get("excluded"):
                                _log(f"  исключено URL по ключевым словам: {cstat['excluded']}")
                            if cstat.get("robots_skipped"):
                                _log(f"  пропущено по robots.txt: {cstat['robots_skipped']}")
                            if _ctx["reused"]:
                                _log(f"  не изменилось (инкремент, из кэша): {len(_ctx['reused'])} стр.")
                            if cstat.get("pages_limit_hit"):
                                site_limits["pages"] = (f"достигнут лимит страниц ({s_maxp}); "
                                                        "часть страниц не обойдена — увеличьте «Макс. страниц»")
                            if cstat.get("depth_limit_hit"):
                                site_depth_urls = cstat.get("depth_pages", [])
                                _nd = len(site_depth_urls)
                                site_limits["depth"] = (f"достигнута глубина обхода ({s_depth}); "
                                                        f"более глубокие ссылки пропущены на {_nd} "
                                                        "страниц(ах) — увеличьте «Глубину обхода» "
                                                        "(список URL — ниже)")
                                _log(f"  глубина превышена на страницах ({_nd}):")
                                for _du in site_depth_urls[:200]:
                                    _log(f"    • {_du}")
                            # скачиваем найденные файлы (любого типа), лимит на сайт
                            all_files = [f for f in file_urls
                                         if not _web_excluded(f, site_excludes)
                                         and not (disallow and _web_path_disallowed(f, disallow))]
                            if s_maxf and len(all_files) > s_maxf:
                                site_limits["files"] = (f"найдено файлов {len(all_files)}, скачано {s_maxf} "
                                                        f"(лимит); {len(all_files) - s_maxf} пропущено — "
                                                        "увеличьте «Макс. файлов»")
                            file_list = all_files[:s_maxf] if s_maxf else all_files
                            nf = len(file_list)
                            _docroot = Path(settings.get("DOCS_DIR")).expanduser()
                            with _wlock:
                                stats_map[u]["progress"] = {"phase": "download", "done": 0, "total": nf, "pct": 40}
                            _save_stats()
                            if nf:
                                _log(f"  файлов к скачиванию: {nf}"
                                     + (f" (параллельно ×{s_workers})" if s_workers > 1 and nf > 1 else ""))

                            def _rec_file(furl, p):
                                okf = p is not None
                                rel, size = None, 0
                                if okf:
                                    try:
                                        rel = str(p.relative_to(_docroot))
                                        size = p.stat().st_size
                                    except Exception:
                                        rel = str(p)
                                site_items.append({"kind": "file", "url": furl,
                                                   "name": (p.name if okf else (furl.rsplit("/", 1)[-1] or furl)),
                                                   "source": rel, "size": size, "ok": okf})
                                done = len(site_items)
                                with _wlock:
                                    stats_map[u]["progress"] = {"phase": "download", "done": done, "total": nf,
                                                                "pct": min(88, 40 + int(done * 45 / max(1, nf)))}
                                if done % 5 == 0:
                                    _save_stats()

                            # Playwright sync привязан к потоку, где создан браузер (поток сайта).
                            # При параллельном скачивании (s_workers>1) файлы качаются в пуле
                            # потоков — из них renderer трогать нельзя, поэтому WAF-фолбэк через
                            # браузер доступен только в последовательном режиме.
                            _dl_browser = renderer if s_workers <= 1 else None

                            # скачивание одного файла: авторизация + инкремент (304 →
                            # переиспользуем файл с диска) + запись отпечатка
                            def _dl_file(furl):
                                prev = fp_map.get(furl) or {}
                                cond = _web_cond_headers(prev) if incremental else None
                                p, reason, fpx, nm = _web_download(furl, filesdir, _log, client,
                                                                   a_headers, a_cookies, cond,
                                                                   browser=_dl_browser)
                                if nm:
                                    relp = prev.get("path")
                                    cachedp = (_docroot / relp) if relp else None
                                    if cachedp and cachedp.exists():
                                        fp_out[furl] = prev
                                        return cachedp, "не изменился (кэш)"
                                    p, reason, fpx, nm = _web_download(furl, filesdir, _log,
                                                                       client, a_headers, a_cookies,
                                                                       None, browser=_dl_browser)
                                if p is not None:
                                    # дедуп по содержимому: одинаковый файл (с любого сайта)
                                    # не дублируем — переиспользуем уже сохранённый
                                    sha = _sha256_file(p)
                                    if sha:
                                        seen = hash_out.get(sha) or hash_map.get(sha)
                                        if seen and (_docroot / seen).exists() and str(p.relative_to(_docroot)) != seen:
                                            try:
                                                p.unlink()
                                            except Exception:
                                                pass
                                            _log(f"    ↺ дубликат (тот же файл): переиспользую {seen}")
                                            reused_p = _docroot / seen
                                            if fpx is not None:
                                                fp_out[furl] = {**fpx, "kind": "file", "path": seen}
                                            return reused_p, "дубликат (переиспользован)"
                                        rel0 = str(p.relative_to(_docroot))
                                        hash_out[sha] = rel0
                                    if fpx is not None:
                                        try:
                                            rel = str(p.relative_to(_docroot))
                                        except Exception:
                                            rel = str(p)
                                        fp_out[furl] = {**fpx, "kind": "file", "path": rel}
                                return p, reason

                            if s_workers > 1 and nf > 1:
                                # параллельное скачивание файлов (потокобезопасно: уникальные имена)
                                from concurrent.futures import ThreadPoolExecutor
                                done_n = [0]
                                def _dl_one(furl):
                                    p, reason = _dl_file(furl)
                                    done_n[0] += 1
                                    _log(f"  [файл {done_n[0]}/{nf}] {int(done_n[0]*100/nf)}%: {furl}"
                                         + ("" if p is not None else f"  — не скачан ({reason})"))
                                    return furl, p, reason
                                with ThreadPoolExecutor(max_workers=min(s_workers, nf)) as ex:
                                    for furl, p, reason in ex.map(_dl_one, file_list):
                                        if p is not None:
                                            web_paths.append(p)
                                            dl += 1
                                        else:
                                            site_errors.append(f"файл не скачан ({reason}): {furl}")
                                        _rec_file(furl, p)
                            else:
                                for fi, furl in enumerate(file_list, 1):
                                    fpct = int(fi * 100 / nf) if nf else 100
                                    _log(f"  [файл {fi}/{nf}] {fpct}% скачиваю: {furl}")
                                    p, reason = _dl_file(furl)
                                    if p is not None:
                                        web_paths.append(p)
                                        dl += 1
                                    else:
                                        site_errors.append(f"файл не скачан ({reason}): {furl}")
                                    _rec_file(furl, p)
                            # текст страниц — в один документ web/<slug>.html
                            if pages:
                                parts, total = [], 0
                                for (pu, pt, ptext) in pages:
                                    parts.append(
                                        "<h2>%s</h2>\n<p><small>%s</small></p>\n<pre>%s</pre>"
                                        % (_html.escape(pt or pu), _html.escape(pu),
                                           _html.escape(ptext)))
                                    total += len(ptext)
                                doc = ("<html><body><!-- source: %s -->\n<h1>%s</h1>\n"
                                       "%s</body></html>"
                                       % (_html.escape(u), _html.escape(pages[0][1] or u),
                                          "\n".join(parts)))
                                out = webdir / (_slug(u) + ".html")
                                out.write_text(doc, encoding="utf-8")
                                web_paths.append(out)
                                try:
                                    prel = str(out.relative_to(_docroot))
                                    psize = out.stat().st_size
                                except Exception:
                                    prel, psize = str(out), 0
                                site_items.insert(0, {"kind": "page", "name": "Страницы сайта (объединено)",
                                                      "source": prel, "size": psize, "ok": True,
                                                      "pages": len(pages)})
                            if pages or dl:
                                _inc("ok")
                                site_ok = True
                                _log(f"ИТОГО {u}: страниц {len(pages)}, файлов {dl}")
                            else:
                                _inc("err")
                                site_errors.append("ни текста, ни файлов (пустая страница "
                                                   "или JS-сайт без headless-браузера)")
                                _log(f"ERR {u}: ни текста, ни файлов (пустая страница "
                                     "или JS-сайт без headless-браузера)")
                        except Exception as e:
                            _inc("err")
                            site_errors.append(str(e))
                            _log(f"ERR {u}: {e}")
                        finally:
                            # свой браузер сайта закрываем всегда (освобождаем Chromium)
                            if renderer is not None:
                                try:
                                    renderer.close()
                                except Exception:
                                    pass
                        with _wlock:
                            stats_map[u] = {"url": u, "ok": site_ok, "pages": len(pages),
                                            "files": dl, "errors": site_errors,
                                            "limits": site_limits, "ts": time.time(),
                                            "items": site_items, "depth_urls": site_depth_urls,
                                            "progress": {"phase": "parsed", "pct": 92}}
                        _save_stats()

                    # запуск сайтов: последовательно или параллельно (каждый — свой поток+браузер)
                    def _site_guarded(u):
                        if _web_job.get("stop"):
                            return
                        _parse_site(u)
                    if site_par > 1 and len(urls) > 1:
                        from concurrent.futures import ThreadPoolExecutor as _TPE
                        with _TPE(max_workers=min(site_par, len(urls))) as _ex:
                            list(_ex.map(_site_guarded, urls))
                    else:
                        for u in urls:
                            if _web_job.get("stop"):
                                _log("⏹ остановлено пользователем — оставшиеся сайты пропущены")
                                break
                            _parse_site(u)
                    ok, err = counters["ok"], counters["err"]
                finally:
                    try:
                        client.close()
                    except Exception:
                        pass
                    _save_stats()
                    # отпечатки для инкрементального парсинга (следующий запуск быстрее)
                    try:
                        if fp_out:
                            fp_map.update(fp_out)
                            _web_fp_save(fp_map)
                    except Exception as _e:
                        print(f"[web] отпечатки не сохранены: {_e}")
                    # карта хэшей файлов для дедупа между сайтами
                    try:
                        if hash_out:
                            hash_map.update(hash_out)
                            _web_filehash_save(hash_map)
                    except Exception as _e:
                        print(f"[web] хэши файлов не сохранены: {_e}")
                # активен каталог PostgreSQL — кладём спарсенные страницы и в него,
                # чтобы индексация из БД их увидела (без папки). catalog_add_paths
                # живёт в admin_ops (домен каталога) — ленивый импорт во избежание
                # цикла на этапе загрузки модулей (вызов идёт уже в рантайме потока).
                import admin_ops as _ao
                added = _ao.catalog_add_paths(web_paths)
                if added:
                    fp.write(f"В PostgreSQL добавлено страниц: {added}\n")
                def _set_all_progress(prog):
                    # под _wlock: мутация и json.dumps stats_map не должны идти конкурентно
                    with _wlock:
                        for _u in stats_map:
                            stats_map[_u]["progress"] = dict(prog)
                        _web_stats_save(stats_map)

                if index:
                    _set_all_progress({"phase": "index", "pct": 96})
                    fp.write(f"Скачано: {ok}, ошибок: {err}. Запускаю индексацию...\n")
                    fp.flush()
                    rc = subprocess.Popen([sys.executable, "-u", "ingest.py"], cwd=ROOT,
                                          stdout=fp, stderr=subprocess.STDOUT).wait(timeout=24 * 3600)
                    fp.write(f"SUMMARY web_ok={ok} web_err={err} index_rc={rc}\n")
                    _set_all_progress({"phase": "done", "pct": 100})
                else:
                    rc = 0
                    fp.write(f"Скачано: {ok}, ошибок: {err}. Индексация ПРОПУЩЕНА "
                             "(только парсинг) — запустите «Переиндексировать», когда нужно.\n")
                    fp.write(f"SUMMARY web_ok={ok} web_err={err} index_rc=skipped\n")
                    _set_all_progress({"phase": "parsed", "pct": 100})
            _web_job["log"] = _tail(logfile)
            _web_job["summary"] = _extract_summary(_web_job["log"])
            # B7: частичный успех — это НЕ провал задачи. Падение одного из нескольких
            # сайтов не должно помечать всю задачу как failed (и слать ложный алерт).
            # Провал = индексация упала (rc != 0) ИЛИ не удалось ни одного сайта (ok == 0
            # при наличии ошибок). Ошибки по отдельным сайтам видны в summary/логе.
            _web_job["ok"] = (rc == 0) and (ok > 0 or err == 0)
            _web_job["partial"] = bool(ok > 0 and err > 0)
        except Exception as e:
            _web_job["ok"] = False
            _web_job["log"] = (_tail(logfile) + "\n" + str(e)).strip()
        _web_job["running"] = False
        _web_job["finished"] = time.time()
        try:
            db.ingest_log_save("Парсинг сайтов", _web_job.get("summary") or "",
                               _read_full_log(logfile))
        except Exception as e:
            print(f"[web] не удалось сохранить лог в БД: {e}")

    threading.Thread(target=run, daemon=True).start()
    tail = "затем — индексация" if index else "без индексации (только парсинг)"
    msg = f"парсинг {len(urls)} сайт(ов) запущен; {tail}"
    if _blocked:
        msg += f"; заблокировано приватных/локальных адресов: {len(_blocked)}"
    return {"ok": True, "msg": msg}
