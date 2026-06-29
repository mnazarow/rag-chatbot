"""Структура компании: парсинг справочника сотрудников по URL раз в час.

Источник — выгрузка в формате SpreadsheetML (Excel 2003 XML, как у 1С/СБИС):
Workbook → Worksheet → Table → Row → Cell → Data. Первая строка — заголовки,
далее по сотруднику на строку. Колонки (по порядку):

  работает · работает до · фамилия · имя · отчество · должность · отдел ·
  тел. рабочий · тел. внутренний · тел. мобильный · электропочта · поставщики

Отдел в источнике приходит с HTML-мусором (`<span>(МСК)</span>`, ссылки на базы
знаний, &nbsp;) — чистим и заодно вынимаем код локации из скобок.

Данные складываются в таблицу org_employees (db.py, полная замена при каждой
синхронизации). Конфиг (URL, включение) и статус последней синхронизации —
в kv_store. Расписание (раз в час) крутит monitor.py.
"""
from __future__ import annotations
import html as _html
import re
import xml.etree.ElementTree as ET

import db

_SS = "urn:schemas-microsoft-com:office:spreadsheet"
_CELL = f"{{{_SS}}}Cell"
_ROW = f"{{{_SS}}}Row"
_DATA = f"{{{_SS}}}Data"
_INDEX = f"{{{_SS}}}Index"

# Позиции колонок в строке (0-based)
COLS = ["active", "work_until", "last_name", "first_name", "middle_name",
        "position", "department", "phone_work", "phone_ext", "phone_mobile",
        "email", "suppliers"]

KV_URL = "org.url"
KV_ENABLED = "org.enabled"
KV_STATUS = "org.status"          # JSON: {ts, ok, count, error}

_TAG_RE = re.compile(r"<[^>]+>")
_LOC_RE = re.compile(r"\(([^()]{1,40})\)\s*$")
_WS_RE = re.compile(r"\s+")


def get_config() -> dict:
    return {"url": db.kv_get(KV_URL) or "",
            "enabled": (db.kv_get(KV_ENABLED) or "0") == "1"}


def set_config(url: str | None = None, enabled: bool | None = None) -> dict:
    if url is not None:
        db.kv_set(KV_URL, url.strip())
    if enabled is not None:
        db.kv_set(KV_ENABLED, "1" if enabled else "0")
    return get_config()


def get_status() -> dict:
    import json
    raw = db.kv_get(KV_STATUS)
    if not raw:
        return {"ts": None, "ok": None, "count": 0, "error": None}
    try:
        return json.loads(raw)
    except Exception:
        return {"ts": None, "ok": None, "count": 0, "error": None}


def _set_status(ok: bool, count: int, error: str | None) -> None:
    import json
    import time
    db.kv_set(KV_STATUS, json.dumps(
        {"ts": time.time(), "ok": ok, "count": count, "error": error},
        ensure_ascii=False))


def clean_department(raw: str) -> tuple[str, str]:
    """(чистое название отдела, код локации). Убираем теги/ссылки/&nbsp;."""
    if not raw:
        return "", ""
    s = _html.unescape(raw)
    # выкидываем блок «База знаний …» и любые ссылки целиком
    s = re.sub(r"<a\b.*?</a>", " ", s, flags=re.IGNORECASE | re.DOTALL)
    s = s.replace("&nbsp;", " ").replace("\xa0", " ")
    # локация из последних скобок до удаления тегов (внутри может быть <span>)
    no_tags = _WS_RE.sub(" ", _TAG_RE.sub(" ", s)).strip()
    loc = ""
    m = _LOC_RE.search(no_tags)
    if m:
        loc = m.group(1).strip()
        no_tags = no_tags[:m.start()].strip()
    no_tags = no_tags.strip(" ,;·-")
    return no_tags, loc


def _cell_values(row) -> list[str]:
    """Значения ячеек строки с учётом ss:Index (пропуски колонок)."""
    vals: list[str] = []
    col = 0
    for cell in row.findall(_CELL):
        idx = cell.get(_INDEX)
        if idx:
            col = int(idx) - 1
        while len(vals) < col:
            vals.append("")
        data = cell.find(_DATA)
        vals.append((data.text or "").strip() if data is not None else "")
        col += 1
    return vals


def parse_xml(text: str) -> list[dict]:
    """Распарсить SpreadsheetML в список сотрудников."""
    text = text.lstrip("﻿")
    root = ET.fromstring(text)
    rows = root.findall(f".//{_ROW}")
    out: list[dict] = []
    for i, row in enumerate(rows):
        vals = _cell_values(row)
        if not vals:
            continue
        # пропускаем строку-заголовок (содержит «фамилия»/«должность»)
        joined = " ".join(vals).lower()
        if i == 0 and ("фамилия" in joined and "должность" in joined):
            continue

        def g(key):
            j = COLS.index(key)
            return vals[j].strip() if j < len(vals) else ""

        last = g("last_name")
        if not last:
            continue
        dept_raw = g("department")
        dept, loc = clean_department(dept_raw)
        rec = {
            "active": 1 if g("active").lower() in ("да", "yes", "1", "true") else 0,
            "work_until": g("work_until"),
            "last_name": last,
            "first_name": g("first_name"),
            "middle_name": g("middle_name"),
            "position": g("position"),
            "department": dept,
            "location": loc,
            "phone_work": g("phone_work"),
            "phone_ext": g("phone_ext"),
            "phone_mobile": g("phone_mobile"),
            "email": g("email"),
            "suppliers": g("suppliers"),
        }
        out.append(rec)
    return out


def _fetch(url: str, timeout: int = 30) -> str:
    """Скачать XML по URL. httpx (как в проекте), иначе urllib. cp1251-фолбэк."""
    data: bytes
    try:
        import httpx
        r = httpx.get(url, timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": "RAG-org/1.0"})
        r.raise_for_status()
        data = r.content
    except ImportError:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "RAG-org/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", "replace")


def sync(url: str | None = None) -> dict:
    """Скачать, распарсить и заменить справочник. Возвращает статус."""
    url = (url or db.kv_get(KV_URL) or "").strip()
    if not url:
        _set_status(False, 0, "URL не задан")
        return {"ok": False, "error": "URL не задан", "count": 0}
    aid = None
    try:
        import activity
        aid = activity.start("parse", "Структура компании", "загрузка по URL")
    except Exception:
        aid = None
    try:
        text = _fetch(url)
        if aid is not None:
            import activity
            activity.update(aid, stage="разбор и сохранение")
        rows = parse_xml(text)
    except Exception as e:
        msg = str(e)[:300]
        _set_status(False, 0, msg)
        if aid is not None:
            import activity
            activity.finish(aid, ok=False, stage="ошибка")
        return {"ok": False, "error": msg, "count": 0}
    n = db.org_replace(rows)
    _set_status(True, n, None)
    if aid is not None:
        import activity
        activity.finish(aid, ok=True, stage=f"загружено {n}")
    return {"ok": True, "count": n, "error": None}


def due_for_sync(interval: int = 3600) -> bool:
    """Пора ли синхронизировать (включено и прошёл интервал)."""
    if (db.kv_get(KV_ENABLED) or "0") != "1":
        return False
    if not (db.kv_get(KV_URL) or "").strip():
        return False
    st = get_status()
    ts = st.get("ts")
    if not ts:
        return True
    import time
    return (time.time() - float(ts)) >= interval
