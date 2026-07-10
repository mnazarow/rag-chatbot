"""Интеграции: API-ключи для внешних систем и веб-хуки на события.

Хранение — в таблице kv_store БД (переживает пересборку контейнера). Секреты ключей
наружу отдаются в маскированном виде. Веб-хуки отправляются в фоне (не блокируют ответ).
"""
from __future__ import annotations
import json
import secrets
import threading
import time

import db

try:
    import httpx
except Exception:
    httpx = None


# ------------------------------------------------------------------ API-ключи

def _keys_load() -> list:
    raw = db.kv_get("api_keys")
    try:
        d = json.loads(raw) if raw else []
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _keys_save(items: list) -> None:
    db.kv_set("api_keys", json.dumps(items, ensure_ascii=False))


def _mask(key: str) -> str:
    return (key[:6] + "…" + key[-4:]) if key and len(key) > 12 else "…"


def api_keys_list() -> list:
    """Список ключей для UI (сам ключ маскирован)."""
    return [{"id": k.get("id"), "label": k.get("label", ""), "ts": k.get("ts"),
             "enabled": k.get("enabled", True), "masked": _mask(k.get("key", "")),
             "calls": k.get("calls", 0), "last": k.get("last")}
            for k in _keys_load()]


def api_key_create(label: str = "") -> dict:
    key = "rag_" + secrets.token_urlsafe(32)
    item = {"id": secrets.token_hex(6), "label": (label or "").strip()[:80],
            "key": key, "ts": time.time(), "enabled": True, "calls": 0, "last": None}
    items = _keys_load()
    items.append(item)
    _keys_save(items)
    # ключ показывается ОДИН раз при создании
    return {"ok": True, "id": item["id"], "key": key, "label": item["label"],
            "msg": "ключ создан — скопируйте его сейчас, позже он не показывается"}


def api_key_revoke(key_id: str) -> dict:
    items = _keys_load()
    n = len(items)
    items = [k for k in items if k.get("id") != key_id]
    _keys_save(items)
    return {"ok": True, "removed": n - len(items)}


def api_key_valid(key: str) -> bool:
    if not key:
        return False
    items = _keys_load()
    hit = None
    for k in items:
        if k.get("key") == key and k.get("enabled", True):
            hit = k
            break
    if not hit:
        return False
    try:
        hit["calls"] = int(hit.get("calls", 0)) + 1
        hit["last"] = time.time()
        _keys_save(items)
    except Exception:
        pass
    return True


# ------------------------------------------------------------------ веб-хуки

def _hooks_load() -> list:
    raw = db.kv_get("webhooks")
    try:
        d = json.loads(raw) if raw else []
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _hooks_save(items: list) -> None:
    db.kv_set("webhooks", json.dumps(items, ensure_ascii=False))


def webhooks_list() -> list:
    return [{"id": h.get("id"), "url": h.get("url", ""), "events": h.get("events", []),
             "enabled": h.get("enabled", True)} for h in _hooks_load()]


def webhook_save(cfg: dict) -> dict:
    cfg = cfg or {}
    url = (cfg.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return {"ok": False, "msg": "укажите URL (http/https)"}
    evs = cfg.get("events") or ["question", "rating"]
    evs = [e for e in evs if e in ("question", "rating")]
    items = _hooks_load()
    hid = cfg.get("id")
    if hid:
        for h in items:
            if h.get("id") == hid:
                h.update(url=url, events=evs, enabled=bool(cfg.get("enabled", True)))
                _hooks_save(items)
                return {"ok": True, "id": hid}
    hid = secrets.token_hex(6)
    items.append({"id": hid, "url": url, "events": evs,
                  "enabled": bool(cfg.get("enabled", True))})
    _hooks_save(items)
    return {"ok": True, "id": hid}


def webhook_delete(hid: str) -> dict:
    items = _hooks_load()
    n = len(items)
    items = [h for h in items if h.get("id") != hid]
    _hooks_save(items)
    return {"ok": True, "removed": n - len(items)}


def fire(event: str, payload: dict) -> None:
    """Отправить событие всем подходящим веб-хукам (в фоне, best-effort)."""
    hooks = [h for h in _hooks_load()
             if h.get("enabled", True) and event in (h.get("events") or [])]
    if not hooks or httpx is None:
        return

    def _send():
        body = {"event": event, "ts": time.time(), "data": payload}
        for h in hooks:
            try:
                httpx.post(h["url"], json=body, timeout=8,
                           headers={"User-Agent": "RAG-Webhook/1"})
            except Exception as e:
                print(f"[webhook] {h.get('url')}: {e}")

    threading.Thread(target=_send, daemon=True).start()
