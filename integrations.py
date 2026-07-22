"""Интеграции: API-ключи для внешних систем и веб-хуки на события.

Хранение — в таблице kv_store БД (переживает пересборку контейнера). Секреты ключей
наружу отдаются в маскированном виде. Веб-хуки отправляются в фоне (не блокируют ответ).
"""
from __future__ import annotations
import atexit
import json
import secrets
import threading
import time

import db

try:
    import httpx
except Exception:
    httpx = None

# M25: _keys_lock сериализует read-modify-write ВСЕГО списка ключей (api_keys в kv_store) —
# только для операций, реально меняющих список: create/revoke. Без лока параллельные
# добавления/отзывы затирали бы друг друга (потеря новых ключей).
#
# M25 (доведение): горячий путь api_key_valid БОЛЬШЕ не переписывает весь список на каждый
# запрос. Счётчик calls/last вынесен в ОТДЕЛЬНЫЙ per-key kv-ключ (api_key_stat:<id>), инкремент
# идёт под локом только этого ключа (см. _bump_key_stat). Это снимает износ (перезапись всего
# JSON-списка на каждую проверку) и потерю параллельных правок соседних ключей/самого списка.
# Строго атомарного инкремента через db.kv_get/kv_set нет (это read-modify-write, не одна SQL-
# операция), а db.py трогать нельзя — поэтому используется per-key лок; в пределах процесса это
# корректно, конкуренция сведена к одному ключу (а не ко всему списку).
_keys_lock = threading.Lock()

# Per-key локи для инкремента счётчика вызовов: конкуренция только на конкретный api-key,
# а не на общий список. Реестр локов защищён своим мета-локом.
_stat_locks_meta = threading.Lock()
_stat_locks: dict = {}

# M23: переиспользуемый httpx-клиент с пулом keep-alive для веб-хуков (вместо нового
# соединения на каждый вызов). Ленивое создание, закрытие при завершении процесса.
_hook_http_lock = threading.Lock()
_hook_http = None


def _hook_client():
    global _hook_http
    if httpx is None:
        return None
    if _hook_http is None or _hook_http.is_closed:
        with _hook_http_lock:
            if _hook_http is None or _hook_http.is_closed:
                _hook_http = httpx.Client(
                    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20,
                                        keepalive_expiry=30.0))
    return _hook_http


@atexit.register
def _close_hook_client():
    global _hook_http
    try:
        if _hook_http is not None and not _hook_http.is_closed:
            _hook_http.close()
    except Exception:
        pass


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


# --- счётчик вызовов ключа: отдельный per-key kv-ключ (не трогаем общий список) ---

def _stat_kv_key(key_id: str) -> str:
    return f"api_key_stat:{key_id}"


def _stat_lock(key_id: str) -> threading.Lock:
    with _stat_locks_meta:
        lk = _stat_locks.get(key_id)
        if lk is None:
            lk = threading.Lock()
            _stat_locks[key_id] = lk
        return lk


def _stat_load(key_id: str) -> dict:
    """{'calls': int, 'last': float|None} из per-key kv (пусто, если ещё не писали)."""
    try:
        raw = db.kv_get(_stat_kv_key(key_id))
        d = json.loads(raw) if raw else {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _bump_key_stat(key_id: str) -> None:
    """Инкремент calls и обновление last в отдельном per-key kv-ключе.

    Не переписывает общий список api_keys → нет износа и потери параллельных правок
    других ключей/списка. Инкремент — read-modify-write под локом ТОЛЬКО этого ключа
    (строго атомарного incr в БД через kv нет, а db.py править нельзя); в пределах
    процесса корректно, конкуренция сведена к одному api-key."""
    if not key_id:
        return
    with _stat_lock(key_id):
        d = _stat_load(key_id)
        d["calls"] = int(d.get("calls", 0)) + 1
        d["last"] = time.time()
        try:
            db.kv_set(_stat_kv_key(key_id), json.dumps(d, ensure_ascii=False))
        except Exception:
            pass


def api_keys_list() -> list:
    """Список ключей для UI (сам ключ маскирован).

    calls/last берём из per-key kv-счётчика (_stat_*); фолбэк на устаревшие поля
    внутри самого ключа — для ключей, созданных до выноса счётчика."""
    out = []
    for k in _keys_load():
        st = _stat_load(k.get("id"))
        out.append({"id": k.get("id"), "label": k.get("label", ""), "ts": k.get("ts"),
                    "enabled": k.get("enabled", True), "masked": _mask(k.get("key", "")),
                    "calls": int(st.get("calls", k.get("calls", 0)) or 0),
                    "last": st.get("last", k.get("last"))})
    return out


def api_key_create(label: str = "") -> dict:
    key = "rag_" + secrets.token_urlsafe(32)
    item = {"id": secrets.token_hex(6), "label": (label or "").strip()[:80],
            "key": key, "ts": time.time(), "enabled": True, "calls": 0, "last": None}
    with _keys_lock:   # M25: не терять параллельные добавления/инкременты
        items = _keys_load()
        items.append(item)
        _keys_save(items)
    # ключ показывается ОДИН раз при создании
    return {"ok": True, "id": item["id"], "key": key, "label": item["label"],
            "msg": "ключ создан — скопируйте его сейчас, позже он не показывается"}


def api_key_revoke(key_id: str) -> dict:
    with _keys_lock:   # M25: согласованно с параллельными valid/create
        items = _keys_load()
        n = len(items)
        items = [k for k in items if k.get("id") != key_id]
        _keys_save(items)
    try:
        db.kv_del(_stat_kv_key(key_id))   # убираем осиротевший per-key счётчик
    except Exception:
        pass
    return {"ok": True, "removed": n - len(items)}


def api_key_valid(key: str) -> bool:
    if not key:
        return False
    # M25 (доведение): проверка ключа — по СНИМКУ списка без мутации (одиночный
    # kv_get + json.loads, тор-ридов нет), поэтому НЕ переписываем весь список api_keys
    # на каждый запрос и не затираем параллельные create/revoke/чужие счётчики.
    hit = None
    for k in _keys_load():
        if k.get("key") == key and k.get("enabled", True):
            hit = k
            break
    if not hit:
        return False
    # Счётчик вызовов/last — атомарный инкремент в отдельном per-key kv-ключе
    # (конкуренция только на этот ключ), не трогая общий список.
    _bump_key_stat(hit.get("id"))
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
        c = _hook_client()               # M23: общий клиент с keep-alive
        for h in hooks:
            try:
                (c or httpx).post(h["url"], json=body, timeout=8,
                                  headers={"User-Agent": "RAG-Webhook/1"})
            except Exception as e:
                print(f"[webhook] {h.get('url')}: {e}")

    threading.Thread(target=_send, daemon=True).start()
