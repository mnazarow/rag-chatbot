"""Опциональный кэш Redis для тяжёлых агрегатов (статистика/аналитика).

По умолчанию ВЫКЛЮЧЕН (REDIS_ENABLED=False). При выключенном или недоступном
Redis всё работает напрямую без кэша — приложение не зависит от Redis.

Инвалидация — по версии: каждая запись в журнал увеличивает счётчик rag:ver,
из-за чего ключи кэша «протухают» мгновенно (а TTL лишь ограничивает память).
"""
from __future__ import annotations
import json
import threading

_client = None
_client_key = None          # параметры подключения, под которые создан клиент
_lock = threading.Lock()
_ver_local = 0              # запасной счётчик версии (если Redis вдруг недоступен)


def _cfg():
    import settings
    return settings


def enabled() -> bool:
    try:
        return bool(_cfg().get("REDIS_ENABLED"))
    except Exception:
        return False


def _params():
    s = _cfg()
    return ((s.get("REDIS_HOST") or "127.0.0.1").strip(),
            int(s.get("REDIS_PORT") or 6379),
            int(s.get("REDIS_DB") or 0),
            s.get("REDIS_PASSWORD") or "")


def client():
    """Вернуть живой Redis-клиент или None (если выключен/недоступен/нет модуля)."""
    global _client, _client_key
    if not enabled():
        return None
    key = _params()
    with _lock:
        if _client is not None and _client_key == key:
            return _client
        try:
            import redis
        except Exception:
            _client = None
            return None
        host, port, db, pw = key
        try:
            c = redis.Redis(host=host, port=port, db=db, password=pw or None,
                            socket_connect_timeout=2, socket_timeout=2,
                            decode_responses=True)
            c.ping()
            _client, _client_key = c, key
            return c
        except Exception:
            _client = None
            return None


def _ver() -> int:
    c = client()
    if not c:
        return _ver_local
    try:
        return int(c.get("rag:ver") or 0)
    except Exception:
        return _ver_local


def bump() -> None:
    """Инвалидировать кэш (вызывается при любой записи в журнал)."""
    global _ver_local
    _ver_local += 1
    c = client()
    if c:
        try:
            c.incr("rag:ver")
        except Exception:
            pass


def get_or_set(name: str, ttl: int, producer):
    """Вернуть кэш по ключу name или вычислить producer() и закэшировать на ttl сек."""
    c = client()
    if not c:
        return producer()
    key = f"rag:cache:{name}:{_ver()}"
    try:
        v = c.get(key)
        if v is not None:
            return json.loads(v)
    except Exception:
        return producer()
    val = producer()
    try:
        c.setex(key, ttl, json.dumps(val, ensure_ascii=False))
    except Exception:
        pass
    return val


def clear() -> int:
    """Удалить все кэш-ключи приложения. Возвращает число удалённых ключей."""
    c = client()
    if not c:
        return 0
    n = 0
    try:
        for k in c.scan_iter("rag:cache:*"):
            c.delete(k)
            n += 1
        c.incr("rag:ver")
    except Exception:
        pass
    return n


def status() -> dict:
    en = enabled()
    out = {"enabled": en, "reachable": False, "keys": 0, "error": "",
           "host": "", "version": "", "used_memory": ""}
    if not en:
        return out
    host, port, db, _ = _params()
    out["host"] = f"{host}:{port}/{db}"
    try:
        import redis  # noqa: F401
    except Exception:
        out["error"] = "модуль redis не установлен (pip install redis)"
        return out
    c = client()
    if not c:
        out["error"] = "Redis недоступен по указанному адресу"
        return out
    out["reachable"] = True
    try:
        out["keys"] = sum(1 for _ in c.scan_iter("rag:cache:*"))
        info = c.info()
        out["used_memory"] = info.get("used_memory_human", "")
        out["version"] = info.get("redis_version", "")
        out["mode"] = info.get("redis_mode", "")
        out["clients"] = info.get("connected_clients")
        out["uptime_sec"] = info.get("uptime_in_seconds")
        out["total_keys"] = c.dbsize()
        hits = info.get("keyspace_hits") or 0
        misses = info.get("keyspace_misses") or 0
        out["hits"] = hits
        out["misses"] = misses
        tot = hits + misses
        out["hit_rate"] = round(hits / tot * 100, 1) if tot else None
        out["evicted_keys"] = info.get("evicted_keys")
        out["version_full"] = info.get("redis_version", "")
    except Exception:
        pass
    return out
