"""Статические DNS-записи: разрешение заданных имён в указанные IP.

Зачем: внутренние адреса компании (например `in.vodokomfort.ru`) часто не
резолвятся из среды, где запущен RAG (особенно в Docker без доступа к внутреннему
DNS). Здесь администратор задаёт пары «имя → IP» (раздел «DNS» в админке), а этот
модуль подменяет системное разрешение имён, чтобы такие хосты работали без правки
`/etc/hosts` и настроек контейнера.

Реализация — аккуратный monkeypatch `socket.getaddrinfo`: если запрашиваемое имя
есть в карте, вместо него подставляется IP (его `getaddrinfo` разрешит тривиально).
Патч глобальный и действует на всё, что ходит по сети через стандартный сокет:
httpx (sync и async — asyncio резолвит через `socket.getaddrinfo` в пуле), urllib,
requests, клиент Qdrant и т. п. Карта кэшируется и обновляется `reload()` после
изменений в админке.
"""
from __future__ import annotations
import socket
import threading

_orig_getaddrinfo = None
_lock = threading.Lock()
_map: dict[str, str] = {}


def set_map(mapping: dict) -> None:
    global _map
    clean = {}
    for k, v in (mapping or {}).items():
        k = (k or "").strip().lower()
        v = (v or "").strip()
        if k and v:
            clean[k] = v
    with _lock:
        _map = clean


def get_map() -> dict:
    with _lock:
        return dict(_map)


def reload() -> int:
    """Перечитать записи из БД. Возвращает их количество."""
    try:
        import db
        m = db.dns_map()
    except Exception as e:
        print(f"[dns] не удалось прочитать записи: {e}")
        m = {}
    set_map(m)
    return len(m)


def _patched_getaddrinfo(host, *args, **kwargs):
    try:
        if isinstance(host, (bytes, bytearray)):
            key = host.decode("ascii", "ignore").lower()
        elif isinstance(host, str):
            key = host.lower()
        else:
            key = None
        if key:
            with _lock:
                ip = _map.get(key)
            if ip:
                host = ip
    except Exception:
        pass
    return _orig_getaddrinfo(host, *args, **kwargs)


def install() -> dict:
    """Установить перехват (идемпотентно) и загрузить записи. Возвращает статус."""
    global _orig_getaddrinfo
    if _orig_getaddrinfo is None:
        _orig_getaddrinfo = socket.getaddrinfo
        socket.getaddrinfo = _patched_getaddrinfo
    n = reload()
    return {"ok": True, "count": n}


def active() -> bool:
    return _orig_getaddrinfo is not None
