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
# M24: опциональный allowlist суффиксов имён, к которым РАЗРЕШЕНО применять переопределение.
# Пустой список (по умолчанию) = применять ко всем записям карты — сохраняет прежнее поведение
# (в т.ч. документированный кейс `in.vodokomfort.ru`, который выглядит как публичный домен и не
# попал бы под жёсткий allowlist «внутренних» суффиксов). Если задан — переопределяются только
# имена, оканчивающиеся на один из суффиксов; прочие резолвятся системно. Это ограничивает
# «радиус поражения» глобального патча (напр. не даёт случайно перехватить api.telegram.org).
_allow_suffixes: list[str] = []


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


def set_allowlist(suffixes) -> None:
    """Ограничить переопределение именами с указанными суффиксами (M24). Пусто = без ограничения.
    Пример: set_allowlist([".corp.local", ".vodokomfort.ru"])."""
    global _allow_suffixes
    clean = [str(s).strip().lower() for s in (suffixes or []) if str(s).strip()]
    with _lock:
        _allow_suffixes = clean


def _allowed(key: str) -> bool:
    with _lock:
        sfx = list(_allow_suffixes)
    if not sfx:
        return True   # без allowlist — прежнее поведение (переопределяем все записи карты)
    for s in sfx:
        bare = s.lstrip(".")          # суффикс без ведущей точки
        if key == bare or key.endswith("." + bare):
            return True
    return False


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
        if key and _allowed(key):
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


def uninstall() -> dict:
    """Снять перехват и восстановить оригинальный socket.getaddrinfo (M24). Идемпотентно.
    Нужно, т.к. install() ставит ГЛОБАЛЬНЫЙ monkeypatch — без снятия он живёт до конца процесса
    и влияет на всё сетевое взаимодействие. После uninstall() карта сохраняется (можно снова
    install())."""
    global _orig_getaddrinfo
    if _orig_getaddrinfo is not None:
        try:
            # восстанавливаем только если патч всё ещё наш (не перекрыт кем-то ещё)
            if socket.getaddrinfo is _patched_getaddrinfo:
                socket.getaddrinfo = _orig_getaddrinfo
        finally:
            _orig_getaddrinfo = None
        return {"ok": True, "restored": True}
    return {"ok": True, "restored": False}


# Псевдоним восстановления (совместимое имя)
restore = uninstall


def active() -> bool:
    return _orig_getaddrinfo is not None


def test(hostname: str, ip: str = "", ports=(443, 80)) -> dict:
    """Проверить статическую DNS-запись и вернуть лог.
    Шаги: резолвинг имени с учётом переопределения, сравнение с системным DNS и
    TCP-проверка доступности адреса. Возвращает {ok, log}."""
    import time
    host = (hostname or "").strip()
    ipv = (ip or "").strip()
    log: list[str] = []
    def add(s): log.append(s)
    if not host:
        return {"ok": False, "log": "Не указано имя хоста."}

    key = host.lower()
    mapped = get_map().get(key)
    add(f"Запись: {host} → {ipv or mapped or '(IP не задан)'}")
    add(f"Перехват DNS активен: {'да' if active() else 'нет'}")
    if mapped:
        add(f"В карте переопределений: {host} → {mapped}")
    elif ipv:
        add("В карте переопределений записи ещё нет — сохраните её (💾), чтобы переопределение применялось приложением.")
    else:
        add("В карте переопределений записи нет.")

    # 1) как имя резолвится в приложении (через переопределённый getaddrinfo)
    resolved: list[str] = []
    try:
        infos = socket.getaddrinfo(host, None)
        resolved = sorted({i[4][0] for i in infos})
        add(f"Резолвинг (с переопределением): {', '.join(resolved) or '—'}")
    except Exception as e:
        add(f"Резолвинг не удался: {e}")

    # 2) системный DNS без переопределения — для сравнения
    if _orig_getaddrinfo is not None:
        try:
            sys_ips = sorted({i[4][0] for i in _orig_getaddrinfo(host, None)})
            add(f"Системный DNS (без переопределения): {', '.join(sys_ips) or '—'}")
        except Exception as e:
            add(f"Системный DNS: имя не резолвится ({e.__class__.__name__}) — для этого и нужна статическая запись.")

    # 3) TCP-доступность адреса (введённый IP → из карты → результат резолвинга)
    target = ipv or mapped or (resolved[0] if resolved else "")
    ok_any = False
    if target:
        add(f"Проверяю доступность {target} (порты {', '.join(str(p) for p in ports)}):")
        for p in ports:
            t0 = time.time()
            try:
                with socket.create_connection((target, p), timeout=3):
                    ms = int((time.time() - t0) * 1000)
                    add(f"  TCP {target}:{p} — доступен ({ms} мс)")
                    ok_any = True
            except Exception as e:
                add(f"  TCP {target}:{p} — недоступен ({e.__class__.__name__})")
    else:
        add("Нет IP для проверки доступности.")

    if ok_any:
        add("Итог: OK — имя резолвится и адрес отвечает.")
    elif resolved:
        add("Итог: имя резолвится, но порты 443/80 не ответили (сервис может слушать другой порт или закрыт фаервол).")
    else:
        add("Итог: проблема — имя не резолвится. Проверьте IP в записи.")
    return {"ok": ok_any, "log": "\n".join(log)}
