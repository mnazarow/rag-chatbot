"""Фоновый сбор метрик загрузки хоста для истории «час/день/неделя/месяц/год».

Лёгкий поток раз в SAMPLE_INTERVAL секунд снимает CPU/память/swap/диск/GPU и
складывает в таблицу server_samples (см. db.py). По этим выборкам строятся
агрегаты и рекомендации по железу (admin_ops.server_history).

Сбор включается автоматически при старте приложения (app.py → monitor.start()).
Без psutil поток не запускается — история будет недоступна, но текущая загрузка
по-прежнему работает. Старые выборки периодически удаляются (db.server_prune).
"""
from __future__ import annotations
import os
import threading
import time

SAMPLE_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "60"))   # секунды
PRUNE_DAYS = int(os.environ.get("MONITOR_PRUNE_DAYS", "400"))      # хранить ~13 мес.

_thread: threading.Thread | None = None
_stop = threading.Event()


def _gpu_metrics():
    """(средняя загрузка GPU %, макс. использование видеопамяти %) или (None, None)."""
    try:
        import admin_ops
        g = admin_ops._gpu_info()
    except Exception:
        return None, None
    devs = (g or {}).get("devices") or []
    if not devs:
        return None, None
    utils, mems = [], []
    for d in devs:
        u = d.get("util")
        if u is not None:
            utils.append(float(u))
        mu, mt = d.get("mem_used"), d.get("mem_total")
        if mu is not None and mt:
            mems.append(float(mu) / float(mt) * 100.0)
    gpu_util = round(sum(utils) / len(utils), 1) if utils else None
    gpu_mem = round(max(mems), 1) if mems else None
    return gpu_util, gpu_mem


def _sample():
    """Снимок ключевых метрик. cpu_percent(interval=1) блокирует ~1с (мы в потоке)."""
    import psutil
    cpu = psutil.cpu_percent(interval=1.0)
    mem = psutil.virtual_memory().percent
    try:
        swap = psutil.swap_memory().percent
    except Exception:
        swap = 0.0
    disk = 0.0
    for part in psutil.disk_partitions(all=False):
        try:
            disk = max(disk, psutil.disk_usage(part.mountpoint).percent)
        except Exception:
            continue
    gpu_util, gpu_mem = _gpu_metrics()
    return cpu, mem, swap, disk, gpu_util, gpu_mem


def _org_tick():
    """Синхронизировать справочник компании, если включено и прошёл час."""
    try:
        import org_structure
        if org_structure.due_for_sync():
            r = org_structure.sync()
            if r.get("ok"):
                print(f"[org] синхронизировано записей: {r.get('count')}")
            else:
                print(f"[org] синхронизация не удалась: {r.get('error')}")
    except Exception as e:
        print(f"[org] tick: {e}")


def _loop(interval: int):
    import db
    last_prune = 0.0
    while not _stop.is_set():
        try:
            db.server_sample_save(*_sample())
            now = time.time()
            if now - last_prune > 6 * 3600:      # прунинг раз в 6 часов
                db.server_prune(PRUNE_DAYS)
                last_prune = now
        except Exception as e:
            print(f"[monitor] выборка не удалась: {e}")
        _org_tick()                              # ежечасная синхронизация справочника
        _stop.wait(interval)


def start(interval: int | None = None) -> dict:
    """Запустить фоновый поток (идемпотентно): сбор метрик + часовая
    синхронизация справочника компании. Возвращает статус."""
    global _thread
    if _thread and _thread.is_alive():
        return {"ok": True, "msg": "уже запущен"}
    try:
        import psutil  # noqa: F401
        have_metrics = True
    except Exception:
        have_metrics = False
        print("[monitor] psutil не установлен — история загрузки недоступна "
              "(pip install psutil); справочник компании синхронизируется как обычно")
    _stop.clear()
    iv = int(interval or SAMPLE_INTERVAL)
    _thread = threading.Thread(target=_loop, args=(iv,), daemon=True, name="monitor")
    _thread.start()
    msg = f"сбор метрик каждые {iv}с" if have_metrics else "фоновый поток запущен (без метрик)"
    return {"ok": True, "msg": msg}


def stop() -> None:
    _stop.set()


def running() -> bool:
    return bool(_thread and _thread.is_alive())
