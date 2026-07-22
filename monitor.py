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


def _web_tick():
    """Раз в сутки в 00:05 заново парсить сохранённые сайты, если включено."""
    try:
        import settings
        if not settings.get("WEB_AUTO_REPARSE"):
            return
        lt = time.localtime()
        # окно 00:05–00:59: цикл раз в минуту — попадём и переспарсим один раз за сутки
        if not (lt.tm_hour == 0 and lt.tm_min >= 5):
            return
        import db
        today = time.strftime("%Y-%m-%d", lt)
        if db.kv_get("web_reparse_last") == today:
            return                               # уже запускали сегодня
        import admin_ops
        urls = admin_ops.web_saved_urls()
        if not urls:
            db.kv_set("web_reparse_last", today)
            return
        r = admin_ops.ingest_web(urls)
        if r.get("ok"):
            db.kv_set("web_reparse_last", today)  # помечаем день только при успешном запуске
            print(f"[web] авто-переспарсинг {len(urls)} сайт(ов) запущен")
        else:
            print(f"[web] авто-переспарсинг отложен: {r.get('msg')}")
    except Exception as e:
        print(f"[web] tick: {e}")


def _web_sched_tick():
    """Пер-сайтовое расписание автопарсинга: перепарсить сайты, у которых истёк
    их индивидуальный интервал (hourly/daily/weekly/Nh)."""
    try:
        import admin_ops
        due = admin_ops.web_sched_due()
        if not due:
            return
        r = admin_ops.ingest_web(due, save=False)
        if r.get("ok"):
            admin_ops.web_sched_mark(due)
            print(f"[web] пер-сайтовое расписание: запущено {len(due)} сайт(ов)")
        else:
            print(f"[web] пер-сайтовое расписание отложено: {r.get('msg')}")
    except Exception as e:
        print(f"[web] sched tick: {e}")


def _reindex_interval(sch) -> int | None:
    s = (sch or "off").strip().lower()
    m = {"hourly": 3600, "6h": 21600, "12h": 43200, "daily": 86400, "weekly": 604800}
    return m.get(s)


def _reindex_tick():
    """Инкрементальная переиндексация папки по расписанию REINDEX_SCHEDULE."""
    try:
        import settings
        iv = _reindex_interval(settings.get("REINDEX_SCHEDULE"))
        if iv is None:
            return
        import db
        last = float(db.kv_get("reindex_last") or 0)
        if time.time() - last < iv:
            return
        # помечаем время СРАЗУ (до запуска), чтобы даже при исключении в reindex не
        # долбить переиндексацию каждую минуту цикла монитора
        db.kv_set("reindex_last", str(time.time()))
        import admin_ops
        r = admin_ops.reindex(reset=False)
        print(f"[reindex] авто-переиндексация по расписанию: {r.get('msg', r)}")
    except Exception as e:
        print(f"[reindex] tick: {e}")


def _alerts_tick():
    """Проверка здоровья компонентов и алерты о переходах (раз в ~5 минут)."""
    try:
        import settings
        if not settings.get("ALERTS_ENABLED"):
            return
        import db
        iv = 300  # проверяем не чаще раза в 5 минут
        last = float(db.kv_get("alerts_last") or 0)
        if time.time() - last < iv:
            return
        import alerts
        # проверку + рассылку (SMTP/Telegram могут быть медленными) уводим в отдельный
        # поток, чтобы не блокировать цикл монитора (сбор метрик и пр.)
        db.kv_set("alerts_last", str(time.time()))

        def _run():
            try:
                r = alerts.health_tick()
                if r.get("down"):
                    print(f"[alerts] недоступны: {', '.join(r['down'])}")
            except Exception as e:
                print(f"[alerts] health_tick: {e}")
        threading.Thread(target=_run, daemon=True, name="alerts-tick").start()
    except Exception as e:
        print(f"[alerts] tick: {e}")


def _loop(interval: int):
    import db
    last_prune = 0.0
    sample_warned = False    # чтобы не спамить в лог каждую итерацию (напр. нет psutil)
    while not _stop.is_set():
        try:
            db.server_sample_save(*_sample())
            now = time.time()
            if now - last_prune > 6 * 3600:      # прунинг раз в 6 часов
                db.server_prune(PRUNE_DAYS)
                last_prune = now
            sample_warned = False
        except Exception as e:
            if not sample_warned:
                print(f"[monitor] выборка не удалась (далее подавляю повтор): {e}")
                sample_warned = True
        _org_tick()                              # ежечасная синхронизация справочника
        _web_tick()                              # ежедневный переспарсинг сайтов (00:05)
        _web_sched_tick()                        # пер-сайтовые расписания автопарсинга
        _reindex_tick()                          # авто-переиндексация папки по расписанию
        _alerts_tick()                           # проверка здоровья + алерты о сбоях
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
