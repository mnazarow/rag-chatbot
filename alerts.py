"""Надёжность: алерты о сбоях.

Каналы: Телеграм (через токен основного бота) и e-mail (SMTP). Есть:
  - троттлинг: повторный алерт по одной проблеме — не чаще ALERT_COOLDOWN секунд;
  - переходы состояния: при падении шлётся «упал», при восстановлении — «снова доступен»
    (состояние хранится в kv_store, поэтому переживает перезапуск);
  - алерты о падении фоновых задач (индексация/парсинг/…).

Точки вызова:
  - monitor.py периодически зовёт `health_tick()`;
  - admin_ops._bg при падении задачи зовёт `job_failed(label, tail)`;
  - endpoint /api/admin/alerts/test зовёт `send_test()`.
"""
from __future__ import annotations

import json
import smtplib
import time
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate

import settings

# Ключ состояния в kv_store: alert:state:<key> → {"status","since","last_alert"}
_KV_PREFIX = "alert:state:"
# Последние отправленные алерты (для UI): кольцевой лог в kv
_KV_LOG = "alert:log"
_LOG_MAX = 25


# ------------------------------------------------------------------ состояние --

def _state_get(key: str) -> dict:
    try:
        import db
        raw = db.kv_get(_KV_PREFIX + key)
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def _state_set(key: str, st: dict) -> None:
    try:
        import db
        db.kv_set(_KV_PREFIX + key, json.dumps(st, ensure_ascii=False))
    except Exception:
        pass


def _state_del(key: str) -> None:
    try:
        import db
        db.kv_del(_KV_PREFIX + key)
    except Exception:
        pass


def _log_add(entry: dict) -> None:
    """Добавить запись в кольцевой лог алертов (для отображения в админке)."""
    try:
        import db
        raw = db.kv_get(_KV_LOG)
        arr = json.loads(raw) if raw else []
    except Exception:
        arr = []
    arr.insert(0, entry)
    arr = arr[:_LOG_MAX]
    try:
        import db
        db.kv_set(_KV_LOG, json.dumps(arr, ensure_ascii=False))
    except Exception:
        pass


def recent(limit: int = _LOG_MAX) -> list[dict]:
    try:
        import db
        raw = db.kv_get(_KV_LOG)
        arr = json.loads(raw) if raw else []
        return arr[:limit]
    except Exception:
        return []


# --------------------------------------------------------------------- каналы --

def _send_telegram(subject: str, body: str) -> tuple[bool, str]:
    chat = (settings.get("ALERT_TG_CHAT") or "").strip()
    if not chat:
        return False, "chat_id не задан"
    try:
        import telegram_bot
        if not telegram_bot._token():
            return False, "токен бота не задан"
        text = f"{subject}\n\n{body}".strip()
        ok = telegram_bot._send_ok(int(chat), text)
        return bool(ok), "" if ok else "Telegram отклонил сообщение"
    except Exception as e:
        return False, str(e)


def _send_email(subject: str, body: str) -> tuple[bool, str]:
    rcpts = [a.strip() for a in (settings.get("ALERT_EMAIL") or "").split(",") if a.strip()]
    host = (settings.get("SMTP_HOST") or "").strip()
    if not rcpts:
        return False, "получатель не задан"
    if not host:
        return False, "SMTP_HOST не задан"
    user = (settings.get("SMTP_USER") or "").strip()
    pwd = settings.get("SMTP_PASSWORD") or ""
    sender = (settings.get("SMTP_FROM") or "").strip() or user or "rag-alerts@localhost"
    port = int(settings.get("SMTP_PORT") or 587)
    mode = (settings.get("SMTP_TLS") or "starttls").strip().lower()
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = formataddr(("RAG алерты", sender))
        msg["To"] = ", ".join(rcpts)
        msg["Date"] = formatdate(localtime=True)
        if mode == "ssl":
            srv = smtplib.SMTP_SSL(host, port, timeout=20)
        else:
            srv = smtplib.SMTP(host, port, timeout=20)
        with srv:
            srv.ehlo()
            if mode == "starttls":
                srv.starttls()
                srv.ehlo()
            if user:
                srv.login(user, pwd)
            srv.sendmail(sender, rcpts, msg.as_string())
        return True, ""
    except Exception as e:
        return False, str(e)


def _dispatch(subject: str, body: str) -> dict:
    """Разослать по всем настроенным каналам. Возвращает {channel: {ok, error}}."""
    res: dict = {}
    if (settings.get("ALERT_TG_CHAT") or "").strip():
        ok, err = _send_telegram(subject, body)
        res["telegram"] = {"ok": ok, "error": err}
    if (settings.get("ALERT_EMAIL") or "").strip():
        ok, err = _send_email(subject, body)
        res["email"] = {"ok": ok, "error": err}
    return res


# --------------------------------------------------------------- публичный API --

def notify(key: str, subject: str, body: str, level: str = "error") -> dict:
    """Единичный алерт с троттлингом по `key`. Возвращает {sent, channels}.
    Не трогает up/down-состояние — для разовых событий (напр. падение задачи)."""
    if not settings.get("ALERTS_ENABLED"):
        return {"sent": False, "reason": "alerts выключены"}
    st = _state_get(key)
    now = time.time()
    cooldown = int(settings.get("ALERT_COOLDOWN") or 900)
    if st.get("last_alert") and (now - st["last_alert"]) < cooldown:
        return {"sent": False, "reason": "cooldown"}
    channels = _dispatch(subject, body)
    st["last_alert"] = now
    _state_set(key, st)
    _log_add({"ts": now, "key": key, "level": level, "subject": subject,
              "channels": channels})
    return {"sent": True, "channels": channels}


def report_down(key: str, subject: str, body: str) -> dict:
    """Компонент упал. Первый переход up→down шлётся сразу; далее — по cooldown."""
    if not settings.get("ALERTS_ENABLED"):
        return {"sent": False, "reason": "alerts выключены"}
    st = _state_get(key)
    now = time.time()
    first = st.get("status") != "down"
    if first:
        st = {"status": "down", "since": now, "last_alert": 0}
    cooldown = int(settings.get("ALERT_COOLDOWN") or 900)
    sent = False
    channels = {}
    if first or (now - (st.get("last_alert") or 0)) >= cooldown:
        channels = _dispatch(f"🔴 {subject}", body)
        st["last_alert"] = now
        sent = True
        _log_add({"ts": now, "key": key, "level": "error",
                  "subject": f"🔴 {subject}", "channels": channels})
    _state_set(key, st)
    return {"sent": sent, "channels": channels, "first": first}


def report_up(key: str, subject: str, body: str = "") -> dict:
    """Компонент восстановился. Если ранее был down — шлём «снова доступен»."""
    st = _state_get(key)
    if st.get("status") != "down":
        return {"sent": False, "reason": "не был в падении"}
    now = time.time()
    dur = int(now - (st.get("since") or now))
    _state_del(key)
    if not settings.get("ALERTS_ENABLED"):
        return {"sent": False, "reason": "alerts выключены"}
    text = body or f"Компонент снова доступен (недоступен был ~{dur} с)."
    channels = _dispatch(f"🟢 {subject}", text)
    _log_add({"ts": now, "key": key, "level": "ok",
              "subject": f"🟢 {subject}", "channels": channels})
    return {"sent": True, "channels": channels, "downtime": dur}


# ------------------------------------------------------- проверка компонентов --
# Лёгкие инфраструктурные пробы (без загрузки моделей и без генерации LLM), чтобы
# фоновый монитор не грел эмбеддер/реранкер и не создавал нагрузку на LLM.

def _probe_vector() -> tuple[bool, str]:
    import vectorstore
    vb = vectorstore.backend()
    ok = vectorstore.ping()
    return ok, (f"{vb}: доступна" if ok else f"{vb}: не отвечает")


def _probe_llm() -> tuple[bool, str]:
    import httpx
    b = (settings.get("LLM_BACKEND") or "ollama").lower()
    try:
        if b == "openai":
            r = httpx.get(f"{settings.get('LLM_BASE_URL')}/models",
                          headers={"Authorization": f"Bearer {settings.get('LLM_API_KEY')}"},
                          timeout=6)
        else:
            r = httpx.get(f"{settings.get('OLLAMA_URL')}/api/tags", timeout=6)
        return r.status_code == 200, f"{b}: HTTP {r.status_code}"
    except Exception as e:
        return False, f"{b}: {e}"


def _probe_db() -> tuple[bool, str]:
    try:
        import db
        db.stats()
        return True, "журнал доступен"
    except Exception as e:
        return False, str(e)


# (человекочитаемое имя, функция-проверка → (ok: bool, detail: str))
def _checks():
    return [
        ("Векторная база", _probe_vector),
        ("LLM (движок)", _probe_llm),
        ("Журнал (БД)", _probe_db),
    ]


def health_tick() -> dict:
    """Один проход проверки критичных компонентов с алертами о переходах.
    Зовётся из фонового монитора. Возвращает сводку {checked, down:[...]}."""
    if not settings.get("ALERTS_ENABLED"):
        return {"checked": 0, "down": []}
    down = []
    checked = 0
    for name, fn in _checks():
        checked += 1
        key = "health:" + name
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, str(e)
        if ok:
            report_up(key, f"{name}: восстановлен")
        else:
            down.append(name)
            report_down(key, f"{name}: недоступен",
                        f"Компонент «{name}» не прошёл проверку.\n\nДетали: {detail}")
    return {"checked": checked, "down": down}


def job_failed(label: str, tail: str = "") -> dict:
    """Алерт о падении фоновой задачи (зовётся из admin_ops._bg)."""
    if not (settings.get("ALERTS_ENABLED") and settings.get("ALERT_ON_JOB_FAIL")):
        return {"sent": False}
    body = f"Фоновая задача «{label}» завершилась с ошибкой."
    if tail:
        body += f"\n\nХвост лога:\n{tail[-1500:]}"
    return notify("job:" + label, f"Задача «{label}» упала", body, level="error")


def send_test() -> dict:
    """Тестовый алерт по всем настроенным каналам (для кнопки в админке)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    channels = _dispatch("🔔 Проверка алертов RAG",
                         f"Это тестовое уведомление о работоспособности каналов.\n"
                         f"Время: {ts}.")
    if not channels:
        return {"ok": False, "msg": "не настроен ни один канал (Telegram chat_id или e-mail)",
                "channels": {}}
    ok = any(c.get("ok") for c in channels.values())
    _log_add({"ts": time.time(), "key": "test", "level": "test",
              "subject": "🔔 Проверка алертов", "channels": channels})
    return {"ok": ok, "channels": channels}


def status() -> dict:
    """Сводка для админки: включено ли, какие каналы настроены, активные падения, лог."""
    active = []
    try:
        import db
        # активные падения: пробегаем известные health-ключи
        for name, _ in _checks():
            st = _state_get("health:" + name)
            if st.get("status") == "down":
                active.append({"name": name, "since": st.get("since")})
    except Exception:
        pass
    return {
        "enabled": bool(settings.get("ALERTS_ENABLED")),
        "telegram": bool((settings.get("ALERT_TG_CHAT") or "").strip()),
        "email": bool((settings.get("ALERT_EMAIL") or "").strip()
                      and (settings.get("SMTP_HOST") or "").strip()),
        "cooldown": int(settings.get("ALERT_COOLDOWN") or 900),
        "active_down": active,
        "recent": recent(),
    }
