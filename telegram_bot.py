"""Телеграм-бот поверх того же RAG-конвейера.

Фоновый поллер (long-polling getUpdates) в отдельном потоке. Доступ пользователей —
с подтверждением администратором: новый пользователь попадает в «ожидание»
(pending), админ подтверждает/блокирует в веб-панели. Запросы и история Телеграм
хранятся ОТДЕЛЬНО (таблицы tg_users/tg_requests в db.py).

Запускается из app.py при старте, если задан TELEGRAM_BOT_TOKEN. Менять токен и
перезапускать бота можно из админки.
"""
from __future__ import annotations
import threading
import time

import httpx

import settings
import db
import prompts
import llm_backend
from retriever import search

_API = "https://api.telegram.org/bot{token}/{method}"

_thread: threading.Thread | None = None
_stop = threading.Event()
_state = {"running": False, "error": None, "username": None, "started": None}
_offset = None


def _token() -> str:
    return (settings.get("TELEGRAM_BOT_TOKEN") or "").strip()


def _proxy() -> str | None:
    """SOCKS5/HTTP-прокси для доступа к api.telegram.org (если задан)."""
    p = (settings.get("TELEGRAM_PROXY") or "").strip()
    return p or None


def _call(method: str, http_timeout: float = 40, **params):
    token = _token()
    if not token:
        return None
    try:
        with httpx.Client(proxy=_proxy(), timeout=http_timeout) as c:
            r = c.post(_API.format(token=token, method=method), json=params)
        return r.json()
    except Exception as e:
        _state["error"] = str(e)
        return None


def send(chat_id: int, text: str) -> None:
    """Отправить сообщение (Telegram режет на 4096 символов)."""
    if not text:
        return
    for i in range(0, len(text), 4096):
        _call("sendMessage", chat_id=chat_id, text=text[i:i + 4096],
              disable_web_page_preview=True)


def notify_approved(chat_id: int) -> None:
    send(chat_id, "✅ Доступ подтверждён. Задайте вопрос по документам компании.")


def notify_blocked(chat_id: int) -> None:
    send(chat_id, "⛔ Доступ к боту закрыт администратором.")


def _answer(question: str):
    """Синхронный RAG-ответ: поиск → контекст → LLM. Возвращает (text, sources, hits)."""
    hits = search(question)
    if not hits:
        return "В доступных документах нет точного ответа на этот вопрос.", [], []
    context = prompts.build_context(hits)
    messages = [{"role": "system", "content": settings.get("SYSTEM_PROMPT")},
                {"role": "user", "content": prompts.build_user_message(question, context)}]
    text = llm_backend.chat(messages, temperature=settings.get("TEMPERATURE"),
                            model=settings.active_model())
    sources = [{"source": h["source"], "page": h.get("page"),
                "score": round(h.get("score", 0), 3)} for h in hits]
    return text, sources, hits


def _handle(msg: dict) -> None:
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return
    frm = msg.get("from") or {}
    username = chat.get("username") or frm.get("username")
    first = chat.get("first_name") or frm.get("first_name")
    text = (msg.get("text") or "").strip()

    user = db.tg_user(chat_id)
    auto = bool(settings.get("TELEGRAM_AUTO_APPROVE"))

    # новый пользователь — регистрируем (pending или авто-approved)
    if user is None:
        status = "approved" if auto else "pending"
        db.tg_user_upsert(chat_id, username, first, status)
        if status == "approved":
            send(chat_id, "Здравствуйте! Доступ открыт. Задайте вопрос по документам "
                          "компании — отвечу по ним.")
        else:
            send(chat_id, "Здравствуйте! Запрос на доступ отправлен администратору. "
                          "Как только его подтвердят, я смогу отвечать на ваши вопросы.")
        return

    # обновим имя/username при изменении
    db.tg_user_upsert(chat_id, username, first, user["status"])

    if user["status"] == "blocked":
        send(chat_id, "⛔ Доступ к боту закрыт.")
        return
    if user["status"] == "pending":
        send(chat_id, "⏳ Ваш доступ ещё не подтверждён администратором. Ожидайте.")
        return

    # approved
    if not text:
        return
    if text.startswith("/start") or text.startswith("/help"):
        send(chat_id, "Просто напишите вопрос — я найду ответ в документах компании "
                      "и пришлю его со ссылками на источники.")
        return
    if text.startswith("/"):
        send(chat_id, "Неизвестная команда. Просто задайте вопрос текстом.")
        return

    t0 = time.time()
    try:
        ans, sources, hits = _answer(text)
        answered = bool(hits)
    except Exception as e:
        print(f"  ! telegram answer error: {e}")
        send(chat_id, "Произошла ошибка при обработке запроса. Попробуйте позже.")
        db.tg_log_request(chat_id, username, text, "", 0, 0.0,
                          int((time.time() - t0) * 1000), False, [])
        return

    latency = int((time.time() - t0) * 1000)
    out = ans
    if sources:
        srctxt = "; ".join(s["source"] + (f", с.{s['page']}" if s.get("page") else "")
                           for s in sources[:6])
        out = f"{ans}\n\n📎 Источники: {srctxt}"
    send(chat_id, out)
    db.tg_log_request(chat_id, username, text, ans, len(sources),
                      hits[0]["score"] if hits else 0.0, latency, answered, sources)


def _loop() -> None:
    global _offset
    backoff = 2
    while not _stop.is_set():
        if not _token():
            _state["running"] = False
            time.sleep(3)
            continue
        # long-poll: Telegram держит соединение до 25 с; httpx-таймаут 40 с
        r = _call("getUpdates", http_timeout=40, offset=_offset, timeout=25)
        if not r or not r.get("ok"):
            _state["running"] = bool(_token()) and r is not None
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        backoff = 2
        _state["running"] = True
        _state["error"] = None
        for upd in r.get("result", []):
            _offset = upd["update_id"] + 1
            m = upd.get("message") or upd.get("edited_message")
            if m:
                try:
                    _handle(m)
                except Exception as e:
                    print(f"  ! telegram handle error: {e}")
    _state["running"] = False


def _drain() -> None:
    """Пропустить сообщения, накопившиеся пока бот был выключен (без ответа на них)."""
    global _offset
    r = _call("getUpdates", http_timeout=15, offset=-1, timeout=0)
    if r and r.get("ok") and r.get("result"):
        _offset = r["result"][-1]["update_id"] + 1


def start() -> dict:
    """Запустить поллер, если задан токен и он ещё не запущен."""
    global _thread
    if _thread and _thread.is_alive():
        return {"ok": True, "msg": "бот уже запущен"}
    if not _token():
        return {"ok": False, "msg": "не задан TELEGRAM_BOT_TOKEN"}
    me = _call("getMe", http_timeout=15)
    if me and me.get("ok"):
        _state["username"] = me["result"].get("username")
    else:
        return {"ok": False, "msg": "неверный токен бота (getMe не прошёл)"}
    _drain()
    _stop.clear()
    _thread = threading.Thread(target=_loop, daemon=True)
    _thread.start()
    _state["started"] = time.time()
    return {"ok": True, "msg": f"бот @{_state.get('username')} запущен"}


def stop() -> None:
    _stop.set()
    _state["running"] = False


def restart() -> dict:
    stop()
    time.sleep(0.6)
    return start()


def status() -> dict:
    return {"running": bool(_state.get("running")),
            "username": _state.get("username"),
            "token_set": bool(_token()),
            "auto_approve": bool(settings.get("TELEGRAM_AUTO_APPROVE")),
            "proxy": _proxy() or "",
            "error": _state.get("error"),
            "started": _state.get("started"),
            **db.tg_counts()}
