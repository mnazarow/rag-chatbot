"""Телеграм-бот поверх того же RAG-конвейера.

Фоновый поллер (long-polling getUpdates) в отдельном потоке. Доступ пользователей —
с подтверждением администратором: новый пользователь попадает в «ожидание»
(pending), админ подтверждает/блокирует в веб-панели. Запросы и история Телеграм
хранятся ОТДЕЛЬНО (таблицы tg_users/tg_requests в db.py).

Запускается из app.py при старте, если задан TELEGRAM_BOT_TOKEN. Менять токен и
перезапускать бота можно из админки.
"""
from __future__ import annotations
import os
import tempfile
import threading
import time

import httpx

import settings
import db
import prompts
import llm_backend
import retriever
from retriever import search

_API = "https://api.telegram.org/bot{token}/{method}"

_thread: threading.Thread | None = None
_stop = threading.Event()
_state = {"running": False, "error": None, "username": None, "started": None}
_offset = None
_pending_comment: dict = {}   # chat_id -> req_id (ждём текст комментария к ответу)


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


def send(chat_id: int, text: str, reply_markup: dict | None = None) -> None:
    """Отправить сообщение (Telegram режет на 4096 символов). reply_markup (если
    задан) прикрепляется к ПОСЛЕДНЕМУ фрагменту (инлайн-клавиатура оценки)."""
    if not text:
        return
    chunks = [text[i:i + 4096] for i in range(0, len(text), 4096)] or [""]
    for idx, ch in enumerate(chunks):
        params = {"chat_id": chat_id, "text": ch, "disable_web_page_preview": True}
        if reply_markup is not None and idx == len(chunks) - 1:
            params["reply_markup"] = reply_markup
        _call("sendMessage", **params)


def _send_ok(chat_id: int, text: str) -> bool:
    """Отправить сообщение (с разбивкой на 4096) и вернуть успех доставки."""
    ok = True
    for i in range(0, len(text), 4096):
        r = _call("sendMessage", chat_id=chat_id, text=text[i:i + 4096],
                  disable_web_page_preview=True)
        ok = ok and bool(r and r.get("ok"))
    return ok


def broadcast(chat_ids, text: str) -> dict:
    """Отправить текст списку пользователей. Возвращает {ok, sent, failed, total}.
    Между сообщениями небольшая пауза — не упираться в лимиты Telegram."""
    text = (text or "").strip()
    ids = [int(c) for c in (chat_ids or [])]
    if not text:
        return {"ok": False, "sent": 0, "failed": 0, "total": 0, "msg": "пустое сообщение"}
    if not _token():
        return {"ok": False, "sent": 0, "failed": 0, "total": 0, "msg": "не задан токен бота"}
    sent = failed = 0
    aid = None
    try:
        import activity
        aid = activity.start("telegram", f"рассылка {len(ids)} польз.", "отправка")
    except Exception:
        aid = None
    for n, cid in enumerate(ids, 1):
        try:
            if _send_ok(cid, text):
                sent += 1
            else:
                failed += 1
        except Exception:
            failed += 1
        if aid is not None and n % 10 == 0:
            try:
                import activity
                activity.update(aid, detail=f"{n}/{len(ids)}")
            except Exception:
                pass
        time.sleep(0.04)
    if aid is not None:
        try:
            import activity
            activity.finish(aid, ok=True, stage=f"отправлено {sent}/{len(ids)}")
        except Exception:
            pass
    return {"ok": True, "sent": sent, "failed": failed, "total": len(ids)}


def _feedback_kb(rid: int) -> dict:
    """Инлайн-клавиатура оценки ответа: 👍 / 👎 / 💬 Комментарий."""
    return {"inline_keyboard": [[
        {"text": "👍", "callback_data": f"rate:{rid}:1"},
        {"text": "👎", "callback_data": f"rate:{rid}:-1"},
        {"text": "💬 Комментарий", "callback_data": f"cmt:{rid}"},
    ]]}


def _download_file(file_id: str) -> str | None:
    """Скачать файл Telegram по file_id во временный файл; вернуть путь (или None)."""
    token = _token()
    if not token or not file_id:
        return None
    r = _call("getFile", file_id=file_id)
    if not r or not r.get("ok"):
        return None
    fp = (r.get("result") or {}).get("file_path")
    if not fp:
        return None
    suffix = os.path.splitext(fp)[1] or ".oga"
    out = tempfile.mktemp(suffix=suffix)
    try:
        url = f"https://api.telegram.org/file/bot{token}/{fp}"
        with httpx.Client(proxy=_proxy(), timeout=120) as c:
            resp = c.get(url)
        if resp.status_code != 200:
            return None
        with open(out, "wb") as f:
            f.write(resp.content)
        return out
    except Exception as e:
        print(f"[tg] загрузка файла не удалась: {e}")
        return None


def send_voice(chat_id: int, ogg_path: str, caption: str = "") -> bool:
    """Отправить голосовое сообщение (OGG/Opus) через sendVoice (multipart)."""
    token = _token()
    if not token:
        return False
    try:
        data = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption[:1024]
        with httpx.Client(proxy=_proxy(), timeout=120) as c, open(ogg_path, "rb") as f:
            r = c.post(_API.format(token=token, method="sendVoice"),
                       data=data, files={"voice": ("voice.ogg", f, "audio/ogg")})
        return bool(r.json().get("ok"))
    except Exception as e:
        print(f"[tg] sendVoice не удался: {e}")
        return False


def send_photo(chat_id: int, path: str, caption: str = "") -> bool:
    """Отправить картинку (превью источника) через sendPhoto (multipart)."""
    token = _token()
    if not token:
        return False
    try:
        data = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption[:1024]
        fname = os.path.basename(path) or "preview.jpg"
        with httpx.Client(proxy=_proxy(), timeout=120) as c, open(path, "rb") as f:
            r = c.post(_API.format(token=token, method="sendPhoto"),
                       data=data, files={"photo": (fname, f, "image/jpeg")})
        return bool(r.json().get("ok"))
    except Exception as e:
        print(f"[tg] sendPhoto не удался: {e}")
        return False


def send_audio(chat_id: int, path: str, caption: str = "") -> bool:
    """Отправить аудиофайл-источник через sendAudio (multipart)."""
    token = _token()
    if not token:
        return False
    try:
        data = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption[:1024]
        fname = os.path.basename(path) or "audio"
        with httpx.Client(proxy=_proxy(), timeout=180) as c, open(path, "rb") as f:
            r = c.post(_API.format(token=token, method="sendAudio"),
                       data=data, files={"audio": (fname, f, "application/octet-stream")})
        return bool(r.json().get("ok"))
    except Exception as e:
        print(f"[tg] sendAudio не удался: {e}")
        return False


def _send_previews(chat_id: int, hits: list) -> None:
    """Отправить визуальные превью источников ответа — как карточки в веб-чате:
    миниатюра картинки/RAW/чертежа/PDF, кадр видео (с таймкодом), аудио-файл.
    Источники дедуплицируются; число ограничено TELEGRAM_PREVIEW_MAX."""
    if not hits:
        return
    try:
        import media
    except Exception:
        return
    try:
        maxn = int(settings.get("TELEGRAM_PREVIEW_MAX") or 4)
    except Exception:
        maxn = 4
    seen: set = set()
    sent = 0
    for h in hits:
        if sent >= maxn:
            break
        src = h.get("source") if isinstance(h, dict) else None
        if not src or src in seen:
            continue
        seen.add(src)
        try:
            k = media.kind_of(src)
            if not media.available(src):
                continue
            page = h.get("page")
            t_start = h.get("t_start")
            cap = src + (f", с.{page}" if page else "")
            if k in ("image", "raw", "cad", "pdf"):
                thumb = media.thumbnail(src, page if k == "pdf" else None)
                if thumb and send_photo(chat_id, str(thumb), cap):
                    sent += 1
            elif k == "video":
                ts = t_start if isinstance(t_start, (int, float)) else 1.0
                thumb = media.thumbnail(src, ts)
                if thumb:
                    vc = cap + (f" · {int(ts // 60):02d}:{int(ts % 60):02d}"
                                if isinstance(t_start, (int, float)) else "")
                    if send_photo(chat_id, str(thumb), "🎬 " + vc):
                        sent += 1
            elif k == "audio":
                p = media.materialize(src)
                if p and send_audio(chat_id, str(p), "🔊 " + cap):
                    sent += 1
        except Exception as e:
            print(f"[tg] превью источника не удалось ({src}): {e}")


def notify_approved(chat_id: int) -> None:
    send(chat_id, "✅ Доступ подтверждён. Задайте вопрос по документам компании.")


def notify_blocked(chat_id: int) -> None:
    send(chat_id, "⛔ Доступ к боту закрыт администратором.")


TRAIN_INSTRUCTIONS = (
    "🎓 Вам открыт доступ к обучению бота — теперь вы можете пополнять базу знаний компании.\n\n"
    "КАК РАБОТАЕТ СИСТЕМА\n"
    "Я — корпоративный ассистент. Отвечаю на вопросы СТРОГО по внутренним документам компании "
    "(прайс-листы, регламенты, презентации, инструкции, договоры и т.п.) и привожу ссылки на "
    "источники. Если ответа в документах нет — честно об этом скажу и не буду выдумывать.\n"
    "Документы хранятся как векторный индекс: текст из файлов разбивается на смысловые фрагменты, "
    "по которым идёт поиск. Чем полнее и аккуратнее документы — тем точнее ответы.\n\n"
    "КАК ОБУЧАТЬ БОТА (добавлять документы)\n"
    "1) Отправьте команду /train — включится режим обучения.\n"
    "2) Пришлите документы по одному в сообщении: PDF, Word (DOCX/DOC), Excel (XLSX/CSV), "
    "PowerPoint, изображения и сканы, архивы и др. Можно фото документа — текст распознается "
    "автоматически (OCR).\n"
    "3) Я скачаю файл, распознаю его и добавлю в базу знаний, присылая статус: "
    "📥 скачиваю → 🔎 распознаю → ✅ добавлено N фрагментов.\n"
    "4) Когда закончите — отправьте /ask, чтобы выйти из режима обучения и снова задавать вопросы.\n\n"
    "РЕКОМЕНДАЦИИ\n"
    "• Присылайте файлы с текстом (а не картинку-обложку) — распознавание будет точнее.\n"
    "• Большие файлы и сканы обрабатываются дольше — дождитесь статуса ✅, прежде чем слать следующий.\n"
    "• Если из файла не удалось извлечь текст, я сообщу — такой файл в базу не попадёт.\n"
    "• Добавленные документы сразу становятся доступны для ответов всем пользователям.\n"
    "• Не загружайте конфиденциальные данные, которые не должны попасть в общие ответы.\n\n"
    "КАК ЗАДАВАТЬ ВОПРОСЫ (вне режима обучения)\n"
    "Просто напишите вопрос текстом, продиктуйте голосовым сообщением или пришлите файл с вопросом "
    "в подписи — я отвечу по документам со ссылками на источники.\n\n"
    "Команды: /train — режим обучения, /ask — вопросы, /help — помощь.")


def send_train_instructions(chat_id: int) -> bool:
    """Отправить пользователю подробное описание системы и инструкцию по обучению."""
    if not _token():
        return False
    send(chat_id, TRAIN_INSTRUCTIONS)
    return True


def _answer(question: str, trace: list | None = None):
    """Синхронный RAG-ответ: поиск → контекст → LLM. Возвращает (text, sources, hits).
    `trace` (если передан) наполняется этапами конвейера — как в веб-чате.
    При включённом ANSWER_CACHE одинаковые вопросы отдаются из Redis без вызова LLM."""
    if trace is None:
        trace = []
    ckey = None
    if settings.get("ANSWER_CACHE"):
        try:
            import cache
            import hashlib
            ckey = "ans:" + hashlib.sha1("|".join([
                question, settings.get("SYSTEM_PROMPT") or "",
                settings.active_model() or "",
                str(settings.get("TEMPERATURE"))]).encode("utf-8")).hexdigest()
            c = cache.get_json(ckey, ns="index")
            if c:
                trace.append({"key": "answer_cache", "ms": 0, "info": {"hit": True}})
                hits = [{"score": c.get("top_score", 0.0)}] if c.get("answered") else []
                return c.get("text", ""), c.get("sources", []), hits
        except Exception:
            ckey = None

    # Тот же выбор движка, что и в веб-чате: LightRAG целиком / KAG / граф для
    # сводных вопросов (hybrid) — иначе вектор+реранк.
    engine = settings.get("ENGINE")
    try:
        if engine == "kag":
            import asyncio
            import kag
            res = asyncio.run(kag.answer(question, trace=trace)) or {}
            hits = res.get("hits", [])
            text = res.get("text", "") or "В доступных документах нет точного ответа на этот вопрос."
            sources = [{"source": h["source"], "page": h.get("page"),
                        "score": round(h.get("score", 0), 3)} for h in hits]
            return text, sources, hits
        use_graph = (engine == "lightrag")
        if not use_graph and settings.get("GRAPH_RAG"):
            try:
                import graph_rag
                use_graph = graph_rag.is_global(question)
            except Exception:
                use_graph = False
        if use_graph:
            import asyncio
            import graph_rag
            t = time.time()
            text = asyncio.run(graph_rag.answer(question)) or ""
            trace.append({"key": "engine", "ms": int((time.time() - t) * 1000),
                          "info": {"engine": "граф знаний (LightRAG)",
                                   "mode": settings.current_mode()}})
            if text.strip():
                return text, [{"source": "граф знаний (LightRAG)", "page": None}], \
                    [{"score": 1.0}]
    except Exception as e:
        print(f"[tg] движок {engine} недоступен, фолбэк на вектор: {e}")

    # Векторный путь + расширенный фолбэк (как в /chat)
    hits = search(question, trace=trace)
    if not hits and settings.get("NO_ANSWER_FALLBACK"):
        try:
            hits = retriever.no_answer_fallback(question, trace=trace) or []
        except Exception as e:
            print(f"[tg] фолбэк-поиск не удался: {e}")
    # прайс-папка: на ценовых вопросах подмешиваем контекст из папки прайсов
    try:
        import price_folder
        if price_folder.enabled() and price_folder.is_price_query(question):
            ph = price_folder.hits(question)
            if ph:
                trace.append({"key": "price", "ms": 0, "info": {"found": len(ph)}})
                seen = set((h.get("source"), (h.get("text") or "")[:60]) for h in ph)
                hits = list(ph) + [h for h in hits
                                   if (h.get("source"), (h.get("text") or "")[:60]) not in seen]
    except Exception as e:
        print(f"[tg] прайс-папка: {e}")
    # внешние API-хуки
    try:
        import api_tools
        frag = api_tools.augment_hit(question)
        if frag:
            trace.append({"key": "api", "ms": 0, "info": {"source": frag["source"]}})
            hits = [frag] + hits
    except Exception as e:
        print(f"[tg] api-хук: {e}")
    if not hits:
        return "В доступных документах нет точного ответа на этот вопрос.", [], []
    context = prompts.build_context(hits)
    trace.append({"key": "context", "ms": 0,
                  "info": {"chunks": len(hits), "chars": len(context)}})
    messages = [{"role": "system", "content": settings.get("SYSTEM_PROMPT")},
                {"role": "user", "content": prompts.build_user_message(question, context)}]
    t = time.time()
    text = llm_backend.chat(messages, temperature=settings.get("TEMPERATURE"),
                            model=settings.active_model(),
                            kind="telegram", label=question)
    trace.append({"key": "generate", "ms": int((time.time() - t) * 1000),
                  "info": {"model": settings.active_model(),
                           "backend": settings.get("LLM_BACKEND"),
                           "temperature": settings.get("TEMPERATURE"),
                           "chars": len(text or "")}})
    sources = [{"source": h["source"], "page": h.get("page"),
                "score": round(h.get("score", 0), 3)} for h in hits]
    if ckey:
        try:
            import cache
            cache.set_json(ckey, 86400, {"text": text, "sources": sources,
                                         "top_score": hits[0]["score"],
                                         "answered": True}, ns="index")
        except Exception:
            pass
    return text, sources, hits


# Этапы конвейера → (иконка, подпись) — зеркало STAGE_META веб-чата.
_STAGE_META = {
    "cache": ("⚡", "Кэш поиска"), "answer_cache": ("⚡", "Кэш ответа"),
    "embed": ("🧮", "Эмбеддинг запроса"), "filter": ("🧭", "Фильтр"),
    "dense": ("🗄", "Векторный поиск (Qdrant)"), "bm25": ("🔤", "Лексика BM25"),
    "rerank": ("🎯", "Реранк (cross-encoder)"),
    "price": ("💲", "Прайс-папка (без индексации)"),
    "api": ("🔌", "Внешний API"),
    "fb_lexical": ("🔎", "Доп. поиск (лексический)"),
    "fb_deep": ("🕵️", "Глубокий поиск (по каталогу)"),
    "attach": ("📎", "Разбор документа"), "context": ("📋", "Сборка контекста"),
    "generate": ("🧠", "Генерация (LLM)"), "engine": ("🧩", "Движок"),
    "kag_decompose": ("🪓", "KAG: декомпозиция вопроса"),
    "kag_retrieve": ("🔗", "KAG: мультихоп-поиск"),
    "kag_graph": ("🕸", "KAG: знания графа"), "kag_generate": ("🧠", "KAG: генерация"),
}


def _stage_params(key: str, info: dict) -> str:
    info = info or {}
    p: list[str] = []
    if key == "embed":
        if info.get("model"):
            p.append(str(info["model"]))
        if info.get("device"):
            p.append(str(info["device"]))
        if info.get("synonyms"):
            p.append("+синонимы")
    elif key == "filter":
        p.append("тип: " + str(info.get("type", "нет")))
    elif key == "dense":
        p.append("top-k " + str(info.get("top_k", "—")))
        p.append("кандидатов " + str(info.get("candidates", "—")))
        if info.get("fallback"):
            p.append("фолбэк без фильтра")
    elif key == "bm25":
        p.append("кандидатов " + str(info.get("candidates", "—")))
    elif key == "rerank":
        if info.get("model"):
            p.append(str(info["model"]))
        if info.get("min_score") is not None:
            p.append("порог " + str(info["min_score"]))
        p.append("оставлено " + str(info.get("kept", "—")) + "/" + str(info.get("candidates", "—")))
    elif key == "attach":
        if info.get("file"):
            p.append(str(info["file"]))
        p.append("фрагментов " + str(info.get("fragments", "—")))
    elif key == "price":
        p.append("фрагментов " + str(info.get("found", "—")))
    elif key == "api":
        if info.get("source"):
            p.append(str(info["source"]))
    elif key in ("fb_lexical", "fb_deep"):
        p.append("найдено " + str(info.get("found", "—")))
        if key == "fb_deep" and info.get("files"):
            p.append("файлы: " + "; ".join(info["files"]))
    elif key == "context":
        p.append("фрагментов " + str(info.get("chunks", "—")))
        if info.get("chars") is not None:
            p.append(str(info["chars"]) + " симв.")
    elif key == "generate":
        if info.get("model"):
            p.append(str(info["model"]))
        if info.get("backend"):
            p.append(str(info["backend"]))
        if info.get("temperature") is not None:
            p.append("t=" + str(info["temperature"]))
        if info.get("chars") is not None:
            p.append(str(info["chars"]) + " симв.")
    elif key == "engine":
        if info.get("engine"):
            p.append(str(info["engine"]))
        if info.get("mode"):
            p.append("режим " + str(info["mode"]))
    elif key == "kag_decompose":
        p.append("под-вопросов " + str(info.get("hops", "—")))
    elif key == "kag_retrieve":
        p.append("шагов " + str(info.get("hops", "—")))
        p.append("фрагментов " + str(info.get("chunks", "—")))
    elif key == "kag_graph":
        p.append(("+" + str(info["chars"]) + " симв. знаний") if info.get("chars")
                 else "граф не дал знаний")
    elif key in ("cache", "answer_cache"):
        p.append("попадание" if info.get("hit") else "мимо")
    return " · ".join(p)


def _format_pipeline(trace: list) -> str:
    """Текстовая «структура формирования ответа» по этапам (как конвейер в чате)."""
    if not trace:
        return ""
    lines = ["⚙️ Структура формирования ответа:"]
    for s in trace:
        key = s.get("key", "")
        icon, label = _STAGE_META.get(key, ("•", key))
        params = _stage_params(key, s.get("info"))
        ms = s.get("ms")
        tail = f" · {ms} мс" if (ms is not None and ms > 0) else ""
        lines.append(f"• {icon} {label}" + (f" — {params}" if params else "") + tail)
    return "\n".join(lines)


def _answer_attachment(path: str, name: str, question: str, trace: list | None = None):
    """Ответ по приложенному файлу: извлечь текст, при наличии вопроса — ответить по
    содержимому файла (реранк), иначе вернуть распознанный фрагмент. Возвращает
    (text, sources, hits) как и _answer; `trace` наполняется этапами."""
    if trace is None:
        trace = []
    import loaders
    from ingest import chunk_text
    items = []
    t = time.time()
    try:
        for part in loaders.load_file(__import__("pathlib").Path(path)):
            for ch in chunk_text(part.get("text") or "", settings.get("CHUNK_SIZE"),
                                 settings.get("CHUNK_OVERLAP")):
                if ch.strip():
                    items.append({"text": ch, "source": name, "page": part.get("page")})
    except Exception as e:
        print(f"[tg] разбор файла {name}: {e}")
    trace.append({"key": "attach", "ms": int((time.time() - t) * 1000),
                  "info": {"file": name, "fragments": len(items)}})
    if not items:
        return "Не удалось извлечь текст из файла (пустой или неподдерживаемый формат).", [], []
    q = (question or "").strip()
    if not q:
        full = " ".join(i["text"] for i in items).strip()[:2500]
        return f"📄 Распознанный текст файла «{name}»:\n\n{full}\n\n" \
               f"Задайте вопрос по этому файлу в подписи к нему.", [], items[:1]
    t = time.time()
    hits = retriever.rerank_texts(q, items)
    trace.append({"key": "rerank", "ms": int((time.time() - t) * 1000),
                  "info": {"model": settings.get("RERANK_MODEL"),
                           "kept": len(hits), "candidates": len(items)}})
    if not hits:
        return "Не удалось найти ответ в приложенном файле.", [], []
    context = prompts.build_context(hits)
    trace.append({"key": "context", "ms": 0,
                  "info": {"chunks": len(hits), "chars": len(context)}})
    messages = [{"role": "system", "content": settings.get("SYSTEM_PROMPT")},
                {"role": "user", "content": prompts.build_user_message(q, context)}]
    t = time.time()
    text = llm_backend.chat(messages, temperature=settings.get("TEMPERATURE"),
                            model=settings.active_model(),
                            kind="telegram", label=q)
    trace.append({"key": "generate", "ms": int((time.time() - t) * 1000),
                  "info": {"model": settings.active_model(),
                           "backend": settings.get("LLM_BACKEND"),
                           "temperature": settings.get("TEMPERATURE"),
                           "chars": len(text or "")}})
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
    can_train = bool(user.get("can_train"))

    # ждём комментарий к ответу (после нажатия «💬 Комментарий»)?
    if text and not text.startswith("/") and chat_id in _pending_comment:
        rid = _pending_comment.pop(chat_id, None)
        if rid:
            db.tg_set_comment(rid, text)
            send(chat_id, "✅ Спасибо, комментарий сохранён.")
            return

    # команды переключения режима обучения
    if text:
        low = text.strip().lower()
        if low in ("/train", "/learn", "/обучение"):
            if not can_train:
                send(chat_id, "🚫 У вас нет разрешения на обучение бота. Обратитесь к администратору.")
                return
            db.tg_set_mode(chat_id, "train")
            send(chat_id, "🎓 Режим обучения включён. Пришлите документы (PDF, DOCX, XLSX, фото и др.) — "
                          "я распознаю их и добавлю в базу знаний. Команда /ask — выйти из режима обучения.")
            return
        if low in ("/ask", "/stop", "/вопросы"):
            db.tg_set_mode(chat_id, "ask")
            send(chat_id, "✅ Режим обучения выключен. Задавайте вопросы по документам компании.")
            return

    # режим обучения: входящие файлы добавляются в базу знаний
    if can_train and user.get("mode") == "train":
        doc = msg.get("document")
        photo = msg.get("photo")
        if doc or photo:
            if doc:
                fid = doc.get("file_id")
                fname = doc.get("file_name") or "файл"
            else:
                big = (photo or [])[-1] if photo else {}
                fid = big.get("file_id")
                fname = "photo.jpg"
            send(chat_id, f"📥 Скачиваю «{fname}»…")
            fp = _download_file(fid)
            if not fp:
                send(chat_id, "Не удалось скачать файл. Попробуйте ещё раз.")
                return
            send(chat_id, "🔎 Распознаю и добавляю в базу знаний…")
            res = {"name": fname, "chunks": 0}
            try:
                import tg_train
                res = tg_train.save_and_index(chat_id, fp, fname)
            except Exception as e:
                print(f"[tg] обучение {fname}: {e}")
            finally:
                try:
                    os.remove(fp)
                except Exception:
                    pass
            if res.get("chunks"):
                send(chat_id, f"✅ «{res['name']}»: добавлено {res['chunks']} фрагментов в базу знаний. "
                              "Пришлите ещё документы или /ask — выйти из обучения.")
            else:
                send(chat_id, f"⚠ Из «{fname}» не удалось извлечь текст (пустой/неподдерживаемый "
                              "формат). В базу не добавлено.")
            return
        # в режиме обучения текст/голос — подсказка, что нужны документы
        send(chat_id, "🎓 Вы в режиме обучения. Пришлите документ — я добавлю его в базу знаний. "
                      "Команда /ask — выйти и задавать вопросы.")
        return

    voice_in = False
    attach_path = attach_name = None
    caption = (msg.get("caption") or "").strip()
    if not text:
        vmsg = msg.get("voice") or msg.get("audio") or msg.get("video_note")
        doc = msg.get("document")
        photo = msg.get("photo")
        # голосовое сообщение → распознаём через Whisper
        if vmsg:
            if not settings.get("TELEGRAM_VOICE_IN"):
                send(chat_id, "Распознавание голосовых сообщений выключено. Напишите вопрос текстом.")
                return
            send(chat_id, "🎙 Распознаю голосовое сообщение…")
            fp = _download_file(vmsg.get("file_id"))
            if fp:
                try:
                    import loaders
                    text = (loaders.transcribe_audio(fp) or "").strip()
                except Exception as e:
                    print(f"[tg] распознавание не удалось: {e}")
                finally:
                    try:
                        os.remove(fp)
                    except Exception:
                        pass
            if not text:
                send(chat_id, "Не удалось распознать голосовое сообщение. "
                              "Попробуйте ещё раз или напишите текстом.")
                return
            voice_in = True
            send(chat_id, f"🔎 Вопрос: {text}")
        # приложенный файл (документ или фото) → распознаём и отвечаем по нему
        elif doc or photo:
            if not settings.get("TELEGRAM_FILES"):
                send(chat_id, "Распознавание приложенных файлов выключено.")
                return
            if doc:
                fid = doc.get("file_id")
                attach_name = doc.get("file_name") or "файл"
            else:  # фото — берём самый крупный размер
                big = (photo or [])[-1] if photo else {}
                fid = big.get("file_id")
                attach_name = "photo.jpg"
            send(chat_id, f"📎 Распознаю файл «{attach_name}»…")
            attach_path = _download_file(fid)
            if not attach_path:
                send(chat_id, "Не удалось скачать файл. Попробуйте ещё раз.")
                return
            text = caption  # подпись к файлу — это вопрос

    if not text and not attach_path:
        return
    if text and (text.startswith("/start") or text.startswith("/help")):
        msg_help = ("Напишите вопрос текстом, продиктуйте голосом или пришлите файл с "
                    "вопросом в подписи — я отвечу по документам компании.")
        if can_train:
            msg_help += ("\n\n🎓 Вам разрешено обучение бота: команда /train — войти в режим "
                         "обучения и присылать документы для добавления в базу знаний; /ask — выйти.")
        send(chat_id, msg_help)
        return
    if text and text.startswith("/") and not attach_path:
        send(chat_id, "Неизвестная команда. Просто задайте вопрос текстом.")
        return

    t0 = time.time()
    trace: list = []
    _aid = None
    try:
        import activity
        _label = (attach_name or text or "").strip().replace("\n", " ")[:80]
        _aid = activity.start("telegram", _label,
                              "разбор файла" if attach_path else "обработка запроса")
    except Exception:
        _aid = None
    try:
        if attach_path:
            try:
                ans, sources, hits = _answer_attachment(attach_path, attach_name, text, trace)
            finally:
                try:
                    os.remove(attach_path)
                except Exception:
                    pass
        else:
            ans, sources, hits = _answer(text, trace)
        answered = bool(hits)
    except Exception as e:
        print(f"  ! telegram answer error: {e}")
        send(chat_id, "Произошла ошибка при обработке запроса. Попробуйте позже.")
        db.tg_log_request(chat_id, username, text, "", 0, 0.0,
                          int((time.time() - t0) * 1000), False, [])
        if _aid is not None:
            try:
                import activity
                activity.finish(_aid, ok=False, stage="ошибка")
            except Exception:
                pass
        return

    latency = int((time.time() - t0) * 1000)
    show_answer = settings.get("TELEGRAM_SHOW_ANSWER")
    parts = []
    if show_answer and ans:
        parts.append(ans)
    if sources:
        srctxt = "; ".join(s["source"] + (f", с.{s['page']}" if s.get("page") else "")
                           for s in sources[:6])
        parts.append(f"📎 Источники: {srctxt}")
    # структура формирования ответа (как конвейер в веб-чате), если включено
    if settings.get("TELEGRAM_PIPELINE"):
        pipe = _format_pipeline(trace)
        if pipe:
            parts.append(pipe)
    out = "\n\n".join(parts) if parts else "✓ Запрос обработан."
    # журналируем заранее, чтобы привязать кнопки оценки к конкретному ответу
    rid = db.tg_log_request(chat_id, username, text or (attach_name or ""), ans,
                            len(sources), (hits[0].get("score", 0.0) if hits else 0.0),
                            latency, answered, sources)
    kb = _feedback_kb(rid) if (rid and settings.get("TELEGRAM_FEEDBACK")) else None
    send(chat_id, out, reply_markup=kb)
    # визуальные превью источников (картинки/чертежи/кадры видео/аудио) — как в веб-чате
    if answered and settings.get("TELEGRAM_PREVIEWS"):
        try:
            _send_previews(chat_id, hits)
        except Exception as e:
            print(f"  ! telegram previews error: {e}")
    # голосовой ответ на голосовой запрос (если включено, доступен TTS и вывод ответа не отключён)
    if show_answer and voice_in and settings.get("TELEGRAM_VOICE_OUT") and ans:
        try:
            import tts
            ogg = tempfile.mktemp(suffix=".ogg")
            if tts.synthesize(ans, ogg):
                send_voice(chat_id, ogg)
            try:
                os.remove(ogg)
            except Exception:
                pass
        except Exception as e:
            print(f"[tg] голосовой ответ не сформирован: {e}")
    if _aid is not None:
        try:
            import activity
            activity.finish(_aid, ok=answered, stage="ответ отправлен")
        except Exception:
            pass


def _handle_callback(cq: dict) -> None:
    """Обработка нажатий инлайн-кнопок оценки/комментария под ответом."""
    cq_id = cq.get("id")
    data = cq.get("data") or ""
    msg = cq.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    message_id = msg.get("message_id")

    def ack(text=""):
        _call("answerCallbackQuery", callback_query_id=cq_id, text=text)

    try:
        if data.startswith("rate:"):
            _, srid, sval = data.split(":")
            rid, v = int(srid), int(sval)
            db.tg_set_rating(rid, v)
            ack("Спасибо за оценку!")
            chosen = "👍" if v > 0 else "👎"
            _call("editMessageReplyMarkup", chat_id=chat_id, message_id=message_id,
                  reply_markup={"inline_keyboard": [[
                      {"text": "✅ " + chosen, "callback_data": f"rate:{rid}:{v}"},
                      {"text": "💬 Комментарий", "callback_data": f"cmt:{rid}"}]]})
        elif data.startswith("cmt:"):
            _, srid = data.split(":")
            if chat_id is not None:
                _pending_comment[chat_id] = int(srid)
            ack("Напишите комментарий сообщением")
            send(chat_id, "💬 Напишите комментарий к ответу одним сообщением.")
        else:
            ack()
    except Exception as e:
        print(f"[tg] callback error: {e}")
        ack()


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
            cq = upd.get("callback_query")
            if cq:
                try:
                    _handle_callback(cq)
                except Exception as e:
                    print(f"  ! telegram callback error: {e}")
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
    try:
        import tts
        tts_info = tts.available()
    except Exception as e:
        tts_info = {"ok": False, "engine": None, "candidates": [], "ffmpeg": False,
                    "error": str(e)}
    return {"running": bool(_state.get("running")),
            "username": _state.get("username"),
            "token_set": bool(_token()),
            "auto_approve": bool(settings.get("TELEGRAM_AUTO_APPROVE")),
            "proxy": _proxy() or "",
            "error": _state.get("error"),
            "started": _state.get("started"),
            "voice_in": bool(settings.get("TELEGRAM_VOICE_IN")),
            "voice_out": bool(settings.get("TELEGRAM_VOICE_OUT")),
            "files": bool(settings.get("TELEGRAM_FILES")),
            "pipeline": bool(settings.get("TELEGRAM_PIPELINE")),
            "show_answer": bool(settings.get("TELEGRAM_SHOW_ANSWER")),
            "feedback": bool(settings.get("TELEGRAM_FEEDBACK")),
            "previews": bool(settings.get("TELEGRAM_PREVIEWS")),
            "tts_engine": settings.get("TTS_ENGINE"),
            "tts_voice": settings.get("TTS_VOICE") or "",
            "tts": tts_info,
            "whisper_backend": settings.get("WHISPER_BACKEND"),
            **db.tg_counts()}
