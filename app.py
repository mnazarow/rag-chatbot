"""FastAPI: чат (стриминг) + API для дашборда, журнала, аналитики и админки.

Запуск:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import tempfile

from fastapi import FastAPI, Header, HTTPException, Body, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

import activity
import dns_override
from pydantic import BaseModel
from qdrant_client import QdrantClient

import config
import prompts
import settings

# В Docker-контейнере адрес Qdrant и коллекция задаются окружением (compose) и должны
# иметь приоритет над сохранёнными настройками — иначе устаревший runtime_config.json
# мог бы указывать на localhost, и приложение «не видело» бы Qdrant. Делаем это ДО
# импорта retriever/admin_ops (они создают клиент Qdrant при импорте). Только в контейнере.
if os.path.exists("/.dockerenv"):
    for _k in ("QDRANT_URL", "QDRANT_COLLECTION"):
        _v = os.environ.get(_k)
        if _v and settings.get(_k) != _v:
            try:
                settings.update({_k: _v})
                print(f"[docker] {_k} зафиксирован из окружения: {_v}")
            except Exception as _e:
                print(f"[docker] не удалось зафиксировать {_k}: {_e}")

import db

# Статические DNS-записи активируем как можно раньше — до создания клиентов Qdrant
# и любых сетевых вызовов в импортируемых ниже модулях.
try:
    dns_override.install()
except Exception as _e:
    print(f"[dns] ранняя инициализация не удалась: {_e}")

import llm_backend
import admin_ops
import graph_rag
import loaders
import retriever
import remote
import media
import telegram_bot
import calibrate
import kag
from ingest import chunk_text, SUPPORTED
from retriever import search, infer_category

app = FastAPI(title="Корпоративный RAG-чатбот")
_qdrant = QdrantClient(url=settings.get("QDRANT_URL"))


@app.on_event("startup")
def _start_dns():
    """Включить статические DNS-записи (имя→IP) до сетевых операций бота/монитора."""
    try:
        r = dns_override.install()
        print(f"[dns] статических записей: {r.get('count', 0)}")
    except Exception as e:
        print(f"[dns] не инициализированы: {e}")


@app.on_event("startup")
def _start_sip():
    """Поднять голосовой мост к АТС (AudioSocket), если включён."""
    try:
        import sip_bridge
        if settings.get("SIP_ENABLED"):
            r = sip_bridge.start()
            print(f"[sip] {r.get('msg')}")
    except Exception as e:
        print(f"[sip] не запущен: {e}")


@app.on_event("startup")
def _start_telegram():
    """Поднять Телеграм-бота, если задан токен (фоновый поток long-polling)."""
    try:
        r = telegram_bot.start()
        if r.get("ok"):
            print(f"[telegram] {r.get('msg')}")
    except Exception as e:
        print(f"[telegram] не запущен: {e}")


@app.on_event("startup")
def _start_monitor():
    """Фоновый сбор метрик загрузки хоста (история час/день/неделя/месяц/год)."""
    try:
        import monitor
        r = monitor.start()
        if r.get("ok"):
            print(f"[monitor] {r.get('msg')}")
    except Exception as e:
        print(f"[monitor] не запущен: {e}")


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []
    filters: dict | None = None
    debug: bool = False
    session_id: str = ""


def _debug_params() -> dict:
    return {k: settings.get(k) for k in
            ("MIN_SCORE", "TOP_K_RETRIEVE", "TOP_K_RERANK", "TEMPERATURE",
             "AUTO_FILTER", "SMART_FILTER")}


def _debug_chunks(hits: list) -> list:
    out = []
    for h in hits:
        out.append({"source": h.get("source"), "page": h.get("page"),
                    "score": round(h.get("score", 0), 3),
                    "snippet": (h.get("text", "") or "")[:240]})
    return out


def _check_admin(token: str | None):
    current = settings.get("ADMIN_TOKEN")
    if current and token != current:
        raise HTTPException(status_code=401, detail="Неверный токен администратора")


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.get("LLM_MODEL"),
            "backend": settings.get("LLM_BACKEND")}


# ============================ ЧАТ ============================
def _stg(key: str, status: str = "done", info: dict | None = None, ms: int = 0) -> str:
    """NDJSON-событие этапа конвейера (для анимации в интерфейсе)."""
    return json.dumps({"type": "stage", "key": key, "status": status,
                       "ms": ms, "info": info or {}}, ensure_ascii=False) + "\n"


def _augment_api(question: str, hits: list, trace: list | None = None) -> list:
    """Подмешать в начало контекста данные внешнего API-хука, если он сработал."""
    try:
        import api_tools
        frag = api_tools.augment_hit(question)
        if not frag:
            return hits
        if trace is not None:
            trace.append({"key": "api", "ms": 0, "info": {"source": frag["source"]}})
        return [frag] + hits
    except Exception as e:
        print(f"[api] augment: {e}")
        return hits


def _augment_price(question: str, hits: list, trace: list | None = None) -> list:
    """На ценовых вопросах добавить в начало контекста фрагменты из прайс-папки
    (без индексации). Дедуп по источнику+началу текста."""
    try:
        import price_folder
        if not (price_folder.enabled() and price_folder.is_price_query(question)):
            return hits
        ph = price_folder.hits(question)
        if not ph:
            return hits
        if trace is not None:
            trace.append({"key": "price", "ms": 0, "info": {"found": len(ph)}})
        seen = set((h.get("source"), (h.get("text") or "")[:60]) for h in ph)
        merged = list(ph)
        for h in hits:
            k = (h.get("source"), (h.get("text") or "")[:60])
            if k not in seen:
                merged.append(h)
        return merged
    except Exception as e:
        print(f"[price] augment: {e}")
        return hits


def _ndjson(gen, aid: int | None = None):
    """StreamingResponse с гарантированным завершением активности после отдачи."""
    bg = BackgroundTask(activity.finish, aid) if aid is not None else None
    return StreamingResponse(gen, media_type="application/x-ndjson", background=bg)


@app.post("/chat")
async def chat(req: ChatRequest):
    t0 = time.time()
    _preview = (req.question or "").strip().replace("\n", " ")[:80]
    aid = activity.start("chat", _preview, "поиск")

    # Движок ответов: LightRAG целиком, либо граф только для сводных вопросов (hybrid)
    engine = settings.get("ENGINE")

    # KAG — знание-усиленная генерация (декомпозиция → мультихоп → знания графа → ответ)
    if engine == "kag" and req.filters is None:
        try:
            ktrace = []
            kres = await kag.answer(req.question, history=req.history, trace=ktrace)
            ktext = kres["text"]
            khits = kres.get("hits", [])
            ksources = [media.cite(h["source"], page=h.get("page"),
                                   t_start=h.get("t_start"), t_end=h.get("t_end"),
                                   score=round(h.get("score", 0.0), 3),
                                   category=h.get("doc_category")) for h in khits]

            async def kstream():
                for s in ktrace:
                    yield _stg(s["key"], "done", s.get("info"), s.get("ms", 0))
                yield _stg("engine", "done", {
                    "engine": "KAG (знание-усиленная генерация)",
                    "hops": len(kres.get("sub", [])), "graph": kres.get("graph"),
                    "model": settings.active_model()})
                for i in range(0, len(ktext), 40):
                    yield json.dumps({"type": "answer", "text": ktext[i:i + 40]},
                                     ensure_ascii=False) + "\n"
                yield json.dumps({"type": "sources", "items": ksources},
                                 ensure_ascii=False) + "\n"
                lat = int((time.time() - t0) * 1000)
                top = round(khits[0].get("score", 0.0), 3) if khits else 0.0
                if req.debug:
                    yield json.dumps({"type": "debug", "info": {
                        "engine": "KAG", "sub_questions": kres.get("sub", []),
                        "graph_used": kres.get("graph"),
                        "mode": settings.current_mode(), "model": settings.active_model(),
                        "backend": settings.get("LLM_BACKEND"),
                        "timings": {"retrieve_ms": 0, "gen_ms": lat, "total_ms": lat},
                        "params": _debug_params(), "chunks": []}}, ensure_ascii=False) + "\n"
                rid = db.log_request(req.question, "kag", len(khits), top, lat,
                                     len(ktext), kres.get("answered", True), ksources,
                                     retrieve_ms=0, gen_ms=lat, session_id=req.session_id,
                                     answer=ktext)
                yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

            activity.update(aid, stage="KAG: генерация ответа")
            return _ndjson(kstream(), aid)
        except Exception as e:
            print(f"KAG недоступен, фолбэк на вектор: {e}")

    use_lightrag_all = engine == "lightrag"
    use_graph_global = (settings.get("GRAPH_RAG") and req.filters is None
                        and graph_rag.is_global(req.question))
    if use_lightrag_all or use_graph_global:
        try:
            text = await graph_rag.answer(req.question)
            cat = "lightrag" if use_lightrag_all else "graph"

            async def gstream():
                yield _stg("engine", "done", {
                    "engine": "LightRAG (граф)" if use_lightrag_all
                    else "граф (hybrid, сводный вопрос)",
                    "mode": settings.get("GRAPH_MODE"), "model": settings.active_model()})
                for i in range(0, len(text), 40):
                    yield json.dumps({"type": "answer", "text": text[i:i + 40]},
                                     ensure_ascii=False) + "\n"
                yield json.dumps({"type": "sources",
                                  "items": [{"source": "граф знаний (LightRAG)", "page": None}]},
                                 ensure_ascii=False) + "\n"
                lat = int((time.time() - t0) * 1000)
                if req.debug:
                    yield json.dumps({"type": "debug", "info": {
                        "engine": "LightRAG (граф)" if use_lightrag_all else "граф (hybrid, сводный вопрос)",
                        "mode": settings.current_mode(), "model": settings.active_model(),
                        "backend": settings.get("LLM_BACKEND"),
                        "timings": {"retrieve_ms": 0, "gen_ms": lat, "total_ms": lat},
                        "params": _debug_params(), "chunks": []}}, ensure_ascii=False) + "\n"
                rid = db.log_request(req.question, cat, 1, 1.0, lat, len(text), True, [],
                                     retrieve_ms=0, gen_ms=lat,
                                     session_id=req.session_id, answer=text)
                yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

            activity.update(aid, stage="генерация ответа (граф)")
            return _ndjson(gstream(), aid)
        except Exception as e:
            print(f"LightRAG недоступен, фолбэк на вектор: {e}")

    # Кэш готовых ответов (опционально, Redis). Только для одиночных вопросов без
    # истории диалога; ключ учитывает фильтры, промпт, модель и температуру; кэш
    # сбрасывается при переиндексации (пространство index).
    acache_key = None
    if settings.get("ANSWER_CACHE") and not req.history:
        acache_key = "ans:" + hashlib.sha1("|".join(str(x) for x in [
            req.question, req.filters, settings.get("SYSTEM_PROMPT"),
            settings.active_model(), settings.get("TEMPERATURE")]).encode("utf-8")).hexdigest()
        try:
            import cache
            cached = cache.get_json(acache_key, ns="index")
        except Exception:
            cached = None
        if cached:
            async def cached_stream():
                yield _stg("answer_cache", "done", {"hit": True})
                txt = cached.get("answer", "")
                for i in range(0, len(txt), 40):
                    yield json.dumps({"type": "answer", "text": txt[i:i + 40]},
                                     ensure_ascii=False) + "\n"
                yield json.dumps({"type": "sources", "items": cached.get("sources", [])},
                                 ensure_ascii=False) + "\n"
                lat = int((time.time() - t0) * 1000)
                if req.debug:
                    yield json.dumps({"type": "debug", "info": {
                        "engine": "векторный (ответ из кэша Redis)", "cached": True,
                        "mode": settings.current_mode(), "model": settings.active_model(),
                        "backend": settings.get("LLM_BACKEND"),
                        "timings": {"retrieve_ms": 0, "gen_ms": 0, "total_ms": lat},
                        "params": _debug_params(), "chunks": []}}, ensure_ascii=False) + "\n"
                rid = db.log_request(req.question, cached.get("category"),
                                     cached.get("n_hits", 0), cached.get("top_score", 0.0),
                                     lat, len(txt), True, cached.get("sources", []),
                                     retrieve_ms=0, gen_ms=0, session_id=req.session_id,
                                     answer=txt)
                yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

            activity.update(aid, stage="ответ из кэша")
            return _ndjson(cached_stream(), aid)

    t_ret = time.time()
    trace = []
    hits = search(req.question, filters=req.filters, trace=trace)
    # расширенный поиск, если ничего не нашлось (опционально): лексический → глубокий
    if not hits and settings.get("NO_ANSWER_FALLBACK"):
        try:
            hits = retriever.no_answer_fallback(req.question, trace=trace) or []
        except Exception as e:
            print(f"  ! фолбэк-поиск не удался: {e}")
    # прайс-папка: на «ценовых» вопросах подмешиваем контекст из папки прайсов
    hits = _augment_price(req.question, hits, trace)
    # внешние API-хуки: подмешиваем данные стороннего сервиса, если хук сработал
    hits = _augment_api(req.question, hits, trace)
    retrieve_ms = int((time.time() - t_ret) * 1000)
    category = (req.filters or {}).get("doc_category") or infer_category(req.question)

    if not hits:
        msg = "В доступных документах нет точного ответа на этот вопрос."
        rid = db.log_request(req.question, category, 0, 0.0,
                             int((time.time() - t0) * 1000), len(msg), False, [],
                             retrieve_ms=retrieve_ms, gen_ms=0,
                             session_id=req.session_id, answer=msg)

        async def empty():
            for s in trace:
                yield _stg(s["key"], "done", s.get("info"), s.get("ms", 0))
            yield _stg("context", "done", {"chunks": 0, "chars": 0})
            yield json.dumps({"type": "answer", "text": msg}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "sources", "items": []}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

        return _ndjson(empty(), aid)

    context = prompts.build_context(hits)
    messages = [{"role": "system", "content": settings.get("SYSTEM_PROMPT")}]
    messages += req.history[-6:]
    messages.append({"role": "user",
                     "content": prompts.build_user_message(req.question, context)})

    sources = [media.cite(h["source"], page=h.get("page"),
                          t_start=h.get("t_start"), t_end=h.get("t_end"),
                          score=round(h["score"], 3), category=h.get("doc_category"))
               for h in hits]
    activity.update(aid, stage="генерация ответа")

    async def stream():
        # анимация конвейера: этапы поиска (измерены), затем контекст и генерация
        for s in trace:
            yield _stg(s["key"], "done", s.get("info"), s.get("ms", 0))
        yield _stg("context", "done",
                   {"chunks": len(hits), "chars": len(context),
                    "retrieve_ms": retrieve_ms})
        yield _stg("generate", "start", {"backend": settings.get("LLM_BACKEND"),
                                         "model": settings.active_model(),
                                         "temperature": settings.get("TEMPERATURE")})
        acc = []
        async for tok in llm_backend.chat_stream(
                messages, temperature=settings.get("TEMPERATURE"),
                model=settings.active_model()):
            acc.append(tok)
            yield json.dumps({"type": "answer", "text": tok}, ensure_ascii=False) + "\n"
        gen_ms = max(0, int((time.time() - t0) * 1000) - retrieve_ms)
        yield _stg("generate", "done", {"chars": len("".join(acc)), "ms": gen_ms}, gen_ms)
        yield json.dumps({"type": "sources", "items": sources}, ensure_ascii=False) + "\n"
        latency = int((time.time() - t0) * 1000)
        if req.debug:
            yield json.dumps({"type": "debug", "info": {
                "engine": "векторный (поиск + реранк)", "mode": settings.current_mode(),
                "model": settings.active_model(), "backend": settings.get("LLM_BACKEND"),
                "filters": req.filters or "авто/по вопросу",
                "timings": {"retrieve_ms": retrieve_ms, "gen_ms": max(0, latency - retrieve_ms),
                            "total_ms": latency},
                "params": _debug_params(), "chunks": _debug_chunks(hits)}}, ensure_ascii=False) + "\n"
        full = "".join(acc)
        rid = db.log_request(req.question, category, len(hits), hits[0]["score"],
                             latency, len(full), True, sources,
                             retrieve_ms=retrieve_ms, gen_ms=max(0, latency - retrieve_ms),
                             session_id=req.session_id, answer=full)
        if acache_key and acc:
            try:
                import cache
                cache.set_json(acache_key, 86400, {
                    "answer": "".join(acc), "sources": sources, "category": category,
                    "n_hits": len(hits), "top_score": hits[0]["score"]}, ns="index")
            except Exception:
                pass
        yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

    return _ndjson(stream(), aid)


@app.post("/api/rate")
def api_rate(payload: dict = Body(...)):
    """Оценка ответа сотрудником: rating 1 (хорошо) / -1 (плохо) / 0 (снять)."""
    rid = payload.get("id")
    rating = int(payload.get("rating", 0))
    if rid is None or rating not in (1, -1, 0):
        return {"ok": False}
    db.set_rating(int(rid), rating)
    note = None
    # авто-калибровка по накоплению оценок
    if rating != 0 and settings.get("AUTO_CALIBRATE"):
        rs = db.rating_stats()
        if rs["rated"] and rs["rated"] % 10 == 0:
            res = admin_ops.apply_recommendations()
            note = res.get("msg")
    return {"ok": True, "note": note}


@app.post("/api/comment")
def api_comment(payload: dict = Body(...)):
    """Комментарий пользователя к ответу веб-чата (по id запроса)."""
    rid = payload.get("id")
    if rid is None:
        return {"ok": False}
    ok = db.set_comment(int(rid), payload.get("comment") or "")
    return {"ok": ok}


@app.get("/api/journal")
def api_journal(limit: int = 100):
    """Объединённый журнал: запросы веб-чата и Телеграм (с пометкой канала),
    с оценками и комментариями."""
    return db.recent_all(min(max(limit, 1), 1000))


@app.get("/api/telegram-recent")
def api_telegram_recent(limit: int = 10):
    """Последние запросы из Телеграм (для дашборда)."""
    return {"items": db.tg_recent(min(max(limit, 1), 200))}


# ============================ ЧАТ С ПРИЛОЖЕННЫМ ДОКУМЕНТОМ ============================
@app.post("/chat-doc")
async def chat_doc(file: UploadFile = File(...), question: str = Form(...),
                   history: str = Form("[]"), debug: str = Form(""),
                   session_id: str = Form("")):
    """Ответ на основе приложенного к вопросу документа (Excel и др.), без индексации."""
    t0 = time.time()
    name = os.path.basename(file.filename or "файл")
    ext = os.path.splitext(name)[1].lower()
    aid = activity.start("attach", name, "разбор документа")
    try:
        hist = json.loads(history) if history else []
    except Exception:
        hist = []

    if ext not in SUPPORTED:
        async def bad():
            msg = f"Тип файла {ext or '?'} не поддерживается."
            yield json.dumps({"type": "answer", "text": msg}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "sources", "items": []}, ensure_ascii=False) + "\n"
        return _ndjson(bad(), aid)

    # сохраняем во временный файл и парсим теми же загрузчиками
    tmp = Path(tempfile.gettempdir()) / f"rag_attach_{int(time.time())}_{name}"
    tmp.write_bytes(await file.read())
    items = []
    try:
        for part in loaders.load_file(tmp):
            for ch in chunk_text(part["text"], settings.get("CHUNK_SIZE"),
                                 settings.get("CHUNK_OVERLAP")):
                items.append({"text": ch, "source": name, "page": part["page"]})
    finally:
        tmp.unlink(missing_ok=True)

    hits = retriever.rerank_texts(question, items)
    if not hits:
        async def empty():
            msg = "Не удалось извлечь данные из файла или он пуст."
            yield json.dumps({"type": "answer", "text": msg}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "sources", "items": []}, ensure_ascii=False) + "\n"
        db.log_request(question, "attached", 0, 0.0,
                       int((time.time() - t0) * 1000), 0, False, [],
                       session_id=session_id,
                       answer="Не удалось извлечь данные из файла или он пуст.")
        return _ndjson(empty(), aid)

    context = prompts.build_context(hits)
    messages = [{"role": "system", "content": settings.get("SYSTEM_PROMPT")}]
    messages += hist[-6:]
    messages.append({"role": "user",
                     "content": prompts.build_user_message(question, context)})
    sources = [{"source": h["source"], "page": h.get("page"),
                "score": round(h["score"], 3)} for h in hits]
    activity.update(aid, stage="генерация ответа")

    async def stream():
        yield _stg("attach", "done", {"file": name, "fragments": len(items)})
        yield _stg("rerank", "done", {"model": settings.get("RERANK_MODEL"),
                                      "kept": len(hits), "candidates": len(items)})
        yield _stg("context", "done", {"chunks": len(hits), "chars": len(context)})
        yield _stg("generate", "start", {"backend": settings.get("LLM_BACKEND"),
                                         "model": settings.active_model(),
                                         "temperature": settings.get("TEMPERATURE")})
        acc = []
        async for tok in llm_backend.chat_stream(
                messages, temperature=settings.get("TEMPERATURE"),
                model=settings.active_model()):
            acc.append(tok)
            yield json.dumps({"type": "answer", "text": tok}, ensure_ascii=False) + "\n"
        yield _stg("generate", "done", {"chars": len("".join(acc))})
        yield json.dumps({"type": "sources", "items": sources}, ensure_ascii=False) + "\n"
        latency = int((time.time() - t0) * 1000)
        if debug in ("1", "true", "on", "yes"):
            yield json.dumps({"type": "debug", "info": {
                "engine": "приложенный документ (rerank)", "mode": settings.current_mode(),
                "model": settings.active_model(), "backend": settings.get("LLM_BACKEND"),
                "timings": {"retrieve_ms": 0, "gen_ms": latency, "total_ms": latency},
                "params": _debug_params(), "chunks": _debug_chunks(hits)}}, ensure_ascii=False) + "\n"
        full = "".join(acc)
        rid = db.log_request(question, "attached", len(hits), hits[0]["score"],
                             latency, len(full), True, sources,
                             session_id=session_id, answer=full)
        yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

    return _ndjson(stream(), aid)


@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile = File(...)):
    """Голосовой ввод: записанный в браузере звук → текст через локальный Whisper
    (тот же бэкенд, что и для индексации аудио/видео). Данные не покидают сервер."""
    name = file.filename or "voice.webm"
    ext = os.path.splitext(name)[1].lower() or ".webm"
    tmp = Path(tempfile.gettempdir()) / f"rag_voice_{int(time.time())}{ext}"
    try:
        tmp.write_bytes(await file.read())
        text = " ".join(p.get("text", "") for p in loaders.load_file(tmp)).strip()
        if not text:
            return {"ok": False, "msg": "речь не распознана (тихо или пусто)"}
        return {"ok": True, "text": text}
    except Exception as e:
        return {"ok": False, "msg": f"ошибка распознавания: {e}"}
    finally:
        tmp.unlink(missing_ok=True)


# ============================ API для UI ============================
@app.get("/api/stats")
def api_stats():
    s = db.stats()
    try:
        s["chunks"] = _qdrant.count(settings.get("QDRANT_COLLECTION"), exact=True).count
    except Exception:
        s["chunks"] = 0
    s["model"] = settings.active_model()
    s["finetuned"] = bool(settings.get("USE_FINETUNED"))
    s["backend"] = settings.get("LLM_BACKEND")
    s["device"] = settings.get("DEVICE")
    # счётчики Телеграм для дашборда (кэшируются ~60с)
    try:
        tg = db.tg_stats()
        s["tg_total"] = tg.get("total", 0)
        s["tg_today"] = tg.get("today", 0)
    except Exception:
        s["tg_total"] = s["tg_today"] = 0
    return s


@app.get("/api/logs")
def api_logs(limit: int = 100):
    return db.recent(min(max(limit, 1), 1000))


@app.get("/api/analytics")
def api_analytics():
    return db.analytics()


@app.get("/api/system")
def api_system():
    return admin_ops.system_info()


@app.get("/api/server-load")
def api_server_load():
    """Текущая загрузка хоста: CPU, память, диски, GPU, сеть, аптайм."""
    return admin_ops.server_load()


@app.get("/api/admin/price/status")
def api_price_status(x_admin_token: str | None = Header(None)):
    """Состояние прайс-папки (включена, путь, число файлов/фрагментов)."""
    _check_admin(x_admin_token)
    import price_folder
    return price_folder.status()


@app.get("/api/admin/sip/status")
def api_sip_status(x_admin_token: str | None = Header(None)):
    """Состояние голосового моста к АТС."""
    _check_admin(x_admin_token)
    import sip_bridge
    return sip_bridge.status()


@app.post("/api/admin/sip/restart")
def api_sip_restart(x_admin_token: str | None = Header(None)):
    """Перезапустить голосовой мост (после смены настроек)."""
    _check_admin(x_admin_token)
    import sip_bridge
    return sip_bridge.restart()


# ===================== Внешние API-хуки =====================

@app.get("/api/admin/api-hooks")
def api_hooks_list(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"hooks": db.api_hooks_list()}


@app.post("/api/admin/api-hooks/save")
def api_hooks_save(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    if not (payload.get("name") or "").strip():
        return {"ok": False, "msg": "укажите название"}
    if not (payload.get("url") or "").strip():
        return {"ok": False, "msg": "укажите URL"}
    hid = db.api_hook_save(payload)
    return {"ok": bool(hid), "id": hid}


@app.post("/api/admin/api-hooks/delete")
def api_hooks_delete(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"ok": db.api_hook_delete(int(payload.get("id")))}


@app.post("/api/admin/api-hooks/test")
def api_hooks_test(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    import api_tools
    q = (payload.get("question") or "").strip()
    if not q:
        return {"matched": False, "msg": "введите тестовый вопрос"}
    return api_tools.test(q)


@app.get("/api/server-history")
def api_server_history():
    """История загрузки за час/день/неделю/месяц/год + рекомендации по железу."""
    return admin_ops.server_history()


@app.get("/api/activity")
def api_activity():
    """Текущие запросы и процессы чат-системы в реальном времени: обработка вопросов
    (веб-чат и Телеграм), генерация ответов, разбор файлов, парсинг справочника, а
    также идущие фоновые задачи (индексация, граф, бенчмарк и т. п.)."""
    snap = activity.snapshot()
    try:
        jobs = admin_ops.active_jobs()
    except Exception:
        jobs = []
    return {"live": snap["items"], "jobs": jobs,
            "active": snap["active"] + sum(1 for j in jobs if j.get("running")),
            "by_kind": snap["by_kind"]}


# ===================== Структура компании =====================

@app.get("/api/admin/org/config")
def api_org_config(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    import org_structure
    return {"config": org_structure.get_config(),
            "status": org_structure.get_status(),
            "meta": db.org_meta()}


@app.post("/api/admin/org/config")
def api_org_config_save(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    import org_structure
    cfg = org_structure.set_config(url=payload.get("url"),
                                   enabled=payload.get("enabled"))
    return {"ok": True, "config": cfg}


@app.post("/api/admin/org/sync")
def api_org_sync(payload: dict = Body(default={}), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    import org_structure
    url = (payload or {}).get("url")
    if url is not None:
        org_structure.set_config(url=url)
    r = org_structure.sync(url)
    r["meta"] = db.org_meta()
    return r


@app.get("/api/admin/org/list")
def api_org_list(search: str = "", department: str = "",
                 x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    import org_structure
    return {"employees": db.org_list(search, department),
            "departments": db.org_departments(),
            "meta": db.org_meta(),
            "status": org_structure.get_status()}


@app.post("/api/admin/org/clear")
def api_org_clear(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    n = db.org_clear()
    try:
        import org_index
        org_index.clear()
    except Exception as e:
        print(f"[org] очистка индекса сотрудников: {e}")
    return {"ok": True, "removed": n}


# ===================== Синонимы =====================

@app.get("/api/admin/synonyms")
def api_syn_list(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    import synonyms
    return {"enabled": synonyms.enabled(), "items": db.syn_list()}


@app.post("/api/admin/synonyms/config")
def api_syn_config(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    import synonyms
    synonyms.set_enabled(bool(payload.get("enabled")))
    return {"ok": True, "enabled": synonyms.enabled()}


@app.post("/api/admin/synonyms/add")
def api_syn_add(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    term = (payload.get("term") or "").strip()
    syns = payload.get("syns") or []
    if isinstance(syns, str):
        import re as _re
        syns = [s.strip() for s in _re.split(r"[\n,;]+", syns) if s.strip()]
    if not term:
        return {"ok": False, "msg": "слово не задано"}
    rid = db.syn_add(term, syns)
    return {"ok": bool(rid), "id": rid}


@app.post("/api/admin/synonyms/update")
def api_syn_update(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    syns = payload.get("syns") or []
    if isinstance(syns, str):
        import re as _re
        syns = [s.strip() for s in _re.split(r"[\n,;]+", syns) if s.strip()]
    ok = db.syn_update(int(payload.get("id")), payload.get("term") or "", syns)
    return {"ok": ok}


@app.post("/api/admin/synonyms/delete")
def api_syn_delete(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"ok": db.syn_delete(int(payload.get("id")))}


@app.post("/api/admin/synonyms/clear")
def api_syn_clear(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"ok": True, "removed": db.syn_clear()}


# ===================== Статические DNS-записи =====================

def _valid_ip(ip: str) -> bool:
    import ipaddress
    try:
        ipaddress.ip_address((ip or "").strip())
        return True
    except Exception:
        return False


def _valid_host(h: str) -> bool:
    h = (h or "").strip()
    if not h or len(h) > 253 or " " in h:
        return False
    # допускаем буквы/цифры/дефис/точку (домены и поддомены)
    import re as _re
    return bool(_re.fullmatch(r"[A-Za-z0-9_.-]+", h))


@app.get("/api/admin/dns")
def api_dns_list(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"items": db.dns_list(), "active": dns_override.active()}


@app.post("/api/admin/dns/add")
def api_dns_add(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    host = (payload.get("hostname") or "").strip()
    ip = (payload.get("ip") or "").strip()
    if not _valid_host(host):
        return {"ok": False, "msg": "некорректное имя хоста"}
    if not _valid_ip(ip):
        return {"ok": False, "msg": "некорректный IP-адрес"}
    rid = db.dns_add(host, ip)
    dns_override.reload()
    return {"ok": bool(rid), "id": rid}


@app.post("/api/admin/dns/update")
def api_dns_update(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    host = (payload.get("hostname") or "").strip()
    ip = (payload.get("ip") or "").strip()
    if not _valid_host(host):
        return {"ok": False, "msg": "некорректное имя хоста"}
    if not _valid_ip(ip):
        return {"ok": False, "msg": "некорректный IP-адрес"}
    ok = db.dns_update(int(payload.get("id")), host, ip)
    dns_override.reload()
    return {"ok": ok}


@app.post("/api/admin/dns/delete")
def api_dns_delete(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    ok = db.dns_delete(int(payload.get("id")))
    dns_override.reload()
    return {"ok": ok}


@app.post("/api/admin/dns/clear")
def api_dns_clear(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    n = db.dns_clear()
    dns_override.reload()
    return {"ok": True, "removed": n}


@app.post("/api/admin/selftest")
def api_selftest(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.self_test()


@app.post("/api/admin/benchmark")
def api_benchmark(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.benchmark()


@app.post("/api/admin/benchmark/stop")
def api_benchmark_stop(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.stop_benchmark()


@app.get("/api/analytics-components")
def api_analytics_components():
    return admin_ops.component_analytics()


@app.get("/api/config")
def api_config():
    return {
        "fields": settings.FIELDS,
        "values": settings.public_settings(),
        "admin_token_set": settings.secret_is_set("ADMIN_TOKEN"),
        "auth_required": bool(settings.get("ADMIN_TOKEN")),
    }


@app.post("/api/config")
def api_set_config(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"ok": True, "values": settings.update(payload)}


@app.post("/api/config/reset")
def api_reset_config(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"ok": True, "values": settings.reset()}


@app.get("/api/mode")
def api_mode():
    return {"current": settings.current_mode(), "modes": settings.modes_catalog()}


@app.post("/api/mode")
def api_set_mode(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    settings.set_mode(payload.get("mode", ""))
    return {"ok": True, "current": settings.current_mode()}


# ============================ АДМИН-ОПЕРАЦИИ ============================
@app.get("/api/admin/status")
def admin_status(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.status()


@app.get("/api/admin/browse")
def admin_browse(path: str | None = None, x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.browse(path)


@app.get("/api/admin/files-catalog")
def admin_files_catalog(limit: int = 100, offset: int = 0, q: str = "",
                        sort: str = "name", order: str = "asc",
                        only_errors: bool = False, method: str = "",
                        x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.files_catalog(limit=limit, offset=offset, query=q,
                                   sort=sort, order=order, only_errors=only_errors,
                                   method=method)


@app.get("/api/admin/kb-graph")
def admin_kb_graph(max_nodes: int = 800, force: bool = False,
                   x_admin_token: str | None = Header(None)):
    """Граф проиндексированной базы знаний (Obsidian-вид): файлы и категории.
    Кэшируется; при отсутствии свежего кэша запускает фоновую сборку с прогрессом."""
    _check_admin(x_admin_token)
    return admin_ops.kb_graph(max_nodes=min(max(max_nodes, 10), 2000), force=force)


@app.get("/api/admin/kb-graph/status")
def admin_kb_graph_status(x_admin_token: str | None = Header(None)):
    """Прогресс сборки графа базы знаний и результат, когда готов."""
    _check_admin(x_admin_token)
    return admin_ops.kb_graph_status()


@app.get("/api/admin/file-text")
def admin_file_text(source: str, x_admin_token: str | None = Header(None)):
    """Извлечённый текст файла (для просмотра транскрипции/распознанного в каталоге)."""
    _check_admin(x_admin_token)
    return admin_ops.file_text(source)


# ---- выдача исходных артефактов (изображения/чертежи/видео) в ответах ----
# доступны без админ-токена: ссылки показываются всем пользователям чата (LAN).
@app.get("/api/media/info")
def media_info(source: str):
    return {"source": source, "kind": media.kind_of(source),
            "exists": media.available(source),
            "has_preview": media.has_preview(source)}


@app.get("/api/media/file")
def media_file(source: str):
    p = media.materialize(source)   # с диска или из PostgreSQL (без папки)
    if p is None:
        raise HTTPException(status_code=404, detail="файл не найден")
    return FileResponse(str(p), filename=Path(source).name)


@app.get("/api/media/thumb")
def media_thumb(source: str, t: float | None = None):
    p = media.thumbnail(source, t)
    if p is None:
        raise HTTPException(status_code=404, detail="превью недоступно")
    return FileResponse(str(p))


@app.get("/api/media/clip")
def media_clip(source: str, start: float, end: float):
    p = media.clip(source, start, end)
    if p is None:
        raise HTTPException(status_code=404, detail="фрагмент недоступен")
    return FileResponse(str(p), media_type="video/mp4")


@app.post("/api/admin/upload-folder")
async def admin_upload_folder(files: list[UploadFile] = File(...),
                              paths: list[str] = Form(...),
                              x_admin_token: str | None = Header(None)):
    """Загрузка целой папки (батчами) в DOCS_DIR с сохранением структуры.
    Веб-интерфейс шлёт файлы порциями — до десятков тысяч файлов суммарно."""
    _check_admin(x_admin_token)
    items = []
    for f, rel in zip(files, paths):
        items.append((rel or os.path.basename(f.filename or ""), await f.read()))
    return admin_ops.save_uploaded_folder(items)


# ---- резервное копирование и восстановление ----
@app.post("/api/admin/backup/create")
def admin_backup_create(payload: dict = Body(...),
                        x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.backup_create(payload.get("scope", ""))


@app.get("/api/admin/backup/list")
def admin_backup_list(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.backup_list()


@app.get("/api/admin/backup/download")
def admin_backup_download(name: str, x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    p = admin_ops.backup_download_path(name)
    if p is None:
        raise HTTPException(status_code=404, detail="архив не найден")
    return FileResponse(str(p), filename=p.name, media_type="application/gzip")


@app.post("/api/admin/backup/delete")
def admin_backup_delete(payload: dict = Body(...),
                        x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.backup_delete(payload.get("name", ""))


@app.post("/api/admin/backup/verify")
async def admin_backup_verify(file: UploadFile = File(...),
                              x_admin_token: str | None = Header(None)):
    """Пред-проверка загружаемого архива: целостность и состав, без восстановления."""
    _check_admin(x_admin_token)
    tmp = Path(tempfile.gettempdir()) / f"rag_verify_{int(time.time())}.tar.gz"
    try:
        tmp.write_bytes(await file.read())
        return admin_ops.backup_verify_file(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/api/admin/backup/restore")
async def admin_backup_restore(file: UploadFile = File(...),
                               x_admin_token: str | None = Header(None)):
    """Восстановление из загруженного архива (с обязательной проверкой целостности)."""
    _check_admin(x_admin_token)
    tmp = Path(tempfile.gettempdir()) / f"rag_restore_{int(time.time())}.tar.gz"
    tmp.write_bytes(await file.read())
    return admin_ops.backup_restore_file(str(tmp))


# ---- Телеграм-бот: статус, подтверждение пользователей, история ----
@app.get("/api/admin/telegram/status")
def admin_tg_status(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return telegram_bot.status()


@app.post("/api/admin/telegram/restart")
def admin_tg_restart(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return telegram_bot.restart()


@app.get("/api/admin/tts/voices")
def admin_tts_voices(engine: str = "", x_admin_token: str | None = Header(None)):
    """Список доступных голосов TTS для выбора движка (или текущего)."""
    _check_admin(x_admin_token)
    import tts
    eng = (engine or "").strip().lower()
    if not eng or eng == "auto":
        eng = (tts.available().get("engine") or "")
    return {"engine": eng, "voices": tts.voices(eng)}


@app.get("/api/admin/telegram/users")
def admin_tg_users(status: str = "", x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"users": db.tg_users(status or None)}


@app.post("/api/admin/telegram/send")
def admin_tg_send(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    """Отправить сообщение пользователям из веб-интерфейса.
    payload: {text, chat_ids?: [...], scope?: 'all'|'approved'}.
    chat_ids имеет приоритет; иначе по scope (all — все, кроме заблокированных)."""
    _check_admin(x_admin_token)
    text = (payload.get("text") or "").strip()
    if not text:
        return {"ok": False, "msg": "пустое сообщение"}
    chat_ids = payload.get("chat_ids") or []
    if not chat_ids:
        scope = (payload.get("scope") or "").strip()
        if scope == "approved":
            chat_ids = [u["chat_id"] for u in db.tg_users("approved")]
        elif scope == "all":
            chat_ids = [u["chat_id"] for u in db.tg_users()
                        if u.get("status") != "blocked"]
        else:
            return {"ok": False, "msg": "не выбраны получатели"}
    if not chat_ids:
        return {"ok": False, "msg": "нет получателей"}
    return telegram_bot.broadcast(chat_ids, text)


@app.post("/api/admin/telegram/map-employee")
def admin_tg_map_employee(payload: dict = Body(...),
                          x_admin_token: str | None = Header(None)):
    """Сопоставить Телеграм-пользователя сотруднику из справочника компании.
    Пустые поля снимают привязку."""
    _check_admin(x_admin_token)
    cid = int(payload.get("chat_id"))
    ok = db.tg_set_employee(cid, payload.get("email") or "", payload.get("name") or "",
                            payload.get("info") or "")
    return {"ok": ok}


@app.post("/api/admin/telegram/approve")
def admin_tg_approve(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    cid = int(payload.get("chat_id"))
    ok = db.tg_set_status(cid, "approved")
    if ok:
        telegram_bot.notify_approved(cid)
    return {"ok": ok}


@app.post("/api/admin/telegram/block")
def admin_tg_block(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    cid = int(payload.get("chat_id"))
    ok = db.tg_set_status(cid, "blocked")
    if ok:
        telegram_bot.notify_blocked(cid)
    return {"ok": ok}


@app.post("/api/admin/telegram/unblock")
def admin_tg_unblock(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    cid = int(payload.get("chat_id"))
    ok = db.tg_set_status(cid, "approved")
    if ok:
        telegram_bot.notify_approved(cid)
    return {"ok": ok}


@app.get("/api/admin/telegram/requests")
def admin_tg_requests(limit: int = 200, x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"items": db.tg_recent(limit), "stats": db.tg_stats()}


@app.post("/api/admin/telegram/clear-history")
def admin_tg_clear(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    n = db.tg_clear_history()
    return {"ok": True, "deleted": n, "msg": f"удалено записей: {n}"}


# ---- Телеграм: обучение (документы от пользователей) ----
@app.get("/api/admin/telegram/train-users")
def admin_tg_train_users(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    import tg_train
    counts = tg_train.user_file_counts()
    users = []
    for u in db.tg_users():
        if u.get("status") == "blocked":
            continue
        users.append({"chat_id": u["chat_id"], "username": u.get("username"),
                      "first_name": u.get("first_name"), "status": u.get("status"),
                      "can_train": bool(u.get("can_train")), "mode": u.get("mode") or "ask",
                      "files": counts.get(u["chat_id"], 0)})
    return {"users": users}


@app.post("/api/admin/telegram/train-allow")
def admin_tg_train_allow(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    cid = int(payload.get("chat_id"))
    allow = bool(payload.get("allow"))
    ok = db.tg_set_train(cid, allow)
    sent = False
    if ok and allow:                       # при выдаче доступа — шлём инструкцию
        try:
            sent = telegram_bot.send_train_instructions(cid)
        except Exception as e:
            print(f"[tg] инструкция не отправлена: {e}")
    return {"ok": ok, "instructions_sent": sent}


@app.post("/api/admin/telegram/train-instruction")
def admin_tg_train_instruction(payload: dict = Body(...),
                               x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    cid = int(payload.get("chat_id"))
    sent = telegram_bot.send_train_instructions(cid)
    return {"ok": sent, "msg": "инструкция отправлена" if sent
            else "не удалось отправить (бот выключен или нет токена)"}


@app.get("/api/admin/telegram/train-files")
def admin_tg_train_files(chat_id: int, x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    import tg_train
    return {"files": tg_train.list_files(chat_id)}


@app.post("/api/admin/telegram/train-delete")
def admin_tg_train_delete(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    import tg_train
    cid = int(payload.get("chat_id"))
    name = payload.get("name")
    if name:
        tg_train.delete_file(cid, name)
        return {"ok": True, "msg": "файл удалён"}
    tg_train.delete_user(cid)
    return {"ok": True, "msg": "все документы пользователя удалены"}


@app.post("/api/admin/telegram/train-delete-all")
def admin_tg_train_delete_all(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    import tg_train
    tg_train.delete_all()
    return {"ok": True, "msg": "все документы из Телеграм удалены"}


@app.get("/api/admin/db/status")
def admin_db_status(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.db_overview()


@app.post("/api/admin/db/test")
def admin_db_test(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.db_test(payload.get("backend", ""))


@app.post("/api/admin/db/copy")
def admin_db_copy(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.db_copy(payload.get("target", ""), migrate=False)


@app.post("/api/admin/db/migrate")
def admin_db_migrate(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.db_copy(payload.get("target", ""), migrate=True)


@app.post("/api/admin/cache/clear")
def admin_cache_clear(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.cache_clear()


@app.get("/api/admin/catalog/status")
def admin_catalog_status(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.catalog_status()


@app.post("/api/admin/catalog/load")
def admin_catalog_load(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.catalog_load()


@app.post("/api/admin/catalog/use")
def admin_catalog_use(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.catalog_use(payload.get("source", ""))


@app.post("/api/admin/catalog/clear-files")
def admin_catalog_clear_files(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.catalog_clear_files()


@app.post("/api/admin/check-data")
def admin_check_data(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.check_data_dir()


@app.get("/api/admin/models")
def admin_models(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.list_models()


@app.get("/api/admin/available-models")
def admin_available_models(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.available_models()


@app.get("/api/admin/vllm-models")
def admin_vllm_models(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.vllm_models()


@app.get("/api/admin/finetune-models")
def admin_finetune_models(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.finetune_models()


@app.post("/api/admin/pull-model")
def admin_pull_model(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.pull_model(payload.get("model", ""))


@app.get("/api/admin/web-urls")
def admin_web_urls(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.get_web_urls()


@app.post("/api/admin/ingest-web")
def admin_ingest_web(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.ingest_web(payload.get("urls", []))


@app.post("/api/admin/web-delete")
def admin_web_delete(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.delete_web(payload.get("url", ""))


@app.post("/api/admin/upload")
async def admin_upload(files: list[UploadFile] = File(...),
                       x_admin_token: str | None = Header(None)):
    """Загрузка файлов (Excel и др.) в DOCS_DIR/uploads. Индексируются при reindex."""
    _check_admin(x_admin_token)
    dest = Path(settings.get("DOCS_DIR")).expanduser() / "uploads"
    dest.mkdir(parents=True, exist_ok=True)
    saved, skipped, saved_paths = [], [], []
    for f in files:
        name = os.path.basename(f.filename or "")
        if not name:
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in admin_ops._SUPPORTED:
            skipped.append(name)
            continue
        try:
            (dest / name).write_bytes(await f.read())
            saved.append(name)
            saved_paths.append(dest / name)
        except Exception as e:
            skipped.append(f"{name} ({e})")
    # если активен каталог PostgreSQL — добавляем загруженные файлы и в него
    catalog_added = admin_ops.catalog_add_paths(saved_paths)
    return {"ok": True, "saved": saved, "skipped": skipped, "dir": str(dest),
            "catalog_added": catalog_added}


@app.get("/api/admin/ingest-logs")
def admin_ingest_logs(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"logs": db.ingest_log_list()}


@app.get("/api/admin/ingest-log")
def admin_ingest_log(id: int, x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return db.ingest_log_get(id) or {"ok": False, "msg": "лог не найден"}


@app.post("/api/admin/ingest-logs/delete")
def admin_ingest_logs_delete(payload: dict = Body(default={}),
                             x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    lid = payload.get("id")
    if lid in (None, "", "all"):
        n = db.ingest_log_clear()
        return {"ok": True, "deleted": n, "msg": f"удалено логов: {n}"}
    n = db.ingest_log_delete(int(lid))
    return {"ok": True, "deleted": n, "msg": "лог удалён" if n else "лог не найден"}


@app.get("/api/admin/calib/testset")
def admin_calib_testset_get(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"items": calibrate.load_testset()}


@app.post("/api/admin/calib/testset")
def admin_calib_testset_set(payload: dict = Body(default={}),
                            x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.save_testset(payload.get("items") or [])


@app.get("/api/admin/calib/example")
def admin_calib_example(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"items": calibrate.example_testset()}


@app.get("/api/admin/calib/modes")
def admin_calib_modes(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"modes": calibrate.available_modes(),
            "current": settings.current_mode(),
            "engines": calibrate.available_engines(),
            "current_engine": settings.get("ENGINE")}


@app.post("/api/admin/calib/run")
def admin_calib_run(payload: dict = Body(default={}),
                    x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.start(use_llm=bool(payload.get("use_llm")),
                           grid=payload.get("grid"),
                           modes=payload.get("modes"),
                           engines=payload.get("engines"))


@app.get("/api/admin/calib/status")
def admin_calib_status(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.status()


@app.post("/api/admin/calib/cancel")
def admin_calib_cancel(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.cancel()


@app.post("/api/admin/calib/save-log")
def admin_calib_save_log(payload: dict = Body(default={}),
                         x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.save_log(payload.get("which", "calib"))


@app.post("/api/admin/calib/apply")
def admin_calib_apply(payload: dict = Body(default={}),
                      x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.apply_params(payload.get("min_score"),
                                  payload.get("k_rerank"),
                                  payload.get("k_retrieve"),
                                  mode=payload.get("mode"))


@app.get("/api/admin/calib/sets")
def admin_calib_sets(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"sets": calibrate.mset_list()}


@app.post("/api/admin/calib/sets/save")
def admin_calib_sets_save(payload: dict = Body(default={}),
                          x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.mset_save(payload.get("name", ""))


@app.post("/api/admin/calib/sets/load")
def admin_calib_sets_load(payload: dict = Body(default={}),
                          x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.mset_load(payload.get("id"))


@app.post("/api/admin/calib/sets/delete")
def admin_calib_sets_delete(payload: dict = Body(default={}),
                            x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.mset_delete(payload.get("id"))


@app.get("/api/admin/calib/auto/testset")
def admin_calib_auto_testset(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"items": calibrate.auto_load()}


@app.post("/api/admin/calib/auto/generate")
def admin_calib_auto_generate(payload: dict = Body(default={}),
                              x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.auto_generate(n=payload.get("n", 50),
                                   folder=payload.get("folder", "test"),
                                   prompt=payload.get("prompt", ""))


@app.get("/api/admin/calib/auto/prompt")
def admin_calib_auto_prompt_get(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"prompt": calibrate.auto_prompt_get(),
            "default": calibrate.DEFAULT_GEN_PROMPT}


@app.post("/api/admin/calib/auto/prompt")
def admin_calib_auto_prompt_set(payload: dict = Body(default={}),
                                x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    calibrate.auto_prompt_set(payload.get("prompt", ""))
    return {"ok": True, "prompt": calibrate.auto_prompt_get()}


@app.get("/api/admin/calib/auto/sets")
def admin_calib_auto_sets(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"sets": calibrate.set_list()}


@app.post("/api/admin/calib/auto/sets/save")
def admin_calib_auto_sets_save(payload: dict = Body(default={}),
                               x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.set_save(payload.get("name", ""))


@app.post("/api/admin/calib/auto/sets/load")
def admin_calib_auto_sets_load(payload: dict = Body(default={}),
                               x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.set_load(payload.get("id"))


@app.post("/api/admin/calib/auto/sets/delete")
def admin_calib_auto_sets_delete(payload: dict = Body(default={}),
                                 x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.set_delete(payload.get("id"))


@app.get("/api/admin/calib/auto/status")
def admin_calib_auto_status(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.auto_status()


@app.post("/api/admin/calib/auto/cancel")
def admin_calib_auto_cancel(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.auto_cancel()


@app.get("/api/admin/calib/opt/variants")
def admin_calib_opt_variants(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"engines": calibrate.optimize_engines(),
            "modes": calibrate.optimize_modes(),
            "current_engine": settings.get("ENGINE"),
            "current_mode": settings.current_mode()}


@app.post("/api/admin/calib/opt/run")
def admin_calib_opt_run(payload: dict = Body(default={}),
                        x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.optimize_start(max_iter=payload.get("max_iter", 50),
                                    deviation=payload.get("deviation", 30),
                                    engine=payload.get("engine"),
                                    mode=payload.get("mode"))


@app.get("/api/admin/calib/opt/status")
def admin_calib_opt_status(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.optimize_status()


@app.post("/api/admin/calib/opt/cancel")
def admin_calib_opt_cancel(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.optimize_cancel()


@app.post("/api/admin/calib/eval-kb")
def admin_calib_eval_kb(payload: dict = Body(default={}),
                        x_admin_token: str | None = Header(None)):
    """Запустить ИИ-оценку всей базы знаний (фоновая задача с прогрессом)."""
    _check_admin(x_admin_token)
    import kb_eval
    return kb_eval.evaluate(force=bool((payload or {}).get("force")))


@app.get("/api/admin/calib/eval-kb/status")
def admin_calib_eval_kb_status(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    import kb_eval
    return kb_eval.status()


@app.post("/api/admin/calib/auto/run")
def admin_calib_auto_run(payload: dict = Body(default={}),
                         x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.auto_run(deviation=payload.get("deviation", 30),
                              engine=payload.get("engine"))


@app.post("/api/admin/reindex")
def admin_reindex(payload: dict = Body(default={}),
                  x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.reindex(bool(payload.get("reset")))


@app.post("/api/admin/apply-llm")
def admin_apply_llm(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.apply_llm()


@app.post("/api/admin/install-qdrant")
def admin_install_qdrant(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.install_qdrant()


@app.post("/api/admin/install-lightrag")
def admin_install_lightrag(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.install_lightrag()


@app.post("/api/admin/build-graph")
def admin_build_graph(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.build_graph()


@app.post("/api/admin/finetune")
def admin_finetune(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.finetune()


@app.post("/api/admin/apply-finetuned")
def admin_apply_finetuned(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.apply_finetuned()


@app.post("/api/admin/reset")
def admin_reset(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.reset(payload.get("targets", []))


@app.post("/api/admin/clear-history")
def admin_clear_history(x_admin_token: str | None = Header(None)):
    """Очистить историю всех чатов и их статистику (журнал запросов и оценки)."""
    _check_admin(x_admin_token)
    n = db.clear()
    return {"ok": True, "deleted": n, "msg": f"удалено записей: {n}"}


@app.get("/api/chat-history")
def api_chat_history(session_id: str):
    """История одного чата (по session_id) — для сохранения в файл."""
    return {"session_id": session_id, "items": db.session_history(session_id)}


@app.get("/api/admin/check-updates")
def admin_check_updates(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.check_updates()


@app.post("/api/admin/update")
def admin_update(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.update_app()


@app.post("/api/admin/reinstall-env")
def admin_reinstall_env(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.reinstall_env()


@app.post("/api/admin/reinstall-full")
def admin_reinstall_full(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.reinstall_full(payload.get("kind", ""))


@app.get("/api/admin/recommendations")
def admin_recommendations(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.recommend()


@app.post("/api/admin/apply-recommendations")
def admin_apply_recommendations(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.apply_recommendations()


@app.post("/api/admin/rollback-calibration")
def admin_rollback_calibration(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.rollback_calibration()


@app.post("/api/admin/restart")
def admin_restart(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.restart()


# ============================ УДАЛЁННЫЕ ХОСТЫ ============================
@app.get("/api/admin/remote/hosts")
def remote_hosts(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {**remote.list_hosts(), "job": remote.status()}


@app.post("/api/admin/remote/save")
def remote_save(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return remote.save_host(payload)


@app.post("/api/admin/remote/delete")
def remote_delete(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return remote.delete_host(payload.get("name", ""))


@app.post("/api/admin/remote/deploy")
def remote_deploy(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return remote.deploy(payload.get("name", ""), payload.get("what", "qdrant"))


@app.post("/api/admin/remote/transfer")
def remote_transfer(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    kind = payload.get("kind", "docs")
    direction = payload.get("direction", "push")
    if kind == "snapshot":
        return remote.transfer_snapshot(payload.get("name", ""), direction)
    return remote.transfer_docs(payload.get("name", ""), direction)


@app.post("/api/admin/remote/switch")
def remote_switch(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return remote.switch(payload.get("name", ""))


@app.post("/api/admin/remote/restore")
def remote_restore(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return remote.restore()


# ============================ статика ============================
@app.get("/")
def index():
    return FileResponse("static/index.html")


try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    pass
