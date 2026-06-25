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
def _start_telegram():
    """Поднять Телеграм-бота, если задан токен (фоновый поток long-polling)."""
    try:
        r = telegram_bot.start()
        if r.get("ok"):
            print(f"[telegram] {r.get('msg')}")
    except Exception as e:
        print(f"[telegram] не запущен: {e}")


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


@app.post("/chat")
async def chat(req: ChatRequest):
    t0 = time.time()

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
                                     retrieve_ms=0, gen_ms=lat, session_id=req.session_id)
                yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

            return StreamingResponse(kstream(), media_type="application/x-ndjson")
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
                                     session_id=req.session_id)
                yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

            return StreamingResponse(gstream(), media_type="application/x-ndjson")
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
                                     retrieve_ms=0, gen_ms=0, session_id=req.session_id)
                yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

            return StreamingResponse(cached_stream(), media_type="application/x-ndjson")

    t_ret = time.time()
    trace = []
    hits = search(req.question, filters=req.filters, trace=trace)
    # расширенный поиск, если ничего не нашлось (опционально): лексический → глубокий
    if not hits and settings.get("NO_ANSWER_FALLBACK"):
        try:
            hits = retriever.no_answer_fallback(req.question, trace=trace) or []
        except Exception as e:
            print(f"  ! фолбэк-поиск не удался: {e}")
    retrieve_ms = int((time.time() - t_ret) * 1000)
    category = (req.filters or {}).get("doc_category") or infer_category(req.question)

    if not hits:
        msg = "В доступных документах нет точного ответа на этот вопрос."
        rid = db.log_request(req.question, category, 0, 0.0,
                             int((time.time() - t0) * 1000), len(msg), False, [],
                             retrieve_ms=retrieve_ms, gen_ms=0,
                             session_id=req.session_id)

        async def empty():
            for s in trace:
                yield _stg(s["key"], "done", s.get("info"), s.get("ms", 0))
            yield _stg("context", "done", {"chunks": 0, "chars": 0})
            yield json.dumps({"type": "answer", "text": msg}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "sources", "items": []}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

        return StreamingResponse(empty(), media_type="application/x-ndjson")

    context = prompts.build_context(hits)
    messages = [{"role": "system", "content": settings.get("SYSTEM_PROMPT")}]
    messages += req.history[-6:]
    messages.append({"role": "user",
                     "content": prompts.build_user_message(req.question, context)})

    sources = [media.cite(h["source"], page=h.get("page"),
                          t_start=h.get("t_start"), t_end=h.get("t_end"),
                          score=round(h["score"], 3), category=h.get("doc_category"))
               for h in hits]

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
        rid = db.log_request(req.question, category, len(hits), hits[0]["score"],
                             latency, len("".join(acc)), True, sources,
                             retrieve_ms=retrieve_ms, gen_ms=max(0, latency - retrieve_ms),
                             session_id=req.session_id)
        if acache_key and acc:
            try:
                import cache
                cache.set_json(acache_key, 86400, {
                    "answer": "".join(acc), "sources": sources, "category": category,
                    "n_hits": len(hits), "top_score": hits[0]["score"]}, ns="index")
            except Exception:
                pass
        yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


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


# ============================ ЧАТ С ПРИЛОЖЕННЫМ ДОКУМЕНТОМ ============================
@app.post("/chat-doc")
async def chat_doc(file: UploadFile = File(...), question: str = Form(...),
                   history: str = Form("[]"), debug: str = Form(""),
                   session_id: str = Form("")):
    """Ответ на основе приложенного к вопросу документа (Excel и др.), без индексации."""
    t0 = time.time()
    name = os.path.basename(file.filename or "файл")
    ext = os.path.splitext(name)[1].lower()
    try:
        hist = json.loads(history) if history else []
    except Exception:
        hist = []

    if ext not in SUPPORTED:
        async def bad():
            msg = f"Тип файла {ext or '?'} не поддерживается."
            yield json.dumps({"type": "answer", "text": msg}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "sources", "items": []}, ensure_ascii=False) + "\n"
        return StreamingResponse(bad(), media_type="application/x-ndjson")

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
                       session_id=session_id)
        return StreamingResponse(empty(), media_type="application/x-ndjson")

    context = prompts.build_context(hits)
    messages = [{"role": "system", "content": settings.get("SYSTEM_PROMPT")}]
    messages += hist[-6:]
    messages.append({"role": "user",
                     "content": prompts.build_user_message(question, context)})
    sources = [{"source": h["source"], "page": h.get("page"),
                "score": round(h["score"], 3)} for h in hits]

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
        rid = db.log_request(question, "attached", len(hits), hits[0]["score"],
                             latency, len("".join(acc)), True, sources,
                             session_id=session_id)
        yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


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


@app.get("/api/admin/telegram/users")
def admin_tg_users(status: str = "", x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"users": db.tg_users(status or None)}


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
            "current": settings.current_mode()}


@app.post("/api/admin/calib/run")
def admin_calib_run(payload: dict = Body(default={}),
                    x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.start(use_llm=bool(payload.get("use_llm")),
                           grid=payload.get("grid"),
                           modes=payload.get("modes"))


@app.get("/api/admin/calib/status")
def admin_calib_status(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.status()


@app.post("/api/admin/calib/apply")
def admin_calib_apply(payload: dict = Body(default={}),
                      x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return calibrate.apply_params(payload.get("min_score"),
                                  payload.get("k_rerank"),
                                  payload.get("k_retrieve"),
                                  mode=payload.get("mode"))


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
