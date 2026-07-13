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
import integrations
from ingest import chunk_text, SUPPORTED
from retriever import search, infer_category

app = FastAPI(title="Корпоративный RAG-чатбот")
_qdrant = QdrantClient(url=settings.get("QDRANT_URL"))

# CORS — чтобы встраиваемый веб-виджет чата работал с других доменов (сайт компании).
try:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
except Exception as _e:
    print(f"[cors] не подключён: {_e}")


def _answer_sync(question: str, filters: dict | None = None) -> dict:
    """Синхронный ответ по базе знаний (без стриминга) — для внешнего API и виджета.
    Возвращает {answer, sources, answered, category, top_score}."""
    hits = search(question, filters=filters)
    if not hits and settings.get("NO_ANSWER_FALLBACK"):
        try:
            hits = retriever.no_answer_fallback(question) or []
        except Exception:
            hits = []
    if not hits:
        return {"answer": prompts.NO_ANSWER_TEXT if hasattr(prompts, "NO_ANSWER_TEXT")
                else "В доступных документах нет точного ответа на этот вопрос.",
                "sources": [], "answered": False, "category": None, "top_score": 0.0}
    context = prompts.build_context(hits)
    messages = [{"role": "system", "content": settings.get("SYSTEM_PROMPT")},
                {"role": "user", "content": prompts.build_user_message(question, context)}]
    try:
        answer = llm_backend.chat(messages, temperature=settings.get("TEMPERATURE"),
                                  model=settings.active_model())
    except Exception as e:
        answer = f"Ошибка генерации: {e}"
    # проверка обоснованности (антигаллюцинации)
    if settings.get("ANSWER_VERIFY") in ("warn", "strict"):
        try:
            import verify
            answer = verify.apply(question, answer, context).get("answer", answer)
        except Exception:
            pass
    answered = not prompts.is_no_answer(answer)
    sources = [media.cite(h["source"], page=h.get("page"), score=round(h["score"], 3),
                          category=h.get("doc_category"), snippet=h.get("text"))
               for h in hits]
    return {"answer": answer, "sources": sources, "answered": answered,
            "category": (filters or {}).get("doc_category") or infer_category(question),
            "top_score": round(hits[0]["score"], 3)}


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
    """Поднять телефонию: AudioSocket-мост и/или нативную SIP-регистрацию."""
    try:
        import sip_bridge
        if settings.get("SIP_ENABLED"):
            r = sip_bridge.start()
            print(f"[sip] {r.get('msg')}")
    except Exception as e:
        print(f"[sip] AudioSocket не запущен: {e}")
    try:
        import sip_phone
        if settings.get("SIP_REGISTER_ENABLED"):
            r = sip_phone.start()
            print(f"[sip-reg] {r.get('msg')}")
    except Exception as e:
        print(f"[sip-reg] не запущен: {e}")


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
def _fix_redis_host_docker():
    """В Docker исправить «залипший» REDIS_HOST=127.0.0.1 (мог остаться от старой кнопки
    «Установить Redis», ставившей одноразовый Redis внутрь контейнера) на сервис compose
    «redis» — чтобы после перезапуска кэш подключался сам, без ручных действий."""
    try:
        if not admin_ops._in_docker() or not settings.get("REDIS_ENABLED"):
            return
        host = (settings.get("REDIS_HOST") or "").strip().lower()
        if host in ("", "127.0.0.1", "localhost", "::1", "0.0.0.0"):
            settings.update({"REDIS_HOST": "redis"})   # сервис redis в сети compose (постоянный)
            print("[redis] Docker: REDIS_HOST исправлен на сервис compose 'redis'")
    except Exception as e:
        print(f"[redis] авто-исправление хоста: {e}")


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
                _ksrc = ([] if (settings.get("HIDE_SOURCES_IF_NO_ANSWER")
                                and prompts.is_no_answer(ktext)) else ksources)
                yield json.dumps({"type": "sources", "items": _ksrc},
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
                _gsrc = ([] if (settings.get("HIDE_SOURCES_IF_NO_ANSWER")
                                and prompts.is_no_answer(text))
                         else [{"source": "граф знаний (LightRAG)", "page": None}])
                yield json.dumps({"type": "sources", "items": _gsrc},
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
    q_emb = None                      # эмбеддинг вопроса для семантического кэша
    if settings.get("ANSWER_CACHE") and not req.history:
        import cache as _cache
        acache_key = "ans:" + hashlib.sha1("|".join(str(x) for x in [
            _cache.norm_q(req.question), req.filters, settings.get("SYSTEM_PROMPT"),
            settings.active_model(), settings.get("TEMPERATURE")]).encode("utf-8")).hexdigest()
        try:
            import cache
            cached = cache.get_json(acache_key, ns="index")
        except Exception:
            cached = None
        # семантический кэш: на похожий по смыслу вопрос — тот же ответ
        if not cached and settings.get("ANSWER_CACHE_SEMANTIC"):
            try:
                import cache
                q_emb = retriever._embed_query(req.question)
                hit = cache.answer_sem_find(q_emb, settings.get("ANSWER_CACHE_SIM"))
                if hit:
                    cached = cache.get_json(hit["key"], ns="index")
            except Exception:
                cached = cached
        if cached:
            async def cached_stream():
                yield _stg("answer_cache", "done", {"hit": True})
                # совместимость с кэшем из Телеграма/VoIP: там текст лежит в "text"
                txt = cached.get("answer") or cached.get("text") or ""
                for i in range(0, len(txt), 40):
                    yield json.dumps({"type": "answer", "text": txt[i:i + 40]},
                                     ensure_ascii=False) + "\n"
                _csrc = ([] if (settings.get("HIDE_SOURCES_IF_NO_ANSWER")
                                and prompts.is_no_answer(txt)) else cached.get("sources", []))
                yield json.dumps({"type": "sources", "items": _csrc},
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
    # разрешение контекста диалога: уточняющий вопрос → самостоятельный для ПОИСКА
    search_q = req.question
    if req.history:
        try:
            search_q = retriever.standalone_question(req.question, req.history)
            if search_q != req.question:
                trace.append({"key": "dialog_rewrite", "ms": 0,
                              "info": {"standalone": search_q[:120]}})
        except Exception:
            search_q = req.question
    hits = search(search_q, filters=req.filters, trace=trace)
    # расширенный поиск, если ничего не нашлось (опционально): лексический → глубокий
    if not hits and settings.get("NO_ANSWER_FALLBACK"):
        try:
            hits = retriever.no_answer_fallback(search_q, trace=trace) or []
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
                          score=round(h["score"], 3), category=h.get("doc_category"),
                          snippet=h.get("text"))
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
        # strict: не стримим по токенам — сперва проверим/перегенерируем, затем отдадим финал
        _strict = settings.get("ANSWER_VERIFY") == "strict"
        acc = []
        try:
            async for tok in llm_backend.chat_stream(
                    messages, temperature=settings.get("TEMPERATURE"),
                    model=settings.active_model(), kind="chat", label=req.question):
                acc.append(tok)
                if not _strict:
                    yield json.dumps({"type": "answer", "text": tok}, ensure_ascii=False) + "\n"
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[chat] ошибка генерации LLM: {e}\n{tb}")
            yield json.dumps({"type": "error", "stage": "generate",
                              "error": f"{type(e).__name__}: {e}",
                              "hint": ("Сбой на этапе генерации (LLM). Проверьте, что движок "
                                       f"{settings.get('LLM_BACKEND')} доступен и модель загружена; "
                                       "при нехватке памяти возможен OOM.")},
                             ensure_ascii=False) + "\n"
            yield _stg("generate", "error", {"error": f"{type(e).__name__}: {e}"})
            return
        gen_ms = max(0, int((time.time() - t0) * 1000) - retrieve_ms)
        yield _stg("generate", "done", {"chars": len("".join(acc)), "ms": gen_ms}, gen_ms)
        # проверка обоснованности (антигаллюцинации). verify.* делают синхронные вызовы LLM —
        # уводим их с event loop через run_in_executor.
        _vmode = settings.get("ANSWER_VERIFY")
        if _vmode in ("warn", "strict") and acc:
            import asyncio as _aio
            _loop = _aio.get_event_loop()
            _ans = "".join(acc)
            try:
                import verify
                if _strict:
                    # перегенерация ДО отдачи, если ответ не обоснован (потому и не стримили)
                    yield _stg("verify", "start", {"mode": "strict"})
                    vr = await _loop.run_in_executor(None, verify.apply,
                                                     req.question, _ans, context)
                    final = (vr or {}).get("answer", _ans)
                    acc = [final]
                    yield _stg("verify", "done", {"grounded": (vr or {}).get("grounded"),
                                                  "changed": (vr or {}).get("changed")})
                    for i in range(0, len(final), 40):
                        yield json.dumps({"type": "answer", "text": final[i:i + 40]},
                                         ensure_ascii=False) + "\n"
                else:  # warn — как раньше: дописываем пометку, если не обосновано
                    grounded = await _loop.run_in_executor(
                        None, verify.is_grounded, req.question, _ans, context)
                    if not prompts.is_no_answer(_ans) and not grounded:
                        acc.append(verify.CAVEAT)
                        yield json.dumps({"type": "answer", "text": verify.CAVEAT},
                                         ensure_ascii=False) + "\n"
            except Exception as _ve:
                print(f"[chat] проверка обоснованности не удалась: {_ve}")
                # strict: мы ещё ничего не отдали — отдаём исходный ответ
                if _strict and _ans:
                    for i in range(0, len(_ans), 40):
                        yield json.dumps({"type": "answer", "text": _ans[i:i + 40]},
                                         ensure_ascii=False) + "\n"
        out_sources = sources
        if settings.get("HIDE_SOURCES_IF_NO_ANSWER") and prompts.is_no_answer("".join(acc)):
            out_sources = []
        yield json.dumps({"type": "sources", "items": out_sources}, ensure_ascii=False) + "\n"
        # «что попало в контекст» — фрагменты, реально поданные модели (сворачиваемый блок)
        try:
            _ctx_items = [{"n": i + 1, "source": h.get("source"), "page": h.get("page"),
                           "score": round(h.get("score", 0), 3),
                           "text": (h.get("parent") or h.get("text") or "")[:4000]}
                          for i, h in enumerate(hits)]
            yield json.dumps({"type": "context", "items": _ctx_items},
                             ensure_ascii=False) + "\n"
        except Exception:
            pass
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
                    "answer": "".join(acc), "text": "".join(acc),  # "text" — для кросс-канального кэша
                    "sources": sources, "category": category,
                    "n_hits": len(hits), "top_score": hits[0]["score"]}, ns="index")
                # семантический кэш: связываем эмбеддинг вопроса с ключом ответа
                if settings.get("ANSWER_CACHE_SEMANTIC"):
                    _qe = q_emb or retriever._embed_query(req.question)
                    cache.answer_sem_add(acache_key, _qe)
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
    try:
        integrations.fire("rating", {"id": int(rid), "rating": rating})
    except Exception:
        pass
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


@app.get("/api/voip-recent")
def api_voip_recent(limit: int = 10):
    """Последние запросы VoIP (для дашборда)."""
    return {"items": db.voip_recent(min(max(limit, 1), 200))}


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
                model=settings.active_model(), kind="chat-doc", label=question):
            acc.append(tok)
            yield json.dumps({"type": "answer", "text": tok}, ensure_ascii=False) + "\n"
        yield _stg("generate", "done", {"chars": len("".join(acc))})
        _dsrc = ([] if (settings.get("HIDE_SOURCES_IF_NO_ANSWER")
                        and prompts.is_no_answer("".join(acc))) else sources)
        yield json.dumps({"type": "sources", "items": _dsrc}, ensure_ascii=False) + "\n"
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


@app.get("/api/admin/quality")
def api_quality(x_admin_token: str | None = Header(None)):
    """Аналитика качества: 👎-ответы, пробелы (вопросы без ответа), топ источников,
    динамика оценок."""
    _check_admin(x_admin_token)
    return db.quality_report()


@app.get("/api/admin/alerts")
def api_alerts(x_admin_token: str | None = Header(None)):
    """Сводка по алертам: включено ли, настроенные каналы, активные падения, лог."""
    _check_admin(x_admin_token)
    import alerts
    return alerts.status()


@app.post("/api/admin/alerts/test")
def api_alerts_test(x_admin_token: str | None = Header(None)):
    """Отправить тестовый алерт по всем настроенным каналам."""
    _check_admin(x_admin_token)
    import alerts
    return alerts.send_test()


# ----- Внешний API для интеграций (по API-ключу) -----

@app.post("/api/v1/ask")
def api_v1_ask(payload: dict = Body(...), x_api_key: str | None = Header(None)):
    """Публичный эндпоинт для внешних систем и встраиваемого виджета: задать вопрос
    базе знаний и получить ответ JSON. Требует заголовок X-API-Key."""
    if not integrations.api_key_valid(x_api_key or (payload or {}).get("api_key") or ""):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    question = ((payload or {}).get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    t0 = time.time()
    res = _answer_sync(question, filters=(payload or {}).get("filters"))
    lat = int((time.time() - t0) * 1000)
    try:
        rid = db.log_request(question, res.get("category"), len(res.get("sources", [])),
                             res.get("top_score", 0.0), lat, len(res.get("answer", "")),
                             res.get("answered", True), res.get("sources", []),
                             channel="api", answer=res.get("answer", ""))
    except Exception:
        rid = None
    integrations.fire("question", {"channel": "api", "id": rid, "question": question,
                                   "answer": res.get("answer", ""),
                                   "answered": res.get("answered"),
                                   "sources": [s.get("source") for s in res.get("sources", [])]})
    return {"id": rid, "answer": res.get("answer"), "answered": res.get("answered"),
            "sources": res.get("sources", []), "latency_ms": lat}


@app.get("/api/admin/integrations")
def api_integrations(x_admin_token: str | None = Header(None)):
    """Список API-ключей (маскированных) и веб-хуков."""
    _check_admin(x_admin_token)
    return {"api_keys": integrations.api_keys_list(),
            "webhooks": integrations.webhooks_list()}


@app.post("/api/admin/api-keys")
def api_keys_create(payload: dict = Body(default={}),
                    x_admin_token: str | None = Header(None)):
    """Создать API-ключ (показывается один раз). payload: {label}."""
    _check_admin(x_admin_token)
    return integrations.api_key_create(payload.get("label", ""))


@app.post("/api/admin/api-keys/revoke")
def api_keys_revoke(payload: dict = Body(...),
                    x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return integrations.api_key_revoke(payload.get("id", ""))


@app.post("/api/admin/webhooks")
def api_webhooks_save(payload: dict = Body(...),
                      x_admin_token: str | None = Header(None)):
    """Создать/обновить веб-хук. payload: {id?, url, events:[question,rating], enabled}."""
    _check_admin(x_admin_token)
    return integrations.webhook_save(payload)


@app.post("/api/admin/webhooks/delete")
def api_webhooks_delete(payload: dict = Body(...),
                        x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return integrations.webhook_delete(payload.get("id", ""))


_WIDGET_JS = r"""
(function(){
  var s=document.currentScript||(function(){var a=document.getElementsByTagName('script');return a[a.length-1];})();
  var u=new URL(s.src); var KEY=u.searchParams.get('key')||''; var BASE=u.origin;
  var TITLE=u.searchParams.get('title')||'Помощник';
  var st=document.createElement('style'); st.textContent=
    '.ragw-btn{position:fixed;right:20px;bottom:20px;width:56px;height:56px;border-radius:50%;background:#2563eb;color:#fff;font-size:26px;border:none;cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,.3);z-index:2147483000}'
   +'.ragw-panel{position:fixed;right:20px;bottom:88px;width:360px;max-width:92vw;height:520px;max-height:76vh;background:#fff;color:#111;border-radius:14px;box-shadow:0 10px 40px rgba(0,0,0,.35);display:none;flex-direction:column;overflow:hidden;z-index:2147483000;font:14px system-ui,Arial}'
   +'.ragw-head{background:#2563eb;color:#fff;padding:12px 14px;font-weight:600;display:flex;justify-content:space-between;align-items:center}'
   +'.ragw-body{flex:1;overflow:auto;padding:12px;background:#f7f7f9}'
   +'.ragw-msg{margin:6px 0;padding:8px 11px;border-radius:12px;max-width:85%;white-space:pre-wrap;line-height:1.4}'
   +'.ragw-me{background:#2563eb;color:#fff;margin-left:auto}.ragw-bot{background:#fff;border:1px solid #e5e7eb}'
   +'.ragw-src{font-size:11px;color:#666;margin-top:4px}'
   +'.ragw-foot{display:flex;gap:6px;padding:10px;border-top:1px solid #eee;background:#fff}'
   +'.ragw-foot input{flex:1;border:1px solid #ddd;border-radius:9px;padding:9px 10px;font:inherit}'
   +'.ragw-foot button{border:none;background:#2563eb;color:#fff;border-radius:9px;padding:0 14px;cursor:pointer}';
  document.head.appendChild(st);
  var btn=document.createElement('button'); btn.className='ragw-btn'; btn.innerHTML='💬';
  var p=document.createElement('div'); p.className='ragw-panel';
  p.innerHTML='<div class="ragw-head"><span>'+TITLE+'</span><span style="cursor:pointer" id="ragw-x">✕</span></div>'
   +'<div class="ragw-body" id="ragw-body"><div class="ragw-msg ragw-bot">Здравствуйте! Задайте вопрос по нашей базе знаний.</div></div>'
   +'<div class="ragw-foot"><input id="ragw-in" placeholder="Ваш вопрос…"><button id="ragw-send">→</button></div>';
  document.body.appendChild(btn); document.body.appendChild(p);
  function esc(t){var d=document.createElement('div');d.textContent=t||'';return d.innerHTML;}
  function add(cls,html){var b=document.getElementById('ragw-body');var m=document.createElement('div');m.className='ragw-msg '+cls;m.innerHTML=html;b.appendChild(m);b.scrollTop=b.scrollHeight;return m;}
  btn.onclick=function(){p.style.display=p.style.display==='flex'?'none':'flex';};
  p.querySelector('#ragw-x').onclick=function(){p.style.display='none';};
  function send(){
    var inp=document.getElementById('ragw-in');var q=(inp.value||'').trim();if(!q)return;inp.value='';
    add('ragw-me',esc(q));var wait=add('ragw-bot','…');
    fetch(BASE+'/api/v1/ask',{method:'POST',headers:{'Content-Type':'application/json','X-API-Key':KEY},body:JSON.stringify({question:q})})
      .then(function(r){return r.json();}).then(function(d){
        wait.innerHTML=esc(d.answer||'Нет ответа');
        var s=(d.sources||[]).map(function(x){return esc((x.source||'').split('/').pop());}).filter(Boolean);
        if(s.length){var e=document.createElement('div');e.className='ragw-src';e.textContent='Источники: '+s.slice(0,5).join(', ');wait.appendChild(e);}
      }).catch(function(){wait.innerHTML='Ошибка соединения';});
  }
  document.getElementById('ragw-send').onclick=send;
  document.getElementById('ragw-in').addEventListener('keydown',function(e){if(e.key==='Enter')send();});
})();
"""


@app.get("/widget.js")
def widget_js():
    """Встраиваемый скрипт чата: <script src="…/widget.js?key=API_KEY&title=…"></script>."""
    from fastapi.responses import Response
    return Response(content=_WIDGET_JS, media_type="application/javascript")


@app.get("/embed")
def widget_demo(key: str = ""):
    """Демо-страница встраиваемого виджета."""
    from fastapi.responses import HTMLResponse
    html = ("<!doctype html><meta charset=utf-8><title>Виджет — демо</title>"
            "<body style='font:16px system-ui;padding:40px'>"
            "<h1>Демо встраиваемого чата</h1>"
            "<p>Кнопка чата — в правом нижнем углу.</p>"
            f"<script src='/widget.js?key={_html_escape(key)}&title=Помощник'></script></body>")
    return HTMLResponse(html)


def _html_escape(s: str) -> str:
    import html as _h
    return _h.escape(s or "", quote=True)


@app.get("/api/system")
def api_system():
    return admin_ops.system_info()


@app.get("/api/server-load")
def api_server_load():
    """Текущая загрузка хоста: CPU, память, диски, GPU, сеть, аптайм."""
    return admin_ops.server_load()


@app.get("/api/component-metrics")
def api_component_metrics():
    """Расширенная статистика по компонентам (Qdrant, эмбеддер, реранкер, LightRAG,
    KAG, SQLite/PostgreSQL, Redis): обращения, ошибки, задержка и ресурсы — в реальном
    времени для графиков на дашборде."""
    return admin_ops.component_metrics()


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


@app.get("/api/admin/sip/register-status")
def api_sip_register_status(x_admin_token: str | None = Header(None)):
    """Состояние нативной SIP-регистрации (без AudioSocket)."""
    _check_admin(x_admin_token)
    import sip_phone
    return sip_phone.status()


@app.get("/api/admin/voip/callers")
def api_voip_callers(x_admin_token: str | None = Header(None)):
    """История VoIP по добавочным номерам (номер, сотрудник, число запросов)."""
    _check_admin(x_admin_token)
    return {"items": db.voip_callers()}


@app.get("/api/admin/voip/history")
def api_voip_history(caller: str, x_admin_token: str | None = Header(None)):
    """Все запросы и ответы VoIP по добавочному номеру (для раскрытия строки)."""
    _check_admin(x_admin_token)
    return {"items": db.voip_history(caller)}


@app.post("/api/admin/voip/delete")
def api_voip_delete(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    """Удалить историю VoIP по добавочному номеру."""
    _check_admin(x_admin_token)
    return {"ok": True, "removed": db.voip_delete_by_caller(payload.get("caller", ""))}


@app.post("/api/admin/sip/register-restart")
def api_sip_register_restart(x_admin_token: str | None = Header(None)):
    """Перерегистрировать SIP-аккаунт (после смены настроек)."""
    _check_admin(x_admin_token)
    import sip_phone
    return sip_phone.restart()


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


@app.get("/api/llm-activity")
def api_llm_activity(limit: int = 60):
    """Запросы к LLM в реальном времени: генерация ответов (чат/Телеграм), описание
    изображений vision-моделью и служебные вызовы (фильтр запроса, API-интент и т. п.) —
    что выполняется сейчас и недавно завершилось, с моделью, объёмом вывода и временем."""
    import llm_activity
    snap = llm_activity.snapshot(min(max(int(limit), 1), 200))
    try:
        import llm_queue
        snap["queue"] = llm_queue.stats()
    except Exception:
        snap["queue"] = {}
    return snap


@app.get("/api/llm-activity/item")
def api_llm_activity_item(id: str):
    """Полная запись одного вызова LLM (с полным текстом запроса) — для раскрытия строки."""
    import llm_activity
    return llm_activity.get(id)


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


@app.post("/api/admin/dns/test")
def api_dns_test(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    """Проверить статическую DNS-запись (резолвинг + доступность) и вернуть лог."""
    _check_admin(x_admin_token)
    return dns_override.test(payload.get("hostname", ""), payload.get("ip", ""))


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


@app.get("/api/admin/check-token")
def api_check_token(x_admin_token: str | None = Header(None)):
    """Проверка токена администратора: 200 — принят, 401 — неверный."""
    _check_admin(x_admin_token)
    return {"ok": True, "auth_required": bool(settings.get("ADMIN_TOKEN"))}


@app.post("/api/config")
def api_set_config(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"ok": True, "values": settings.update(payload)}


@app.post("/api/config/reset")
def api_reset_config(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return {"ok": True, "values": settings.reset()}


@app.get("/api/admin/settings/export")
def api_settings_export(secrets: int = 0, x_admin_token: str | None = Header(None)):
    """Снимок настроек (JSON) для переноса между стендами. secrets=1 — включить секреты."""
    _check_admin(x_admin_token)
    import json as _json
    data = settings.export_settings(include_secrets=bool(secrets))
    body = _json.dumps({"settings": data}, ensure_ascii=False, indent=2)
    from fastapi.responses import Response
    fn = "rag_settings.json"
    return Response(content=body, media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{fn}"'})


@app.post("/api/admin/settings/import")
def api_settings_import(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    """Применить настройки из снимка (JSON). Пустые секреты не затирают заданные."""
    _check_admin(x_admin_token)
    return settings.import_settings(payload)


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


@app.get("/api/admin/kb-search")
def admin_kb_search(q: str = "", x_admin_token: str | None = Header(None)):
    """Поиск по словам в графе базы знаний: возвращает источники (файлы), где
    встречаются слова запроса — для подсветки узлов в графе."""
    _check_admin(x_admin_token)
    return admin_ops.kb_search(q)


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


# ---- Клонирование голоса (XTTS): «обучение» вывода на образце ----
@app.get("/api/admin/tts/clone")
def admin_tts_clone_status(x_admin_token: str | None = Header(None)):
    """Готовность клонирования голоса: установлен ли пакет, есть ли образец."""
    _check_admin(x_admin_token)
    import tts
    return tts.xtts_status()


@app.post("/api/admin/tts/clone/sample")
async def admin_tts_clone_sample(file: UploadFile = File(...),
                                 x_admin_token: str | None = Header(None)):
    """Загрузить образец голоса (любой аудиоформат) — нормализуется в WAV 16 кГц/моно,
    сохраняется как образец, путь прописывается в XTTS_SAMPLE, движок → xtts."""
    _check_admin(x_admin_token)
    import tempfile
    import tts
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="пустой файл")
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()
        res = tts.save_voice_sample(tmp.name)
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("msg", "ошибка обработки образца"))
    # запомнить путь к образцу и включить движок клонирования
    settings.update({"XTTS_SAMPLE": res["path"], "TTS_ENGINE": "xtts"})
    return {"ok": True, "seconds": res.get("seconds", 0.0),
            "status": tts.xtts_status()}


@app.post("/api/admin/tts/clone/delete")
def admin_tts_clone_delete(x_admin_token: str | None = Header(None)):
    """Удалить образец голоса и вернуть движок TTS в auto."""
    _check_admin(x_admin_token)
    import tts
    p = tts.clone_sample_path()
    removed = False
    try:
        if p and os.path.exists(p):
            os.remove(p)
            removed = True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"не удалось удалить образец: {e}")
    changes = {"XTTS_SAMPLE": ""}
    if (settings.get("TTS_ENGINE") or "") == "xtts":
        changes["TTS_ENGINE"] = "auto"
    settings.update(changes)
    return {"ok": True, "removed": removed, "status": tts.xtts_status()}


@app.post("/api/admin/tts/clone/install")
def admin_tts_clone_install(x_admin_token: str | None = Header(None)):
    """Установить пакет Coqui XTTS (pip) в окружение сервиса."""
    _check_admin(x_admin_token)
    return admin_ops.xtts_install()


@app.post("/api/admin/tts/preview")
def admin_tts_preview(payload: dict = Body(default={}),
                      x_admin_token: str | None = Header(None)):
    """Синтезировать пробную фразу текущим движком/голосом и вернуть OGG для прослушки."""
    _check_admin(x_admin_token)
    import tempfile
    import tts
    text = (payload.get("text") or "Здравствуйте! Это пример синтезированного голоса.").strip()
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
    out.close()
    ok = False
    try:
        ok = tts.synthesize(text, out.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"синтез не удался: {e}")
    if not ok or not os.path.exists(out.name) or os.path.getsize(out.name) == 0:
        try:
            os.remove(out.name)
        except Exception:
            pass
        raise HTTPException(status_code=400,
                            detail="синтез недоступен: проверьте движок TTS, образец и ffmpeg")
    return FileResponse(out.name, media_type="audio/ogg", filename="preview.ogg",
                        background=BackgroundTask(lambda: _safe_unlink(out.name)))


def _safe_unlink(p: str):
    try:
        os.remove(p)
    except Exception:
        pass


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


@app.post("/api/admin/redis/install")
def admin_redis_install(x_admin_token: str | None = Header(None)):
    """Установить и запустить Redis-сервер средствами ОС, включить REDIS_ENABLED."""
    _check_admin(x_admin_token)
    return admin_ops.redis_install()


@app.get("/api/admin/milvus/status")
def admin_milvus_status(x_admin_token: str | None = Header(None)):
    """Состояние Milvus/Qdrant: установлен ли клиент, режим, доступность, число точек,
    активный бэкенд и ход миграции."""
    _check_admin(x_admin_token)
    return admin_ops.milvus_status()


@app.post("/api/admin/milvus/install")
def admin_milvus_install(payload: dict = Body(default={}),
                         x_admin_token: str | None = Header(None)):
    """Установить клиент Milvus (pip pymilvus) и зафиксировать режим/тип индекса.
    payload: {mode: lite|standalone, index_type: HNSW|GPU_CAGRA|...}."""
    _check_admin(x_admin_token)
    return admin_ops.milvus_install(mode=payload.get("mode"),
                                    index_type=payload.get("index_type"))


@app.post("/api/admin/milvus/migrate")
def admin_milvus_migrate(payload: dict = Body(...),
                         x_admin_token: str | None = Header(None)):
    """Запустить полную миграцию векторов. payload: {direction: to_milvus|to_qdrant}."""
    _check_admin(x_admin_token)
    return admin_ops.milvus_migrate(payload.get("direction", ""))


@app.post("/api/admin/milvus/switch")
def admin_milvus_switch(payload: dict = Body(...),
                        x_admin_token: str | None = Header(None)):
    """Переключить активную векторную базу. payload: {target: qdrant|milvus}."""
    _check_admin(x_admin_token)
    return admin_ops.milvus_switch(payload.get("target", ""))


@app.post("/api/admin/milvus/verify")
def admin_milvus_verify(x_admin_token: str | None = Header(None)):
    """Сверка Qdrant ↔ Milvus: число точек и совпадение результатов поиска."""
    _check_admin(x_admin_token)
    return admin_ops.milvus_verify()


@app.get("/api/admin/embed-finetune/info")
def admin_embed_ft_info(x_admin_token: str | None = Header(None)):
    """Данные о дообучении эмбеддингов: база, число пар из оценок, статус."""
    _check_admin(x_admin_token)
    return admin_ops.embed_finetune_info()


@app.post("/api/admin/embed-finetune")
def admin_embed_ft(payload: dict = Body(default={}),
                   x_admin_token: str | None = Header(None)):
    """Запустить дообучение эмбеддингов на оценках 👍 (фон). payload: {epochs, batch}."""
    _check_admin(x_admin_token)
    return admin_ops.embed_finetune(epochs=payload.get("epochs", 1),
                                    batch=payload.get("batch", 16))


@app.post("/api/admin/embed-finetune/activate")
def admin_embed_ft_activate(x_admin_token: str | None = Header(None)):
    """Переключить EMBED_MODEL на дообученную модель."""
    _check_admin(x_admin_token)
    return admin_ops.embed_finetune_activate()


@app.post("/api/admin/retrieval/eval")
def admin_retrieval_eval(x_admin_token: str | None = Header(None)):
    """Оценка качества поиска по золотому набору (hit@k, MRR)."""
    _check_admin(x_admin_token)
    return admin_ops.retrieval_eval()


@app.post("/api/admin/retrieval/autotune")
def admin_retrieval_autotune(payload: dict = Body(default={}),
                             x_admin_token: str | None = Header(None)):
    """Авто-подбор MIN_SCORE/TOP_K по золотому набору (фон). payload: {apply}."""
    _check_admin(x_admin_token)
    return admin_ops.retrieval_autotune(apply=bool(payload.get("apply")))


@app.get("/api/admin/retrieval/autotune/status")
def admin_retrieval_autotune_status(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.retrieval_tune_status()


@app.post("/api/admin/oda/install")
def admin_oda_install(x_admin_token: str | None = Header(None)):
    """Установить/проверить ODA File Converter (запасной конвертер DWG→DXF)."""
    _check_admin(x_admin_token)
    return admin_ops.oda_install()


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
    return admin_ops.ingest_web(payload.get("urls", []),
                                index=payload.get("index", True),
                                save=payload.get("save", True))


@app.post("/api/admin/web-delete")
def admin_web_delete(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.delete_web(payload.get("url", ""))


@app.post("/api/admin/web-reparse-fresh")
def admin_web_reparse_fresh(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    """Удалить файлы сайта и его чанки, сбросить инкремент и спарсить заново с нуля."""
    _check_admin(x_admin_token)
    return admin_ops.web_reparse_fresh(payload.get("url", ""),
                                       index=payload.get("index", True))


@app.post("/api/admin/web-parse/stop")
def admin_web_parse_stop(x_admin_token: str | None = Header(None)):
    """Кооперативно остановить текущий парсинг сайтов (между страницами/сайтами)."""
    _check_admin(x_admin_token)
    return admin_ops.stop_web_parse()


@app.post("/api/admin/web-excludes")
def admin_web_excludes(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    """Задать исключения по ключевым словам для сайта: URL, содержащие любое из слов,
    пропускаются при парсинге. payload: {url, keywords: строка или список}."""
    _check_admin(x_admin_token)
    return admin_ops.web_set_excludes(payload.get("url", ""),
                                      payload.get("keywords", ""))


@app.post("/api/admin/web-excludes-all")
def admin_web_excludes_all(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    """Глобальные исключения по ключевым словам — для ВСЕХ сайтов. Объединяются с
    исключениями конкретного сайта. payload: {keywords: строка или список}."""
    _check_admin(x_admin_token)
    return admin_ops.web_set_excludes_all(payload.get("keywords", ""))


@app.post("/api/admin/web-site-cfg")
def admin_web_site_cfg(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    """Пер-сайтовые настройки обхода (пустые поля = глобальные).
    payload: {url, cfg:{depth,max_pages,max_files,concurrency,same_domain,js_render}}."""
    _check_admin(x_admin_token)
    return admin_ops.web_set_site_cfg(payload.get("url", ""), payload.get("cfg", {}))


@app.post("/api/admin/web-auth")
def admin_web_auth(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    """Авторизация на сайт для парсинга (секреты). payload: {url, auth:{type, ...}}.
    type: none|basic(user,password)|cookie(cookie)|header(hname,hvalue)."""
    _check_admin(x_admin_token)
    return admin_ops.web_set_auth(payload.get("url", ""), payload.get("auth", {}))


@app.get("/api/admin/web-structure")
def admin_web_structure(x_admin_token: str | None = Header(None)):
    """Структура спарсенных сайтов в реальном времени: прогресс + дерево файлов/страниц
    со статусом в БД, чанками, типом, размером, LLM-описанием, датой и временем обработки."""
    _check_admin(x_admin_token)
    return admin_ops.web_structure()


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


@app.get("/api/admin/ingest-log/download")
def admin_ingest_log_download(id: int, token: str | None = None,
                              x_admin_token: str | None = Header(None)):
    """Скачать полный лог как текстовый файл. Токен можно передать заголовком или
    query-параметром ?token= (для прямой ссылки-скачивания из браузера)."""
    _check_admin(x_admin_token or token)
    from fastapi.responses import Response
    rec = db.ingest_log_get(id)
    if not rec:
        raise HTTPException(status_code=404, detail="лог не найден")
    from datetime import datetime as _dt
    day = _dt.fromtimestamp(rec.get("ts") or time.time()).strftime("%Y%m%d_%H%M%S")
    label = "".join(c if c.isalnum() else "_" for c in (rec.get("label") or "log"))[:40]
    fname = f"ingest_{label}_{day}.txt"
    body = (f"{rec.get('label','')} — {rec.get('day','')}\n"
            f"{rec.get('summary','')}\n{'=' * 60}\n{rec.get('log','')}")
    return Response(content=body, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


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


@app.get("/api/admin/index-log")
def admin_index_log(x_admin_token: str | None = Header(None)):
    """Живой лог индексации: статус, прогресс и хвост лог-файла (лёгкий, для частого опроса)."""
    _check_admin(x_admin_token)
    return admin_ops.index_log()


@app.post("/api/admin/reindex/stop")
def admin_reindex_stop(x_admin_token: str | None = Header(None)):
    """Остановить текущую индексацию (прибить процесс ingest и его воркеры)."""
    _check_admin(x_admin_token)
    return admin_ops.stop_reindex()


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
