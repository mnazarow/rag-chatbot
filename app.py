"""FastAPI: чат (стриминг) + API для дашборда, журнала, аналитики и админки.

Запуск:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
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
import db
import llm_backend
import admin_ops
import graph_rag
import loaders
import retriever
import remote
from ingest import chunk_text, SUPPORTED
from retriever import search, infer_category

app = FastAPI(title="Корпоративный RAG-чатбот")
_qdrant = QdrantClient(url=settings.get("QDRANT_URL"))


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []
    filters: dict | None = None
    debug: bool = False


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
@app.post("/chat")
async def chat(req: ChatRequest):
    t0 = time.time()

    # Движок ответов: LightRAG целиком, либо граф только для сводных вопросов (hybrid)
    engine = settings.get("ENGINE")
    use_lightrag_all = engine == "lightrag"
    use_graph_global = (settings.get("GRAPH_RAG") and req.filters is None
                        and graph_rag.is_global(req.question))
    if use_lightrag_all or use_graph_global:
        try:
            text = await graph_rag.answer(req.question)
            cat = "lightrag" if use_lightrag_all else "graph"

            async def gstream():
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
                                     retrieve_ms=0, gen_ms=lat)
                yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

            return StreamingResponse(gstream(), media_type="application/x-ndjson")
        except Exception as e:
            print(f"LightRAG недоступен, фолбэк на вектор: {e}")

    t_ret = time.time()
    hits = search(req.question, filters=req.filters)
    retrieve_ms = int((time.time() - t_ret) * 1000)
    category = (req.filters or {}).get("doc_category") or infer_category(req.question)

    if not hits:
        msg = "В доступных документах нет точного ответа на этот вопрос."
        rid = db.log_request(req.question, category, 0, 0.0,
                             int((time.time() - t0) * 1000), len(msg), False, [],
                             retrieve_ms=retrieve_ms, gen_ms=0)

        async def empty():
            yield json.dumps({"type": "answer", "text": msg}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "sources", "items": []}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

        return StreamingResponse(empty(), media_type="application/x-ndjson")

    context = prompts.build_context(hits)
    messages = [{"role": "system", "content": settings.get("SYSTEM_PROMPT")}]
    messages += req.history[-6:]
    messages.append({"role": "user",
                     "content": prompts.build_user_message(req.question, context)})

    sources = [{"source": h["source"], "page": h.get("page"),
                "category": h.get("doc_category"), "score": round(h["score"], 3)}
               for h in hits]

    async def stream():
        acc = []
        async for tok in llm_backend.chat_stream(
                messages, temperature=settings.get("TEMPERATURE"),
                model=settings.active_model()):
            acc.append(tok)
            yield json.dumps({"type": "answer", "text": tok}, ensure_ascii=False) + "\n"
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
                             retrieve_ms=retrieve_ms, gen_ms=max(0, latency - retrieve_ms))
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
                   history: str = Form("[]"), debug: str = Form("")):
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
                       int((time.time() - t0) * 1000), 0, False, [])
        return StreamingResponse(empty(), media_type="application/x-ndjson")

    context = prompts.build_context(hits)
    messages = [{"role": "system", "content": settings.get("SYSTEM_PROMPT")}]
    messages += hist[-6:]
    messages.append({"role": "user",
                     "content": prompts.build_user_message(question, context)})
    sources = [{"source": h["source"], "page": h.get("page"),
                "score": round(h["score"], 3)} for h in hits]

    async def stream():
        acc = []
        async for tok in llm_backend.chat_stream(
                messages, temperature=settings.get("TEMPERATURE"),
                model=settings.active_model()):
            acc.append(tok)
            yield json.dumps({"type": "answer", "text": tok}, ensure_ascii=False) + "\n"
        yield json.dumps({"type": "sources", "items": sources}, ensure_ascii=False) + "\n"
        latency = int((time.time() - t0) * 1000)
        if debug in ("1", "true", "on", "yes"):
            yield json.dumps({"type": "debug", "info": {
                "engine": "приложенный документ (rerank)", "mode": settings.current_mode(),
                "model": settings.active_model(), "backend": settings.get("LLM_BACKEND"),
                "timings": {"retrieve_ms": 0, "gen_ms": latency, "total_ms": latency},
                "params": _debug_params(), "chunks": _debug_chunks(hits)}}, ensure_ascii=False) + "\n"
        rid = db.log_request(question, "attached", len(hits), hits[0]["score"],
                             latency, len("".join(acc)), True, sources)
        yield json.dumps({"type": "meta", "id": rid}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


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
def admin_files_catalog(limit: int = 500, offset: int = 0, q: str = "",
                        x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.files_catalog(limit=limit, offset=offset, query=q)


@app.get("/api/admin/models")
def admin_models(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.list_models()


@app.get("/api/admin/available-models")
def admin_available_models(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.available_models()


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
    saved, skipped = [], []
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
        except Exception as e:
            skipped.append(f"{name} ({e})")
    return {"ok": True, "saved": saved, "skipped": skipped, "dir": str(dest)}


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
