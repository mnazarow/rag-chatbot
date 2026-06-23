"""FastAPI: чат (стриминг) + API для дашборда, журнала, аналитики и админки.

Запуск:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Body, UploadFile, File
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
from retriever import search, infer_category

app = FastAPI(title="Корпоративный RAG-чатбот")
_qdrant = QdrantClient(url=settings.get("QDRANT_URL"))


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []
    filters: dict | None = None


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
                db.log_request(req.question, cat, 1, 1.0,
                               int((time.time() - t0) * 1000), len(text), True, [])

            return StreamingResponse(gstream(), media_type="application/x-ndjson")
        except Exception as e:
            print(f"LightRAG недоступен, фолбэк на вектор: {e}")

    hits = search(req.question, filters=req.filters)
    category = (req.filters or {}).get("doc_category") or infer_category(req.question)

    if not hits:
        msg = "В доступных документах нет точного ответа на этот вопрос."
        db.log_request(req.question, category, 0, 0.0,
                       int((time.time() - t0) * 1000), len(msg), False, [])

        async def empty():
            yield json.dumps({"type": "answer", "text": msg}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "sources", "items": []}, ensure_ascii=False) + "\n"

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
        db.log_request(req.question, category, len(hits), hits[0]["score"],
                       int((time.time() - t0) * 1000), len("".join(acc)), True, sources)

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


@app.get("/api/admin/models")
def admin_models(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.list_models()


@app.post("/api/admin/pull-model")
def admin_pull_model(payload: dict = Body(...), x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.pull_model(payload.get("model", ""))


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


@app.post("/api/admin/reinstall-env")
def admin_reinstall_env(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.reinstall_env()


@app.post("/api/admin/restart")
def admin_restart(x_admin_token: str | None = Header(None)):
    _check_admin(x_admin_token)
    return admin_ops.restart()


# ============================ статика ============================
@app.get("/")
def index():
    return FileResponse("static/index.html")


try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    pass
