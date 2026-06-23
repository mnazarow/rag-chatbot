"""Операции администратора, вызываемые из веб-панели:

  - status()    : доступность Qdrant и LLM, число чанков, статус индексации
  - reindex()   : запуск переиндексации (фоновый процесс ingest.py)
  - apply_llm() : перезапуск контейнера vLLM с текущей моделью (GPU-вариант)
  - restart()   : перезапуск самого сервиса (через systemd Restart=always)

Все вызовы защищены токеном на уровне app.py. Лёгкие зависимости (httpx,
subprocess) — без импорта тяжёлых ML-библиотек.
"""
from __future__ import annotations
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import importlib.util as _iu
import json as _json

import httpx

import settings
import db

ROOT = Path(__file__).resolve().parent

_job = {"running": False, "started": None, "finished": None, "ok": None, "log": ""}
_ft_job = {"running": False, "started": None, "finished": None, "ok": None, "log": ""}
_graph_job = {"running": False, "started": None, "finished": None, "ok": None, "log": ""}

# зависимости LightRAG (как в lightrag_variant/requirements-lightrag.txt)
_LIGHTRAG_DEPS = ["lightrag-hku==1.3.0", "nano-vectordb==0.0.4.3",
                  "tiktoken==0.8.0", "networkx==3.4.2"]

# поддерживаемые типы — для подсказки «сколько документов в папке»
_SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv", ".txt", ".md",
              ".html", ".htm", ".mp3", ".wav", ".m4a", ".aac", ".mp4", ".mov",
              ".mkv", ".webm"}


def status() -> dict:
    out: dict = {}
    # Qdrant + число чанков через REST
    try:
        coll = settings.get("QDRANT_COLLECTION")
        r = httpx.get(f"{settings.get('QDRANT_URL')}/collections/{coll}", timeout=3)
        out["qdrant"] = r.status_code == 200
        out["chunks"] = (r.json().get("result", {}) or {}).get("points_count", 0) \
            if r.status_code == 200 else 0
    except Exception:
        out["qdrant"] = False
        out["chunks"] = 0
    # LLM
    try:
        if settings.get("LLM_BACKEND") == "openai":
            r = httpx.get(
                f"{settings.get('LLM_BASE_URL')}/models",
                headers={"Authorization": f"Bearer {settings.get('LLM_API_KEY')}"},
                timeout=3,
            )
        else:
            r = httpx.get(f"{settings.get('OLLAMA_URL')}/api/tags", timeout=3)
        out["llm"] = r.status_code == 200
    except Exception:
        out["llm"] = False
    out["backend"] = settings.get("LLM_BACKEND")
    out["index_job"] = dict(_job)
    out["finetune_job"] = dict(_ft_job)
    out["adapter_ready"] = (ROOT / "finetune" / "adapter").exists()
    out["use_finetuned"] = bool(settings.get("USE_FINETUNED"))
    out["graph_job"] = dict(_graph_job)
    out["graph_ready"] = (ROOT / "graph_storage").exists()
    out["engine"] = settings.get("ENGINE")
    return out


def reindex(reset: bool = False) -> dict:
    if _job["running"]:
        return {"ok": False, "msg": "индексация уже идёт"}

    def run():
        _job.update(running=True, started=time.time(), finished=None, ok=None, log="")
        cmd = [sys.executable, "ingest.py"] + (["--reset"] if reset else [])
        try:
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                               timeout=24 * 3600)
            _job["log"] = (p.stdout[-4000:] + "\n" + p.stderr[-2000:]).strip()
            _job["ok"] = p.returncode == 0
        except Exception as e:
            _job["ok"] = False
            _job["log"] = str(e)
        _job["running"] = False
        _job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": "индексация запущена"}


def apply_llm() -> dict:
    """Перезапуск vLLM с текущей моделью из настроек (только GPU-вариант)."""
    script = ROOT / "gpu_variant" / "apply_llm.sh"
    env_file = ROOT / "gpu_variant" / ".env"
    if not script.exists():
        return {"ok": False, "msg": "apply_llm.sh не найден — это операция только для GPU-варианта"}
    _update_env(env_file, {
        "VLLM_MODEL": settings.get("VLLM_MODEL"),
        "VLLM_MAX_LEN": settings.get("VLLM_MAX_LEN"),
        "VLLM_TP": settings.get("VLLM_TP"),
        "LLM_MODEL": settings.get("LLM_MODEL"),
    })
    try:
        p = subprocess.run(["bash", str(script)], cwd=ROOT / "gpu_variant",
                           capture_output=True, text=True, timeout=1800)
        return {"ok": p.returncode == 0,
                "msg": (p.stdout + p.stderr)[-1500:].strip() or "vLLM перезапускается"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


def build_graph() -> dict:
    """Установить LightRAG и построить граф знаний из документов (в фоне).
    Эквивалент gpu_variant/setup_hybrid.sh + `python -m graph_rag ingest`."""
    if _graph_job["running"]:
        return {"ok": False, "msg": "построение графа уже идёт"}

    def run():
        _graph_job.update(running=True, started=time.time(), finished=None, ok=None, log="")
        log = []
        try:
            log.append("[1/2] Установка LightRAG...")
            p1 = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                                 *_LIGHTRAG_DEPS],
                                cwd=ROOT, capture_output=True, text=True, timeout=3600)
            log.append((p1.stdout[-1000:] + p1.stderr[-1500:]).strip())
            if p1.returncode != 0:
                raise RuntimeError("не удалось установить LightRAG")
            log.append("[2/2] Построение графа (graph_rag ingest)...")
            p2 = subprocess.run([sys.executable, "-m", "graph_rag", "ingest"],
                                cwd=ROOT, capture_output=True, text=True, timeout=24 * 3600)
            log.append((p2.stdout[-4000:] + p2.stderr[-2000:]).strip())
            _graph_job["ok"] = p2.returncode == 0
        except Exception as e:
            _graph_job["ok"] = False
            log.append(str(e))
        _graph_job["log"] = "\n".join(log).strip()
        _graph_job["running"] = False
        _graph_job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": "построение графа запущено (может занять часы)"}


def finetune() -> dict:
    """Запустить пайплайн дообучения (датасет + LoRA) в фоне."""
    script = ROOT / "finetune" / "run_pipeline.sh"
    if not script.exists():
        return {"ok": False, "msg": "run_pipeline.sh не найден"}
    if _ft_job["running"]:
        return {"ok": False, "msg": "дообучение уже идёт"}

    def run():
        _ft_job.update(running=True, started=time.time(), finished=None, ok=None, log="")
        try:
            p = subprocess.run(["bash", str(script)], cwd=ROOT,
                               capture_output=True, text=True, timeout=24 * 3600)
            _ft_job["log"] = (p.stdout[-4000:] + "\n" + p.stderr[-3000:]).strip()
            _ft_job["ok"] = p.returncode == 0
        except Exception as e:
            _ft_job["ok"] = False
            _ft_job["log"] = str(e)
        _ft_job["running"] = False
        _ft_job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": "дообучение запущено (может занять часы)"}


def apply_finetuned() -> dict:
    """Перезапустить vLLM с LoRA-адаптером (GPU)."""
    script = ROOT / "gpu_variant" / "apply_finetuned.sh"
    env_file = ROOT / "gpu_variant" / ".env"
    if not script.exists():
        return {"ok": False, "msg": "apply_finetuned.sh не найден — операция только для GPU-варианта"}
    if not (ROOT / "finetune" / "adapter").exists():
        return {"ok": False, "msg": "адаптер не найден — сначала запустите дообучение"}
    _update_env(env_file, {
        "VLLM_MODEL": settings.get("VLLM_MODEL"),
        "VLLM_MAX_LEN": settings.get("VLLM_MAX_LEN"),
        "VLLM_TP": settings.get("VLLM_TP"),
        "FINETUNED_MODEL": settings.get("FINETUNED_MODEL"),
    })
    try:
        p = subprocess.run(["bash", str(script)], cwd=ROOT / "gpu_variant",
                           capture_output=True, text=True, timeout=1800)
        return {"ok": p.returncode == 0,
                "msg": (p.stdout + p.stderr)[-1500:].strip() or "vLLM перезапускается с адаптером"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


def restart() -> dict:
    """Завершить процесс — systemd (Restart=always) поднимет его заново."""
    def killer():
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=killer, daemon=True).start()
    return {"ok": True, "msg": "перезапуск сервиса через ~1 сек..."}


def _qcount(base: str, coll: str, flt: dict) -> int:
    try:
        r = httpx.post(f"{base}/collections/{coll}/points/count", timeout=4,
                       json={"filter": flt, "exact": True})
        if r.status_code == 200:
            return r.json().get("result", {}).get("count", 0)
    except Exception:
        pass
    return 0


def _dir_size_mb(path: Path) -> float:
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / 1e6, 1)
    except Exception:
        return 0.0


def system_info() -> dict:
    """Полная сводка по компонентам: Qdrant, граф (LightRAG), дообучение, hybrid+."""
    coll = settings.get("QDRANT_COLLECTION")
    qbase = settings.get("QDRANT_URL")

    # ---- Qdrant ----
    qd: dict = {"online": False}
    try:
        r = httpx.get(f"{qbase}/collections/{coll}", timeout=4)
        if r.status_code == 200:
            res = r.json().get("result", {}) or {}
            params = (res.get("config", {}) or {}).get("params", {}) or {}
            vec = params.get("vectors", {}) or {}
            qd = {
                "online": True, "collection": coll,
                "points": res.get("points_count", 0),
                "segments": res.get("segments_count", 0),
                "status": res.get("status"),
                "vector_size": vec.get("size"),
                "distance": vec.get("distance"),
                "payload_fields": sorted((res.get("payload_schema") or {}).keys()),
                "by_category": {
                    cat: _qcount(qbase, coll,
                                 {"must": [{"key": "doc_category",
                                            "match": {"value": cat}}]})
                    for cat in ("price", "presentation", "training", "document")
                },
                "with_product": _qcount(qbase, coll,
                                        {"must_not": [{"is_empty": {"key": "product"}}]}),
            }
    except Exception as e:
        qd = {"online": False, "error": str(e)}

    # ---- LightRAG / граф ----
    gdir = ROOT / "graph_storage"
    graph: dict = {
        "ready": gdir.exists(),
        "installed": _iu.find_spec("lightrag") is not None,
        "engine": settings.get("ENGINE"),
        "mode": settings.get("GRAPH_MODE"),
        "job": dict(_graph_job),
    }
    if gdir.exists():
        def _jlen(name, key=None):
            f = gdir / name
            if not f.exists():
                return None
            try:
                d = _json.loads(f.read_text(encoding="utf-8"))
                v = d.get(key) if key else d
                return len(v) if hasattr(v, "__len__") else None
            except Exception:
                return None
        graph["entities"] = _jlen("vdb_entities.json", "data")
        graph["relations"] = _jlen("vdb_relationships.json", "data")
        graph["chunks"] = _jlen("kv_store_text_chunks.json")
        graph["docs"] = _jlen("kv_store_full_docs.json")
        gml = gdir / "graph_chunk_entity_relation.graphml"
        if gml.exists() and graph.get("entities") is None:
            t = gml.read_text(errors="ignore")
            graph["entities"] = t.count("<node ")
            graph["relations"] = t.count("<edge ")
        graph["size_mb"] = _dir_size_mb(gdir)

    # ---- Дообучение ----
    adapter = ROOT / "finetune" / "adapter"
    ds = ROOT / "finetune" / "data" / "train.jsonl"
    ft: dict = {
        "adapter_ready": adapter.exists(),
        "use_finetuned": bool(settings.get("USE_FINETUNED")),
        "finetuned_model": settings.get("FINETUNED_MODEL"),
        "base_model": settings.get("VLLM_MODEL"),
        "deps_installed": (_iu.find_spec("peft") is not None
                           and _iu.find_spec("trl") is not None),
        "job": dict(_ft_job),
    }
    if ds.exists():
        try:
            ft["dataset_pairs"] = sum(1 for _ in ds.open(encoding="utf-8"))
        except Exception:
            ft["dataset_pairs"] = None
    if adapter.exists():
        ft["adapter_config"] = (adapter / "adapter_config.json").exists()
        ft["adapter_size_mb"] = _dir_size_mb(adapter)

    # ---- Hybrid+ ----
    hybrid = {
        "mode": settings.current_mode(),
        "LLM_METADATA": bool(settings.get("LLM_METADATA")),
        "SMART_FILTER": bool(settings.get("SMART_FILTER")),
        "GRAPH_RAG": bool(settings.get("GRAPH_RAG")),
        "AUTO_FILTER": bool(settings.get("AUTO_FILTER")),
        "GRAPH_MODE": settings.get("GRAPH_MODE"),
    }

    return {"qdrant": qd, "graph": graph, "finetune": ft,
            "hybrid": hybrid, "usage": db.engine_usage()}


def browse(path: str | None = None) -> dict:
    """Обзор папок на сервере для выбора DOCS_DIR.
    Возвращает текущий путь, родителя, список подпапок и число документов в папке."""
    try:
        base = Path(path).expanduser() if path else Path(settings.get("DOCS_DIR")).expanduser()
        if not base.exists() or not base.is_dir():
            base = Path.home()
        base = base.resolve()
    except Exception:
        base = Path.home()

    dirs = []
    n_docs = 0
    try:
        for p in sorted(base.iterdir(), key=lambda x: x.name.lower()):
            if p.name.startswith("."):
                continue
            if p.is_dir():
                dirs.append(p.name)
            elif p.suffix.lower() in _SUPPORTED:
                n_docs += 1
    except PermissionError:
        pass

    return {
        "path": str(base),
        "parent": str(base.parent),
        "dirs": dirs,
        "docs_here": n_docs,        # документов непосредственно в этой папке
    }


def _update_env(path: Path, kv: dict) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    seen = set()
    out = []
    for ln in lines:
        key = ln.split("=", 1)[0] if "=" in ln else None
        if key in kv:
            out.append(f"{key}={kv[key]}")
            seen.add(key)
        else:
            out.append(ln)
    for k, v in kv.items():
        if k not in seen:
            out.append(f"{k}={v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
