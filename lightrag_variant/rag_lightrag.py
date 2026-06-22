"""Фабрика LightRAG, настроенного на локальные Ollama (LLM) и bge-m3 (эмбеддинги).

LightRAG строит граф знаний (сущности + связи) поверх документов и поддерживает
режимы поиска: naive (как обычный вектор), local, global, hybrid, mix.

Требуется:
  ollama pull qwen2.5:32b-instruct-q4_K_M
  ollama pull bge-m3
"""
from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

load_dotenv()

WORKING_DIR = Path(os.getenv("LIGHTRAG_DIR", "./lightrag_storage"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:32b-instruct-q4_K_M")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
EMBED_DIM = 1024


async def build_rag() -> LightRAG:
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={
            "host": OLLAMA_URL,
            "options": {"num_ctx": 16384, "temperature": 0.1},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model=EMBED_MODEL, host=OLLAMA_URL
            ),
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag
