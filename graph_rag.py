"""Лёгкий граф-RAG слой поверх основной системы (опциональный).

Идея: вектор+реранк остаётся основой и отвечает на точечные вопросы; а «сводные»
вопросы (сравнения, обзоры, связи между сущностями) маршрутизируются в граф
знаний LightRAG. Граф строится на ТОЙ ЖЕ инфраструктуре, что и основная система:
  - генерация: через llm_backend (vLLM на GPU или Ollama);
  - эмбеддинги: bge-m3 из retriever (CUDA/MPS).
Поэтому отдельный Ollama/конфиг не нужен.

LightRAG и тяжёлые зависимости импортируются ЛЕНИВО — модуль безопасно
импортируется, даже если LightRAG не установлен (граф просто выключен).

Включается настройкой GRAPH_RAG. Построение графа:
    python -m graph_rag ingest
"""
from __future__ import annotations
import asyncio
from pathlib import Path

import settings

WORKING_DIR = Path(__file__).resolve().parent / "graph_storage"

# вопросы, для которых граф обычно полезнее вектора
_GLOBAL_KW = [
    "сравни", "сравнен", "обзор", "все ", "всех", "суммируй", "сводк", "итого",
    "тенденц", "динамик", "связь", "связан", "соотнос", "перечисли все",
    "в целом", "общая картина", "across", "overview", "summary",
]


def is_global(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _GLOBAL_KW)


# ----- ленивое построение экземпляра LightRAG на нашем стеке -----
_rag = None
_lock = asyncio.Lock()


async def _llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    import llm_backend
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs += history_messages or []
    msgs.append({"role": "user", "content": prompt})
    return await asyncio.to_thread(llm_backend.chat, msgs, 0.1)


async def _embed_func(texts):
    from retriever import _embedder
    return await asyncio.to_thread(
        lambda: _embedder().encode(texts, normalize_embeddings=True))


def _build_sync():
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    return LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=_llm_func,
        llm_model_name=settings.get("LLM_MODEL"),
        embedding_func=EmbeddingFunc(
            embedding_dim=1024, max_token_size=8192, func=_embed_func),
    )


async def get_rag():
    global _rag
    async with _lock:
        if _rag is None:
            from lightrag.kg.shared_storage import initialize_pipeline_status
            _rag = _build_sync()
            await _rag.initialize_storages()
            await initialize_pipeline_status()
    return _rag


async def answer(question: str) -> str:
    from lightrag import QueryParam
    rag = await get_rag()
    return await rag.aquery(question, param=QueryParam(mode=settings.get("GRAPH_MODE")))


# ----- построение графа из папки документов -----
async def _ingest():
    import sys
    from loaders import load_file
    docs = Path(settings.get("DOCS_DIR")).expanduser()
    if not docs.exists():
        sys.exit(f"FATAL: DOCS_DIR не найдена: {docs}")
    # инициализация LightRAG — фатальная ошибка (нет пакета/не строится граф)
    try:
        rag = await get_rag()
    except ModuleNotFoundError as e:
        sys.exit(f"FATAL: LightRAG не установлен ({e}). Запустите «Построить граф» из админки "
                 "или установите зависимости вручную.")
    except Exception as e:
        sys.exit(f"FATAL: не удалось инициализировать граф (проверьте Qdrant/LLM/эмбеддер): {e}")

    files = [p for p in docs.rglob("*") if p.is_file()]
    print(f"Файлов: {len(files)}")
    ok = skip = 0
    errors = []
    for path in files:
        name = str(path.relative_to(docs))
        try:
            text = "\n\n".join(part["text"] for part in load_file(path)
                               if part["text"].strip())
            if not text.strip():
                skip += 1
                continue
            print(f"  + {name}")
            await rag.ainsert(text, ids=[name], file_paths=[name])
            ok += 1
        except KeyboardInterrupt:
            raise
        except Exception as e:
            errors.append((name, str(e)[:200]))
            print(f"  ! ошибка {name}: {e}")
            continue

    print(f"Граф построен в {WORKING_DIR}. Добавлено: {ok}, пропущено: {skip}, ошибок: {len(errors)}")
    if errors:
        print("Файлы с ошибками:")
        for s, e in errors[:50]:
            print(f"  - {s}: {e}")
    print(f"SUMMARY files_ok={ok} skipped={skip} errors={len(errors)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        asyncio.run(_ingest())
    else:
        print("Использование: python -m graph_rag ingest")
