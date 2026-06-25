"""KAG — Knowledge Augmented Generation (знание-усиленная генерация).

Движок ответов поверх существующего стека (вектор+реранк, опц. граф LightRAG):

  1) Декомпозиция вопроса на под-вопросы (логические шаги) — опционально.
  2) Мультихоп-поиск: по каждому под-вопросу — вектор+реранк (retriever.search).
  3) Взаимное индексирование (mutual indexing): объединяем найденные фрагменты и
     переоцениваем их кросс-энкодером под ИСХОДНЫЙ вопрос.
  4) Аугментация знаниями графа знаний (LightRAG) — опционально.
  5) Генерация финального ответа по собранному знанию со ссылками на источники.

Граф/LightRAG не обязательны: при их отсутствии KAG работает как мультихоп-RAG.
Все тяжёлые синхронные вызовы (поиск, реранк, генерация) выполняются в пуле потоков,
чтобы не блокировать событийный цикл FastAPI.
"""
from __future__ import annotations
import asyncio
import re
import time

import settings
import prompts
import retriever


def _g(key, default=None):
    v = settings.get(key)
    return default if v is None else v


async def _decompose(question: str, max_hops: int) -> list:
    """LLM раскладывает вопрос на самостоятельные под-вопросы (не больше max_hops)."""
    import llm_backend
    sys_p = ("Разбей вопрос пользователя на самостоятельные под-вопросы для поиска по "
             f"корпоративным документам (не более {max_hops}). Если вопрос простой — "
             "верни его как есть. Выведи ТОЛЬКО под-вопросы, по одному на строку, без "
             "нумерации и пояснений.")
    try:
        out = await asyncio.to_thread(
            llm_backend.chat,
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": question}], 0.0)
    except Exception:
        return [question]
    subs = []
    for line in (out or "").splitlines():
        s = re.sub(r"^\s*\d+[.)]\s*", "", line.strip(" -•\t").strip())
        if len(s) >= 4:
            subs.append(s)
    # всегда ищем и по исходному вопросу; уникализируем
    subs = [question] + subs
    seen, uniq = set(), []
    for s in subs:
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        uniq.append(s)
    return uniq[:max_hops + 1]


async def _graph_knowledge(question: str) -> str:
    """Выжимка из графа знаний (LightRAG); мягкий фолбэк на пустую строку."""
    try:
        import graph_rag
        from lightrag import QueryParam
        rag = await graph_rag.get_rag()
        return await rag.aquery(
            question, param=QueryParam(mode=_g("KAG_GRAPH_MODE", "local"))) or ""
    except Exception as e:
        print(f"[kag] знания графа недоступны: {e}")
        return ""


async def answer(question: str, history=None, trace=None) -> dict:
    """Вернуть {text, hits, sub, graph, answered}. trace (если список) наполняется
    этапами {key, ms, info} для анимации конвейера."""
    t0 = time.time()
    max_hops = int(_g("KAG_MAX_HOPS", 3))
    per_hop = int(_g("KAG_CHUNKS_PER_HOP", 4))
    ctx_chunks = int(_g("KAG_CONTEXT_CHUNKS", 8))

    # 1) декомпозиция
    subs = [question]
    if _g("KAG_DECOMPOSE", True):
        subs = await _decompose(question, max_hops)
    if trace is not None:
        trace.append({"key": "kag_decompose", "ms": int((time.time() - t0) * 1000),
                      "info": {"sub": subs, "hops": len(subs)}})

    # 2) мультихоп-поиск + 3) объединение пула
    t = time.time()
    pool, seen = [], set()
    for sq in subs:
        try:
            hits = await asyncio.to_thread(retriever.search, sq) or []
        except Exception:
            hits = []
        for h in hits[:per_hop]:
            key = (h.get("source"), h.get("page"), (h.get("text") or "")[:60])
            if key in seen:
                continue
            seen.add(key)
            pool.append(h)
    # взаимное индексирование: переоценка объединённого пула под исходный вопрос
    if pool and _g("KAG_MUTUAL_INDEX", True):
        try:
            pool = await asyncio.to_thread(
                retriever.rerank_texts, question, pool, ctx_chunks)
        except Exception:
            pool = pool[:ctx_chunks]
    else:
        pool = pool[:ctx_chunks]
    if trace is not None:
        trace.append({"key": "kag_retrieve", "ms": int((time.time() - t) * 1000),
                      "info": {"hops": len(subs), "chunks": len(pool)}})

    # 4) знания графа
    graph_txt = ""
    if _g("KAG_GRAPH", False):
        t = time.time()
        graph_txt = await _graph_knowledge(question)
        if trace is not None:
            trace.append({"key": "kag_graph", "ms": int((time.time() - t) * 1000),
                          "info": {"chars": len((graph_txt or "").strip())}})

    if not pool and not (graph_txt or "").strip():
        return {"text": "В доступных документах нет точного ответа на этот вопрос.",
                "hits": [], "sub": subs, "graph": False, "answered": False}

    # 5) генерация по собранному знанию
    t = time.time()
    context = prompts.build_context(pool)
    parts = []
    if context:
        parts.append("КОНТЕКСТ (фрагменты документов):\n" + context)
    if (graph_txt or "").strip():
        parts.append("ЗНАНИЯ ГРАФА:\n" + graph_txt.strip())
    cite = ("\n6. Обязательно укажи источники [источник: имя_файла, стр. N]."
            if _g("KAG_REQUIRE_CITATIONS", True) else "")
    user_msg = ("\n\n".join(parts) + f"\n\nВОПРОС СОТРУДНИКА:\n{question}\n\n"
                "Дай ответ по правилам выше." + cite)
    sys_prompt = _g("KAG_SYSTEM_PROMPT") or prompts.KAG_SYSTEM_PROMPT
    messages = [{"role": "system", "content": sys_prompt}]
    messages += (history or [])[-6:]
    messages.append({"role": "user", "content": user_msg})

    import llm_backend
    text = await asyncio.to_thread(
        llm_backend.chat, messages, float(_g("KAG_TEMPERATURE", 0.1)),
        settings.active_model())
    if trace is not None:
        trace.append({"key": "kag_generate", "ms": int((time.time() - t) * 1000),
                      "info": {"model": settings.active_model()}})
    return {"text": text, "hits": pool, "sub": subs,
            "graph": bool((graph_txt or "").strip()), "answered": True}
