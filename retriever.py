"""Гибридный поиск: плотный (Qdrant) + лексический (BM25) -> реранк bge-reranker.

Модели грузятся один раз при импорте (синглтоны).
"""
from __future__ import annotations
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from rank_bm25 import BM25Okapi

import settings

# клиенты/модели создаются при старте процесса из текущих настроек
# (поля scope=restart применяются после перезапуска сервиса)
_client = QdrantClient(url=settings.get("QDRANT_URL"))
_COLLECTION = settings.get("QDRANT_COLLECTION")


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.get("EMBED_MODEL"), device=settings.get("DEVICE"))


@lru_cache(maxsize=1)
def _reranker() -> FlagReranker:
    return FlagReranker(settings.get("RERANK_MODEL"), use_fp16=True)


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


# слова в вопросе -> категория документа (мягкий интент-роутер)
_INTENT = {
    "price": ["цена", "цены", "стоит", "стоимость", "прайс", "тариф", "сколько стоит",
              "почём", "почем", "расценк"],
    "training": ["обучен", "тренинг", "вебинар", "курс", "урок", "онбординг",
                 "как научиться", "инструктаж"],
    "presentation": ["презентац", "слайд", "питч"],
}


def infer_category(question: str) -> str | None:
    """Угадать категорию по вопросу. None — фильтр не применять."""
    q = question.lower()
    for cat, kws in _INTENT.items():
        if any(kw in q for kw in kws):
            return cat
    return None


def _build_filter(filters: dict | None) -> qm.Filter | None:
    if not filters:
        return None
    must = [qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
            for k, v in filters.items() if v]
    return qm.Filter(must=must) if must else None


def _dense_search(qvec, qfilter):
    res = _client.query_points(
        _COLLECTION,
        query=qvec.tolist(),
        query_filter=qfilter,
        limit=settings.get("TOP_K_RETRIEVE"),
        with_payload=True,
    ).points
    return [
        {"text": p.payload["text"], "source": p.payload["source"],
         "page": p.payload.get("page"), "doc_category": p.payload.get("doc_category"),
         "date": p.payload.get("date"), "dense": p.score}
        for p in res
    ]


def search(question: str, filters: dict | None = None,
           auto_filter: bool | None = None) -> list[dict]:
    """filters: явные фильтры из API, напр. {'doc_category': 'price', 'date': '2024-05'}.
    auto_filter: если явных нет — попробовать угадать категорию по вопросу
    (по умолчанию берётся из рантайм-настроек AUTO_FILTER)."""
    if auto_filter is None:
        auto_filter = settings.get("AUTO_FILTER")
    qvec = _embedder().encode([question], normalize_embeddings=True)[0]

    # 1) определяем фильтр: явный приоритетнее авто-угаданного
    if filters is None and auto_filter:
        cat = infer_category(question)
        filters = {"doc_category": cat} if cat else None

    # 2) плотный поиск с фильтром; если фильтр дал мало — фолбэк без фильтра
    cands = _dense_search(qvec, _build_filter(filters))
    if len(cands) < 3 and filters:
        cands = _dense_search(qvec, None)
    if not cands:
        return []

    # 2) лексический реранж по BM25 внутри кандидатов (дешёвый гибрид)
    bm25 = BM25Okapi([_tokenize(c["text"]) for c in cands])
    bm_scores = bm25.get_scores(_tokenize(question))
    for c, s in zip(cands, bm_scores):
        c["bm25"] = float(s)

    # 3) кросс-энкодер реранк — финальная релевантность 0..1
    pairs = [[question, c["text"]] for c in cands]
    scores = _reranker().compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    for c, s in zip(cands, scores):
        c["score"] = float(s)

    cands.sort(key=lambda c: c["score"], reverse=True)
    min_score = settings.get("MIN_SCORE")
    top_k = settings.get("TOP_K_RERANK")
    top = [c for c in cands if c["score"] >= min_score][:top_k]
    return top
