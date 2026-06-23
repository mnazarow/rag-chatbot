"""Умные фильтры: извлечение структурированных условий из вопроса через LLM.

Пример: «сколько стоила подписка Pro в мае 2024» ->
{doc_category: price, product: "Pro", date: "2024-05"}.
Включается настройкой SMART_FILTER. Фильтры применяются в Qdrant; если они
дают слишком мало результатов, retriever автоматически откатывается без фильтра.
"""
from __future__ import annotations
import json
import re

import llm_backend

_PROMPT = (
    "Определи фильтры для поиска по корпоративной базе документов из вопроса.\n"
    "Верни СТРОГО JSON, включая ТОЛЬКО явно упомянутые поля:\n"
    '  "doc_category": одно из price|presentation|training|document;\n'
    '  "product": название продукта/услуги;\n'
    '  "date": "ГГГГ-ММ", если указан конкретный период.\n'
    "Если конкретики нет — верни {}. Только JSON.\n\nВОПРОС:\n"
)
_VALID_CAT = {"price", "presentation", "training", "document"}


def extract(question: str) -> dict:
    try:
        out = llm_backend.chat(
            [{"role": "user", "content": _PROMPT + question}], temperature=0)
    except Exception:
        return {}
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except Exception:
        return {}
    res: dict = {}
    if d.get("doc_category") in _VALID_CAT:
        res["doc_category"] = d["doc_category"]
    if isinstance(d.get("product"), str) and d["product"].strip():
        res["product"] = d["product"].strip()[:80]
    if isinstance(d.get("date"), str) and re.match(r"20\d\d-\d\d", d["date"]):
        res["date"] = d["date"][:7]
    return res
