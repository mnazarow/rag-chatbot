"""LLM-извлечение структурированных метаданных при индексации.

Дополняет rule-based metadata.py полями, которые сложно вытащить правилами:
product (продукт/услуга), topic (тема), doc_type (тип документа), а при
необходимости уточняет category. Дёшево на GPU (vLLM), поэтому включается
настройкой LLM_METADATA в админке. Вызывается один раз на документ.
"""
from __future__ import annotations
import json
import re

import llm_backend

_PROMPT = (
    "Ты извлекаешь структурированные метаданные из фрагмента корпоративного документа.\n"
    "Верни СТРОГО JSON с полями (значения по-русски):\n"
    '  "product"  — название продукта/услуги, о котором документ, или "" если неясно;\n'
    '  "topic"    — тема в 2-4 словах;\n'
    '  "category" — одно из: price, presentation, training, document;\n'
    '  "doc_type" — короткий тип (прайс-лист, презентация, инструкция, договор и т.п.).\n'
    "Только JSON, без пояснений.\n\nФРАГМЕНТ:\n"
)

_VALID_CAT = {"price", "presentation", "training", "document"}


def _parse_json(s: str) -> dict:
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def extract_structured(text_sample: str) -> dict:
    """Вернуть {product, topic, doc_type, category?} или {} при сбое."""
    try:
        out = llm_backend.chat(
            [{"role": "user", "content": _PROMPT + (text_sample or "")[:2000]}],
            temperature=0,
        )
    except Exception:
        return {}
    d = _parse_json(out)
    res: dict = {}
    for k in ("product", "topic", "doc_type"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            res[k] = v.strip()[:80]
    cat = d.get("category")
    if isinstance(cat, str) and cat in _VALID_CAT:
        res["category"] = cat
    return res
