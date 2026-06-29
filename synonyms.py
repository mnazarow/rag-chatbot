"""Словарь синонимов для улучшения понимания вопросов и формирования ответов.

Администратор задаёт слова и несколько синонимов к каждому (раздел «Синонимы»).
Когда функция включена, синонимы используются двумя способами:

  1. Расширение поискового запроса (`expand_query`) — если в вопросе встретилось
     слово из группы, к запросу добавляются остальные члены группы. Это повышает
     полноту поиска (особенно лексического BM25 и плотного по эмбеддингу).
  2. Подсказка модели (`hint`) — в сообщение к LLM добавляется список групп
     синонимов, встретившихся в вопросе, чтобы ответ учитывал равнозначность слов.

Группы кэшируются по сигнатуре таблицы (сбрасываются при любом изменении словаря).
Состояние вкл/выкл хранится в kv_store (ключ `syn.enabled`). Сопоставление слов —
по границам слова с поддержкой кириллицы (Unicode \\w).
"""
from __future__ import annotations
import hashlib
import json
import re
import threading

import db

KV_ENABLED = "syn.enabled"

_cache = {"sig": None, "groups": None}
_lock = threading.Lock()


def enabled() -> bool:
    return (db.kv_get(KV_ENABLED) or "0") == "1"


def set_enabled(value: bool) -> None:
    db.kv_set(KV_ENABLED, "1" if value else "0")


def signature() -> str:
    """Короткий хеш текущего словаря + флага включения (для кэш-ключей поиска)."""
    try:
        rows = db.syn_list()
    except Exception:
        rows = []
    raw = json.dumps([[r.get("id"), r.get("term"), r.get("syns")] for r in rows],
                     ensure_ascii=False, sort_keys=True)
    raw += "|" + ("1" if enabled() else "0")
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _load_groups() -> list[list[str]]:
    groups: list[list[str]] = []
    try:
        rows = db.syn_list()
    except Exception:
        rows = []
    for r in rows:
        members = [(r.get("term") or "").strip()]
        members += [s.strip() for s in (r.get("syns") or [])]
        members = [m for m in members if m]
        # уникальные, сохраняя порядок; группа значима при ≥2 членах
        seen, uniq = set(), []
        for m in members:
            k = m.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(m)
        if len(uniq) >= 2:
            groups.append(uniq)
    return groups


def _groups() -> list[list[str]]:
    sig = signature()
    if _cache["sig"] != sig:
        with _lock:
            _cache["groups"] = _load_groups()
            _cache["sig"] = sig
    return _cache["groups"] or []


def _present(member: str, text_lower: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(member.lower()) + r"(?!\w)",
                     text_lower) is not None


def matched_groups(text: str) -> list[list[str]]:
    """Группы синонимов, члены которых встретились в тексте (если функция включена)."""
    if not enabled() or not text:
        return []
    tl = text.lower()
    out = []
    for g in _groups():
        if any(_present(m, tl) for m in g):
            out.append(g)
    return out


def expand_query(text: str) -> str:
    """Дополнить запрос синонимами встретившихся слов (для поиска). Идемпотентно."""
    if not enabled() or not text:
        return text
    tl = text.lower()
    additions: list[str] = []
    for g in _groups():
        if any(_present(m, tl) for m in g):
            for m in g:
                if not _present(m, tl):
                    additions.append(m)
    if not additions:
        return text
    # убираем дубликаты добавлений, сохраняя порядок
    uniq = list(dict.fromkeys(additions))
    return text + " " + " ".join(uniq)


def hint(text: str) -> str:
    """Краткая подсказка для LLM по встретившимся группам синонимов (или '')."""
    gs = matched_groups(text)
    if not gs:
        return ""
    lines = "\n".join("- " + " = ".join(g) for g in gs)
    return ("СИНОНИМЫ (считай слова в каждой группе равнозначными при понимании "
            "вопроса и ответе):\n" + lines)
