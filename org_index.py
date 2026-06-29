"""Индексация справочника сотрудников в базу знаний (Qdrant).

После каждой синхронизации структуры компании карточки сотрудников добавляются в
векторный индекс, чтобы ассистент мог отвечать на вопросы вроде «телефон Иванова»,
«кто руководит отделом ТКП», «почта менеджера проектных продаж в Казани» и т. п.

Индексация **инкрементальная**: для каждого сотрудника считается хеш карточки;
заэмбеддиваются и обновляются только новые/изменённые, удалённые из справочника —
убираются из индекса. Карта «ключ → хеш» хранится в kv_store (`org.index.map`).
Идентификатор точки Qdrant детерминированно выводится из ключа (uuid5), поэтому
повторная индексация перезаписывает ту же точку. Служебные поля payload:
`org=True`, `org_key=<ключ>` — по ним выполняется выборочное удаление/очистка.
"""
from __future__ import annotations
import hashlib
import json
import time
import uuid

import db

CATEGORY = "сотрудники"
KV_MAP = "org.index.map"
_NS = uuid.uuid5(uuid.NAMESPACE_URL, "rag-org-employees")


def _fio(e: dict) -> str:
    return " ".join(x for x in [e.get("last_name"), e.get("first_name"),
                                e.get("middle_name")] if x).strip()


def _source(e: dict) -> str:
    """Персональный источник сотрудника (для ссылок и графа). Каждый сотрудник —
    отдельная сущность; для различения тёзок добавляем должность/отдел."""
    fio = _fio(e)
    if not fio:
        return (e.get("email") or "Сотрудник")
    extra = " · ".join(x for x in [e.get("position"), e.get("department")] if x)
    return f"{fio} — {extra}" if extra else fio


def _key(e: dict) -> str:
    """Стабильный и уникальный ключ сотрудника. Нельзя опираться только на e-mail —
    в справочнике встречаются общие ящики (один email на нескольких людей), поэтому
    ключ строим по полному набору полей идентичности."""
    parts = [e.get("last_name"), e.get("first_name"), e.get("middle_name"),
             e.get("position"), e.get("department"), e.get("location"), e.get("email")]
    s = "|".join((p or "").strip() for p in parts).lower()
    return ("emp:" + s) if s.strip("|") else ""


def _point_id(key: str) -> str:
    return str(uuid.uuid5(_NS, key))


def _card(e: dict) -> str:
    fio = " ".join(x for x in [e.get("last_name"), e.get("first_name"),
                               e.get("middle_name")] if x).strip()
    lines = [f"Сотрудник: {fio}" if fio else "Сотрудник"]
    add = lines.append
    if e.get("position"):
        add(f"Должность: {e['position']}")
    if e.get("department"):
        add(f"Отдел: {e['department']}")
    if e.get("location"):
        add(f"Локация: {e['location']}")
    if e.get("phone_work"):
        add(f"Телефон рабочий: {e['phone_work']}")
    if e.get("phone_ext"):
        add(f"Телефон внутренний (добавочный): {e['phone_ext']}")
    if e.get("phone_mobile"):
        add(f"Телефон мобильный: {e['phone_mobile']}")
    if e.get("email"):
        add(f"Электронная почта: {e['email']}")
    if e.get("suppliers"):
        add(f"Поставщики: {e['suppliers']}")
    if not e.get("active"):
        wu = (e.get("work_until") or "").strip()
        add("Статус: не работает" + (f" (работал до {wu})" if wu else ""))
    else:
        add("Статус: работает")
    return "\n".join(lines)


def _bump():
    try:
        import cache
        cache.bump("index")
        cache.bump("stats")
    except Exception:
        pass


def sync_index(rows: list[dict]) -> dict:
    """Привести индекс сотрудников в соответствие со справочником. Инкрементально:
    добавляет/обновляет изменённые карточки, удаляет исчезнувшие. Возвращает
    {added, updated, removed, total}."""
    import retriever
    from qdrant_client.http import models as qm

    try:
        prev = json.loads(db.kv_get(KV_MAP) or "{}")
        if not isinstance(prev, dict):
            prev = {}
    except Exception:
        prev = {}

    cur: dict = {}
    cards: dict = {}          # key -> (card, payload_extra)
    to_embed: list = []       # (key, card, emp)
    for e in rows:
        k = _key(e)
        if not k:
            continue
        card = _card(e)
        h = hashlib.sha1(card.encode("utf-8")).hexdigest()
        cur[k] = h
        cards[k] = (card, e)
        if prev.get(k) != h:
            to_embed.append((k, card, e))

    removed = [k for k in prev if k not in cur]
    added = [k for k, _c, _e in to_embed if k not in prev]
    updated = [k for k, _c, _e in to_embed if k in prev]

    # upsert новых/изменённых
    if to_embed:
        try:
            batch = max(1, int(settings_batch()))
        except Exception:
            batch = 32
        vecs = retriever._embedder().encode([c for _k, c, _e in to_embed],
                                            normalize_embeddings=True, batch_size=batch)
        day = time.strftime("%Y-%m-%d")
        pts = []
        for (k, card, e), v in zip(to_embed, vecs):
            pts.append(qm.PointStruct(
                id=_point_id(k),
                vector=(v.tolist() if hasattr(v, "tolist") else list(v)),
                payload={"text": card, "source": _source(e), "page": None,
                         "doc_category": CATEGORY, "indexed_at": day,
                         "org": True, "org_key": k,
                         "emp_name": _fio(e), "emp_email": e.get("email") or "",
                         "department": e.get("department") or ""}))
        retriever._client.upsert(retriever._COLLECTION, wait=False, points=pts)

    # удаление исчезнувших
    if removed:
        retriever._client.delete(
            retriever._COLLECTION,
            points_selector=qm.PointIdsList(points=[_point_id(k) for k in removed]))

    db.kv_set(KV_MAP, json.dumps(cur, ensure_ascii=False))
    _bump()
    return {"added": len(added), "updated": len(updated),
            "removed": len(removed), "total": len(cur)}


def settings_batch():
    import settings
    return settings.get("EMBED_BATCH") or 32


def clear() -> int:
    """Убрать все карточки сотрудников из индекса и очистить карту."""
    try:
        import retriever
        from qdrant_client.http import models as qm
        retriever._client.delete(
            retriever._COLLECTION,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(
                    key="org", match=qm.MatchValue(value=True))])))
    except Exception as e:
        print(f"[org_index] очистка индекса: {e}")
    db.kv_set(KV_MAP, "{}")
    _bump()
    return 0
