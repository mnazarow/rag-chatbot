"""Единый фасад векторной базы: Qdrant ⇄ Milvus.

Весь остальной код (поиск, индексация, статистика) обращается к векторной базе
ТОЛЬКО через этот модуль. Активный бэкенд выбирается настройкой VECTOR_BACKEND
(«qdrant» — по умолчанию, или «milvus»). Благодаря единому API можно установить
Milvus, полностью перенести в него данные и переключить поиск — не трогая логику
RAG, а также в любой момент вернуться на Qdrant.

Нейтральный формат данных:
  point  = {"id": str, "vector": list[float], "payload": dict}
  filter = {field: value, ...}   — логическое И равенств (value != None учитывается)
  hit    = {"id": str, "score": float, "payload": dict}

Qdrant реализован через REST (те же эндпоинты, что и раньше), поэтому при
backend=qdrant поведение идентично прежнему. Milvus — через pymilvus MilvusClient
(режимы Lite и Standalone, индексы CPU HNSW / GPU CAGRA), импортируется лениво.
"""
from __future__ import annotations
import threading
import time
from pathlib import Path

import httpx

import settings

# ------------------------------------------------------------------ выбор бэкенда


def backend() -> str:
    b = (settings.get("VECTOR_BACKEND") or "qdrant").strip().lower()
    return "milvus" if b == "milvus" else "qdrant"


def is_milvus() -> bool:
    return backend() == "milvus"


# ============================================================ Qdrant (REST) ====

def _qbase() -> str:
    return (settings.get("QDRANT_URL") or "").rstrip("/")


def _qcoll() -> str:
    return settings.get("QDRANT_COLLECTION")


def _qtimeout() -> int:
    try:
        return int(settings.get("QDRANT_TIMEOUT") or 60)
    except Exception:
        return 60


def _q_filter(flt: dict | None) -> dict | None:
    if not flt:
        return None
    must = [{"key": k, "match": {"value": v}} for k, v in flt.items() if v is not None]
    return {"must": must} if must else None


def _q_ping() -> bool:
    try:
        r = httpx.get(f"{_qbase()}/collections/{_qcoll()}", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def _q_ensure(dim: int, reset: bool) -> None:
    base, coll = _qbase(), _qcoll()
    exists = False
    try:
        exists = httpx.get(f"{base}/collections/{coll}", timeout=6).status_code == 200
    except Exception:
        pass
    if reset and exists:
        httpx.delete(f"{base}/collections/{coll}", timeout=30)
        exists = False
    if not exists:
        httpx.put(f"{base}/collections/{coll}", timeout=30,
                  json={"vectors": {"size": int(dim), "distance": "Cosine"}})
        for field in ("source", "fhash", "doc_category", "date", "ftype",
                      "product", "topic", "doc_type", "tags", "vision_desc"):
            try:
                httpx.put(f"{base}/collections/{coll}/index", timeout=15,
                          json={"field_name": field, "field_schema": "keyword"})
            except Exception:
                pass


def _q_search(vector, limit, flt, with_payload) -> list[dict]:
    base, coll = _qbase(), _qcoll()
    body = {"query": list(vector), "limit": int(limit),
            "with_payload": bool(with_payload)}
    qf = _q_filter(flt)
    if qf:
        body["filter"] = qf
    r = httpx.post(f"{base}/collections/{coll}/points/query", json=body, timeout=_qtimeout())
    if r.status_code == 404:
        # коллекция ещё не создана (база знаний не проиндексирована) — не роняем чат в 500,
        # а возвращаем «ничего не найдено»: пайплайн ответит, что данных нет.
        print(f"[vectorstore] коллекция '{coll}' не найдена (404) — база пуста? Верните пусто.")
        return []
    r.raise_for_status()
    pts = (r.json().get("result", {}) or {}).get("points", [])
    return [{"id": p.get("id"), "score": p.get("score"), "payload": p.get("payload") or {}}
            for p in pts]


def _q_scroll(flt, limit, offset, with_vectors, with_payload):
    base, coll = _qbase(), _qcoll()
    body = {"limit": int(limit), "with_payload": bool(with_payload),
            "with_vector": bool(with_vectors)}
    qf = _q_filter(flt)
    if qf:
        body["filter"] = qf
    if offset is not None:
        body["offset"] = offset
    r = httpx.post(f"{base}/collections/{coll}/points/scroll", json=body, timeout=_qtimeout())
    if r.status_code == 404:
        return [], None                       # коллекция ещё не создана — пусто, без 500
    r.raise_for_status()
    res = r.json().get("result", {}) or {}
    pts = [{"id": p.get("id"), "payload": p.get("payload") or {},
            "vector": p.get("vector")} for p in res.get("points", [])]
    return pts, res.get("next_page_offset")


def _q_facet(key, limit, flt) -> list[dict]:
    base, coll = _qbase(), _qcoll()
    body = {"key": key, "limit": int(limit), "exact": True}
    qf = _q_filter(flt)
    if qf:
        body["filter"] = qf
    r = httpx.post(f"{base}/collections/{coll}/facet", json=body, timeout=_qtimeout())
    r.raise_for_status()
    return [{"value": h.get("value"), "count": h.get("count", 0)}
            for h in (r.json().get("result", {}) or {}).get("hits", [])]


def _q_count(flt) -> int:
    base, coll = _qbase(), _qcoll()
    body = {"exact": True}
    qf = _q_filter(flt)
    if qf:
        body["filter"] = qf
    r = httpx.post(f"{base}/collections/{coll}/points/count", json=body, timeout=_qtimeout())
    r.raise_for_status()
    return int((r.json().get("result", {}) or {}).get("count", 0))


def _q_upsert(points, wait) -> None:
    base, coll = _qbase(), _qcoll()
    body = {"points": [{"id": p["id"], "vector": list(p["vector"]),
                        "payload": p.get("payload") or {}} for p in points]}
    url = f"{base}/collections/{coll}/points"
    if not wait:
        url += "?wait=false"
    r = httpx.put(url, json=body, timeout=int(settings.get("QDRANT_INGEST_TIMEOUT") or 480))
    r.raise_for_status()


def _q_delete(flt) -> None:
    base, coll = _qbase(), _qcoll()
    httpx.post(f"{base}/collections/{coll}/points/delete", timeout=30,
               json={"filter": _q_filter(flt)})


def _q_info() -> dict:
    base, coll = _qbase(), _qcoll()
    try:
        r = httpx.get(f"{base}/collections/{coll}", timeout=4)
        if r.status_code != 200:
            return {"exists": False, "points_count": 0, "status": "missing"}
        res = r.json().get("result", {}) or {}
        cfg = (((res.get("config") or {}).get("params") or {}).get("vectors") or {})
        return {"exists": True, "points_count": res.get("points_count", 0),
                "status": res.get("status"), "dim": cfg.get("size"),
                "indexed": res.get("indexed_vectors_count")}
    except Exception as e:
        return {"exists": False, "points_count": 0, "status": f"error: {e}"}


# ============================================================ Milvus (pymilvus) =

_mlock = threading.Lock()
_mclient = None
_mclient_key = None


def milvus_uri() -> str:
    """URI активного Milvus. Standalone — http://host:port; Lite — путь к файлу БД."""
    mode = (settings.get("MILVUS_MODE") or "lite").strip().lower()
    if mode == "standalone":
        uri = (settings.get("MILVUS_URI") or "").strip()
        if uri:
            return uri
        host = (settings.get("MILVUS_HOST") or "127.0.0.1").strip() or "127.0.0.1"
        port = int(settings.get("MILVUS_PORT") or 19530)
        return f"http://{host}:{port}"
    # Lite: локальный файл-хранилище
    p = (settings.get("MILVUS_LITE_PATH") or "").strip()
    if not p:
        p = str(Path(__file__).resolve().parent / "milvus_lite.db")
    return p


def _milvus():
    """Синглтон MilvusClient, пересоздаётся при смене URI. Ленивый импорт pymilvus."""
    global _mclient, _mclient_key
    try:
        from pymilvus import MilvusClient
    except ImportError as e:
        # Понятная ошибка вместо голого ImportError, если бэкенд Milvus, а пакета нет.
        raise RuntimeError(
            "Выбран бэкенд Milvus (VECTOR_BACKEND=milvus), но пакет pymilvus не установлен. "
            "Установите его кнопкой «Установить/проверить Milvus» в админке или переключитесь "
            "на Qdrant.") from e
    uri = milvus_uri()
    token = (settings.get("MILVUS_TOKEN") or "").strip()
    key = (uri, token)
    with _mlock:
        if _mclient is not None and _mclient_key == key:
            return _mclient
        kwargs = {"uri": uri}
        if token:
            kwargs["token"] = token
        _mclient = MilvusClient(**kwargs)
        _mclient_key = key
        return _mclient


def _mcoll() -> str:
    return settings.get("MILVUS_COLLECTION") or "company_kb"


def _m_metric() -> str:
    return (settings.get("MILVUS_METRIC") or "COSINE").strip().upper()


def _m_expr(flt: dict | None) -> str:
    """Нейтральный фильтр → булево выражение Milvus (динамические поля по имени)."""
    if not flt:
        return ""
    parts = []
    for k, v in flt.items():
        if v is None:
            continue
        if isinstance(v, bool):
            parts.append(f'{k} == {"true" if v else "false"}')
        elif isinstance(v, (int, float)):
            parts.append(f"{k} == {v}")
        else:
            s = str(v).replace('"', '\\"')
            parts.append(f'{k} == "{s}"')
    return " and ".join(parts)


def _m_index_params(client):
    """Параметры индекса вектора из настроек (CPU HNSW / GPU CAGRA / IVF)."""
    itype = (settings.get("MILVUS_INDEX_TYPE") or "HNSW").strip().upper()
    metric = _m_metric()
    ip = client.prepare_index_params()
    if itype in ("GPU_CAGRA", "GPU_IVF_FLAT", "GPU_IVF_PQ"):
        params = {}
        if itype == "GPU_CAGRA":
            params = {"intermediate_graph_degree": 64, "graph_degree": 32}
        else:
            params = {"nlist": int(settings.get("MILVUS_NLIST") or 1024)}
        ip.add_index(field_name="vector", index_type=itype, metric_type=metric,
                     params=params)
    elif itype in ("IVF_FLAT", "IVF_SQ8", "IVF_PQ"):
        ip.add_index(field_name="vector", index_type=itype, metric_type=metric,
                     params={"nlist": int(settings.get("MILVUS_NLIST") or 1024)})
    elif itype == "FLAT":
        ip.add_index(field_name="vector", index_type="FLAT", metric_type=metric)
    else:  # HNSW (по умолчанию, CPU)
        ip.add_index(field_name="vector", index_type="HNSW", metric_type=metric,
                     params={"M": int(settings.get("MILVUS_HNSW_M") or 16),
                             "efConstruction": int(settings.get("MILVUS_HNSW_EF_CONSTRUCTION") or 200)})
    return ip


def _m_search_params() -> dict:
    itype = (settings.get("MILVUS_INDEX_TYPE") or "HNSW").strip().upper()
    if itype == "GPU_CAGRA":
        return {"itopk_size": int(settings.get("MILVUS_SEARCH_EF") or 128)}
    if itype.startswith("IVF") or itype.startswith("GPU_IVF"):
        return {"nprobe": int(settings.get("MILVUS_NPROBE") or 16)}
    if itype == "FLAT":
        return {}
    return {"ef": int(settings.get("MILVUS_SEARCH_EF") or 128)}   # HNSW


def _m_ping() -> bool:
    try:
        c = _milvus()
        c.list_collections()
        return True
    except Exception:
        return False


def _m_ensure(dim: int, reset: bool) -> None:
    from pymilvus import DataType
    c = _milvus()
    coll = _mcoll()
    has = coll in c.list_collections()
    if reset and has:
        c.drop_collection(coll)
        has = False
    if has:
        return
    schema = c.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=int(dim))
    c.create_collection(coll, schema=schema, index_params=_m_index_params(c))
    c.load_collection(coll)


def _m_search(vector, limit, flt, with_payload) -> list[dict]:
    c = _milvus()
    res = c.search(_mcoll(), data=[list(vector)], limit=int(limit),
                   filter=_m_expr(flt),
                   search_params={"metric_type": _m_metric(), "params": _m_search_params()},
                   output_fields=["*"] if with_payload else None)
    out = []
    for hit in (res[0] if res else []):
        ent = dict(hit.get("entity") or {})
        ent.pop("vector", None)
        pk = hit.get("id") if hit.get("id") is not None else ent.pop("pk", None)
        ent.pop("pk", None)
        out.append({"id": pk, "score": hit.get("distance"), "payload": ent})
    return out


def _m_query(flt, limit, offset, with_vectors, with_payload):
    c = _milvus()
    fields = ["*"] if with_payload else ["pk"]
    if with_vectors and "vector" not in fields:
        fields = fields + ["vector"]
    rows = c.query(_mcoll(), filter=_m_expr(flt) or "", output_fields=fields,
                   limit=int(limit), offset=int(offset or 0))
    pts = []
    for row in rows:
        row = dict(row)
        vec = row.pop("vector", None)
        pk = row.pop("pk", None)
        pts.append({"id": pk, "payload": row, "vector": vec})
    # Milvus не отдаёт «курсор»: следующий offset считает вызывающий (по числу строк)
    nxt = (int(offset or 0) + len(pts)) if len(pts) >= int(limit) else None
    return pts, nxt


def _m_iter(batch: int = 1000):
    """Итератор по всем точкам коллекции (для миграции). Отдаёт нейтральные points."""
    c = _milvus()
    coll = _mcoll()
    try:
        it = c.query_iterator(coll, filter="", output_fields=["*", "vector"],
                              batch_size=int(batch))
        while True:
            rows = it.next()
            if not rows:
                break
            for row in rows:
                row = dict(row)
                vec = row.pop("vector", None)
                pk = row.pop("pk", None)
                yield {"id": pk, "vector": vec, "payload": row}
        it.close()
        return
    except Exception:
        pass
    # Фолбэк: постраничный обход (ограничение offset+limit до 16384)
    offset = 0
    while offset < 16384:
        rows = c.query(coll, filter="", output_fields=["*", "vector"],
                       limit=batch, offset=offset)
        if not rows:
            break
        for row in rows:
            row = dict(row)
            vec = row.pop("vector", None)
            pk = row.pop("pk", None)
            yield {"id": pk, "vector": vec, "payload": row}
        if len(rows) < batch:
            break
        offset += batch


def _m_facet(key, limit, flt) -> list[dict]:
    """Эмуляция facet: считаем распределение значений поля по строкам (с фильтром)."""
    c = _milvus()
    counts: dict = {}
    cap = int(limit) if limit else 100000
    expr = _m_expr(flt)
    try:
        it = c.query_iterator(_mcoll(), filter=expr or "", output_fields=[key], batch_size=2000)
        while True:
            rows = it.next()
            if not rows:
                break
            for row in rows:
                v = row.get(key)
                if v is None:
                    continue
                counts[v] = counts.get(v, 0) + 1
        it.close()
    except Exception:
        offset = 0
        while offset < 16384:
            rows = c.query(_mcoll(), filter=expr or "", output_fields=[key],
                           limit=2000, offset=offset)
            if not rows:
                break
            for row in rows:
                v = row.get(key)
                if v is not None:
                    counts[v] = counts.get(v, 0) + 1
            if len(rows) < 2000:
                break
            offset += 2000
    hits = [{"value": k, "count": v} for k, v in counts.items()]
    hits.sort(key=lambda h: h["count"], reverse=True)
    return hits[:cap] if cap else hits


def _m_count(flt) -> int:
    c = _milvus()
    coll = _mcoll()
    expr = _m_expr(flt)
    try:
        rows = c.query(coll, filter=expr or "", output_fields=["count(*)"])
        if rows:
            for k in ("count(*)", "count"):
                if k in rows[0]:
                    return int(rows[0][k])
    except Exception:
        pass
    if not expr:
        try:
            st = c.get_collection_stats(coll)
            return int(st.get("row_count", 0))
        except Exception:
            return 0
    # с фильтром без count(*): считаем перебором pk
    n, offset = 0, 0
    while offset < 16384:
        rows = c.query(coll, filter=expr, output_fields=["pk"], limit=2000, offset=offset)
        if not rows:
            break
        n += len(rows)
        if len(rows) < 2000:
            break
        offset += 2000
    return n


def _m_upsert(points, wait) -> None:
    c = _milvus()
    data = []
    for p in points:
        row = {"pk": str(p["id"]), "vector": list(p["vector"])}
        row.update(p.get("payload") or {})
        data.append(row)
    if data:
        c.upsert(_mcoll(), data=data)


def _m_delete(flt) -> None:
    c = _milvus()
    expr = _m_expr(flt)
    if expr:
        c.delete(_mcoll(), filter=expr)


def _m_info() -> dict:
    c = _milvus()
    coll = _mcoll()
    try:
        if coll not in c.list_collections():
            return {"exists": False, "points_count": 0, "status": "missing"}
        n = 0
        try:
            n = int(c.get_collection_stats(coll).get("row_count", 0))
        except Exception:
            n = _m_count(None)
        dim = None
        try:
            desc = c.describe_collection(coll)
            for f in desc.get("fields", []):
                if f.get("name") == "vector":
                    dim = (f.get("params") or {}).get("dim")
        except Exception:
            pass
        return {"exists": True, "points_count": n, "status": "green",
                "dim": dim, "indexed": n}
    except Exception as e:
        return {"exists": False, "points_count": 0, "status": f"error: {e}"}


def milvus_reset_client() -> None:
    """Сбросить кэшированный клиент (после смены режима/URI)."""
    global _mclient, _mclient_key
    with _mlock:
        try:
            if _mclient is not None:
                _mclient.close()
        except Exception:
            pass
        _mclient, _mclient_key = None, None


# ============================================================ единый API =======

def ping(backend_name: str | None = None) -> bool:
    b = backend_name or backend()
    return _m_ping() if b == "milvus" else _q_ping()


def ensure_collection(dim: int, reset: bool = False, backend_name: str | None = None) -> None:
    b = backend_name or backend()
    (_m_ensure if b == "milvus" else _q_ensure)(dim, reset)


def search(vector, limit: int, flt: dict | None = None, with_payload: bool = True) -> list[dict]:
    if not is_milvus():
        return _q_search(vector, limit, flt, with_payload)
    try:
        return _m_search(vector, limit, flt, with_payload)
    except Exception as e:
        # Milvus недоступен/pymilvus не установлен — не роняем ответ 500, деградируем к
        # «ничего не найдено» (получится честное «нет ответа»).
        print(f"[vectorstore] Milvus search недоступен, пустой результат: {e}")
        return []


def search_on(backend_name: str, vector, limit: int, flt: dict | None = None,
              with_payload: bool = True) -> list[dict]:
    """Поиск в КОНКРЕТНОМ бэкенде (для сверки Qdrant↔Milvus), независимо от активного."""
    return _m_search(vector, limit, flt, with_payload) if backend_name == "milvus" \
        else _q_search(vector, limit, flt, with_payload)


def scroll(flt: dict | None = None, limit: int = 256, offset=None,
           with_vectors: bool = False, with_payload: bool = True):
    return _m_query(flt, limit, offset, with_vectors, with_payload) if is_milvus() \
        else _q_scroll(flt, limit, offset, with_vectors, with_payload)


def facet(key: str, limit: int = 100000, flt: dict | None = None) -> list[dict]:
    return _m_facet(key, limit, flt) if is_milvus() else _q_facet(key, limit, flt)


def list_values(key: str, limit: int = 100000, flt: dict | None = None) -> list:
    return [h["value"] for h in facet(key, limit, flt) if h.get("value") is not None]


def count(flt: dict | None = None) -> int:
    if not is_milvus():
        return _q_count(flt)
    try:
        return _m_count(flt)
    except Exception as e:
        print(f"[vectorstore] Milvus count недоступен: {e}")
        return 0


def upsert(points, wait: bool = False) -> None:
    (_m_upsert if is_milvus() else _q_upsert)(points, wait)


def delete(flt: dict) -> None:
    (_m_delete if is_milvus() else _q_delete)(flt)


def collection_info(backend_name: str | None = None) -> dict:
    b = backend_name or backend()
    if b != "milvus":
        return _q_info()
    try:
        return _m_info()
    except Exception as e:
        return {"exists": False, "points_count": 0, "status": f"milvus недоступен: {e}"}


def iterate_all(batch: int = 1000, backend_name: str | None = None):
    """Итератор по всем точкам активного (или указанного) бэкенда — для миграции."""
    b = backend_name or backend()
    if b == "milvus":
        yield from _m_iter(batch)
        return
    # Qdrant: постраничный scroll с векторами
    offset = None
    while True:
        pts, offset = _q_scroll(None, batch, offset, with_vectors=True, with_payload=True)
        for p in pts:
            yield p
        if offset is None or not pts:
            break


def upsert_to(backend_name: str, points, dim: int | None = None) -> None:
    """Запись точек в КОНКРЕТНЫЙ бэкенд (для миграции, независимо от активного)."""
    if backend_name == "milvus":
        _m_upsert(points, wait=True)
    else:
        _q_upsert(points, wait=True)
