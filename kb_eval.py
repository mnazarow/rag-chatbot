"""ИИ-оценка базы знаний: профилирование индекса Qdrant + аудит силами LLM.

Собирает «портрет» базы (число чанков/файлов, категории, типы файлов, покрытие
метаданными, длины чанков, доля дубликатов, примеры фрагментов, текущие настройки
поиска/чанкинга) и передаёт его модели с просьбой оценить базу и дать конкретные
рекомендации по оптимизации и настройкам (чанкинг, метаданные/категории, пробелы,
дубликаты, MIN_SCORE/TOP_K/температура, движок/режим, синонимы).

Выполняется фоновой задачей с прогрессом (профилирование → анализ ИИ); результат
кэшируется до следующего запуска. Эндпоинты — в app.py.
"""
from __future__ import annotations
import hashlib
import os
import threading
import time
from collections import Counter

import httpx
import settings

_JOB = {"running": False, "stage": "", "scrolled": 0, "total": 0,
        "result": None, "error": None, "ts": 0.0}


def _total_points() -> int:
    try:
        r = httpx.get(f"{settings.get('QDRANT_URL')}/collections/"
                      f"{settings.get('QDRANT_COLLECTION')}", timeout=4)
        if r.status_code == 200:
            return int((r.json().get("result", {}) or {}).get("points_count", 0) or 0)
    except Exception:
        pass
    return 0


def _gather(sample_per_cat: int = 2, max_samples: int = 24) -> dict:
    base = settings.get("QDRANT_URL")
    coll = settings.get("QDRANT_COLLECTION")
    total = 0
    cats: Counter = Counter()
    ftypes: Counter = Counter()
    sources: set = set()
    meta = {"category": 0, "date": 0, "product": 0, "topic": 0}
    char_sum = 0
    char_min = 10 ** 9
    char_max = 0
    short = 0          # очень короткие чанки (<120 симв.)
    long_ = 0          # очень длинные (>2500)
    seen_hash: set = set()
    dup = 0
    samples: list = []
    samples_cat: Counter = Counter()
    next_off = None
    _JOB["total"] = _total_points()
    for _ in range(200000):
        body = {"limit": 256, "with_payload": True, "with_vector": False}
        if next_off is not None:
            body["offset"] = next_off
        try:
            r = httpx.post(f"{base}/collections/{coll}/points/scroll",
                           json=body, timeout=60)
        except Exception as e:
            _JOB["error"] = str(e)[:160]
            break
        if r.status_code != 200:
            _JOB["error"] = f"Qdrant HTTP {r.status_code}"
            break
        res = r.json().get("result", {}) or {}
        for p in res.get("points", []):
            pl = p.get("payload") or {}
            txt = (pl.get("text") or "")
            total += 1
            cat = pl.get("doc_category") or "—"
            cats[cat] += 1
            for k in ("category", "date", "product", "topic"):
                key = "doc_category" if k == "category" else k
                if pl.get(key):
                    meta[k] += 1
            src = pl.get("source") or ""
            if src:
                sources.add(src)
            ft = pl.get("ftype") or os.path.splitext(src)[1].lstrip(".").lower()
            if ft:
                ftypes[ft] += 1
            L = len(txt)
            char_sum += L
            char_min = min(char_min, L)
            char_max = max(char_max, L)
            if L < 120:
                short += 1
            if L > 2500:
                long_ += 1
            h = hashlib.sha1(txt.strip().lower().encode("utf-8")).hexdigest()
            if h in seen_hash:
                dup += 1
            else:
                seen_hash.add(h)
            if (txt.strip() and len(samples) < max_samples
                    and samples_cat[cat] < sample_per_cat):
                samples.append({"category": cat, "source": os.path.basename(src),
                                "text": txt.strip()[:280]})
                samples_cat[cat] += 1
        _JOB["scrolled"] = total
        next_off = res.get("next_page_offset")
        if next_off is None:
            break

    avg = round(char_sum / total) if total else 0
    return {
        "chunks": total, "files": len(sources),
        "categories": cats.most_common(),
        "file_types": ftypes.most_common(),
        "metadata_coverage_pct": {k: (round(v * 100 / total, 1) if total else 0)
                                  for k, v in meta.items()},
        "chunk_chars": {"avg": avg, "min": (char_min if total else 0), "max": char_max,
                        "short_lt120": short, "long_gt2500": long_},
        "duplicates": dup,
        "samples": samples,
        "settings": {
            "embed_model": settings.get("EMBED_MODEL"),
            "rerank_model": settings.get("RERANK_MODEL"),
            "chunk_size": settings.get("CHUNK_SIZE"),
            "chunk_overlap": settings.get("CHUNK_OVERLAP"),
            "min_score": settings.get("MIN_SCORE"),
            "top_k_retrieve": settings.get("TOP_K_RETRIEVE"),
            "top_k_rerank": settings.get("TOP_K_RERANK"),
            "engine": settings.get("ENGINE"),
            "temperature": settings.get("TEMPERATURE"),
            "smart_filter": settings.get("SMART_FILTER"),
            "auto_filter": settings.get("AUTO_FILTER"),
        },
    }


def _fmt_profile(p: dict) -> str:
    s = p["settings"]
    cats = ", ".join(f"{c}={n}" for c, n in p["categories"][:15]) or "нет"
    fts = ", ".join(f"{c}={n}" for c, n in p["file_types"][:15]) or "нет"
    mc = p["metadata_coverage_pct"]
    cc = p["chunk_chars"]
    lines = [
        f"Чанков: {p['chunks']}, файлов: {p['files']}",
        f"Категории (чанков): {cats}",
        f"Типы файлов: {fts}",
        f"Покрытие метаданными, %: категория={mc['category']}, дата={mc['date']}, "
        f"продукт={mc['product']}, тема={mc['topic']}",
        f"Длина чанка (симв.): сред={cc['avg']}, мин={cc['min']}, макс={cc['max']}, "
        f"очень коротких(<120)={cc['short_lt120']}, очень длинных(>2500)={cc['long_gt2500']}",
        f"Точные дубликаты чанков: {p['duplicates']}",
        f"Настройки: эмбеддер={s['embed_model']}, реранкер={s['rerank_model']}, "
        f"chunk_size={s['chunk_size']}, overlap={s['chunk_overlap']}, "
        f"MIN_SCORE={s['min_score']}, TOP_K_RETRIEVE={s['top_k_retrieve']}, "
        f"TOP_K_RERANK={s['top_k_rerank']}, движок={s['engine']}, "
        f"температура={s['temperature']}, умный_фильтр={s['smart_filter']}, "
        f"авто_фильтр={s['auto_filter']}",
    ]
    if p["samples"]:
        lines.append("\nПримеры фрагментов:")
        for i, sm in enumerate(p["samples"][:24], 1):
            lines.append(f"{i}. [{sm['category']} · {sm['source']}] {sm['text']}")
    return "\n".join(lines)


_SYS = ("Ты — аудитор корпоративной базы знаний для RAG-системы (поиск + реранк + LLM). "
        "Тебе дают профиль текущего индекса. Оцени базу и дай КОНКРЕТНЫЕ рекомендации. "
        "Отвечай по-русски, структурировано, по делу, без воды. Разделы:\n"
        "1) Краткая оценка (1–2 абзаца) и общий балл качества базы 0–100.\n"
        "2) Сильные стороны.\n"
        "3) Проблемы и риски (дубликаты, слишком короткие/длинные чанки, слабое покрытие "
        "метаданными, перекос по категориям, мало источников и т. п.).\n"
        "4) Рекомендации по контенту и индексации (чанкинг chunk_size/overlap, метаданные/"
        "категории, что дозагрузить, дедупликация).\n"
        "5) Рекомендации по настройкам поиска (MIN_SCORE, TOP_K_RETRIEVE, TOP_K_RERANK, "
        "температура, движок/режим, умный фильтр, синонимы) — с конкретными значениями.\n"
        "Опирайся только на профиль; если данных мало — так и скажи.")


def _run():
    try:
        _JOB.update(running=True, stage="профилирование базы", scrolled=0,
                    error=None, result=None)
        profile = _gather()
        if _JOB.get("error") and not profile.get("chunks"):
            _JOB["running"] = False
            return
        _JOB["stage"] = "анализ ИИ"
        import llm_backend
        text = llm_backend.chat(
            [{"role": "system", "content": _SYS},
             {"role": "user", "content": "Профиль базы знаний:\n\n" + _fmt_profile(profile)}],
            temperature=0.2, model=settings.active_model())
        _JOB["result"] = {"profile": profile, "assessment": (text or "").strip()}
        _JOB["ts"] = time.time()
    except Exception as e:
        _JOB["error"] = str(e)[:300]
    finally:
        _JOB["running"] = False
        _JOB["stage"] = ""


def evaluate(force: bool = False) -> dict:
    if _JOB["running"]:
        return {"running": True, "stage": _JOB["stage"],
                "progress": {"scrolled": _JOB["scrolled"], "total": _JOB["total"]}}
    if _JOB["result"] and not force:
        return {"running": False, "cached": True,
                "age_sec": int(time.time() - _JOB["ts"]), **_JOB["result"]}
    threading.Thread(target=_run, daemon=True).start()
    return {"running": True, "stage": "запуск", "progress": {"scrolled": 0, "total": 0}}


def status() -> dict:
    if _JOB["running"]:
        tot = _JOB.get("total", 0)
        pct = round(_JOB["scrolled"] * 100.0 / tot, 1) if tot else None
        return {"running": True, "stage": _JOB["stage"],
                "progress": {"scrolled": _JOB["scrolled"], "total": tot, "pct": pct}}
    if _JOB["result"]:
        return {"running": False, "done": True, "error": _JOB.get("error"),
                "cached": True, "age_sec": int(time.time() - _JOB["ts"]), **_JOB["result"]}
    return {"running": False, "done": False, "error": _JOB.get("error")}
