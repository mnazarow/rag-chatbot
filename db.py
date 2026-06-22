"""Журнал запросов и аналитика на SQLite (лёгкая, без внешних сервисов)."""
from __future__ import annotations
import json
import sqlite3
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "rag_logs.db"
_LOCK = threading.Lock()

_STOPWORDS = set(
    "и в во не на по с со о об а но что как так это для из у к до за от же бы ли "
    "the a of to is при или есть быть какой какая какие чем тип где когда".split()
)


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init() -> None:
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS requests(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, day TEXT, question TEXT, category TEXT,
                n_hits INTEGER, top_score REAL, latency_ms INTEGER,
                answer_chars INTEGER, answered INTEGER, sources TEXT)"""
        )


init()


def log_request(question: str, category: str | None, n_hits: int,
                top_score: float, latency_ms: int, answer_chars: int,
                answered: bool, sources: list) -> None:
    now = datetime.now()
    with _LOCK, _conn() as c:
        c.execute(
            """INSERT INTO requests
               (ts,day,question,category,n_hits,top_score,latency_ms,
                answer_chars,answered,sources)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (now.timestamp(), now.strftime("%Y-%m-%d"), question, category,
             n_hits, round(top_score, 4), latency_ms, answer_chars,
             int(answered), json.dumps(sources, ensure_ascii=False)),
        )


def recent(limit: int = 100) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d.get("sources") or "[]")
        out.append(d)
    return out


def stats() -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) n FROM requests").fetchone()["n"]
        today_n = c.execute(
            "SELECT COUNT(*) n FROM requests WHERE day=?", (today,)
        ).fetchone()["n"]
        agg = c.execute(
            "SELECT AVG(latency_ms) lat, AVG(top_score) sc, AVG(answered) ans "
            "FROM requests"
        ).fetchone()
    return {
        "total": total,
        "today": today_n,
        "avg_latency_ms": round(agg["lat"] or 0),
        "avg_top_score": round(agg["sc"] or 0, 3),
        "answer_rate": round((agg["ans"] or 0) * 100, 1),
    }


def analytics() -> dict:
    with _conn() as c:
        per_day = c.execute(
            "SELECT day, COUNT(*) n FROM requests GROUP BY day ORDER BY day DESC LIMIT 14"
        ).fetchall()
        per_cat = c.execute(
            "SELECT COALESCE(NULLIF(category,''),'—') cat, COUNT(*) n "
            "FROM requests GROUP BY cat ORDER BY n DESC"
        ).fetchall()
        answered_rows = c.execute(
            "SELECT sources FROM requests WHERE answered=1"
        ).fetchall()
        lat_rows = c.execute("SELECT latency_ms FROM requests").fetchall()
        q_rows = c.execute("SELECT question FROM requests").fetchall()

    src = Counter()
    for r in answered_rows:
        for s in json.loads(r["sources"] or "[]"):
            if s.get("source"):
                src[s["source"]] += 1

    kw = Counter()
    for r in q_rows:
        for w in (r["question"] or "").lower().split():
            w = w.strip("?.,!:;()\"'«»—-")
            if len(w) > 3 and w not in _STOPWORDS:
                kw[w] += 1

    buckets = {"<1с": 0, "1–3с": 0, "3–6с": 0, ">6с": 0}
    for r in lat_rows:
        ms = r["latency_ms"] or 0
        if ms < 1000:
            buckets["<1с"] += 1
        elif ms < 3000:
            buckets["1–3с"] += 1
        elif ms < 6000:
            buckets["3–6с"] += 1
        else:
            buckets[">6с"] += 1

    return {
        "per_day": [{"day": r["day"], "n": r["n"]} for r in reversed(per_day)],
        "per_category": [{"cat": r["cat"], "n": r["n"]} for r in per_cat],
        "top_sources": [{"source": s, "n": n} for s, n in src.most_common(10)],
        "top_keywords": [{"word": w, "n": n} for w, n in kw.most_common(15)],
        "latency_buckets": buckets,
    }
