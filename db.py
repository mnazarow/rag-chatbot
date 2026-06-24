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
        # миграции колонок (идемпотентно)
        for col in ("rating INTEGER", "retrieve_ms INTEGER", "gen_ms INTEGER",
                    "session_id TEXT"):
            try:
                c.execute(f"ALTER TABLE requests ADD COLUMN {col}")
            except Exception:
                pass
        # --- Telegram: пользователи бота (с подтверждением доступа) ---
        c.execute(
            """CREATE TABLE IF NOT EXISTS tg_users(
                chat_id INTEGER PRIMARY KEY,
                username TEXT, first_name TEXT,
                status TEXT,            -- pending | approved | blocked
                created REAL, updated REAL, n_requests INTEGER DEFAULT 0)"""
        )
        # --- Telegram: запросы/история (хранятся ОТДЕЛЬНО от веб-чата) ---
        c.execute(
            """CREATE TABLE IF NOT EXISTS tg_requests(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, day TEXT, chat_id INTEGER, username TEXT,
                question TEXT, answer TEXT, n_hits INTEGER, top_score REAL,
                latency_ms INTEGER, answer_chars INTEGER, answered INTEGER,
                sources TEXT)"""
        )


init()


def log_request(question: str, category: str | None, n_hits: int,
                top_score: float, latency_ms: int, answer_chars: int,
                answered: bool, sources: list,
                retrieve_ms: int = 0, gen_ms: int = 0,
                session_id: str = "") -> int:
    now = datetime.now()
    with _LOCK, _conn() as c:
        cur = c.execute(
            """INSERT INTO requests
               (ts,day,question,category,n_hits,top_score,latency_ms,
                answer_chars,answered,sources,retrieve_ms,gen_ms,session_id)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (now.timestamp(), now.strftime("%Y-%m-%d"), question, category,
             n_hits, round(top_score, 4), latency_ms, answer_chars,
             int(answered), json.dumps(sources, ensure_ascii=False),
             int(retrieve_ms), int(gen_ms), session_id or ""),
        )
        return cur.lastrowid


def set_rating(req_id: int, rating: int) -> bool:
    with _LOCK, _conn() as c:
        cur = c.execute("UPDATE requests SET rating=? WHERE id=?",
                        (int(rating), int(req_id)))
        return cur.rowcount > 0


def rating_stats() -> dict:
    with _conn() as c:
        g = c.execute("SELECT COUNT(*) n FROM requests WHERE rating=1").fetchone()["n"]
        b = c.execute("SELECT COUNT(*) n FROM requests WHERE rating=-1").fetchone()["n"]
        tot = c.execute("SELECT COUNT(*) n FROM requests").fetchone()["n"]
    rated = g + b
    return {"good": g, "bad": b, "rated": rated, "unrated": tot - rated,
            "satisfaction": round(g / rated * 100, 1) if rated else None}


def rating_analysis() -> dict:
    with _conn() as c:
        good = c.execute("SELECT AVG(top_score) s, COUNT(*) n FROM requests WHERE rating=1").fetchone()
        bad = c.execute("SELECT AVG(top_score) s, COUNT(*) n FROM requests WHERE rating=-1").fetchone()
        bad_no = c.execute("SELECT COUNT(*) n FROM requests WHERE rating=-1 AND answered=0").fetchone()["n"]
        bad_yes = c.execute("SELECT COUNT(*) n FROM requests WHERE rating=-1 AND answered=1").fetchone()["n"]
    return {"good_avg_score": good["s"], "good_n": good["n"],
            "bad_avg_score": bad["s"], "bad_n": bad["n"],
            "bad_no_answer": bad_no, "bad_answered": bad_yes}


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
            "SELECT AVG(latency_ms) lat, AVG(top_score) sc, AVG(answered) ans, "
            "AVG(retrieve_ms) rt, AVG(gen_ms) gn FROM requests"
        ).fetchone()
    return {
        "total": total,
        "today": today_n,
        "avg_latency_ms": round(agg["lat"] or 0),
        "avg_retrieve_ms": round(agg["rt"] or 0),
        "avg_gen_ms": round(agg["gn"] or 0),
        "avg_top_score": round(agg["sc"] or 0, 3),
        "answer_rate": round((agg["ans"] or 0) * 100, 1),
    }


def clear() -> int:
    """Очистить журнал запросов (история всех чатов и их статистика). Возвращает
    число удалённых записей."""
    with _LOCK, _conn() as c:
        n = c.execute("SELECT COUNT(*) n FROM requests").fetchone()["n"]
        c.execute("DELETE FROM requests")
    return n


def session_history(session_id: str) -> list[dict]:
    """Все записи одного чата (по session_id), от старых к новым — для экспорта."""
    if not session_id:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM requests WHERE session_id=? ORDER BY id ASC",
            (session_id,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d.get("sources") or "[]")
        out.append(d)
    return out


def engine_usage() -> dict:
    """Сколько ответов дано каждым движком (по полю category в журнале)."""
    with _conn() as c:
        g = c.execute("SELECT COUNT(*) n FROM requests WHERE category='graph'").fetchone()["n"]
        l = c.execute("SELECT COUNT(*) n FROM requests WHERE category='lightrag'").fetchone()["n"]
        tot = c.execute("SELECT COUNT(*) n FROM requests").fetchone()["n"]
    return {"graph": g, "lightrag": l, "vector": max(tot - g - l, 0), "total": tot}


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
        tm = c.execute("SELECT AVG(retrieve_ms) rt, AVG(gen_ms) gn, AVG(latency_ms) lt "
                       "FROM requests").fetchone()

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
        "ratings": rating_stats(),
        "timings": {"retrieve": round(tm["rt"] or 0), "gen": round(tm["gn"] or 0),
                    "total": round(tm["lt"] or 0)},
    }


# ===================== Telegram: пользователи и запросы =====================

def tg_user(chat_id: int) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM tg_users WHERE chat_id=?", (int(chat_id),)).fetchone()
    return dict(r) if r else None


def tg_user_upsert(chat_id: int, username: str | None, first_name: str | None,
                   status: str) -> dict:
    """Создать пользователя (если нет) с указанным статусом, либо обновить имя."""
    now = datetime.now().timestamp()
    with _LOCK, _conn() as c:
        ex = c.execute("SELECT chat_id FROM tg_users WHERE chat_id=?",
                       (int(chat_id),)).fetchone()
        if ex:
            c.execute("UPDATE tg_users SET username=?, first_name=?, updated=? "
                      "WHERE chat_id=?",
                      (username or "", first_name or "", now, int(chat_id)))
        else:
            c.execute("INSERT INTO tg_users(chat_id,username,first_name,status,"
                      "created,updated,n_requests) VALUES(?,?,?,?,?,?,0)",
                      (int(chat_id), username or "", first_name or "", status, now, now))
    return tg_user(chat_id)


def tg_set_status(chat_id: int, status: str) -> bool:
    with _LOCK, _conn() as c:
        cur = c.execute("UPDATE tg_users SET status=?, updated=? WHERE chat_id=?",
                        (status, datetime.now().timestamp(), int(chat_id)))
        return cur.rowcount > 0


def tg_users(status: str | None = None) -> list[dict]:
    with _conn() as c:
        if status:
            rows = c.execute("SELECT * FROM tg_users WHERE status=? "
                             "ORDER BY updated DESC", (status,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM tg_users ORDER BY updated DESC").fetchall()
    return [dict(r) for r in rows]


def tg_counts() -> dict:
    with _conn() as c:
        rows = c.execute("SELECT status, COUNT(*) n FROM tg_users GROUP BY status").fetchall()
    d = {r["status"]: r["n"] for r in rows}
    return {"pending": d.get("pending", 0), "approved": d.get("approved", 0),
            "blocked": d.get("blocked", 0),
            "total": sum(d.values())}


def tg_log_request(chat_id: int, username: str | None, question: str, answer: str,
                   n_hits: int, top_score: float, latency_ms: int,
                   answered: bool, sources: list) -> int:
    now = datetime.now()
    with _LOCK, _conn() as c:
        cur = c.execute(
            """INSERT INTO tg_requests(ts,day,chat_id,username,question,answer,
               n_hits,top_score,latency_ms,answer_chars,answered,sources)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (now.timestamp(), now.strftime("%Y-%m-%d"), int(chat_id), username or "",
             question, answer, n_hits, round(top_score, 4), latency_ms,
             len(answer or ""), int(answered),
             json.dumps(sources, ensure_ascii=False)))
        c.execute("UPDATE tg_users SET n_requests=n_requests+1 WHERE chat_id=?",
                  (int(chat_id),))
        return cur.lastrowid


def tg_recent(limit: int = 200) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM tg_requests ORDER BY id DESC LIMIT ?",
                         (int(limit),)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d.get("sources") or "[]")
        out.append(d)
    return out


def tg_stats() -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) n FROM tg_requests").fetchone()["n"]
        today_n = c.execute("SELECT COUNT(*) n FROM tg_requests WHERE day=?",
                            (today,)).fetchone()["n"]
        agg = c.execute("SELECT AVG(latency_ms) lat, AVG(answered) ans "
                        "FROM tg_requests").fetchone()
        users = c.execute("SELECT COUNT(DISTINCT chat_id) n FROM tg_requests").fetchone()["n"]
    counts = tg_counts()
    return {"total": total, "today": today_n, "users": users,
            "avg_latency_ms": round(agg["lat"] or 0),
            "answer_rate": round((agg["ans"] or 0) * 100, 1),
            "pending": counts["pending"], "approved": counts["approved"],
            "blocked": counts["blocked"]}


def tg_clear_history() -> int:
    with _LOCK, _conn() as c:
        n = c.execute("SELECT COUNT(*) n FROM tg_requests").fetchone()["n"]
        c.execute("DELETE FROM tg_requests")
    return n
