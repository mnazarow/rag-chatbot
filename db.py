"""Журнал запросов, история Телеграм и аналитика.

Хранилище переключаемое: SQLite (по умолчанию, без внешних сервисов), MySQL или
PostgreSQL. Бэкенд выбирается настройкой DB_BACKEND; параметры подключения —
MYSQL_* / PG_*. Все функции работают одинаково на любом бэкенде через тонкий
диалект-слой (плейсхолдеры, DDL, lastrowid/RETURNING).

Копирование и миграция данных между СУБД — функции copy_all() / migrate().
Настройки приложения (runtime_config.json) при копировании сохраняются в таблицу
kv_store приёмника (для переносимости между хостами).

Тяжёлые агрегаты (stats/analytics/tg_stats) при включённом Redis кэшируются
(см. cache.py); кэш сбрасывается при любой записи в журнал.
"""
from __future__ import annotations
import json
import sqlite3
import threading
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "rag_logs.db"
_RUNTIME = Path(__file__).resolve().parent / "runtime_config.json"
_LOCK = threading.Lock()

_STOPWORDS = set(
    "и в во не на по с со о об а но что как так это для из у к до за от же бы ли "
    "the a of to is при или есть быть какой какая какие чем тип где когда".split()
)


# ===================== Диалект-слой (sqlite / mysql / postgresql) =====================

def _settings():
    """Ленивая ссылка на settings (избегаем циклов импорта на старте)."""
    import settings
    return settings


def _norm(d: str | None) -> str:
    d = (d or "sqlite").lower()
    if d in ("postgres", "postgresql", "pg"):
        return "postgresql"
    if d in ("mysql", "mariadb"):
        return "mysql"
    return "sqlite"


def _dialect() -> str:
    try:
        return _norm(_settings().get("DB_BACKEND"))
    except Exception:
        return "sqlite"


def _params_for(dialect: str) -> dict:
    s = _settings()
    d = _norm(dialect)
    if d == "mysql":
        return {"host": (s.get("MYSQL_HOST") or "").strip(),
                "port": int(s.get("MYSQL_PORT") or 3306),
                "user": s.get("MYSQL_USER") or "",
                "password": s.get("MYSQL_PASSWORD") or "",
                "db": (s.get("MYSQL_DB") or "").strip()}
    if d == "postgresql":
        return {"host": (s.get("PG_HOST") or "").strip(),
                "port": int(s.get("PG_PORT") or 5432),
                "user": s.get("PG_USER") or "",
                "password": s.get("PG_PASSWORD") or "",
                "db": (s.get("PG_DB") or "").strip()}
    return {}


def _driver_present(dialect: str) -> bool:
    d = _norm(dialect)
    try:
        if d == "mysql":
            import pymysql  # noqa: F401
        elif d == "postgresql":
            import psycopg2  # noqa: F401
        return True
    except Exception:
        return False


def _connect_for(dialect: str):
    d = _norm(dialect)
    if d == "mysql":
        import pymysql
        p = _params_for("mysql")
        conn = pymysql.connect(host=p["host"], port=p["port"], user=p["user"],
                               password=p["password"], database=p["db"],
                               charset="utf8mb4", autocommit=True, connect_timeout=5)
        return "mysql", conn
    if d == "postgresql":
        import psycopg2
        p = _params_for("postgresql")
        # client_encoding='UTF8' — заставляем сервер отдавать данные/сообщения в UTF-8
        # (иначе на сервере с русской локалью psycopg2 спотыкается на cp1251).
        conn = psycopg2.connect(host=p["host"], port=p["port"], user=p["user"],
                                password=p["password"], dbname=p["db"],
                                connect_timeout=5, client_encoding="UTF8")
        conn.autocommit = True
        return "postgresql", conn
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return "sqlite", conn


def _mk_cursor(dialect: str, conn):
    if dialect == "mysql":
        import pymysql
        return conn.cursor(pymysql.cursors.DictCursor)
    if dialect == "postgresql":
        import psycopg2.extras
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    return conn.cursor()


@contextmanager
def _cursor(dialect: str | None = None, conn=None):
    own = conn is None
    if own:
        dialect, conn = _connect_for(dialect or _dialect())
    try:
        cur = _mk_cursor(dialect, conn)
        yield dialect, conn, cur
        if dialect == "sqlite" and own:
            conn.commit()
    finally:
        if own:
            try:
                conn.close()
            except Exception:
                pass


def _ph(sql: str, d: str) -> str:
    return sql if d == "sqlite" else sql.replace("?", "%s")


def _all(sql: str, params: tuple = ()) -> list[dict]:
    with _cursor() as (d, conn, cur):
        cur.execute(_ph(sql, d), params)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def _one(sql: str, params: tuple = ()) -> dict | None:
    with _cursor() as (d, conn, cur):
        cur.execute(_ph(sql, d), params)
        r = cur.fetchone()
    return dict(r) if r else None


def _exec(sql: str, params: tuple = ()) -> int:
    with _cursor() as (d, conn, cur):
        cur.execute(_ph(sql, d), params)
        return cur.rowcount


def _insert(sql: str, params: tuple = ()):
    with _cursor() as (d, conn, cur):
        if d == "postgresql":
            cur.execute(_ph(sql, d) + " RETURNING id", params)
            r = cur.fetchone()
            if not r:
                return None
            return r["id"] if isinstance(r, dict) else r[0]
        cur.execute(_ph(sql, d), params)
        return cur.lastrowid


def _f(x) -> float:
    """Безопасное приведение к float (AVG в mysql/pg возвращает Decimal)."""
    try:
        return float(x or 0)
    except Exception:
        return 0.0


def _fn(x):
    return None if x is None else float(x)


def _safe_err(e) -> str:
    """Текст ошибки, устойчивый к не-UTF-8 сообщениям СУБД.

    Сервер PostgreSQL/MySQL с русской локалью присылает текст ошибки в cp1251, и
    psycopg2 пытается декодировать его как UTF-8 → получаем загадочное «'utf-8'
    codec can't decode…». Здесь достаём исходные байты и декодируем cp1251/latin1,
    чтобы показать настоящую причину (неверный пароль, нет БД, pg_hba и т. п.)."""
    obj = getattr(e, "object", None)
    if isinstance(e, UnicodeDecodeError) and isinstance(obj, (bytes, bytearray)):
        for enc in ("cp1251", "latin1"):
            try:
                return obj.decode(enc, "replace").strip()
            except Exception:
                pass
    try:
        return str(e)
    except Exception:
        return repr(e)


def _bump() -> None:
    try:
        import cache
        cache.bump()
    except Exception:
        pass


# ===================== Схема =====================

def _ddl(d: str) -> list[str]:
    if d == "mysql":
        return [
            """CREATE TABLE IF NOT EXISTS requests(
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                ts DOUBLE, day VARCHAR(16), question MEDIUMTEXT, category VARCHAR(128),
                n_hits INT, top_score DOUBLE, latency_ms INT,
                answer_chars INT, answered INT, sources MEDIUMTEXT,
                rating INT, retrieve_ms INT, gen_ms INT, session_id VARCHAR(80)
                ) CHARACTER SET utf8mb4""",
            """CREATE TABLE IF NOT EXISTS tg_users(
                chat_id BIGINT PRIMARY KEY, username VARCHAR(255), first_name VARCHAR(255),
                status VARCHAR(16), created DOUBLE, updated DOUBLE, n_requests INT DEFAULT 0
                ) CHARACTER SET utf8mb4""",
            """CREATE TABLE IF NOT EXISTS tg_requests(
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                ts DOUBLE, day VARCHAR(16), chat_id BIGINT, username VARCHAR(255),
                question MEDIUMTEXT, answer LONGTEXT, n_hits INT, top_score DOUBLE,
                latency_ms INT, answer_chars INT, answered INT, sources MEDIUMTEXT
                ) CHARACTER SET utf8mb4""",
            """CREATE TABLE IF NOT EXISTS kv_store(
                k VARCHAR(190) PRIMARY KEY, v LONGTEXT) CHARACTER SET utf8mb4""",
            """CREATE TABLE IF NOT EXISTS doc_catalog(
                rel_path VARCHAR(700) PRIMARY KEY, fname VARCHAR(512), ext VARCHAR(32),
                size BIGINT, mtime BIGINT, n_chars INT, method VARCHAR(32),
                sha256 VARCHAR(64), content LONGBLOB, content_oid BIGINT, txt LONGTEXT,
                updated DOUBLE) CHARACTER SET utf8mb4""",
        ]
    if d == "postgresql":
        return [
            """CREATE TABLE IF NOT EXISTS requests(
                id BIGSERIAL PRIMARY KEY,
                ts DOUBLE PRECISION, day TEXT, question TEXT, category TEXT,
                n_hits INTEGER, top_score DOUBLE PRECISION, latency_ms INTEGER,
                answer_chars INTEGER, answered INTEGER, sources TEXT,
                rating INTEGER, retrieve_ms INTEGER, gen_ms INTEGER, session_id TEXT)""",
            """CREATE TABLE IF NOT EXISTS tg_users(
                chat_id BIGINT PRIMARY KEY, username TEXT, first_name TEXT,
                status TEXT, created DOUBLE PRECISION, updated DOUBLE PRECISION,
                n_requests INTEGER DEFAULT 0)""",
            """CREATE TABLE IF NOT EXISTS tg_requests(
                id BIGSERIAL PRIMARY KEY,
                ts DOUBLE PRECISION, day TEXT, chat_id BIGINT, username TEXT,
                question TEXT, answer TEXT, n_hits INTEGER, top_score DOUBLE PRECISION,
                latency_ms INTEGER, answer_chars INTEGER, answered INTEGER, sources TEXT)""",
            """CREATE TABLE IF NOT EXISTS kv_store(k TEXT PRIMARY KEY, v TEXT)""",
            """CREATE TABLE IF NOT EXISTS doc_catalog(
                rel_path TEXT PRIMARY KEY, fname TEXT, ext TEXT, size BIGINT,
                mtime BIGINT, n_chars INTEGER, method TEXT, sha256 TEXT,
                content BYTEA, content_oid BIGINT, txt TEXT,
                updated DOUBLE PRECISION)""",
        ]
    # sqlite (по умолчанию)
    return [
        """CREATE TABLE IF NOT EXISTS requests(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, day TEXT, question TEXT, category TEXT,
            n_hits INTEGER, top_score REAL, latency_ms INTEGER,
            answer_chars INTEGER, answered INTEGER, sources TEXT,
            rating INTEGER, retrieve_ms INTEGER, gen_ms INTEGER, session_id TEXT)""",
        """CREATE TABLE IF NOT EXISTS tg_users(
            chat_id INTEGER PRIMARY KEY,
            username TEXT, first_name TEXT, status TEXT,
            created REAL, updated REAL, n_requests INTEGER DEFAULT 0)""",
        """CREATE TABLE IF NOT EXISTS tg_requests(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, day TEXT, chat_id INTEGER, username TEXT,
            question TEXT, answer TEXT, n_hits INTEGER, top_score REAL,
            latency_ms INTEGER, answer_chars INTEGER, answered INTEGER, sources TEXT)""",
        """CREATE TABLE IF NOT EXISTS kv_store(k TEXT PRIMARY KEY, v TEXT)""",
        """CREATE TABLE IF NOT EXISTS doc_catalog(
            rel_path TEXT PRIMARY KEY, fname TEXT, ext TEXT, size INTEGER,
            mtime INTEGER, n_chars INTEGER, method TEXT, sha256 TEXT,
            content BLOB, content_oid BIGINT, txt TEXT, updated REAL)""",
    ]


def init(dialect: str | None = None) -> None:
    """Создать таблицы для указанного (или текущего) бэкенда. Идемпотентно и
    устойчиво к недоступности внешней СУБД (не валит старт приложения)."""
    d = _norm(dialect or _dialect())
    try:
        dd, conn = _connect_for(d)
    except Exception as e:
        print(f"[db] init: бэкенд {d} недоступен: {e}")
        return
    try:
        cur = _mk_cursor(dd, conn)
        for stmt in _ddl(dd):
            try:
                cur.execute(stmt)
            except Exception as e:
                print(f"[db] init {dd}: {e}")
        if dd == "sqlite":
            # миграции колонок для старых баз (идемпотентно)
            for col in ("rating INTEGER", "retrieve_ms INTEGER", "gen_ms INTEGER",
                        "session_id TEXT"):
                try:
                    cur.execute(f"ALTER TABLE requests ADD COLUMN {col}")
                except Exception:
                    pass
        # миграции колонок doc_catalog (хранение файлов целиком) — все диалекты
        _blob = {"mysql": "LONGBLOB", "postgresql": "BYTEA", "sqlite": "BLOB"}[dd]
        _txt = {"mysql": "VARCHAR(512)", "postgresql": "TEXT", "sqlite": "TEXT"}[dd]
        _sha = {"mysql": "VARCHAR(64)", "postgresql": "TEXT", "sqlite": "TEXT"}[dd]
        for _c, _t in (("fname", _txt), ("sha256", _sha), ("content", _blob),
                       ("content_oid", "BIGINT")):
            try:
                cur.execute(f"ALTER TABLE doc_catalog ADD COLUMN {_c} {_t}")
            except Exception:
                pass
        if dd == "sqlite":
            conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


init()


# ===================== Веб-чат: журнал и аналитика =====================

def log_request(question: str, category: str | None, n_hits: int,
                top_score: float, latency_ms: int, answer_chars: int,
                answered: bool, sources: list,
                retrieve_ms: int = 0, gen_ms: int = 0,
                session_id: str = "") -> int:
    now = datetime.now()
    try:
        with _LOCK:
            rid = _insert(
                """INSERT INTO requests
                   (ts,day,question,category,n_hits,top_score,latency_ms,
                    answer_chars,answered,sources,retrieve_ms,gen_ms,session_id)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now.timestamp(), now.strftime("%Y-%m-%d"), question, category,
                 n_hits, round(top_score, 4), latency_ms, answer_chars,
                 int(answered), json.dumps(sources, ensure_ascii=False),
                 int(retrieve_ms), int(gen_ms), session_id or ""),
            )
        _bump()
        return rid or 0
    except Exception as e:
        print(f"[db] log_request: {e}")
        return 0


def set_rating(req_id: int, rating: int) -> bool:
    try:
        with _LOCK:
            n = _exec("UPDATE requests SET rating=? WHERE id=?",
                      (int(rating), int(req_id)))
        _bump()
        return n > 0
    except Exception as e:
        print(f"[db] set_rating: {e}")
        return False


def rating_stats() -> dict:
    g = _one("SELECT COUNT(*) n FROM requests WHERE rating=1")["n"]
    b = _one("SELECT COUNT(*) n FROM requests WHERE rating=-1")["n"]
    tot = _one("SELECT COUNT(*) n FROM requests")["n"]
    rated = g + b
    return {"good": g, "bad": b, "rated": rated, "unrated": tot - rated,
            "satisfaction": round(g / rated * 100, 1) if rated else None}


def rating_analysis() -> dict:
    good = _one("SELECT AVG(top_score) s, COUNT(*) n FROM requests WHERE rating=1")
    bad = _one("SELECT AVG(top_score) s, COUNT(*) n FROM requests WHERE rating=-1")
    bad_no = _one("SELECT COUNT(*) n FROM requests WHERE rating=-1 AND answered=0")["n"]
    bad_yes = _one("SELECT COUNT(*) n FROM requests WHERE rating=-1 AND answered=1")["n"]
    return {"good_avg_score": _fn(good["s"]), "good_n": good["n"],
            "bad_avg_score": _fn(bad["s"]), "bad_n": bad["n"],
            "bad_no_answer": bad_no, "bad_answered": bad_yes}


def recent(limit: int = 100) -> list[dict]:
    rows = _all("SELECT * FROM requests ORDER BY id DESC LIMIT ?", (int(limit),))
    for d in rows:
        d["sources"] = json.loads(d.get("sources") or "[]")
    return rows


def stats() -> dict:
    import cache
    return cache.get_or_set("stats", 60, _stats_raw)


def _stats_raw() -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    total = _one("SELECT COUNT(*) n FROM requests")["n"]
    today_n = _one("SELECT COUNT(*) n FROM requests WHERE day=?", (today,))["n"]
    agg = _one("SELECT AVG(latency_ms) lat, AVG(top_score) sc, AVG(answered) ans, "
               "AVG(retrieve_ms) rt, AVG(gen_ms) gn FROM requests")
    return {
        "total": total,
        "today": today_n,
        "avg_latency_ms": round(_f(agg["lat"])),
        "avg_retrieve_ms": round(_f(agg["rt"])),
        "avg_gen_ms": round(_f(agg["gn"])),
        "avg_top_score": round(_f(agg["sc"]), 3),
        "answer_rate": round(_f(agg["ans"]) * 100, 1),
    }


def clear() -> int:
    """Очистить журнал запросов (история всех чатов и их статистика)."""
    with _LOCK:
        n = _one("SELECT COUNT(*) n FROM requests")["n"]
        _exec("DELETE FROM requests")
    _bump()
    return n


def session_history(session_id: str) -> list[dict]:
    if not session_id:
        return []
    rows = _all("SELECT * FROM requests WHERE session_id=? ORDER BY id ASC",
                (session_id,))
    for d in rows:
        d["sources"] = json.loads(d.get("sources") or "[]")
    return rows


def engine_usage() -> dict:
    g = _one("SELECT COUNT(*) n FROM requests WHERE category='graph'")["n"]
    l = _one("SELECT COUNT(*) n FROM requests WHERE category='lightrag'")["n"]
    tot = _one("SELECT COUNT(*) n FROM requests")["n"]
    return {"graph": g, "lightrag": l, "vector": max(tot - g - l, 0), "total": tot}


def analytics() -> dict:
    import cache
    return cache.get_or_set("analytics", 60, _analytics_raw)


def _analytics_raw() -> dict:
    per_day = _all("SELECT day, COUNT(*) n FROM requests GROUP BY day "
                   "ORDER BY day DESC LIMIT 14")
    per_cat = _all("SELECT COALESCE(NULLIF(category,''),'—') cat, COUNT(*) n "
                   "FROM requests GROUP BY COALESCE(NULLIF(category,''),'—') "
                   "ORDER BY n DESC")
    answered_rows = _all("SELECT sources FROM requests WHERE answered=1")
    lat_rows = _all("SELECT latency_ms FROM requests")
    q_rows = _all("SELECT question FROM requests")
    tm = _one("SELECT AVG(retrieve_ms) rt, AVG(gen_ms) gn, AVG(latency_ms) lt "
              "FROM requests")

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
        "timings": {"retrieve": round(_f(tm["rt"])), "gen": round(_f(tm["gn"])),
                    "total": round(_f(tm["lt"]))},
    }


# ===================== Telegram: пользователи и запросы =====================

def tg_user(chat_id: int) -> dict | None:
    try:
        return _one("SELECT * FROM tg_users WHERE chat_id=?", (int(chat_id),))
    except Exception as e:
        print(f"[db] tg_user: {e}")
        return None


def tg_user_upsert(chat_id: int, username: str | None, first_name: str | None,
                   status: str) -> dict | None:
    now = datetime.now().timestamp()
    try:
        with _LOCK, _cursor() as (d, conn, cur):
            cur.execute(_ph("SELECT chat_id FROM tg_users WHERE chat_id=?", d),
                        (int(chat_id),))
            ex = cur.fetchone()
            if ex:
                cur.execute(_ph("UPDATE tg_users SET username=?, first_name=?, "
                                "updated=? WHERE chat_id=?", d),
                            (username or "", first_name or "", now, int(chat_id)))
            else:
                cur.execute(_ph("INSERT INTO tg_users(chat_id,username,first_name,"
                                "status,created,updated,n_requests) "
                                "VALUES(?,?,?,?,?,?,0)", d),
                            (int(chat_id), username or "", first_name or "", status,
                             now, now))
            if d == "sqlite":
                conn.commit()
    except Exception as e:
        print(f"[db] tg_user_upsert: {e}")
    return tg_user(chat_id)


def tg_set_status(chat_id: int, status: str) -> bool:
    n = _exec("UPDATE tg_users SET status=?, updated=? WHERE chat_id=?",
              (status, datetime.now().timestamp(), int(chat_id)))
    return n > 0


def tg_users(status: str | None = None) -> list[dict]:
    if status:
        return _all("SELECT * FROM tg_users WHERE status=? ORDER BY updated DESC",
                    (status,))
    return _all("SELECT * FROM tg_users ORDER BY updated DESC")


def tg_counts() -> dict:
    rows = _all("SELECT status, COUNT(*) n FROM tg_users GROUP BY status")
    d = {r["status"]: r["n"] for r in rows}
    return {"pending": d.get("pending", 0), "approved": d.get("approved", 0),
            "blocked": d.get("blocked", 0), "total": sum(d.values())}


def tg_log_request(chat_id: int, username: str | None, question: str, answer: str,
                   n_hits: int, top_score: float, latency_ms: int,
                   answered: bool, sources: list) -> int:
    now = datetime.now()
    params = (now.timestamp(), now.strftime("%Y-%m-%d"), int(chat_id), username or "",
              question, answer, n_hits, round(top_score, 4), latency_ms,
              len(answer or ""), int(answered),
              json.dumps(sources, ensure_ascii=False))
    ins = ("INSERT INTO tg_requests(ts,day,chat_id,username,question,answer,"
           "n_hits,top_score,latency_ms,answer_chars,answered,sources)"
           "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)")
    try:
        with _LOCK, _cursor() as (d, conn, cur):
            if d == "postgresql":
                cur.execute(_ph(ins, d) + " RETURNING id", params)
                row = cur.fetchone()
                rid = (row["id"] if isinstance(row, dict) else row[0]) if row else 0
            else:
                cur.execute(_ph(ins, d), params)
                rid = cur.lastrowid
            cur.execute(_ph("UPDATE tg_users SET n_requests=n_requests+1 "
                            "WHERE chat_id=?", d), (int(chat_id),))
            if d == "sqlite":
                conn.commit()
        _bump()
        return rid or 0
    except Exception as e:
        print(f"[db] tg_log_request: {e}")
        return 0


def tg_recent(limit: int = 200) -> list[dict]:
    rows = _all("SELECT * FROM tg_requests ORDER BY id DESC LIMIT ?", (int(limit),))
    for d in rows:
        d["sources"] = json.loads(d.get("sources") or "[]")
    return rows


def tg_stats() -> dict:
    import cache
    return cache.get_or_set("tg_stats", 60, _tg_stats_raw)


def _tg_stats_raw() -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    total = _one("SELECT COUNT(*) n FROM tg_requests")["n"]
    today_n = _one("SELECT COUNT(*) n FROM tg_requests WHERE day=?", (today,))["n"]
    agg = _one("SELECT AVG(latency_ms) lat, AVG(answered) ans FROM tg_requests")
    users = _one("SELECT COUNT(DISTINCT chat_id) n FROM tg_requests")["n"]
    counts = tg_counts()
    return {"total": total, "today": today_n, "users": users,
            "avg_latency_ms": round(_f(agg["lat"])),
            "answer_rate": round(_f(agg["ans"]) * 100, 1),
            "pending": counts["pending"], "approved": counts["approved"],
            "blocked": counts["blocked"]}


def tg_clear_history() -> int:
    with _LOCK:
        n = _one("SELECT COUNT(*) n FROM tg_requests")["n"]
        _exec("DELETE FROM tg_requests")
    _bump()
    return n


# ===================== Копирование / миграция между СУБД =====================

# Полные списки колонок (с явным id для сохранения связей оценок/истории).
_TABLES = {
    "requests": ["id", "ts", "day", "question", "category", "n_hits", "top_score",
                 "latency_ms", "answer_chars", "answered", "sources", "rating",
                 "retrieve_ms", "gen_ms", "session_id"],
    "tg_users": ["chat_id", "username", "first_name", "status", "created", "updated",
                 "n_requests"],
    "tg_requests": ["id", "ts", "day", "chat_id", "username", "question", "answer",
                    "n_hits", "top_score", "latency_ms", "answer_chars", "answered",
                    "sources"],
}


def _runtime_text() -> str:
    return _RUNTIME.read_text(encoding="utf-8") if _RUNTIME.exists() else "{}"


def _kv_set(d: str, cur, k: str, v: str) -> None:
    if d == "mysql":
        cur.execute("INSERT INTO kv_store(k,v) VALUES(%s,%s) "
                    "ON DUPLICATE KEY UPDATE v=VALUES(v)", (k, v))
    elif d == "postgresql":
        cur.execute("INSERT INTO kv_store(k,v) VALUES(%s,%s) "
                    "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))
    else:
        cur.execute("INSERT INTO kv_store(k,v) VALUES(?,?) "
                    "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))


def kv_get(k: str) -> str | None:
    r = _one("SELECT v FROM kv_store WHERE k=?", (k,))
    return r["v"] if r else None


def _fix_seq(d: str, cur) -> None:
    """Выровнять автоинкремент приёмника после вставки явных id."""
    try:
        if d == "mysql":
            for t in ("requests", "tg_requests"):
                cur.execute(f"SELECT MAX(id) m FROM {t}")
                row = cur.fetchone()
                m = (row["m"] if isinstance(row, dict) else row[0]) or 0
                cur.execute(f"ALTER TABLE {t} AUTO_INCREMENT = {int(m) + 1}")
        elif d == "postgresql":
            for t in ("requests", "tg_requests"):
                cur.execute(
                    f"SELECT setval(pg_get_serial_sequence('{t}','id'), "
                    f"GREATEST((SELECT COALESCE(MAX(id),0) FROM {t}),1))")
    except Exception as e:
        print(f"[db] _fix_seq {d}: {e}")


def copy_all(target: str) -> dict:
    """Скопировать все данные (журналы) текущего бэкенда в target, а настройки
    (runtime_config) — в kv_store приёмника. Таблицы приёмника создаются по
    необходимости и очищаются перед вставкой (полная замена)."""
    target = _norm(target)
    src = _dialect()
    log: list[str] = [f"Источник: {src} → приёмник: {target}"]
    if not _driver_present(target):
        return {"ok": False, "target": target,
                "log": "\n".join(log) + f"\nДрайвер для {target} не установлен "
                       f"({'PyMySQL' if target == 'mysql' else 'psycopg2'})."}
    # читаем данные источника заранее (на текущем бэкенде)
    data = {}
    for table, cols in _TABLES.items():
        data[table] = _all(f"SELECT {','.join(cols)} FROM {table}")
    cfg = _runtime_text()

    init(target)  # гарантируем схему на приёмнике
    counts = {}
    try:
        td, tconn = _connect_for(target)
    except Exception as e:
        return {"ok": False, "target": target,
                "log": "\n".join(log) + f"\nПодключение к {target}: {_safe_err(e)}"}
    try:
        tcur = _mk_cursor(td, tconn)
        for table, cols in _TABLES.items():
            rows = data[table]
            try:
                tcur.execute(f"DELETE FROM {table}")
            except Exception as e:
                log.append(f"{table}: очистка приёмника: {e}")
            ins = _ph(f"INSERT INTO {table} ({','.join(cols)}) "
                      f"VALUES ({','.join(['?'] * len(cols))})", td)
            n = 0
            for r in rows:
                try:
                    tcur.execute(ins, [r.get(c) for c in cols])
                    n += 1
                except Exception as e:
                    log.append(f"{table}: строка пропущена: {e}")
            if td == "sqlite":
                tconn.commit()
            counts[table] = n
            log.append(f"{table}: скопировано {n} из {len(rows)}")
        _fix_seq(td, tcur)
        try:
            _kv_set(td, tcur, "runtime_config", cfg)
            if td == "sqlite":
                tconn.commit()
            counts["settings"] = 1
            log.append("Настройки (runtime_config) сохранены в kv_store приёмника.")
        except Exception as e:
            log.append(f"Настройки: {e}")
    finally:
        try:
            tconn.close()
        except Exception:
            pass
    return {"ok": True, "target": target, "counts": counts, "log": "\n".join(log)}


def migrate(target: str) -> dict:
    """Скопировать данные и переключить активный бэкенд на target."""
    res = copy_all(target)
    if res.get("ok"):
        try:
            _settings().update({"DB_BACKEND": _norm(target)})
            res["migrated"] = True
            res["log"] = res.get("log", "") + \
                f"\nАктивная БД переключена на {_norm(target)}."
        except Exception as e:
            res["log"] = res.get("log", "") + f"\nНе удалось переключить бэкенд: {e}"
    return res


def test_connection(backend: str) -> dict:
    d = _norm(backend)
    if d == "sqlite":
        return {"ok": True, "backend": "sqlite",
                "msg": "SQLite (локальный файл) доступен"}
    if not _driver_present(d):
        drv = "PyMySQL" if d == "mysql" else "psycopg2"
        return {"ok": False, "backend": d, "msg": f"драйвер не установлен ({drv})"}
    p = _params_for(d)
    if not (p.get("host") and p.get("db")):
        return {"ok": False, "backend": d,
                "msg": "не заданы хост/имя БД в настройках"}
    try:
        dd, conn = _connect_for(d)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        conn.close()
        return {"ok": True, "backend": d,
                "msg": f"{d}: подключение успешно ({p['host']}:{p['port']}/{p['db']})"}
    except Exception as e:
        return {"ok": False, "backend": d,
                "msg": f"{d}: ошибка подключения: {_safe_err(e)}"}


def _rv(row, key):
    if row is None:
        return None
    try:
        return row[key]
    except Exception:
        try:
            return row[0]
        except Exception:
            return None


def _backend_detail(dialect: str) -> dict:
    """Расширенная статистика по одному бэкенду: версия сервера, размер БД,
    число строк в таблицах (если доступен)."""
    d = _norm(dialect)
    cur = _dialect()
    info = {"name": d, "active": cur == d, "configured": d == "sqlite",
            "driver": True if d == "sqlite" else _driver_present(d),
            "reachable": False, "error": "", "version": "", "size_mb": None,
            "host": "", "counts": {}}
    if d != "sqlite":
        p = _params_for(d)
        info["configured"] = bool(p.get("host") and p.get("db"))
        if info["configured"]:
            info["host"] = f"{p['host']}:{p['port']}/{p['db']}"
    if d == "sqlite":
        try:
            info["host"] = str(DB_PATH)
            if DB_PATH.exists():
                info["size_mb"] = round(DB_PATH.stat().st_size / 1048576, 2)
        except Exception:
            pass
    elif not info["configured"] or not info["driver"]:
        return info
    try:
        dd, conn = _connect_for(d)
    except Exception as e:
        info["error"] = _safe_err(e)[:200]
        return info
    try:
        c = _mk_cursor(dd, conn)
        info["reachable"] = True
        if dd == "sqlite":
            c.execute("SELECT sqlite_version() v")
            info["version"] = _rv(c.fetchone(), "v")
        elif dd == "mysql":
            c.execute("SELECT VERSION() v")
            info["version"] = _rv(c.fetchone(), "v")
            try:
                c.execute("SELECT ROUND(SUM(data_length+index_length)/1048576,2) m "
                          "FROM information_schema.tables WHERE table_schema=DATABASE()")
                info["size_mb"] = _f(_rv(c.fetchone(), "m"))
            except Exception:
                pass
        else:  # postgresql
            try:
                c.execute("SELECT current_setting('server_version') v")
                info["version"] = _rv(c.fetchone(), "v")
            except Exception:
                pass
            try:
                c.execute("SELECT ROUND(pg_database_size(current_database())/1048576.0,2) m")
                info["size_mb"] = _f(_rv(c.fetchone(), "m"))
            except Exception:
                pass
        for t in ("requests", "tg_requests", "tg_users", "kv_store", "doc_catalog"):
            try:
                c.execute(f"SELECT COUNT(*) n FROM {t}")
                info["counts"][t] = _rv(c.fetchone(), "n")
            except Exception:
                info["counts"][t] = None
        conn.close()
    except Exception as e:
        info["error"] = _safe_err(e)[:200]
        try:
            conn.close()
        except Exception:
            pass
    return info


def system_stats() -> dict:
    """Развёрнутая статистика по всем бэкендам БД (для раздела «Система»)."""
    return {"active": _dialect(),
            "backends": {d: _backend_detail(d)
                         for d in ("sqlite", "mysql", "postgresql")}}


def db_status() -> dict:
    cur = _dialect()
    out = {"backend": cur, "backends": {}}
    out["backends"]["sqlite"] = {"configured": True, "driver": True,
                                 "reachable": True, "active": cur == "sqlite",
                                 "error": ""}
    for d in ("mysql", "postgresql"):
        p = _params_for(d)
        info = {"configured": bool(p.get("host") and p.get("db")),
                "driver": _driver_present(d), "reachable": False,
                "active": cur == d, "error": ""}
        if info["configured"] and info["driver"]:
            try:
                dd, conn = _connect_for(d)
                c = conn.cursor()
                c.execute("SELECT 1")
                c.fetchone()
                conn.close()
                info["reachable"] = True
            except Exception as e:
                info["error"] = _safe_err(e)[:200]
        out["backends"][d] = info
    return out


# ===================== Каталог документов в БД (doc_catalog) =====================
# Хранит документы ЦЕЛИКОМ: метаданные (имя, расширение, размер, дата, способ),
# контрольную сумму sha256 (для пропуска повторной загрузки одинаковых файлов),
# само содержимое файла (content) и извлечённый текст (txt, для предпросмотра).

# полный набор колонок для записи (порядок важен)
_CAT_COLS = "rel_path,fname,ext,size,mtime,n_chars,method,sha256,content,txt,updated"


def catalog_existing() -> dict:
    """Карта rel_path -> {sha256, size, mtime, has_content} для инкрементальной
    загрузки (без выборки самих файлов)."""
    try:
        rows = _all("SELECT rel_path, sha256, size, mtime, "
                    "(content IS NOT NULL OR content_oid IS NOT NULL) AS has_content "
                    "FROM doc_catalog")
    except Exception:
        return {}
    out = {}
    for r in rows:
        out[r["rel_path"]] = {"sha256": r.get("sha256"), "size": r.get("size"),
                              "mtime": r.get("mtime"),
                              "has_content": bool(r.get("has_content"))}
    return out


def catalog_store_many(rows: list) -> int:
    """Пакетная вставка/обновление записей каталога вместе с содержимым файлов.

    rows: список кортежей (rel_path, fname, ext, size, mtime, n_chars, method,
    sha256, content_bytes|None, txt). Одно соединение на пакет; для PostgreSQL —
    execute_values, для MySQL/SQLite — executemany. content передаётся как bytes
    (psycopg2/pymysql адаптируют в bytea/BLOB; для sqlite оборачиваем в Binary)."""
    if not rows:
        return 0
    now = datetime.now().timestamp()
    # content_oid=NULL: при сохранении небольшого файла в bytea сбрасываем ссылку на
    # возможный прежний Large Object (для крупных файлов используется отдельный путь).
    upd_pg = ("fname=excluded.fname,ext=excluded.ext,size=excluded.size,"
              "mtime=excluded.mtime,n_chars=excluded.n_chars,method=excluded.method,"
              "sha256=excluded.sha256,content=excluded.content,content_oid=NULL,"
              "txt=excluded.txt,updated=excluded.updated")
    upd_my = ("fname=VALUES(fname),ext=VALUES(ext),size=VALUES(size),"
              "mtime=VALUES(mtime),n_chars=VALUES(n_chars),method=VALUES(method),"
              "sha256=VALUES(sha256),content=VALUES(content),content_oid=NULL,"
              "txt=VALUES(txt),updated=VALUES(updated)")
    with _LOCK, _cursor() as (d, conn, cur):
        if d == "postgresql":
            from psycopg2 import Binary
            from psycopg2.extras import execute_values
            data = [(r[0], r[1] or "", r[2] or "", int(r[3] or 0), int(r[4] or 0),
                     int(r[5] or 0), r[6] or "", r[7] or "",
                     (Binary(r[8]) if r[8] is not None else None), r[9] or "", now)
                    for r in rows]
            execute_values(
                cur, f"INSERT INTO doc_catalog({_CAT_COLS}) VALUES %s "
                f"ON CONFLICT(rel_path) DO UPDATE SET {upd_pg}", data, page_size=50)
        elif d == "mysql":
            data = [(r[0], r[1] or "", r[2] or "", int(r[3] or 0), int(r[4] or 0),
                     int(r[5] or 0), r[6] or "", r[7] or "", r[8], r[9] or "", now)
                    for r in rows]
            cur.executemany(
                f"INSERT INTO doc_catalog({_CAT_COLS}) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                f"ON DUPLICATE KEY UPDATE {upd_my}", data)
        else:
            data = [(r[0], r[1] or "", r[2] or "", int(r[3] or 0), int(r[4] or 0),
                     int(r[5] or 0), r[6] or "", r[7] or "",
                     (sqlite3.Binary(r[8]) if r[8] is not None else None),
                     r[9] or "", now) for r in rows]
            cur.executemany(
                f"INSERT INTO doc_catalog({_CAT_COLS}) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                f"ON CONFLICT(rel_path) DO UPDATE SET {upd_pg}", data)
        if d == "sqlite":
            conn.commit()
    return len(rows)


def catalog_clear() -> int:
    """Полностью очистить каталог (метаданные, текст и файлы)."""
    try:
        with _LOCK:
            return _exec("DELETE FROM doc_catalog")
    except Exception as e:
        print(f"[db] catalog_clear: {e}")
        return 0


def _connect_pg_tx():
    """Отдельное соединение PostgreSQL в транзакции (autocommit=False) — нужно для
    операций с Large Object (lobject)."""
    import psycopg2
    p = _params_for("postgresql")
    return psycopg2.connect(host=p["host"], port=p["port"], user=p["user"],
                            password=p["password"], dbname=p["db"],
                            connect_timeout=5, client_encoding="UTF8")


def catalog_store_large_pg(rel_path, fname, ext, size, mtime, method, src_path,
                           sha256, txt="") -> bool:
    """Сохранить КРУПНЫЙ файл в PostgreSQL как Large Object (потоково, без загрузки в
    память; обходит лимит bytea в 1 ГБ). Содержимое читается из src_path по кускам."""
    conn = _connect_pg_tx()
    try:
        cur = conn.cursor()
        cur.execute("SELECT content_oid FROM doc_catalog WHERE rel_path=%s", (rel_path,))
        row = cur.fetchone()
        if row and row[0]:
            try:
                conn.lobject(oid=int(row[0]), mode="n").unlink()
            except Exception:
                pass
        lo = conn.lobject(0, "wb")
        with open(src_path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                lo.write(chunk)
        oid = lo.oid
        lo.close()
        now = datetime.now().timestamp()
        cur.execute(
            "INSERT INTO doc_catalog(rel_path,fname,ext,size,mtime,n_chars,method,"
            "sha256,content,content_oid,txt,updated) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,NULL,%s,%s,%s) "
            "ON CONFLICT(rel_path) DO UPDATE SET fname=excluded.fname,ext=excluded.ext,"
            "size=excluded.size,mtime=excluded.mtime,n_chars=excluded.n_chars,"
            "method=excluded.method,sha256=excluded.sha256,content=NULL,"
            "content_oid=excluded.content_oid,txt=excluded.txt,updated=excluded.updated",
            (rel_path, fname or "", ext or "", int(size or 0), int(mtime or 0),
             len(txt or ""), method or "", sha256 or "", oid, txt or "", now))
        conn.commit()
        return True
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"[db] catalog_store_large_pg {rel_path}: {e}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def catalog_export_to(rel_path: str, dst_path) -> bool:
    """Выгрузить содержимое файла из каталога в dst_path потоково (Large Object или
    bytea/BLOB). Работает без загрузки гигабайтных файлов в память."""
    d = _dialect()
    if d == "postgresql":
        conn = _connect_pg_tx()
        try:
            cur = conn.cursor()
            cur.execute("SELECT content_oid, content FROM doc_catalog "
                        "WHERE rel_path=%s", (rel_path,))
            row = cur.fetchone()
            if not row:
                return False
            oid, content = row[0], row[1]
            if oid:
                lo = conn.lobject(oid=int(oid), mode="rb")
                with open(dst_path, "wb") as f:
                    while True:
                        chunk = lo.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk if isinstance(chunk, (bytes, bytearray))
                                else bytes(chunk))
                lo.close()
                conn.commit()
                return True
            if content is not None:
                b = content.tobytes() if isinstance(content, memoryview) else bytes(content)
                Path(dst_path).write_bytes(b)
                conn.commit()
                return True
            return False
        except Exception as e:
            print(f"[db] catalog_export_to {rel_path}: {e}")
            return False
        finally:
            try:
                conn.close()
            except Exception:
                pass
    # MySQL/SQLite — содержимое в bytea/BLOB
    c = catalog_get_content(rel_path)
    if c is None:
        return False
    try:
        Path(dst_path).write_bytes(c)
        return True
    except Exception:
        return False


def catalog_file_info(rel_path: str) -> dict:
    """Дёшево: есть ли содержимое, его sha256 и размер (без выборки самого файла)."""
    try:
        r = _one("SELECT sha256, size, "
                 "(content IS NOT NULL OR content_oid IS NOT NULL) AS has "
                 "FROM doc_catalog WHERE rel_path=?", (rel_path,))
    except Exception:
        r = None
    if not r:
        return {"has": False, "sha": "", "size": 0}
    return {"has": bool(r.get("has")), "sha": r.get("sha256") or "",
            "size": int(r.get("size") or 0)}


def catalog_clear_files() -> int:
    """Удалить ТОЛЬКО содержимое файлов (bytea и Large Object) и контрольные суммы,
    оставив метаданные и текст. Освобождает место; повторная загрузка снова сохранит."""
    d = _dialect()
    try:
        if d == "postgresql":
            conn = _connect_pg_tx()
            try:
                cur = conn.cursor()
                cur.execute("SELECT content_oid FROM doc_catalog "
                            "WHERE content_oid IS NOT NULL")
                for (oid,) in cur.fetchall():
                    try:
                        conn.lobject(oid=int(oid), mode="n").unlink()
                    except Exception:
                        pass
                cur.execute("UPDATE doc_catalog SET content=NULL, content_oid=NULL, "
                            "sha256=NULL WHERE content IS NOT NULL "
                            "OR content_oid IS NOT NULL")
                n = cur.rowcount
                conn.commit()
                return n
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        with _LOCK:
            return _exec("UPDATE doc_catalog SET content=NULL, sha256=NULL "
                         "WHERE content IS NOT NULL")
    except Exception as e:
        print(f"[db] catalog_clear_files: {e}")
        return 0


def catalog_count() -> int:
    try:
        return _one("SELECT COUNT(*) n FROM doc_catalog")["n"]
    except Exception:
        return 0


def catalog_meta() -> dict:
    try:
        m = _one("SELECT COUNT(*) n, COALESCE(SUM(size),0) sz, MAX(updated) up, "
                 "SUM(CASE WHEN content IS NOT NULL OR content_oid IS NOT NULL "
                 "THEN 1 ELSE 0 END) fs FROM doc_catalog")
        by = _all("SELECT ext, COUNT(*) n FROM doc_catalog GROUP BY ext ORDER BY n DESC")
        return {"count": m["n"], "total_size": int(_f(m["sz"])),
                "files_stored": m.get("fs", 0) or 0,
                "updated": _fn(m["up"]),
                "by_ext": {r["ext"]: r["n"] for r in by}}
    except Exception:
        return {"count": 0, "total_size": 0, "files_stored": 0,
                "updated": None, "by_ext": {}}


def catalog_rows() -> list[dict]:
    """Метаданные всех записей каталога (без текста) — для списка."""
    try:
        return _all("SELECT rel_path, ext, size, mtime, n_chars, method "
                    "FROM doc_catalog")
    except Exception:
        return []


def catalog_index_list() -> list[dict]:
    """Список файлов каталога с содержимым (для индексации напрямую из БД, без папки)."""
    try:
        return _all("SELECT rel_path, fname, ext, sha256 FROM doc_catalog "
                    "WHERE content IS NOT NULL OR content_oid IS NOT NULL "
                    "ORDER BY rel_path")
    except Exception as e:
        print(f"[db] catalog_index_list: {e}")
        return []


def catalog_has_content(rel_path: str) -> bool:
    """Есть ли в каталоге сохранённое содержимое файла (без выборки самого блоба)."""
    try:
        r = _one("SELECT 1 AS x FROM doc_catalog WHERE rel_path=? "
                 "AND (content IS NOT NULL OR content_oid IS NOT NULL)", (rel_path,))
        return bool(r)
    except Exception:
        return False


def catalog_get_content(rel_path: str):
    """Содержимое файла (bytes) из каталога или None."""
    try:
        r = _one("SELECT content FROM doc_catalog WHERE rel_path=?", (rel_path,))
    except Exception as e:
        print(f"[db] catalog_get_content: {e}")
        return None
    if not r:
        return None
    c = r.get("content")
    if c is None:
        return None
    if isinstance(c, memoryview):       # psycopg2 отдаёт bytea как memoryview
        return c.tobytes()
    if isinstance(c, bytearray):
        return bytes(c)
    return c


def catalog_update_text(rel_path: str, txt: str, n_chars: int | None = None) -> None:
    """Записать извлечённый текст (предпросмотр) в запись каталога — при индексации
    напрямую из БД, чтобы предпросмотр работал без папки с файлами."""
    try:
        with _LOCK:
            _exec("UPDATE doc_catalog SET txt=?, n_chars=? WHERE rel_path=?",
                  (txt or "",
                   int(n_chars if n_chars is not None else len(txt or "")), rel_path))
    except Exception as e:
        print(f"[db] catalog_update_text: {e}")


def catalog_text(rel_path: str, max_chars: int = 20000) -> dict | None:
    try:
        r = _one("SELECT txt, method, n_chars FROM doc_catalog WHERE rel_path=?",
                 (rel_path,))
    except Exception:
        r = None
    if not r:
        return None
    t = r.get("txt") or ""
    return {"text": t[:max_chars], "method": r.get("method"),
            "n_chars": r.get("n_chars") if r.get("n_chars") is not None else len(t),
            "truncated": len(t) > max_chars}
