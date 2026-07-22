"""Лёгкий реестр обращений к компонентам конвейера (Qdrant, эмбеддер, реранкер,
БД, LightRAG, KAG). Потокобезопасно, в памяти процесса приложения.

Хранит кумулятивные счётчики: число вызовов, ошибок, суммарное время. Из них
дашборд считает скорость (запросов/с) и среднюю задержку в реальном времени —
разностью между двумя опросами. Компоненты, работающие в отдельном процессе
индексации (ingest.py), здесь не учитываются — это метрики «горячего» пути
запросов веб-чата/Телеграма/VoIP.
"""
from __future__ import annotations
import threading
import time

_lock = threading.Lock()
_data: dict = {}   # name -> {"calls", "errors", "total_ms", "last_ts"}


def _slot(name: str) -> dict:
    d = _data.get(name)
    if d is None:
        d = {"calls": 0, "errors": 0, "total_ms": 0.0, "last_ts": 0.0}
        _data[name] = d
    return d


def record(name: str, ms: float = 0.0, ok: bool = True) -> None:
    """Зафиксировать один вызов компонента name с длительностью ms (мс)."""
    with _lock:
        d = _slot(name)
        d["calls"] += 1
        if not ok:
            d["errors"] += 1
        d["total_ms"] += float(ms or 0.0)
        d["last_ts"] = time.time()


class timer:
    """Контекст-менеджер замера: `with metrics.timer("qdrant"): ...`."""

    __slots__ = ("name", "_t")

    def __init__(self, name: str):
        self.name = name
        self._t = 0.0

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        record(self.name, (time.perf_counter() - self._t) * 1000.0, exc_type is None)
        return False


def snapshot() -> dict:
    """Копия кумулятивных счётчиков по всем компонентам + метка времени сервера."""
    with _lock:
        comps = {k: dict(v) for k, v in _data.items()}
    for v in comps.values():
        v["avg_ms"] = round(v["total_ms"] / v["calls"], 1) if v["calls"] else 0.0
    return {"ts": time.time(), "components": comps}


def reset() -> None:
    with _lock:
        _data.clear()
        _counters.clear()
        _gauges.clear()
        _hist.clear()


# ======================================================================
# Расширенный реестр метрик (для /metrics в формате Prometheus 0.0.4).
# Три семейства: counters (монотонные), gauges (мгновенные), histograms
# (латентность). Всё под тем же _lock — операции дешёвые (dict + арифметика).
# ======================================================================

# ключ = (name, labels_tuple), где labels_tuple = tuple(sorted(labels.items()))
_counters: dict = {}          # -> float
_gauges: dict = {}            # -> float
_hist: dict = {}              # -> {"count": int, "sum": float, "buckets": [int,...]}

# Границы бакетов гистограмм латентности (секунды), как в клиентах Prometheus.
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

# Опциональные строки HELP по имени метрики (для читаемого /metrics).
_HELP: dict = {}


def _labels_key(labels: dict) -> tuple:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def inc(name: str, n: float = 1, **labels) -> None:
    """Увеличить счётчик name на n (по умолчанию 1) с метками labels."""
    k = (name, _labels_key(labels))
    with _lock:
        _counters[k] = _counters.get(k, 0.0) + float(n)


def set_gauge(name: str, value: float, **labels) -> None:
    """Установить текущее значение gauge (глубина очереди, активные слоты и т.п.)."""
    k = (name, _labels_key(labels))
    with _lock:
        _gauges[k] = float(value)


def observe(name: str, seconds: float, **labels) -> None:
    """Зафиксировать наблюдение латентности (секунды) в гистограмму name."""
    try:
        v = float(seconds)
    except Exception:
        return
    k = (name, _labels_key(labels))
    with _lock:
        h = _hist.get(k)
        if h is None:
            h = {"count": 0, "sum": 0.0, "buckets": [0] * len(_BUCKETS)}
            _hist[k] = h
        h["count"] += 1
        h["sum"] += v
        b = h["buckets"]
        for i, ub in enumerate(_BUCKETS):
            if v <= ub:
                b[i] += 1


def set_help(name: str, text: str) -> None:
    """Задать строку HELP для метрики (необязательно)."""
    with _lock:
        _HELP[name] = text


_METRIC_NAME_OK = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:")


def _safe_name(name: str) -> str:
    s = "".join(ch if ch in _METRIC_NAME_OK else "_" for ch in str(name))
    if s and (s[0].isdigit()):
        s = "_" + s
    return s or "_"


def _esc_label(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _fmt_labels(label_tuple: tuple, extra: tuple | None = None) -> str:
    items = list(label_tuple)
    if extra:
        items = items + [extra]
    if not items:
        return ""
    inner = ",".join(f'{_safe_name(k)}="{_esc_label(v)}"' for k, v in items)
    return "{" + inner + "}"


def _fmt_num(x: float) -> str:
    xf = float(x)
    if xf == int(xf):
        return str(int(xf))
    return repr(xf)


def render_prometheus() -> str:
    """Сериализовать все метрики в Prometheus text exposition format 0.0.4.

    Включает: расширенные counters/gauges/histograms (inc/set_gauge/observe) и
    кумулятивные счётчики компонентов из record()/timer() (calls/errors/seconds).
    Предназначено для эндпоинта /metrics (его подключает app.py)."""
    with _lock:
        counters = dict(_counters)
        gauges = dict(_gauges)
        hist = {k: {"count": v["count"], "sum": v["sum"],
                    "buckets": list(v["buckets"])} for k, v in _hist.items()}
        comps = {k: dict(v) for k, v in _data.items()}
        help_map = dict(_HELP)

    lines: list = []

    # --- группируем по имени, чтобы HELP/TYPE выводились один раз на метрику ---
    def _by_name(d: dict) -> dict:
        g: dict = {}
        for (name, lbls), val in d.items():
            g.setdefault(name, []).append((lbls, val))
        return g

    # counters
    for name, series in _by_name(counters).items():
        mn = _safe_name(name)
        if name in help_map:
            lines.append(f"# HELP {mn} {help_map[name]}")
        lines.append(f"# TYPE {mn} counter")
        for lbls, val in series:
            lines.append(f"{mn}{_fmt_labels(lbls)} {_fmt_num(val)}")

    # gauges
    for name, series in _by_name(gauges).items():
        mn = _safe_name(name)
        if name in help_map:
            lines.append(f"# HELP {mn} {help_map[name]}")
        lines.append(f"# TYPE {mn} gauge")
        for lbls, val in series:
            lines.append(f"{mn}{_fmt_labels(lbls)} {_fmt_num(val)}")

    # histograms
    for name, series in _by_name(hist).items():
        mn = _safe_name(name)
        if name in help_map:
            lines.append(f"# HELP {mn} {help_map[name]}")
        lines.append(f"# TYPE {mn} histogram")
        for lbls, h in series:
            # h["buckets"][i] уже кумулятивно (число наблюдений <= _BUCKETS[i]).
            for i, ub in enumerate(_BUCKETS):
                lines.append(f"{mn}_bucket{_fmt_labels(lbls, ('le', repr(ub)))} {h['buckets'][i]}")
            lines.append(f"{mn}_bucket{_fmt_labels(lbls, ('le', '+Inf'))} {h['count']}")
            lines.append(f"{mn}_sum{_fmt_labels(lbls)} {_fmt_num(h['sum'])}")
            lines.append(f"{mn}_count{_fmt_labels(lbls)} {h['count']}")

    # --- кумулятивные компоненты из record()/timer() как counters ---
    if comps:
        lines.append("# HELP rag_component_calls_total Число вызовов компонента конвейера")
        lines.append("# TYPE rag_component_calls_total counter")
        for name, v in comps.items():
            lbl = _fmt_labels((("component", name),))
            lines.append(f"rag_component_calls_total{lbl} {_fmt_num(v.get('calls', 0))}")
        lines.append("# HELP rag_component_errors_total Число ошибок компонента конвейера")
        lines.append("# TYPE rag_component_errors_total counter")
        for name, v in comps.items():
            lbl = _fmt_labels((("component", name),))
            lines.append(f"rag_component_errors_total{lbl} {_fmt_num(v.get('errors', 0))}")
        lines.append("# HELP rag_component_seconds_total Суммарное время вызовов компонента")
        lines.append("# TYPE rag_component_seconds_total counter")
        for name, v in comps.items():
            lbl = _fmt_labels((("component", name),))
            secs = float(v.get("total_ms", 0.0)) / 1000.0
            lines.append(f"rag_component_seconds_total{lbl} {_fmt_num(secs)}")

    return "\n".join(lines) + "\n"
