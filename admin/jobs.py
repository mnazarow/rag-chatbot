"""Примитивы фоновых задач админ-подсистемы (вынесено из admin_ops.py).

Централизует чтение живых логов и БЕЗОПАСНУЮ сериализацию статуса задачи. Раньше
``status()`` делал ``dict(job)`` напрямую, из-за чего в JSON-ответ /api/admin/status
попадал несериализуемый объект ``Popen`` (ключ ``_proc``) → 500 во время индексации.
Единый ``jobview`` выкидывает служебные (начинающиеся с «_») поля.
"""
from __future__ import annotations

from pathlib import Path

# Ключи состояния задачи, которые нельзя сериализовать в JSON (Popen и т.п.).
# Общее правило: любое поле, имя которого начинается с «_», считается служебным.
_NONSERIAL_PREFIX = "_"


def _tail(path: str, n: int = 6000) -> str:
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read()[-n:]
    except Exception:
        return ""


def _read_full_log(path: str, cap: int = 5_000_000) -> str:
    """Полный текст лог-файла (с ограничением размера)."""
    try:
        t = Path(path).read_text(errors="ignore")
        return t[-cap:] if len(t) > cap else t
    except Exception:
        return ""


def jobview(jb: dict) -> dict:
    """JSON-безопасный снимок задачи для /api/admin/status.

    - выкидывает служебные поля (``_proc`` и любые «_*») — они несериализуемы;
    - для идущей задачи подставляет живой хвост её logfile.
    """
    d = dict(jb)
    for k in list(d.keys()):
        if isinstance(k, str) and k.startswith(_NONSERIAL_PREFIX):
            d.pop(k, None)
    if d.get("running") and d.get("logfile"):
        d["log"] = _tail(d["logfile"])
    return d
