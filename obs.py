"""Единый структурный логгер поверх stdlib logging — без внешних зависимостей.

Заменяет разрозненные print(...) и «немые» except: pass на единообразное
структурное логирование. Управляется настройками:

  LOG_LEVEL  — уровень корневого логгера (DEBUG/INFO/WARNING/ERROR); по умолч. INFO;
  LOG_JSON   — если истина, каждая запись выводится JSON-строкой
               {ts, level, logger, msg, ...extra}; иначе — человекочитаемо.

Инициализация идемпотентна: повторные вызовы setup() не плодят хендлеры.
get_logger(name) гарантированно возвращает настроенный logging.Logger.
log_exc(logger, msg, exc) — единый способ логировать пойманное исключение:
сообщение на WARNING/ERROR, полный трейс — на DEBUG.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time

_setup_lock = threading.Lock()
_setup_done = False
_HANDLER_TAG = "_rag_obs"          # метка нашего хендлера (для идемпотентности)

# Стандартные атрибуты LogRecord — всё остальное считаем «extra» и выносим в структуру.
_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {
    "message", "asctime", "taskName",
}


def _cfg(key: str, default):
    """Безопасно прочитать настройку; при любой проблеме — дефолт (нет жёсткой связи)."""
    try:
        import settings
        v = settings.get(key)
        return default if v is None else v
    except Exception:
        return default


def _extras(record: logging.LogRecord) -> dict:
    """Собрать переданные через logging(..., extra={...}) поля."""
    out = {}
    for k, v in record.__dict__.items():
        if k in _RESERVED or k.startswith("_"):
            continue
        out[k] = v
    return out


class _JsonFormatter(logging.Formatter):
    """JSON-строка на запись: {ts, level, logger, msg, ...extra[, exc]}."""

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
                  + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in _extras(record).items():
            base[k] = v
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        try:
            return json.dumps(base, ensure_ascii=False, default=str)
        except Exception:
            # последний рубеж — не дать логированию упасть
            return json.dumps({"level": record.levelname, "logger": record.name,
                               "msg": record.getMessage()}, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    """Человекочитаемо: `2026-07-22T12:00:00.123 LEVEL logger: msg key=val`."""

    def format(self, record: logging.LogRecord) -> str:
        ts = (time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
              + f".{int(record.msecs):03d}")
        head = f"{ts} {record.levelname:<7} {record.name}: {record.getMessage()}"
        extras = _extras(record)
        if extras:
            head += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            head += "\n" + self.formatException(record.exc_info)
        return head


def _level_value(name) -> int:
    try:
        if isinstance(name, int):
            return name
        return getattr(logging, str(name).strip().upper(), logging.INFO)
    except Exception:
        return logging.INFO


def setup(force: bool = False) -> None:
    """Идемпотентно настроить корневой логгер из настроек (LOG_LEVEL, LOG_JSON).

    Не плодит хендлеры: наш хендлер помечен _HANDLER_TAG, при повторном вызове он
    переиспользуется (обновляются уровень и форматтер). force=True — пересобрать
    форматтер/уровень (напр. после смены LOG_JSON в админке)."""
    global _setup_done
    with _setup_lock:
        if _setup_done and not force:
            return
        level = _level_value(_cfg("LOG_LEVEL", "INFO"))
        use_json = bool(_cfg("LOG_JSON", False))
        fmt = _JsonFormatter() if use_json else _TextFormatter()
        root = logging.getLogger()
        root.setLevel(level)
        handler = None
        for h in root.handlers:
            if getattr(h, _HANDLER_TAG, False):
                handler = h
                break
        if handler is None:
            handler = logging.StreamHandler(sys.stderr)
            setattr(handler, _HANDLER_TAG, True)
            root.addHandler(handler)
        handler.setLevel(level)
        handler.setFormatter(fmt)
        _setup_done = True


def get_logger(name: str) -> logging.Logger:
    """Вернуть настроенный логгер. Ленивая идемпотентная инициализация корня."""
    if not _setup_done:
        setup()
    return logging.getLogger(name)


def log_exc(logger: logging.Logger, msg: str, exc: BaseException,
            level: int = logging.WARNING, **extra) -> None:
    """Единообразно залогировать пойманное исключение.

    Короткое сообщение (msg + тип/текст исключения) — на заданном уровне
    (по умолчанию WARNING; для фатальных передайте logging.ERROR). Полный трейс —
    отдельной записью на DEBUG (не шумит в обычном режиме, доступен при отладке).
    Любые именованные аргументы попадут в структурные поля записи."""
    try:
        logger.log(level, "%s: %s: %s", msg, type(exc).__name__, exc,
                   extra=extra or None)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s: трейс", msg,
                         exc_info=(type(exc), exc, exc.__traceback__))
    except Exception:
        # логирование не должно ломать вызывающий код
        pass
