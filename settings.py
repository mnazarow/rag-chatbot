"""Централизованные настройки — единый источник правды в рантайме.

Значения по умолчанию берутся из config (.env) при первом запуске. Дальше
правятся из админ-панели и сохраняются в runtime_config.json.

scope каждого поля определяет, как применяется изменение:
  - live    : сразу (поиск/генерация читают на каждом запросе)
  - reindex : вступит в силу после переиндексации (кнопка в админке)
  - restart : вступит в силу после перезапуска сервиса (кнопка в админке)
  - vllm    : требует перезапуска контейнера vLLM (кнопка «Применить модель»)
  - secret  : пароль; в API не отдаётся открытым текстом
"""
from __future__ import annotations
import json
import threading
from pathlib import Path

import config
import prompts

_RUNTIME = Path(__file__).resolve().parent / "runtime_config.json"
_LOCK = threading.Lock()

# Полная схема: ключ, подпись, группа, тип, scope, доп. параметры.
FIELDS: list[dict] = [
    # --- Поиск и генерация (на лету) ---
    {"key": "TOP_K_RETRIEVE", "label": "Кандидатов из БД", "group": "Поиск и генерация",
     "type": "range", "scope": "live", "min": 5, "max": 50, "step": 1, "default": config.TOP_K_RETRIEVE},
    {"key": "TOP_K_RERANK", "label": "Контекста после реранка", "group": "Поиск и генерация",
     "type": "range", "scope": "live", "min": 1, "max": 15, "step": 1, "default": config.TOP_K_RERANK},
    {"key": "MIN_SCORE", "label": "Порог релевантности", "group": "Поиск и генерация",
     "type": "range", "scope": "live", "min": 0, "max": 1, "step": 0.01, "default": config.MIN_SCORE},
    {"key": "TEMPERATURE", "label": "Температура генерации", "group": "Поиск и генерация",
     "type": "range", "scope": "live", "min": 0, "max": 1, "step": 0.05, "default": 0.1},
    {"key": "AUTO_FILTER", "label": "Авто-фильтр по категории вопроса", "group": "Поиск и генерация",
     "type": "bool", "scope": "live", "default": True},
    {"key": "AUTO_CALIBRATE", "label": "Авто-калибровка настроек по оценкам ответов",
     "group": "Поиск и генерация", "type": "bool", "scope": "live", "default": False},
    {"key": "SYSTEM_PROMPT", "label": "Системный промпт", "group": "Поиск и генерация",
     "type": "textarea", "scope": "live", "default": prompts.SYSTEM_PROMPT},

    # --- Генерация / бэкенд ---
    {"key": "LLM_BACKEND", "label": "Бэкенд LLM", "group": "Генерация (LLM)",
     "type": "select", "scope": "restart", "options": ["ollama", "openai"], "default": config.LLM_BACKEND},
    {"key": "LLM_MODEL", "label": "Имя модели (для запросов)", "group": "Генерация (LLM)",
     "type": "text", "scope": "live", "default": config.LLM_MODEL},
    {"key": "OLLAMA_URL", "label": "URL Ollama", "group": "Генерация (LLM)",
     "type": "text", "scope": "live", "default": config.OLLAMA_URL},
    {"key": "LLM_BASE_URL", "label": "URL vLLM (OpenAI API)", "group": "Генерация (LLM)",
     "type": "text", "scope": "live", "default": config.LLM_BASE_URL},
    {"key": "LLM_API_KEY", "label": "API-ключ vLLM", "group": "Генерация (LLM)",
     "type": "text", "scope": "live", "default": config.LLM_API_KEY},

    # --- vLLM контейнер (GPU) ---
    {"key": "VLLM_MODEL", "label": "Модель vLLM (контейнер)", "group": "vLLM (GPU)",
     "type": "text", "scope": "vllm", "default": config.VLLM_MODEL},
    {"key": "VLLM_MAX_LEN", "label": "Макс. длина контекста", "group": "vLLM (GPU)",
     "type": "int", "scope": "vllm", "default": config.VLLM_MAX_LEN},
    {"key": "VLLM_TP", "label": "Tensor-parallel (число GPU)", "group": "vLLM (GPU)",
     "type": "int", "scope": "vllm", "default": config.VLLM_TP},

    # --- Эмбеддинги / устройство ---
    {"key": "EMBED_MODEL", "label": "Модель эмбеддингов", "group": "Эмбеддинги и устройство",
     "type": "text", "scope": "restart", "default": config.EMBED_MODEL},
    {"key": "RERANK_MODEL", "label": "Модель реранка", "group": "Эмбеддинги и устройство",
     "type": "text", "scope": "restart", "default": config.RERANK_MODEL},
    {"key": "DEVICE", "label": "Устройство", "group": "Эмбеддинги и устройство",
     "type": "select", "scope": "restart", "options": ["cuda", "mps", "cpu"], "default": config.DEVICE},

    # --- Хранилище ---
    {"key": "QDRANT_URL", "label": "URL Qdrant", "group": "Хранилище",
     "type": "text", "scope": "restart", "default": config.QDRANT_URL},
    {"key": "QDRANT_COLLECTION", "label": "Коллекция", "group": "Хранилище",
     "type": "text", "scope": "restart", "default": config.QDRANT_COLLECTION},
    {"key": "QDRANT_TIMEOUT", "label": "Таймаут запросов Qdrant, с", "group": "Хранилище",
     "type": "int", "scope": "restart", "default": config.QDRANT_TIMEOUT},
    {"key": "QDRANT_INGEST_TIMEOUT", "label": "Таймаут индексации Qdrant, с", "group": "Хранилище",
     "type": "int", "scope": "reindex", "default": config.QDRANT_INGEST_TIMEOUT},

    # --- Документы и индексация ---
    {"key": "DOCS_DIR", "label": "Папка с документами", "group": "Документы и индексация",
     "type": "text", "scope": "reindex", "default": str(config.DOCS_DIR)},
    {"key": "CHUNK_SIZE", "label": "Размер чанка (символов)", "group": "Документы и индексация",
     "type": "int", "scope": "reindex", "default": config.CHUNK_SIZE},
    {"key": "CHUNK_OVERLAP", "label": "Перекрытие чанков", "group": "Документы и индексация",
     "type": "int", "scope": "reindex", "default": config.CHUNK_OVERLAP},

    # --- Транскрибация ---
    {"key": "WHISPER_BACKEND", "label": "Бэкенд Whisper", "group": "Транскрибация",
     "type": "select", "scope": "reindex", "options": ["faster", "mlx"], "default": config.WHISPER_BACKEND},
    {"key": "WHISPER_MODEL", "label": "Модель Whisper", "group": "Транскрибация",
     "type": "text", "scope": "reindex", "default": config.WHISPER_MODEL},

    # --- Расширенный поиск (hybrid+) ---
    {"key": "LLM_METADATA", "label": "LLM-метаданные при индексации (продукт/тема/тип)",
     "group": "Расширенный поиск (hybrid+)", "type": "bool", "scope": "reindex", "default": False},
    {"key": "SMART_FILTER", "label": "Умные фильтры из вопроса (LLM)",
     "group": "Расширенный поиск (hybrid+)", "type": "bool", "scope": "live", "default": False},
    {"key": "GRAPH_RAG", "label": "Граф-RAG для сводных вопросов (нужен LightRAG)",
     "group": "Расширенный поиск (hybrid+)", "type": "bool", "scope": "live", "default": False},
    {"key": "GRAPH_MODE", "label": "Режим графа", "group": "Расширенный поиск (hybrid+)",
     "type": "select", "scope": "live", "options": ["mix", "hybrid", "local", "global", "naive"],
     "default": "mix"},

    # --- Движок ответов ---
    {"key": "ENGINE", "label": "Движок ответов", "group": "Движок ответов",
     "type": "select", "scope": "live", "options": ["vector", "lightrag"],
     "default": "vector"},

    # --- Дообучение (fine-tuning) ---
    {"key": "USE_FINETUNED", "label": "Использовать дообученную модель",
     "group": "Дообучение (fine-tuning)", "type": "bool", "scope": "live", "default": False},
    {"key": "FINETUNED_MODEL", "label": "Имя дообученной модели (LoRA в vLLM)",
     "group": "Дообучение (fine-tuning)", "type": "text", "scope": "live", "default": "company-lora"},

    # --- Доступ ---
    {"key": "ADMIN_TOKEN", "label": "Токен администратора", "group": "Доступ",
     "type": "secret", "scope": "live", "default": config.ADMIN_TOKEN},
]

DEFAULTS: dict = {f["key"]: f["default"] for f in FIELDS}
_TYPES: dict = {f["key"]: f["type"] for f in FIELDS}
_OPTIONS: dict = {f["key"]: f.get("options") for f in FIELDS if f["type"] == "select"}

# Режимы работы — пресеты, которые разом включают связку улучшений.
MODES: dict = {
    "basic": {
        "label": "Базовый",
        "desc": "Вектор + реранк. Быстро, без дополнительных вызовов LLM.",
        "flags": {"SMART_FILTER": False, "GRAPH_RAG": False, "AUTO_FILTER": True},
    },
    "meta": {
        "label": "С метаданными",
        "desc": "+ умные фильтры из вопроса (категория / продукт / период).",
        "flags": {"SMART_FILTER": True, "GRAPH_RAG": False, "AUTO_FILTER": True},
    },
    "hybrid": {
        "label": "Hybrid+ (граф)",
        "desc": "+ граф-RAG для сводных вопросов (требуется LightRAG).",
        "flags": {"SMART_FILTER": True, "GRAPH_RAG": True, "AUTO_FILTER": True},
    },
}

_state: dict = dict(DEFAULTS)
_state["MODE"] = "basic"


def _load() -> None:
    if _RUNTIME.exists():
        try:
            data = json.loads(_RUNTIME.read_text(encoding="utf-8"))
            for k, v in data.items():
                if k in _state:
                    _state[k] = v
        except Exception:
            pass


_load()


def get(key: str):
    return _state.get(key, DEFAULTS.get(key))


def active_model() -> str:
    """Имя модели для генерации: дообученная (если включена) или базовая."""
    if _state.get("USE_FINETUNED"):
        return _state.get("FINETUNED_MODEL") or _state.get("LLM_MODEL")
    return _state.get("LLM_MODEL")


def all_settings() -> dict:
    return dict(_state)


def public_settings() -> dict:
    """Значения для UI: секреты маскируются."""
    out = dict(_state)
    for f in FIELDS:
        if f["type"] == "secret":
            out[f["key"]] = ""  # не отдаём пароль наружу
    return out


def secret_is_set(key: str) -> bool:
    return bool(_state.get(key))


def _coerce(key: str, value):
    t = _TYPES.get(key, "text")
    if t == "bool":
        return bool(value)
    if t == "int" or t == "range" and isinstance(DEFAULTS[key], int):
        return int(value)
    if t == "range" or (t in ("float",)):
        return float(value)
    return str(value)


def update(changes: dict) -> dict:
    with _LOCK:
        for k, v in changes.items():
            if k not in DEFAULTS:
                continue
            # пустой секрет = «не менять»
            if _TYPES[k] == "secret" and (v is None or v == ""):
                continue
            # select: только значения из списка опций
            if _TYPES[k] == "select" and v not in (_OPTIONS.get(k) or []):
                continue
            try:
                _state[k] = _coerce(k, v)
            except (TypeError, ValueError):
                continue
        _RUNTIME.write_text(json.dumps(_state, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return public_settings()


def reset() -> dict:
    with _LOCK:
        _state.clear()
        _state.update(DEFAULTS)
        _state["MODE"] = "basic"
        if _RUNTIME.exists():
            _RUNTIME.unlink()
    return public_settings()


# ----- режимы работы -----
def modes_catalog() -> list[dict]:
    return [{"key": k, "label": m["label"], "desc": m["desc"]} for k, m in MODES.items()]


def current_mode() -> str:
    """Текущий режим; 'custom', если флаги настроены вручную и не совпадают с пресетом."""
    cur = _state.get("MODE", "basic")
    flags = MODES.get(cur, {}).get("flags", {})
    if flags and all(_state.get(k) == v for k, v in flags.items()):
        return cur
    return "custom"


def set_mode(name: str) -> dict:
    if name not in MODES:
        return public_settings()
    with _LOCK:
        _state["MODE"] = name
        for k, v in MODES[name]["flags"].items():
            _state[k] = v
        _RUNTIME.write_text(json.dumps(_state, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return public_settings()
