"""Единая конфигурация (читается из .env)."""
import os
from pathlib import Path
from dotenv import load_dotenv

# HF-токенайзеры: отключаем внутренний параллелизм — иначе при fork (распаковка
# архивов, dwg2dxf, ffmpeg) сыплется предупреждение и возможны зависания/замедления.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

load_dotenv()

def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))

def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))

# Документы
DOCS_DIR = Path(os.getenv("DOCS_DIR", "/opt/db")).expanduser()

# Модели
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6:35b-a3b-q4_K_M")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
DEVICE = os.getenv("DEVICE", "mps")          # mps (Apple) | cuda (GPU) | cpu

# Бэкенд генерации: ollama (Apple/CPU) | openai (vLLM, OpenAI-совместимый API)
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8001/v1")  # для vLLM
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "company_kb")
QDRANT_TIMEOUT = _int("QDRANT_TIMEOUT", 60)            # таймаут запросов (чат), с
QDRANT_INGEST_TIMEOUT = _int("QDRANT_INGEST_TIMEOUT", 480)  # таймаут индексации, с

# RAG-параметры
CHUNK_SIZE = _int("CHUNK_SIZE", 900)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
TOP_K_RETRIEVE = _int("TOP_K_RETRIEVE", 20)
TOP_K_RERANK = _int("TOP_K_RERANK", 6)
MIN_SCORE = _float("MIN_SCORE", 0.30)

# Транскрибация
WHISPER_BACKEND = os.getenv("WHISPER_BACKEND", "mlx")  # mlx (Apple) | faster (GPU/CPU)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

# vLLM (GPU-вариант): параметры контейнера генерации
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3.6-35B-A3B")
VLLM_MAX_LEN = _int("VLLM_MAX_LEN", 16384)
VLLM_TP = _int("VLLM_TP", 1)

# Дообучение (QLoRA): базовая fp16-модель. Пусто = берётся из VLLM_MODEL
# (с отбрасыванием суффиксов квантизации -AWQ/-GPTQ/-Int4).
FINETUNE_BASE = os.getenv("FINETUNE_BASE", "")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0") not in ("0", "false", "False", "no", "")


# Индексация: какие тяжёлые экстракторы включать (отключение ускоряет индексацию).
OCR_IMAGES = _bool("OCR_IMAGES", True)        # OCR изображений (jpg/png/…) — самый долгий
OCR_RAW = _bool("OCR_RAW", True)              # OCR RAW-фото (CR2/NEF/…)
PARSE_CAD = _bool("PARSE_CAD", True)          # чертежи DXF/DWG и 3D-CAD (конвертация DWG долгая)
TRANSCRIBE_AV = _bool("TRANSCRIBE_AV", True)  # транскрибация аудио/видео (Whisper, минуты на файл)
FILE_PARSE_TIMEOUT = _int("FILE_PARSE_TIMEOUT", 0)  # лимит времени на файл, c (0 = без лимита)

# Телеграм-бот: токен от @BotFather (пусто = бот выключен) и авто-подтверждение
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_AUTO_APPROVE = _bool("TELEGRAM_AUTO_APPROVE", False)
# Прокси для доступа к api.telegram.org (где Telegram заблокирован). Поддерживаются
# socks5://, socks5h://, http://, https:// (можно с user:pass@). ВНИМАНИЕ: MTProto-прокси
# (tg://proxy-ссылки) — для клиентов Telegram и НЕ работают с Bot API; нужен SOCKS5/HTTP.
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY", "")

# --- База данных приложения (журнал/история/настройки) ---
# sqlite (по умолчанию, без внешних сервисов) | mysql | postgresql
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")
MYSQL_HOST = os.getenv("MYSQL_HOST", "")
MYSQL_PORT = _int("MYSQL_PORT", 3306)
MYSQL_USER = os.getenv("MYSQL_USER", "")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "rag")
PG_HOST = os.getenv("PG_HOST", "")
PG_PORT = _int("PG_PORT", 5432)
PG_USER = os.getenv("PG_USER", "")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_DB = os.getenv("PG_DB", "rag")

# Источник каталога документов: filesystem (папка DOCS_DIR) | postgresql (таблица
# doc_catalog в активной PostgreSQL). Переключается кнопками в админке.
CATALOG_SOURCE = os.getenv("CATALOG_SOURCE", "filesystem")

# --- Кэш Redis (по умолчанию выключен) ---
REDIS_ENABLED = _bool("REDIS_ENABLED", False)
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = _int("REDIS_PORT", 6379)
REDIS_DB = _int("REDIS_DB", 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
# Кэшировать готовые ответы LLM в Redis (для одинаковых вопросов). По умолчанию выкл.:
# экономит время/нагрузку, но один и тот же вопрос будет получать один ответ до
# переиндексации/смены модели. Требует REDIS_ENABLED.
ANSWER_CACHE = _bool("ANSWER_CACHE", False)

# Доступ
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = _int("API_PORT", 8000)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # пусто = админка без пароля (только LAN!)

EMBED_DIM = 1024  # размерность bge-m3
