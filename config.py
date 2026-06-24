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
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
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

# Доступ
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = _int("API_PORT", 8000)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # пусто = админка без пароля (только LAN!)

EMBED_DIM = 1024  # размерность bge-m3
