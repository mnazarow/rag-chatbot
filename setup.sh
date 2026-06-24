#!/usr/bin/env bash
# =============================================================================
#  Корпоративный RAG-чатбот — bootstrap для чистого Mac Studio (Apple Silicon)
#  Запуск:  chmod +x setup.sh && ./setup.sh
#  Идемпотентен: можно запускать повторно.
# =============================================================================
set -euo pipefail

# ----- настройки ------------------------------------------------------------
LLM_MODEL="${LLM_MODEL:-qwen2.5:32b-instruct-q4_K_M}"   # основная модель генерации
EMBED_MODEL_HF="${EMBED_MODEL_HF:-BAAI/bge-m3}"          # эмбеддинги (многоязычные, сильный RU)
RERANK_MODEL_HF="${RERANK_MODEL_HF:-BAAI/bge-reranker-v2-m3}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python3.11"

log() { printf "\033[1;32m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }

# ----- 0. проверка платформы ------------------------------------------------
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  warn "Скрипт рассчитан на Apple Silicon (macOS arm64). Текущая платформа: $(uname -s)/$(uname -m)."
  warn "На Linux/NVIDIA замените Ollama на vLLM, а Metal — на CUDA. Остальное переносимо."
fi

# ----- 1. Homebrew ----------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  log "Устанавливаю Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
else
  log "Homebrew уже установлен."
fi
eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || true

# ----- 2. системные зависимости --------------------------------------------
log "Устанавливаю системные пакеты (python, ffmpeg, ollama, docker)..."
brew install python@3.11 ffmpeg libmagic poppler tesseract tesseract-lang || true
brew install libredwg || true   # dwg2dxf: конвертация DWG в DXF (необязательно)
brew install --cask docker || true          # для Qdrant (Docker Desktop)
brew install ollama || true

# ----- 3. Ollama сервис + модель генерации ---------------------------------
log "Запускаю Ollama как фоновый сервис..."
brew services start ollama || ollama serve >/dev/null 2>&1 &
sleep 5
log "Скачиваю LLM: ${LLM_MODEL} (это надолго при первом запуске)..."
ollama pull "${LLM_MODEL}"

# ----- 4. Qdrant (векторная БД) через Docker -------------------------------
log "Поднимаю Qdrant..."
open -a Docker || true
# ждём демон Docker
for i in {1..30}; do docker info >/dev/null 2>&1 && break || sleep 2; done
docker compose -f "${PROJECT_DIR}/docker-compose.yml" up -d

# ----- 5. Python окружение --------------------------------------------------
log "Создаю виртуальное окружение и ставлю зависимости..."
cd "${PROJECT_DIR}"
${PYTHON_BIN} -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
pip install -q ezdxf rawpy pytesseract || true   # DWG/DXF + OCR для RAW-фото (CR2)

# ----- 6. .env --------------------------------------------------------------
if [[ ! -f "${PROJECT_DIR}/.env" ]]; then
  cp "${PROJECT_DIR}/.env.example" "${PROJECT_DIR}/.env"
  sed -i '' "s|^LLM_MODEL=.*|LLM_MODEL=${LLM_MODEL}|" "${PROJECT_DIR}/.env"
  log "Создан .env (папка документов по умолчанию: /opt/db)."
fi

# папка документов по умолчанию /opt/db (в /opt нужны права sudo)
if [[ ! -d /opt/db ]]; then
  log "Создаю /opt/db (может потребоваться пароль sudo)..."
  sudo mkdir -p /opt/db && sudo chown "$(whoami)" /opt/db \
    || warn "Не удалось создать /opt/db — создайте вручную или укажите другую папку в админке."
fi

# ----- 7. прогрев моделей эмбеддинга/реранка -------------------------------
log "Прогреваю модели эмбеддинга и реранка (скачивание весов с HF)..."
python - <<PY
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
SentenceTransformer("${EMBED_MODEL_HF}", device="mps")
FlagReranker("${RERANK_MODEL_HF}", use_fp16=True)
print("OK")
PY

# ----- 8. автозапуск API через launchd -----
log "Настраиваю автозапуск (launchd)..."
PORT="$(grep -E '^API_PORT=' "${PROJECT_DIR}/.env" 2>/dev/null | cut -d= -f2)"
PORT="${PORT:-8000}"
TPL="${PROJECT_DIR}/mac_variant/com.rag.api.plist.tpl"
if [[ -f "$TPL" ]]; then
  mkdir -p "$HOME/Library/LaunchAgents"
  PLIST="$HOME/Library/LaunchAgents/com.rag.api.plist"
  sed -e "s|__ROOT__|${PROJECT_DIR}|g" -e "s|__PORT__|${PORT}|g" "$TPL" > "$PLIST"
  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load "$PLIST"
  log "Сервис rag-api зарегистрирован и запущен (автозапуск при входе в систему)."
else
  warn "Шаблон launchd не найден — автозапуск не настроен; запускайте вручную uvicorn."
fi

cat <<EOF

============================================================
  Готово. Сервис запущен автоматически (launchd).

  Веб-панель:   http://localhost:${PORT}
  Раздел «Администратор» — укажите папку с документами и нажмите
  «Переиндексировать» (или загрузите файлы в админке).

  Управление:   bash mac_variant/manage_mac.sh {status|logs|restart|stop|start}
============================================================
EOF
