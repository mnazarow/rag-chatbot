#!/usr/bin/env bash
# =============================================================================
#  Корпоративный RAG-чатбот — bootstrap для чистого Mac Studio (Apple Silicon)
#  Запуск:  chmod +x setup.sh && ./setup.sh
#  Идемпотентен: можно запускать повторно.
# =============================================================================
set -euo pipefail

# ----- настройки ------------------------------------------------------------
# Выбор модели генерации (Ollama, по памяти) — подробно в docs/MODELS.md:
#   8–16 ГБ : qwen3:8b, qwen3.6:35b-a3b-q4_K_M (MoE), gemma3:12b
#   24 ГБ   : qwen3.6:35b-a3b, qwen2.5:32b, gemma3:27b
#   48–96 ГБ: qwen2.5:72b, qwen3.6:27b, llama3.3:70b   (Mac Studio)
#   CPU/слаб: qwen3:1.7b–qwen3:4b, llama3.2:3b
_USER_LLM="${LLM_MODEL:-}"                          # если задана через env — меню пропускаем
LLM_MODEL="${LLM_MODEL:-qwen3.6:35b-a3b-q4_K_M}"   # основная модель генерации (MoE, быстрая)
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
brew install antiword || true    # чтение старого .doc (Word 97-2003)
brew install p7zip unar || true  # распаковка архивов (.7z/.rar и др.)
brew install espeak || true      # синтез речи (TTS); на macOS есть и системный `say`
brew install --cask docker || true          # для Qdrant (Docker Desktop)
brew install ollama || true
[ "${INSTALL_REDIS:-1}" = "1" ] && brew install redis || true   # кэш агрегатов + семантический кэш

# ----- 3. Ollama сервис + модель генерации ---------------------------------
log "Запускаю Ollama как фоновый сервис..."
brew services start ollama || ollama serve >/dev/null 2>&1 &
sleep 5
# Redis — запускаем как фоновый сервис (кэш агрегатов + семантический кэш)
if [ "${INSTALL_REDIS:-1}" = "1" ] && command -v redis-server >/dev/null 2>&1; then
  log "Запускаю Redis (кэш)..."
  brew services start redis 2>/dev/null || redis-server --daemonize yes >/dev/null 2>&1 || true
fi
# Интерактивный выбор модели Ollama (терминал; пропуск при заданной LLM_MODEL)
if [ -z "${_USER_LLM}" ] && [ -t 0 ]; then
  echo
  echo "============================================================"
  echo "  Выбор модели генерации (Ollama). Рекомендуется: ${LLM_MODEL}"
  echo "------------------------------------------------------------"
  echo "  1) qwen2.5:7b-instruct           ~4.7 ГБ  — быстрая"
  echo "  2) qwen2.5:14b-instruct          ~9 ГБ    — баланс"
  echo "  3) qwen2.5:32b-instruct-q4_K_M   ~20 ГБ   — сильный RU"
  echo "  4) qwen3:8b                      ~5 ГБ    — reasoning, лёгкая"
  echo "  5) qwen3.6:35b-a3b-q4_K_M        ~20 ГБ   — MoE 35B, топ (мощный Mac)"
  echo "  6) glm4:9b                       ~6 ГБ    — GLM-4 (Zhipu), RU/CN"
  echo "  7) Ввести свою (ollama-тег)"
  echo "  0) Рекомендованную (Enter)"
  echo "============================================================"
  printf "Выбор [0-7]: "; read -r _lm || _lm=""
  case "${_lm}" in
    1) LLM_MODEL="qwen2.5:7b-instruct" ;;
    2) LLM_MODEL="qwen2.5:14b-instruct" ;;
    3) LLM_MODEL="qwen2.5:32b-instruct-q4_K_M" ;;
    4) LLM_MODEL="qwen3:8b" ;;
    5) LLM_MODEL="qwen3.6:35b-a3b-q4_K_M" ;;
    6) LLM_MODEL="glm4:9b" ;;
    7) printf "ollama-тег: "; read -r _ct || _ct=""; [ -n "${_ct}" ] && LLM_MODEL="${_ct}" ;;
    *) : ;;
  esac
  log "Модель: ${LLM_MODEL}"
fi
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
pip install -q ezdxf rawpy pytesseract Pillow matplotlib extract-msg py7zr rarfile psutil || true   # DWG/DXF + OCR (RAW/фото) + Outlook .msg + архивы + метрики сервера
# headless-браузер для парсинга JS-сайтов (на macOS зависимости идут с браузером)
python -m playwright install chromium 2>/dev/null || true
# XTTS (клонирование голоса) — ОТДЕЛЬНЫЙ venv .venv-xtts + микросервис (launchd com.rag.xtts
# ниже), чтобы coqui-tts (transformers>=4.57) не конфликтовал с ядром RAG (.venv,
# transformers==4.44.2). Приложение обращается к сервису по HTTP (XTTS_URL). Отключить: INSTALL_XTTS=0.
XTTS_PORT="${XTTS_PORT:-8020}"
INSTALL_XTTS="${INSTALL_XTTS:-1}"
if [ "${INSTALL_XTTS}" = "1" ]; then
  log "Ставлю XTTS в отдельное окружение .venv-xtts (изолированно от ядра)..."
  if ( deactivate 2>/dev/null; cd "${PROJECT_DIR}" \
        && ${PYTHON_BIN} -m venv .venv-xtts \
        && ./.venv-xtts/bin/pip install -q --upgrade pip wheel \
        && ./.venv-xtts/bin/pip install -q "coqui-tts>=0.24.0" "fastapi" "uvicorn[standard]" ); then
    log "XTTS-окружение готово (сервис будет на 127.0.0.1:${XTTS_PORT})."
  else
    warn "XTTS-окружение не собралось — голос-клон будет недоступен (остальное работает)."
    INSTALL_XTTS=0
  fi
  # вернуться в основной venv (мы могли из него выйти в подоболочке — на всякий случай)
  source .venv/bin/activate 2>/dev/null || true
fi

# ----- 6. .env --------------------------------------------------------------
if [[ ! -f "${PROJECT_DIR}/.env" ]]; then
  cp "${PROJECT_DIR}/.env.example" "${PROJECT_DIR}/.env"
  sed -i '' "s|^LLM_MODEL=.*|LLM_MODEL=${LLM_MODEL}|" "${PROJECT_DIR}/.env"
  log "Создан .env (папка документов по умолчанию: /opt/db)."
fi
# адрес сервиса XTTS (клонирование голоса) — приложение шлёт синтез сюда по HTTP
if [ "${INSTALL_XTTS}" = "1" ]; then
  if grep -qE '^XTTS_URL=' "${PROJECT_DIR}/.env" 2>/dev/null; then
    sed -i '' "s|^XTTS_URL=.*|XTTS_URL=http://127.0.0.1:${XTTS_PORT}|" "${PROJECT_DIR}/.env"
  else
    echo "XTTS_URL=http://127.0.0.1:${XTTS_PORT}" >> "${PROJECT_DIR}/.env"
  fi
fi
# Redis включён по умолчанию (кэш агрегатов + семантический кэш), если сервер отвечает
if [ "${INSTALL_REDIS:-1}" = "1" ] && (redis-cli ping >/dev/null 2>&1); then
  _envset(){ if grep -qE "^$1=" "${PROJECT_DIR}/.env"; then sed -i '' "s|^$1=.*|$1=$2|" "${PROJECT_DIR}/.env"; else echo "$1=$2" >> "${PROJECT_DIR}/.env"; fi; }
  _envset REDIS_ENABLED true; _envset REDIS_HOST 127.0.0.1; _envset REDIS_PORT 6379
  log "Redis работает — REDIS_ENABLED=true."
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

# микросервис XTTS (клонирование голоса) — отдельный launchd-агент com.rag.xtts
XTTS_TPL="${PROJECT_DIR}/mac_variant/com.rag.xtts.plist.tpl"
if [ "${INSTALL_XTTS}" = "1" ] && [ -x "${PROJECT_DIR}/.venv-xtts/bin/python" ] && [ -f "$XTTS_TPL" ]; then
  XPLIST="$HOME/Library/LaunchAgents/com.rag.xtts.plist"
  sed -e "s|__ROOT__|${PROJECT_DIR}|g" -e "s|__PORT__|${XTTS_PORT}|g" "$XTTS_TPL" > "$XPLIST"
  launchctl unload "$XPLIST" 2>/dev/null || true
  launchctl load "$XPLIST"
  log "Сервис клонирования голоса (rag-xtts) зарегистрирован (порт ${XTTS_PORT})."
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

# ----- 9. полная проверка установки (пакеты и компоненты) -------------------
bash "${PROJECT_DIR}/scripts/checklist.sh" "${PROJECT_DIR}" || true
