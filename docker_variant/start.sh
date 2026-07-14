#!/usr/bin/env bash
# Автоматическая установка RAG в Docker с Redis — Linux и macOS, одной командой.
#   ./start.sh
# Сам ставит Docker и Ollama (если их нет), качает модель, поднимает контейнеры
# qdrant + redis + app (Redis включён) и печатает статус.
#
# Параметры (необязательно):
#   DOCS_DIR_HOST=/path/to/docs   — папка с документами (по умолчанию ./docs)
#   LLM_MODEL=qwen3.6:35b-a3b-q4_K_M
#   NO_AUTOINSTALL=1              — не устанавливать автоматически, только проверить
set -uo pipefail
cd "$(dirname "$0")"

green(){ printf '\033[32m%s\033[0m\n' "$1"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$1"; }
red(){ printf '\033[31m%s\033[0m\n' "$1"; }
log(){ printf '\033[36m==> %s\033[0m\n' "$1"; }

OS="$(uname -s)"
_USER_LLM="${LLM_MODEL:-}"                       # если задана через env — меню пропускаем
LLM_MODEL="${LLM_MODEL:-qwen3.6:35b-a3b-q4_K_M}"
AUTO="${NO_AUTOINSTALL:-0}"

# Интерактивный выбор модели Ollama (терминал; пропуск при заданной LLM_MODEL)
choose_ollama_model(){
  [ -n "${_USER_LLM}" ] && return 0
  [ -t 0 ] || return 0
  echo
  echo "============================================================"
  echo "  Выбор модели генерации (Ollama). Рекомендуется: ${LLM_MODEL}"
  echo "------------------------------------------------------------"
  echo "  1) qwen2.5:7b-instruct           ~4.7 ГБ  — быстрая (CPU/слабый GPU)"
  echo "  2) qwen2.5:14b-instruct          ~9 ГБ    — баланс"
  echo "  3) qwen2.5:32b-instruct-q4_K_M   ~20 ГБ   — сильный RU (24 ГБ+ GPU)"
  echo "  4) qwen3:8b                      ~5 ГБ    — reasoning, лёгкая"
  echo "  5) qwen3.6:35b-a3b-q4_K_M        ~20 ГБ   — MoE 35B, топ (мощный сервер)"
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
}
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"

# ----- 0. базовые утилиты (нужны для установки Docker/Ollama и git-обновлений) -----
# Все прикладные пакеты (Python, ffmpeg, tesseract, playwright/chromium, espeak и т.п.)
# ставятся ВНУТРИ образа приложения (Dockerfile) — на хосте достаточно этих утилит.
if [ "$OS" = "Linux" ] && command -v apt-get >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || ! command -v gpg >/dev/null 2>&1; then
    log "Ставлю базовые пакеты (curl, ca-certificates, git, gnupg)…"
    $SUDO apt-get update -y && $SUDO apt-get install -y curl ca-certificates git gnupg || true
  fi
fi

# ----- 1. Docker -----
if ! command -v docker >/dev/null 2>&1; then
  if [ "$AUTO" = "1" ]; then red "Docker не найден (NO_AUTOINSTALL=1)."; exit 1; fi
  if [ "$OS" = "Linux" ]; then
    log "Docker не найден. Устанавливаю Docker Engine (get.docker.com; потребуется sudo)…"
    if curl -fsSL https://get.docker.com -o /tmp/get-docker.sh; then
      $SUDO sh /tmp/get-docker.sh || { red "Не удалось установить Docker автоматически."; exit 1; }
      $SUDO systemctl enable --now docker 2>/dev/null || true
      [ -n "$SUDO" ] && $SUDO usermod -aG docker "$USER" 2>/dev/null || true
      yellow "Docker установлен. Если команды докера требуют sudo — перелогиньтесь (группа docker)."
    else
      red "Не удалось скачать установщик Docker. Установите вручную: https://docs.docker.com/engine/install/"; exit 1
    fi
  else  # macOS
    if command -v brew >/dev/null 2>&1; then
      log "Docker не найден. Устанавливаю Docker Desktop (brew --cask docker)…"
      brew install --cask docker || { red "Не удалось установить Docker Desktop через brew."; exit 1; }
      open -a Docker || true
      log "Запускаю Docker Desktop, жду движок (до ~3 минут)…"
    else
      red "Docker не найден и Homebrew недоступен. Установите Docker Desktop:"
      echo "  https://www.docker.com/products/docker-desktop/  затем повторите ./start.sh"; exit 1
    fi
  fi
fi

# дождаться движка Docker
if ! docker info >/dev/null 2>&1; then
  [ "$OS" = "Darwin" ] && open -a Docker 2>/dev/null || true
  log "Жду запуска движка Docker…"
  for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 5; done
fi
if ! docker info >/dev/null 2>&1; then
  red "Движок Docker не запустился. Запустите Docker и повторите ./start.sh"; exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  red "Нужен Docker Compose v2 (команда 'docker compose'). Обновите Docker."; exit 1
fi
green "Docker готов."

# ----- 2. Ollama (на хосте) + модель -----
if ! command -v ollama >/dev/null 2>&1; then
  if [ "$AUTO" = "1" ]; then
    yellow "Ollama не найдена (NO_AUTOINSTALL=1) — пропускаю."
  elif [ "$OS" = "Linux" ]; then
    log "Устанавливаю Ollama (ollama.com/install.sh)…"
    curl -fsSL https://ollama.com/install.sh | sh || yellow "Не удалось установить Ollama автоматически."
  elif command -v brew >/dev/null 2>&1; then
    log "Устанавливаю Ollama (brew)…"; brew install ollama || yellow "brew install ollama не удался."
    brew services start ollama 2>/dev/null || (ollama serve >/dev/null 2>&1 &) || true
  else
    yellow "Ollama не установлена. Установите с https://ollama.com — без неё ответы генерироваться не будут."
  fi
fi
if command -v ollama >/dev/null 2>&1; then
  # поднять сервер, если не запущен
  curl -fs http://localhost:11434/api/tags >/dev/null 2>&1 || (ollama serve >/dev/null 2>&1 &) ; sleep 2
  choose_ollama_model               # интерактивный выбор модели (если терминал)
  log "Скачиваю модель Ollama: $LLM_MODEL (при первом запуске долго)…"
  ollama pull "$LLM_MODEL" || yellow "Модель не скачалась — выполните позже: ollama pull $LLM_MODEL"
else
  yellow "Ollama недоступна — контейнеры поднимутся, но отвечать на вопросы не смогут."
fi

# ----- 3. Конфиг и состояние -----
mkdir -p state backups docs
[ -f .env.docker ] || { cp .env.docker.example .env.docker; yellow "Создан .env.docker из примера."; }
# прописать выбранную модель
if grep -q '^LLM_MODEL=' .env.docker 2>/dev/null; then
  sed -i.bak "s|^LLM_MODEL=.*|LLM_MODEL=${LLM_MODEL}|" .env.docker && rm -f .env.docker.bak
fi
[ -f state/runtime_config.json ] || echo '{}' > state/runtime_config.json
[ -f state/rag_logs.db ]         || : > state/rag_logs.db
[ -f state/ingest_stats.json ]   || echo '{}' > state/ingest_stats.json

# ----- 4. Сборка и запуск -----
echo
log "Собираю и запускаю контейнеры (qdrant + redis + app)…"
docker compose up -d --build || { red "docker compose не выполнился."; exit 1; }

# ----- 5. Чеклист (полная проверка: контейнеры, сервисы, инструменты и Python-пакеты в образе) -----
echo; echo "Ожидание готовности приложения…"
ok_app=0
for i in $(seq 1 60); do curl -fs http://localhost:8000/health >/dev/null 2>&1 && { ok_app=1; break; }; sleep 3; done

if [ -f ../scripts/checklist_docker.sh ]; then
  bash ../scripts/checklist_docker.sh || true
else
  # запасной короткий статус, если общий скрипт недоступен
  line(){ if [ "$2" = "1" ]; then green "  [OK]  $1"; else red "  [X]   $1"; fi; }
  echo; echo "=================== Статус ==================="
  qok=0; curl -fs http://localhost:6333/collections >/dev/null 2>&1 && qok=1; line "Qdrant (векторная база)" "$qok"
  rping=$(docker compose exec -T redis redis-cli ping 2>/dev/null | tr -d '\r' || true)
  rok=0; [ "$rping" = "PONG" ] && rok=1; line "Redis (кэш) отвечает PONG" "$rok"
  line "Приложение (http://localhost:8000)" "$ok_app"
  oll=0; curl -fs http://localhost:11434/api/tags >/dev/null 2>&1 && oll=1; line "Ollama на хосте" "$oll"
  echo "=============================================="; echo
fi

if [ "$ok_app" = "1" ]; then
  green "Готово! Веб-интерфейс: http://localhost:8000"
  echo "Раздел «Система» → панель «⚡ Кэш Redis» показывает статистику кэша."
  command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:8000 >/dev/null 2>&1 || true
  [ "$OS" = "Darwin" ] && open http://localhost:8000 >/dev/null 2>&1 || true
else
  yellow "Приложение ещё поднимается (первая сборка качает модели). Подождите 1–2 минуты: http://localhost:8000"
  echo "Логи: docker compose logs -f app"
fi
