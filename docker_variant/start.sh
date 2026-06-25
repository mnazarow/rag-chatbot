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
LLM_MODEL="${LLM_MODEL:-qwen3.6:35b-a3b-q4_K_M}"
AUTO="${NO_AUTOINSTALL:-0}"
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"

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

# ----- 5. Чеклист -----
echo; echo "Ожидание готовности приложения…"
ok_app=0
for i in $(seq 1 60); do curl -fs http://localhost:8000/health >/dev/null 2>&1 && { ok_app=1; break; }; sleep 3; done

line(){ if [ "$2" = "1" ]; then green "  [OK]  $1"; else red "  [X]   $1"; fi; }
echo; echo "=================== Статус ==================="
qok=0; curl -fs http://localhost:6333/collections >/dev/null 2>&1 && qok=1; line "Qdrant (векторная база)" "$qok"
rping=$(docker compose exec -T redis redis-cli ping 2>/dev/null | tr -d '\r' || true)
rok=0; [ "$rping" = "PONG" ] && rok=1; line "Redis (кэш) отвечает PONG" "$rok"
line "Приложение (http://localhost:8000)" "$ok_app"
seen=0; [ "$ok_app" = "1" ] && curl -fs http://localhost:8000/api/system 2>/dev/null | grep -Eq '"enabled": *true' && seen=1
line "Приложение: кэш Redis включён" "$seen"
oll=0; curl -fs http://localhost:11434/api/tags >/dev/null 2>&1 && oll=1; line "Ollama на хосте" "$oll"
echo "=============================================="; echo

if [ "$ok_app" = "1" ]; then
  green "Готово! Веб-интерфейс: http://localhost:8000"
  echo "Раздел «Система» → панель «⚡ Кэш Redis» показывает статистику кэша."
  command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:8000 >/dev/null 2>&1 || true
  [ "$OS" = "Darwin" ] && open http://localhost:8000 >/dev/null 2>&1 || true
else
  yellow "Приложение ещё поднимается (первая сборка качает модели). Подождите 1–2 минуты: http://localhost:8000"
  echo "Логи: docker compose logs -f app"
fi
