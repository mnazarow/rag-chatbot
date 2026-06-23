#!/usr/bin/env bash
# =============================================================================
#  Деплой RAG-чатбота из GitHub на чистый Mac (Apple Silicon).
#  Клонирует репозиторий, запускает установку (setup.sh: Homebrew, Ollama,
#  Qdrant, Python-окружение, модели) и регистрирует автозапуск через launchd.
#  Все прикладные настройки потом — в веб-админке.
#
#  Запуск (репозиторий аргументом):
#     bash deploy_mac.sh https://github.com/USER/rag-chatbot.git
#
#  Или одной строкой прямо из GitHub (замените USER):
#     curl -fsSL https://raw.githubusercontent.com/USER/rag-chatbot/main/mac_variant/deploy_mac.sh \
#       | REPO=https://github.com/USER/rag-chatbot.git ADMIN_TOKEN='пароль' bash
#
#  Переменные (необязательные):
#     REPO         URL репозитория (если не первым аргументом)
#     BRANCH       ветка (по умолчанию main)
#     TARGET_DIR   куда клонировать (по умолчанию ~/rag-chatbot)
#     ADMIN_TOKEN  пароль админ-панели (рекомендуется)
#     DOCS_DIR     папку с документами можно задать сразу
# =============================================================================
set -euo pipefail

REPO="${REPO:-${1:-}}"
BRANCH="${BRANCH:-main}"
TARGET_DIR="${TARGET_DIR:-$HOME/rag-chatbot}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"
DOCS_DIR_ENV="${DOCS_DIR:-}"

log(){ printf "\033[1;35m[deploy-mac]\033[0m %s\n" "$*"; }

[[ "$(uname -s)" == "Darwin" ]] || { echo "Этот скрипт только для macOS."; exit 1; }
[[ -n "$REPO" ]] || { echo "Укажите репозиторий: bash deploy_mac.sh <git-url>  (или REPO=...)"; exit 1; }

# ----- 1. git (Command Line Tools) -----
if ! command -v git >/dev/null 2>&1; then
  log "git не найден — запускаю установку Command Line Tools..."
  xcode-select --install || true
  echo "Дождитесь окончания установки CLT и запустите скрипт повторно."; exit 1
fi

# ----- 2. клон или обновление -----
if [[ -d "$TARGET_DIR/.git" ]]; then
  log "Обновляю существующий репозиторий в $TARGET_DIR..."
  git -C "$TARGET_DIR" fetch --all -q
  git -C "$TARGET_DIR" reset --hard "origin/$BRANCH"
else
  log "Клонирую $REPO (ветка $BRANCH) в $TARGET_DIR..."
  git clone -q -b "$BRANCH" "$REPO" "$TARGET_DIR"
fi

cd "$TARGET_DIR"

# ----- 3. установка (Apple) -----
chmod +x setup.sh
log "Запускаю setup.sh (Homebrew, Ollama, Qdrant, venv, модели)..."
./setup.sh

# ----- 4. правки .env -----
[[ -f .env ]] || cp .env.example .env
if [[ -n "$ADMIN_TOKEN" ]]; then sed -i '' "s|^ADMIN_TOKEN=.*|ADMIN_TOKEN=${ADMIN_TOKEN}|" .env; fi
if [[ -n "$DOCS_DIR_ENV" ]]; then sed -i '' "s|^DOCS_DIR=.*|DOCS_DIR=${DOCS_DIR_ENV}|" .env; fi

# ----- 5. автозапуск через launchd -----
mkdir -p "$HOME/Library/LaunchAgents"
PLIST="$HOME/Library/LaunchAgents/com.rag.api.plist"
sed -e "s|__ROOT__|${TARGET_DIR}|g" -e "s|__PORT__|8000|g" \
    mac_variant/com.rag.api.plist.tpl > "$PLIST"
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
log "API зарегистрирован в launchd (автозапуск + перезапуск при сбое)."

cat <<EOF

============================================================
  Готово. Сервис запущен и стартует автоматически.

  Веб-панель:   http://localhost:8000
  Раздел «Администратор» — там настраивается всё:
    • папка с документами → кнопка «Переиндексировать»
    • модель Ollama, параметры поиска, режимы — в админке

  Управление:   bash mac_variant/manage_mac.sh {status|logs|restart|stop|start}
  Обновление:   bash mac_variant/update_mac.sh
============================================================
EOF
