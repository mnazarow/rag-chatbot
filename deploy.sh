#!/usr/bin/env bash
# =============================================================================
#  Деплой RAG-чатбота из GitHub на ЧИСТЫЙ сервер (Linux + NVIDIA GPU).
#  Клонирует репозиторий и запускает полную GPU-установку (run_gpu.sh):
#  Docker + NVIDIA toolkit, vLLM + Qdrant, Python-окружение, systemd-сервис.
#  Все прикладные настройки потом делаются в веб-админке.
#
#  Запуск (репозиторий аргументом):
#     sudo -E bash deploy.sh https://github.com/USER/rag-chatbot.git
#
#  Или одной строкой прямо из GitHub (замените USER/REPO):
#     curl -fsSL https://raw.githubusercontent.com/USER/rag-chatbot/main/deploy.sh \
#       | sudo -E REPO=https://github.com/USER/rag-chatbot.git ADMIN_TOKEN='пароль' bash
#
#  Переменные окружения (необязательные):
#     REPO         URL репозитория (если не передан первым аргументом)
#     BRANCH       ветка (по умолчанию main)
#     TARGET_DIR   куда клонировать (по умолчанию /opt/rag)
#     ADMIN_TOKEN  пароль админ-панели (настоятельно рекомендуется)
#     VLLM_MODEL   стартовая модель vLLM (по умолчанию Qwen2.5-14B-AWQ)
#     VLLM_TP      число GPU (tensor-parallel), по умолчанию 1
# =============================================================================
set -euo pipefail

REPO="${REPO:-${1:-}}"
BRANCH="${BRANCH:-main}"
TARGET_DIR="${TARGET_DIR:-/opt/rag}"
RUN_USER="${SUDO_USER:-$(whoami)}"

log(){ printf "\033[1;36m[deploy]\033[0m %s\n" "$*"; }

[[ $EUID -eq 0 ]] || { echo "Запустите через sudo (нужны установка пакетов и systemd)."; exit 1; }
[[ -n "$REPO" ]] || { echo "Укажите репозиторий: sudo -E bash deploy.sh <git-url>  (или REPO=...)"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi не найден — установите драйвер NVIDIA."; exit 1; }

# ----- 1. git -----
command -v git >/dev/null || { log "Ставлю git..."; apt-get update -y && apt-get install -y git; }

# ----- 2. клон или обновление -----
if [[ -d "${TARGET_DIR}/.git" ]]; then
  log "Репозиторий уже есть в ${TARGET_DIR} — обновляю до origin/${BRANCH}..."
  git -C "${TARGET_DIR}" fetch --all -q
  git -C "${TARGET_DIR}" reset --hard "origin/${BRANCH}"
else
  log "Клонирую ${REPO} (ветка ${BRANCH}) в ${TARGET_DIR}..."
  mkdir -p "$(dirname "${TARGET_DIR}")"
  git clone -q -b "${BRANCH}" "${REPO}" "${TARGET_DIR}"
fi
chown -R "${RUN_USER}:${RUN_USER}" "${TARGET_DIR}"

# ----- 3. запуск GPU-установки -----
log "Запускаю GPU-установку (run_gpu.sh)..."
cd "${TARGET_DIR}/gpu_variant"
chmod +x run_gpu.sh apply_llm.sh manage.sh 2>/dev/null || true
# run_gpu.sh сам ставит всё и поднимает сервис; env (ADMIN_TOKEN, VLLM_*) наследуется
exec bash run_gpu.sh
