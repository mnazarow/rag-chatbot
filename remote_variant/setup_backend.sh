#!/usr/bin/env bash
# =============================================================================
#  БЭКЕНД-СЕРВЕР (split-развёртывание): Qdrant + vLLM, доступные по сети.
#  Ставит Docker + NVIDIA Container Toolkit и поднимает контейнеры из
#  gpu_variant/docker-compose.gpu.yml (порты 6333 — Qdrant, 8001 — vLLM).
#  Сервер приложения подключается к ним по сети (см. setup_app.sh).
#
#  Запуск:  sudo bash remote_variant/setup_backend.sh
#  Опции (env): VLLM_MODEL, VLLM_MAX_LEN, VLLM_TP, APP_HOST (IP сервера
#               приложения — кому открыть порты в firewall).
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GV="${ROOT}/gpu_variant"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-35B-A3B}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-16384}"
VLLM_TP="${VLLM_TP:-1}"
APP_HOST="${APP_HOST:-}"

log(){ printf "\033[1;34m[backend]\033[0m %s\n" "$*"; }
[[ $EUID -eq 0 ]] || { echo "Запустите через sudo."; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi не найден — установите драйвер NVIDIA."; exit 1; }
nvidia-smi -L

# ----- Docker + NVIDIA Container Toolkit -----
apt-get update -y && apt-get install -y curl ca-certificates gnupg ufw
command -v docker >/dev/null || { log "Docker..."; curl -fsSL https://get.docker.com | sh; }
if ! docker info 2>/dev/null | grep -qi nvidia; then
  log "NVIDIA Container Toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -y && apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
fi

# ----- .env для compose -----
cd "$GV"
[[ -f .env ]] || cp .env.gpu.example .env
upd(){ grep -q "^$1=" .env && sed -i "s|^$1=.*|$1=$2|" .env || echo "$1=$2" >> .env; }
upd VLLM_MODEL "${VLLM_MODEL}"; upd VLLM_MAX_LEN "${VLLM_MAX_LEN}"; upd VLLM_TP "${VLLM_TP}"

# ----- поднимаем Qdrant + vLLM (порты уже проброшены на 0.0.0.0) -----
log "Поднимаю Qdrant + vLLM (первый старт качает веса — долго)..."
docker compose --env-file .env -f docker-compose.gpu.yml up -d
log "Жду готовности vLLM..."
for i in {1..120}; do curl -sf http://localhost:8001/health >/dev/null 2>&1 && break || sleep 10; done

# ----- firewall -----
if command -v ufw >/dev/null 2>&1; then
  if [[ -n "$APP_HOST" ]]; then
    log "Открываю порты 6333/8001 только для ${APP_HOST}..."
    ufw allow from "$APP_HOST" to any port 6333 proto tcp || true
    ufw allow from "$APP_HOST" to any port 8001 proto tcp || true
  else
    log "APP_HOST не задан — открываю порты 6333/8001 всем (НЕБЕЗОПАСНО для публичной сети)."
    ufw allow 6333/tcp || true; ufw allow 8001/tcp || true
  fi
fi

IP="$(hostname -I | awk '{print $1}')"
cat <<EOF

============================================================
  Бэкенд готов. На сервере приложения выполните:

    sudo BACKEND_HOST=${IP} BACKEND_MODEL=${VLLM_MODEL} \\
         ADMIN_TOKEN='пароль' bash remote_variant/setup_app.sh

  Адреса бэкенда:
    Qdrant : http://${IP}:6333
    vLLM   : http://${IP}:8001/v1   (модель ${VLLM_MODEL})

  ВАЖНО: Qdrant и vLLM без авторизации — держите их в приватной сети/VPN
  и ограничивайте доступ по APP_HOST (firewall), не выставляйте в интернет.
============================================================
EOF
