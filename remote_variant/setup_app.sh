#!/usr/bin/env bash
# =============================================================================
#  СЕРВЕР ПРИЛОЖЕНИЯ (split-развёртывание): FastAPI + эмбеддинги/реранк + граф,
#  подключается к удалённому бэкенду (Qdrant + vLLM на другом сервере).
#  Генерация — на удалённом vLLM; эмбеддинги, реранк и граф LightRAG — локально.
#
#  Запуск:  sudo BACKEND_HOST=<ip-бэкенда> bash remote_variant/setup_app.sh
#  Опции (env):
#     BACKEND_HOST   IP/хост бэкенда (обязательно)
#     BACKEND_MODEL  имя модели, которую отдаёт vLLM (должно совпадать)
#     DEVICE         cpu (по умолч.) | cuda — на чём считать эмбеддинги/реранк
#     TORCH_CUDA     cu124/cu126 (если DEVICE=cuda)
#     ADMIN_TOKEN    пароль админ-панели
#     API_PORT       порт приложения (8000)
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_HOST="${BACKEND_HOST:-${1:-}}"
BACKEND_MODEL="${BACKEND_MODEL:-Qwen/Qwen3.6-35B-A3B}"
DEVICE="${DEVICE:-cpu}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"
API_PORT="${API_PORT:-8000}"
RUN_USER="${SUDO_USER:-$(whoami)}"

log(){ printf "\033[1;34m[app]\033[0m %s\n" "$*"; }
[[ $EUID -eq 0 ]] || { echo "Запустите через sudo (нужны пакеты и systemd)."; exit 1; }
[[ -n "$BACKEND_HOST" ]] || { echo "Укажите BACKEND_HOST=<ip-бэкенда>"; exit 1; }

# ----- системные пакеты -----
apt-get update -y
apt-get install -y python3 python3-venv python3-pip ffmpeg curl ca-certificates
PYBIN="$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)"

# ----- .env (указывает на удалённый бэкенд) -----
cd "$ROOT"
[[ -f .env ]] || cp gpu_variant/.env.gpu.example .env
upd(){ grep -q "^$1=" .env && sed -i "s|^$1=.*|$1=$2|" .env || echo "$1=$2" >> .env; }
upd LLM_BACKEND "openai"
upd LLM_BASE_URL "http://${BACKEND_HOST}:8001/v1"
upd LLM_MODEL "${BACKEND_MODEL}"
upd QDRANT_URL "http://${BACKEND_HOST}:6333"
upd DEVICE "${DEVICE}"
upd WHISPER_BACKEND "faster"
upd API_PORT "${API_PORT}"
upd ADMIN_TOKEN "${ADMIN_TOKEN}"

mkdir -p /opt/db && chown "${RUN_USER}:${RUN_USER}" /opt/db || true
chown -R "${RUN_USER}:${RUN_USER}" "${ROOT}"

# ----- Python-окружение -----
log "Окружение и зависимости (DEVICE=${DEVICE})..."
sudo -u "${RUN_USER}" "${PYBIN}" -m venv .venv
sudo -u "${RUN_USER}" ./.venv/bin/pip install -U pip wheel
if [[ "${DEVICE}" == "cuda" ]]; then
  sudo -u "${RUN_USER}" ./.venv/bin/pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    || sudo -u "${RUN_USER}" ./.venv/bin/pip install torch --index-url "https://download.pytorch.org/whl/cu126"
else
  sudo -u "${RUN_USER}" ./.venv/bin/pip install torch --index-url "https://download.pytorch.org/whl/cpu"
fi
sudo -u "${RUN_USER}" ./.venv/bin/pip install -r gpu_variant/requirements-gpu.txt

# ----- проверка связи с бэкендом -----
log "Проверяю доступность бэкенда..."
curl -sf "http://${BACKEND_HOST}:6333/healthz" >/dev/null && log "Qdrant доступен" || log "ВНИМАНИЕ: Qdrant недоступен (проверьте firewall)"
curl -sf "http://${BACKEND_HOST}:8001/health" >/dev/null && log "vLLM доступен" || log "ВНИМАНИЕ: vLLM недоступен (проверьте firewall)"

# ----- systemd-сервис приложения -----
log "Регистрирую systemd-сервис rag-api..."
sed -e "s|__USER__|${RUN_USER}|g" -e "s|__ROOT__|${ROOT}|g" -e "s|__PORT__|${API_PORT}|g" \
    "${ROOT}/gpu_variant/rag-api.service.tpl" > /etc/systemd/system/rag-api.service
systemctl daemon-reload
systemctl enable --now rag-api

IP="$(hostname -I | awk '{print $1}')"
cat <<EOF

============================================================
  Сервер приложения готов и подключён к бэкенду ${BACKEND_HOST}.

  Веб-панель:  http://${IP}:${API_PORT}
  Документы:   /opt/db  → загрузите файлы и нажмите «Переиндексировать».
  Граф LightRAG строится здесь (кнопка «Построить граф»), генерация — на
  удалённом vLLM, эмбеддинги/реранк — локально (DEVICE=${DEVICE}).
============================================================
EOF
