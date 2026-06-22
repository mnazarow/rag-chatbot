#!/usr/bin/env bash
# =============================================================================
#  Корпоративный RAG-чатбот — bootstrap для Linux-сервера с NVIDIA GPU
#  Стек: vLLM (генерация) + Qdrant + Python-приложение (эмбеддинги/реранк на CUDA)
#  Тестировалось на Ubuntu 22.04 / 24.04. Запуск:  sudo bash setup_gpu.sh
# =============================================================================
set -euo pipefail

# ----- настройки (можно переопределить через env) ---------------------------
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-14B-Instruct-AWQ}"   # см. таблицу в README
VLLM_MAX_LEN="${VLLM_MAX_LEN:-16384}"
VLLM_TP="${VLLM_TP:-1}"                                      # = число GPU
TORCH_CUDA="${TORCH_CUDA:-cu124}"                            # cu121 / cu124 ...
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"                  # код приложения в корне

log() { printf "\033[1;32m[setup-gpu]\033[0m %s\n" "$*"; }

# ----- 0. проверка GPU ------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi не найден. Установите драйвер NVIDIA и повторите."; exit 1
fi
nvidia-smi -L

# ----- 1. системные пакеты --------------------------------------------------
log "Устанавливаю базовые пакеты..."
apt-get update -y
apt-get install -y python3 python3-venv python3-pip ffmpeg curl ca-certificates gnupg
PYBIN="$(command -v python3.11 || command -v python3.12 || command -v python3.10 || command -v python3)"

# ----- 2. Docker + Compose --------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  log "Устанавливаю Docker..."
  curl -fsSL https://get.docker.com | sh
fi

# ----- 3. NVIDIA Container Toolkit (GPU внутри контейнеров) ------------------
if ! docker info 2>/dev/null | grep -qi nvidia; then
  log "Устанавливаю NVIDIA Container Toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -y
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

# ----- 4. .env --------------------------------------------------------------
if [[ ! -f "${PROJECT_DIR}/.env" ]]; then
  cp "${PROJECT_DIR}/.env.gpu.example" "${PROJECT_DIR}/.env"
  sed -i "s|^LLM_MODEL=.*|LLM_MODEL=${VLLM_MODEL}|" "${PROJECT_DIR}/.env"
  log "Создан .env (отредактируйте DOCS_DIR)."
fi
# vLLM compose читает переменные модели из .env
{ echo "VLLM_MODEL=${VLLM_MODEL}"; echo "VLLM_MAX_LEN=${VLLM_MAX_LEN}"; echo "VLLM_TP=${VLLM_TP}"; } \
  >> "${PROJECT_DIR}/.env"

# ----- 5. поднимаем vLLM + Qdrant ------------------------------------------
log "Запускаю vLLM + Qdrant (первый старт качает веса модели — долго)..."
cd "${PROJECT_DIR}"
docker compose --env-file .env -f docker-compose.gpu.yml up -d
log "Жду готовности vLLM (/health на :8001)..."
for i in {1..120}; do
  curl -sf http://localhost:8001/health >/dev/null 2>&1 && { log "vLLM готов."; break; }
  sleep 10
done

# ----- 6. Python-окружение приложения --------------------------------------
log "Ставлю Python-зависимости (torch ${TORCH_CUDA} + RAG)..."
cd "${ROOT_DIR}"
"${PYBIN}" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel
pip install "torch==2.5.1" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
pip install -r "${PROJECT_DIR}/requirements-gpu.txt"
# приложение читает .env из текущей папки — кладём симлинк на gpu-конфиг
ln -sf "${PROJECT_DIR}/.env" "${ROOT_DIR}/.env"

# ----- 7. прогрев эмбеддера/реранкера на CUDA ------------------------------
log "Прогреваю эмбеддинги и реранк на GPU..."
DEVICE=cuda python - <<'PY'
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
SentenceTransformer("BAAI/bge-m3", device="cuda")
FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
print("OK")
PY

cat <<EOF

============================================================
  Готово. Дальше:
  1) Отредактируйте gpu_variant/.env -> DOCS_DIR
  2) Индексация:   source .venv/bin/activate && python ingest.py
  3) Запуск API:   uvicorn app:app --host 0.0.0.0 --port 8000
  4) Веб-чат:      http://<ip-сервера>:8000
  vLLM API:        http://localhost:8001/v1  (модель ${VLLM_MODEL})
============================================================
EOF
