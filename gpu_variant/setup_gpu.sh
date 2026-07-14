#!/usr/bin/env bash
# =============================================================================
#  Корпоративный RAG-чатбот — bootstrap для Linux-сервера с NVIDIA GPU
#  Стек: vLLM (генерация) + Qdrant + Python-приложение (эмбеддинги/реранк на CUDA)
#  Тестировалось на Ubuntu 22.04 / 24.04. Запуск:  sudo bash setup_gpu.sh
# =============================================================================
set -euo pipefail

# ----- настройки (можно переопределить через env) ---------------------------
# Выбор модели vLLM по VRAM (подробно — docs/MODELS.md):
#   24 ГБ (3090/4090): Qwen/Qwen2.5-14B-Instruct-AWQ, Qwen/Qwen3-8B, Qwen/Qwen3.6-35B-A3B
#   48 ГБ (A6000)    : Qwen/Qwen2.5-32B-Instruct-AWQ, Qwen/Qwen3-32B-AWQ, google/gemma-3-27b-it
#   80 ГБ (A100/H100): Qwen/Qwen2.5-72B-Instruct-AWQ, Qwen/Qwen3-32B, Llama-3.1-70B-AWQ-INT4
# Модель и tensor-parallel: если не заданы явно через env — подбираются АВТОМАТИЧЕСКИ
# по VRAM и числу GPU (ниже) + предлагается выбор в меню. Образ vLLM v0.19.0 понимает
# Qwen3.6 (qwen3_5_moe) и GLM-5.2 (старый v0.6.6 их не грузит).
_USER_MODEL="${VLLM_MODEL:-}"; _USER_TP="${VLLM_TP:-}"      # пусто = авто-подбор
VLLM_MODEL="${VLLM_MODEL:-QuantTrio/Qwen3.6-35B-A3B-AWQ}"   # см. docs/MODELS.md
VLLM_MAX_LEN="${VLLM_MAX_LEN:-16384}"
VLLM_TP="${VLLM_TP:-1}"                                      # число GPU для tensor-parallel
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.19.0}"        # версия vLLM (нужна ≥0.19 для Qwen3.6/GLM-5.2)
TORCH_CUDA="${TORCH_CUDA:-cu124}"                            # cu121 / cu124 ...
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"                  # код приложения в корне

log() { printf "\033[1;32m[setup-gpu]\033[0m %s\n" "$*"; }

# ----- 0. проверка GPU ------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi не найден — этот скрипт для сервера с видеокартой NVIDIA (vLLM на CUDA)."
  if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -qi nvidia; then
    echo "GPU NVIDIA обнаружена, но драйвер не установлен. Установите его и перезагрузитесь:"
    echo "  sudo ubuntu-drivers install    (в старых версиях: sudo ubuntu-drivers autoinstall)"
    echo "  либо:  sudo apt install -y nvidia-driver-535    (версию см. в: ubuntu-drivers devices)"
    echo "  затем: sudo reboot  и повторите скрипт."
  else
    echo "GPU NVIDIA не обнаружена. Для сервера БЕЗ видеокарты используйте CPU-вариант:"
    echo "  cd ../docker_variant && ./start.sh     (Docker + Ollama, генерация на CPU)"
  fi
  exit 1
fi
nvidia-smi -L

# ----- 0b. авто-подбор модели Qwen3.6 по VRAM (рекомендация для меню) --------
# Ориентируемся на VRAM одной карты (модель split'ится по картам через TP).
_gpu_mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)"
_gpu_cnt="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | grep -c .)"
_gpu_mem="${_gpu_mem:-0}"; _gpu_cnt="${_gpu_cnt:-1}"
if [ -z "${_USER_MODEL}" ]; then
  if   [ "${_gpu_mem}" -ge 40000 ] && [ "${_gpu_cnt}" -ge 2 ]; then VLLM_MODEL="QuantTrio/Qwen3.6-35B-A3B-AWQ"; _atp=2
  elif [ "${_gpu_mem}" -ge 22000 ]; then VLLM_MODEL="QuantTrio/Qwen3.6-35B-A3B-AWQ"; _atp=1
  elif [ "${_gpu_mem}" -ge 16000 ]; then VLLM_MODEL="QuantTrio/Qwen3.6-27B-AWQ";     _atp=1
  else VLLM_MODEL="QuantTrio/Qwen3.6-27B-AWQ"; _atp=1
  fi
  [ -z "${_USER_TP}" ] && VLLM_TP="${_atp}"
  log "Авто-подбор: ${_gpu_cnt}×${_gpu_mem} МБ VRAM → ${VLLM_MODEL}, tensor-parallel=${VLLM_TP}"
else
  log "Модель задана вручную: ${VLLM_MODEL}, TP=${VLLM_TP}"
fi

# ----- 0c. интерактивный выбор модели (терминал; пропуск при заданном VLLM_MODEL) --
if [ -z "${_USER_MODEL}" ] && [ -t 0 ]; then
  echo
  echo "============================================================"
  echo "  Выбор модели генерации (vLLM ${VLLM_IMAGE##*:})"
  echo "  Обнаружено GPU: ${_gpu_cnt} × ${_gpu_mem} МБ VRAM"
  echo "  Рекомендуется:  ${VLLM_MODEL}  (карт: ${VLLM_TP})"
  echo "------------------------------------------------------------"
  echo "  Qwen3.6:"
  echo "   1) QuantTrio/Qwen3.6-27B-AWQ        ~15 ГБ  — плотная 27B (24–48 ГБ)"
  echo "   2) QuantTrio/Qwen3.6-35B-A3B-AWQ    ~20 ГБ  — MoE 35B/3B актив. (24 ГБ+; реком.)"
  echo "   3) Qwen/Qwen3.6-27B-FP8             ~28 ГБ  — выше точность (нужно 48 ГБ)"
  echo "  GLM (Zhipu):"
  echo "   4) QuantTrio/GLM-4.7-Flash-AWQ      ~18 ГБ  — MoE 30B/3B актив., быстрая (24–48 ГБ) ✅"
  echo "   5) QuantTrio/GLM-4.6-AWQ            ~176 ГБ — 357B MoE (нужно ~4×48 ГБ)"
  echo "   6) cyankiwi/GLM-5.2-AWQ-INT4        ~372 ГБ — 744B MoE (нужно ~4×H200/5×A100)"
  echo "  Прочее:"
  echo "   7) Ввести свою модель (HF-идентификатор)"
  echo "   0) Рекомендованную (Enter)"
  echo "============================================================"
  printf "Выбор [0-7]: "; read -r _ans || _ans=""
  case "${_ans}" in
    1) VLLM_MODEL="QuantTrio/Qwen3.6-27B-AWQ";     VLLM_TP=1 ;;
    2) VLLM_MODEL="QuantTrio/Qwen3.6-35B-A3B-AWQ"; if [ "${_gpu_cnt}" -ge 2 ]; then VLLM_TP=2; else VLLM_TP=1; fi ;;
    3) VLLM_MODEL="Qwen/Qwen3.6-27B-FP8";          VLLM_TP=1 ;;
    4) VLLM_MODEL="QuantTrio/GLM-4.7-Flash-AWQ";   VLLM_TP=1
       echo "  GLM-4.7-Flash — 30B/3B MoE (~18 ГБ), нужен образ vLLM ≥0.14 (у нас ${VLLM_IMAGE##*:})." ;;
    5) VLLM_MODEL="QuantTrio/GLM-4.6-AWQ";         VLLM_TP="${_gpu_cnt}"
       echo "  ВНИМАНИЕ: GLM-4.6 — 357B, ~176 ГБ в AWQ. Нужно ~4×48 ГБ (192 ГБ). На 3×48 может не влезть." ;;
    6) VLLM_MODEL="cyankiwi/GLM-5.2-AWQ-INT4";     VLLM_TP="${_gpu_cnt}"
       echo "  ВНИМАНИЕ: GLM-5.2 — 744B, ~372 ГБ даже в AWQ INT4 (нужно ~4×H200). На малом железе не загрузится." ;;
    7) printf "HF-идентификатор: "; read -r _cm || _cm=""
       [ -n "${_cm}" ] && VLLM_MODEL="${_cm}"
       echo "  Если модель очень новая — может понадобиться свежее образа vLLM (VLLM_IMAGE в .env)." ;;
    *) : ;;   # 0 или Enter — рекомендованная
  esac
  if [ "${_gpu_cnt}" -ge 2 ]; then
    printf "Сколько карт задействовать (tensor-parallel) [%s из %s]: " "${VLLM_TP}" "${_gpu_cnt}"; read -r _tp || _tp=""
    [ -n "${_tp}" ] && VLLM_TP="${_tp}"
  fi
  log "Выбрано: ${VLLM_MODEL}, карт (TP)=${VLLM_TP}"
fi

# ----- 1. системные пакеты --------------------------------------------------
log "Устанавливаю базовые пакеты..."
apt-get update -y
apt-get install -y python3 python3-venv python3-pip ffmpeg curl ca-certificates gnupg git
apt-get install -y libgl1 libglib2.0-0 espeak-ng 2>/dev/null || true   # OpenCV/pymupdf/rawpy + TTS (espeak)
apt-get install -y tesseract-ocr tesseract-ocr-rus libredwg-tools antiword p7zip-full unar 2>/dev/null || true   # OCR (rus) + DWG + .doc + архивы
# ODA File Converter (запасной конвертер DWG→DXF) из локального дистрибутива vendor/oda/*.deb + xvfb
bash "${ROOT_DIR}/scripts/install_oda.sh" "${ROOT_DIR}" || true
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
  log "Создан .env (отредактируйте DOCS_DIR)."
fi
# upsert: обновить ключ, если есть; иначе добавить (без дублей при повторном запуске)
upd(){ if grep -q "^$1=" "${PROJECT_DIR}/.env"; then \
         sed -i "s|^$1=.*|$1=$2|" "${PROJECT_DIR}/.env"; \
       else echo "$1=$2" >> "${PROJECT_DIR}/.env"; fi; }
# имя модели для приложения = served-model-name у vLLM (иначе «модель не найдена»)
upd LLM_MODEL "${VLLM_MODEL}"
upd VLLM_MODEL "${VLLM_MODEL}"
upd VLLM_MAX_LEN "${VLLM_MAX_LEN}"
upd VLLM_TP "${VLLM_TP}"
upd VLLM_IMAGE "${VLLM_IMAGE}"

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
pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
pip install -r "${PROJECT_DIR}/requirements-gpu.txt"
# headless-браузер для парсинга JS-сайтов (браузер + системные зависимости)
python -m playwright install --with-deps chromium 2>/dev/null || python -m playwright install chromium 2>/dev/null || true
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

# ----- 8. полная проверка установки (пакеты и компоненты) -------------------
bash "${ROOT_DIR}/scripts/checklist.sh" "${ROOT_DIR}" || true
