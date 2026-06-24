#!/usr/bin/env bash
# =============================================================================
#  Обновление уже развёрнутого сервера из GitHub (новый релиз).
#  Делает git pull, обновляет зависимости, перезапускает сервис и контейнеры.
#
#  Запуск:            sudo bash update.sh
#  С переиндексацией: sudo REINDEX=1 bash update.sh
# =============================================================================
set -euo pipefail

TARGET_DIR="${TARGET_DIR:-/opt/rag}"
BRANCH="${BRANCH:-main}"
REINDEX="${REINDEX:-0}"

log(){ printf "\033[1;36m[update]\033[0m %s\n" "$*"; }
cd "${TARGET_DIR}"

# update.sh — только для уже развёрнутого сервера. Если установки нет — направляем на run_gpu.sh
if [[ ! -x ./.venv/bin/pip ]] || ! command -v docker >/dev/null; then
  echo "Сервер ещё не развёрнут (нет .venv или Docker)."
  echo "Запустите первичную установку:"
  echo "  sudo bash -c \"ADMIN_TOKEN='пароль' bash ${TARGET_DIR}/gpu_variant/run_gpu.sh\""
  exit 1
fi

OLD="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
log "Обновляю код до origin/${BRANCH}..."
git fetch --all -q
git reset --hard "origin/${BRANCH}"
NEW="$(git rev-parse --short HEAD)"

log "Обновляю системные пакеты (OCR/конвертеры)..."
apt-get install -y tesseract-ocr tesseract-ocr-rus libredwg-tools antiword p7zip-full unar 2>/dev/null || true

log "Обновляю Python-зависимости..."
./.venv/bin/pip install -q -r gpu_variant/requirements-gpu.txt || true
./.venv/bin/pip install -q ezdxf rawpy pytesseract Pillow extract-msg py7zr rarfile psutil xlrd python-multipart paramiko || true   # новые зависимости

log "Перезапускаю контейнеры (vLLM + Qdrant)..."
docker compose --env-file gpu_variant/.env -f gpu_variant/docker-compose.gpu.yml up -d

log "Перезапускаю сервис API..."
systemctl restart rag-api

if [[ "${REINDEX}" == "1" ]]; then
  log "Запускаю переиндексацию..."
  ./.venv/bin/python ingest.py || true
fi

log "Готово: ${OLD} → ${NEW}"
