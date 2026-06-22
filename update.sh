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

OLD="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
log "Обновляю код до origin/${BRANCH}..."
git fetch --all -q
git reset --hard "origin/${BRANCH}"
NEW="$(git rev-parse --short HEAD)"

log "Обновляю Python-зависимости..."
./.venv/bin/pip install -q -r gpu_variant/requirements-gpu.txt || true

log "Перезапускаю контейнеры (vLLM + Qdrant)..."
docker compose --env-file gpu_variant/.env -f gpu_variant/docker-compose.gpu.yml up -d

log "Перезапускаю сервис API..."
systemctl restart rag-api

if [[ "${REINDEX}" == "1" ]]; then
  log "Запускаю переиндексацию..."
  ./.venv/bin/python ingest.py || true
fi

log "Готово: ${OLD} → ${NEW}"
