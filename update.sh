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
RUN_USER="${SUDO_USER:-$(whoami)}"
FORCE="${FORCE:-0}"          # FORCE=1 — затирать локальные изменения без stash (неинтерактивно)

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
# git reset --hard затирает локальные правки. Проверяем рабочее дерево и сохраняем изменения.
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  if [[ "${FORCE}" == "1" ]]; then
    log "Локальные изменения в ${TARGET_DIR} будут ЗАТЁРТЫ (FORCE=1)."
  else
    log "Обнаружены локальные изменения — сохраняю их в git stash (запустите с FORCE=1, чтобы затереть)."
    git stash push -u -q -m "update.sh auto $(date +%F_%T)" || {
      echo "Не удалось сохранить локальные изменения (git stash). Прерываю. FORCE=1 для перезаписи."; exit 1; }
  fi
fi
git reset --hard "origin/${BRANCH}"
NEW="$(git rev-parse --short HEAD)"

log "Обновляю системные пакеты (OCR/конвертеры)..."
apt-get install -y ffmpeg tesseract-ocr tesseract-ocr-rus libredwg-tools antiword p7zip-full unar 2>/dev/null || true   # ffmpeg — TTS/аудио для VoIP (SIP)
# ODA File Converter (запасной конвертер DWG→DXF) из локального дистрибутива vendor/oda/*.deb + xvfb
bash "${TARGET_DIR}/scripts/install_oda.sh" "${TARGET_DIR}" || true

log "Обновляю Python-зависимости..."
# Обязательный шаг: код уже обновлён до NEW, зависимости должны соответствовать ему.
# Если pip -r падает — откатываем код к OLD (иначе сервис перезапустится на новом коде
# со старыми зависимостями) и выходим с ошибкой.
if ! ./.venv/bin/pip install -q -r gpu_variant/requirements-gpu.txt; then
  log "ОШИБКА установки зависимостей — откатываю код к ${OLD} и прерываю обновление."
  git reset --hard "${OLD}" -q || true
  chown -R "${RUN_USER}:${RUN_USER}" "${TARGET_DIR}" 2>/dev/null || true
  exit 1
fi
./.venv/bin/pip install -q ezdxf rawpy pytesseract Pillow matplotlib extract-msg py7zr rarfile psutil xlrd python-multipart paramiko || true   # доп. (необязательные) зависимости
# вернуть владельца файлов рабочему пользователю (как в deploy.sh)
chown -R "${RUN_USER}:${RUN_USER}" "${TARGET_DIR}" 2>/dev/null || true

log "Перезапускаю контейнеры (vLLM + Qdrant)..."
docker compose --env-file gpu_variant/.env -f gpu_variant/docker-compose.gpu.yml up -d

log "Перезапускаю сервис API..."
systemctl restart rag-api

if [[ "${REINDEX}" == "1" ]]; then
  log "Запускаю переиндексацию..."
  ./.venv/bin/python ingest.py || true
fi

log "Готово: ${OLD} → ${NEW}"

# полная проверка пакетов и компонентов после обновления
bash "${TARGET_DIR}/scripts/checklist.sh" "${TARGET_DIR}" || true
