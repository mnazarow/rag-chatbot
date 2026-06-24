#!/usr/bin/env bash
# =============================================================================
#  Запуск GPU-варианта RAG на ЧИСТОМ сервере одной командой.
#  Делает всё: Docker + NVIDIA toolkit, vLLM + Qdrant, Python-окружение,
#  systemd-сервис API с автозапуском. Все остальные настройки — в веб-админке.
#
#  Использование:   sudo bash run_gpu.sh
#  Опционально:     VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ sudo -E bash run_gpu.sh
# =============================================================================
set -euo pipefail

# ----- параметры первого запуска (дальше всё меняется в админке) -------------
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-14B-Instruct-AWQ}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-16384}"
VLLM_TP="${VLLM_TP:-1}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"
API_PORT="${API_PORT:-8000}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"                 # рекомендуется задать!
RUN_USER="${SUDO_USER:-$(whoami)}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

log(){ printf "\033[1;32m[run-gpu]\033[0m %s\n" "$*"; }

[[ $EUID -eq 0 ]] || { echo "Запустите через sudo (нужны установка пакетов и systemd)."; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi не найден — установите драйвер NVIDIA."; exit 1; }
nvidia-smi -L

# ----- 1. системные пакеты + Docker + NVIDIA toolkit ------------------------
log "Системные пакеты..."
apt-get update -y
# системный python3 (3.10–3.12 подходят) + venv + pip; версия-специфичный пакет не требуется
apt-get install -y python3 python3-venv python3-pip ffmpeg curl ca-certificates gnupg
apt-get install -y libredwg-tools 2>/dev/null || true   # dwg2dxf: конвертация DWG (необязательно)
apt-get install -y tesseract-ocr tesseract-ocr-rus 2>/dev/null || true   # OCR для CR2/фото
apt-get install -y antiword 2>/dev/null || true   # чтение старого .doc
apt-get install -y p7zip-full unar 2>/dev/null || true   # распаковка архивов (.7z/.rar)
PYBIN="$(command -v python3.11 || command -v python3.12 || command -v python3.10 || command -v python3)"
log "Использую интерпретатор: ${PYBIN} ($(${PYBIN} --version 2>&1))"
command -v docker >/dev/null || { log "Docker..."; curl -fsSL https://get.docker.com | sh; }
usermod -aG docker "${RUN_USER}" || true

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

# ----- 2. .env (минимальный: дальше всё в админке) --------------------------
if [[ ! -f "${PROJECT_DIR}/.env" ]]; then
  cp "${PROJECT_DIR}/.env.gpu.example" "${PROJECT_DIR}/.env"
fi
upd(){ grep -q "^$1=" "${PROJECT_DIR}/.env" \
        && sed -i "s|^$1=.*|$1=$2|" "${PROJECT_DIR}/.env" \
        || echo "$1=$2" >> "${PROJECT_DIR}/.env"; }
upd LLM_MODEL "${VLLM_MODEL}"; upd VLLM_MODEL "${VLLM_MODEL}"
upd VLLM_MAX_LEN "${VLLM_MAX_LEN}"; upd VLLM_TP "${VLLM_TP}"
upd API_PORT "${API_PORT}"; upd ADMIN_TOKEN "${ADMIN_TOKEN}"
ln -sf "${PROJECT_DIR}/.env" "${ROOT_DIR}/.env"

# папка документов по умолчанию
mkdir -p /opt/db && chown "${RUN_USER}:${RUN_USER}" /opt/db || true

# ----- 3. vLLM + Qdrant -----------------------------------------------------
log "Поднимаю vLLM + Qdrant (первый старт качает веса — долго)..."
cd "${PROJECT_DIR}"
docker compose --env-file .env -f docker-compose.gpu.yml up -d
log "Жду готовности vLLM..."
for i in {1..120}; do curl -sf http://localhost:8001/health >/dev/null 2>&1 && break || sleep 10; done

# ----- 4. Python-окружение --------------------------------------------------
log "Python-окружение + зависимости (torch ${TORCH_CUDA})..."
cd "${ROOT_DIR}"
# каталог должен принадлежать пользователю сервиса: и для venv, и чтобы приложение
# могло писать runtime_config.json, журнал и т.п.
chown -R "${RUN_USER}:${RUN_USER}" "${ROOT_DIR}"
sudo -u "${RUN_USER}" "${PYBIN}" -m venv .venv
sudo -u "${RUN_USER}" ./.venv/bin/pip install --upgrade pip wheel
# torch без жёсткой версии — pip подберёт совместимый с вашим Python и CUDA-каналом
sudo -u "${RUN_USER}" ./.venv/bin/pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" || {
  echo "Не удалось поставить torch из канала ${TORCH_CUDA}.";
  echo "Попробуйте другой CUDA-канал: TORCH_CUDA=cu121 (или cu126) повторно запустите скрипт,";
  echo "или используйте Python 3.10–3.12 (для самых новых версий Python колёс может не быть).";
  exit 1; }
sudo -u "${RUN_USER}" ./.venv/bin/pip install -r "${PROJECT_DIR}/requirements-gpu.txt"
sudo -u "${RUN_USER}" ./.venv/bin/pip install -q ezdxf rawpy pytesseract Pillow extract-msg py7zr rarfile || true   # DWG/DXF + OCR + Outlook .msg + архивы
chmod +x "${PROJECT_DIR}/apply_llm.sh"

# ----- 5. systemd-сервис API (автозапуск + Restart=always) ------------------
log "Регистрирую systemd-сервис rag-api..."
sed -e "s|__USER__|${RUN_USER}|g" -e "s|__ROOT__|${ROOT_DIR}|g" -e "s|__PORT__|${API_PORT}|g" \
    "${PROJECT_DIR}/rag-api.service.tpl" > /etc/systemd/system/rag-api.service
systemctl daemon-reload
systemctl enable --now rag-api

IP="$(hostname -I | awk '{print $1}')"
cat <<EOF

============================================================
  Готово! Сервер запущен и стартует автоматически.

  Откройте веб-панель:   http://${IP}:${API_PORT}
  Раздел «Администратор» — там настраивается ВСЁ:
    • папка с документами (DOCS_DIR) → кнопка «Переиндексировать»
    • модель vLLM, контекст, число GPU → «Применить модель LLM»
    • параметры поиска, промпт, пороги — применяются на лету
    • при смене моделей эмбеддингов/устройства → «Перезапустить сервис»

  Управление:   bash gpu_variant/manage.sh {status|logs|restart|stop|start}
============================================================
EOF
