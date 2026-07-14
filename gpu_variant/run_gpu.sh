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
# Модель и TP подбираются АВТОМАТИЧЕСКИ по VRAM/числу GPU (ниже) + меню выбора.
# Образ vLLM v0.19.0 понимает Qwen3.6 (qwen3_5_moe) и GLM-5.2.
_USER_MODEL="${VLLM_MODEL:-}"; _USER_TP="${VLLM_TP:-}"      # пусто = авто-подбор
VLLM_MODEL="${VLLM_MODEL:-QuantTrio/Qwen3.6-35B-A3B-AWQ}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-16384}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.19.0}"        # ≥0.19 нужна для Qwen3.6/GLM-5.2
TORCH_CUDA="${TORCH_CUDA:-cu124}"
API_PORT="${API_PORT:-8000}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"                 # рекомендуется задать!
RUN_USER="${SUDO_USER:-$(whoami)}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

log(){ printf "\033[1;32m[run-gpu]\033[0m %s\n" "$*"; }

[[ $EUID -eq 0 ]] || { echo "Запустите через sudo (нужны установка пакетов и systemd)."; exit 1; }
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi не найден — этот скрипт для сервера с видеокартой NVIDIA (vLLM на CUDA)."
  if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -qi nvidia; then
    echo "GPU NVIDIA обнаружена, но драйвер не установлен. Установите и перезагрузитесь:"
    echo "  sudo ubuntu-drivers install   (старые версии: sudo ubuntu-drivers autoinstall)  &&  sudo reboot"
  else
    echo "GPU нет. Для сервера БЕЗ видеокарты используйте CPU-вариант:  cd ../docker_variant && ./start.sh"
  fi
  exit 1
fi
nvidia-smi -L

# ----- 0a. проверка свободного места (torch + образ vLLM + веса модели: 40+ ГБ) --
_free_gb="$(df -PBG / 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4);print $4}')"
if [ -n "${_free_gb}" ] && [ "${_free_gb}" -lt 40 ]; then
  echo "[!] На корне (/) свободно ~${_free_gb} ГБ. Нужно 40+ ГБ (torch, образ vLLM, веса модели)."
  echo "    LVM с маленьким корнем — расширьте на свободное место группы:"
  echo "      sudo lvextend -l +100%FREE /dev/mapper/ubuntu--vg-ubuntu--lv && sudo resize2fs /dev/mapper/ubuntu--vg-ubuntu--lv"
fi

# ----- 0b. авто-подбор модели vLLM по VRAM и числу GPU (если не задано вручную) -----
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
       echo "  ВНИМАНИЕ: GLM-4.6 — 357B, ~176 ГБ в AWQ (нужно ~4×48 ГБ). На 3×48 может не влезть." ;;
    6) VLLM_MODEL="cyankiwi/GLM-5.2-AWQ-INT4";     VLLM_TP="${_gpu_cnt}"
       echo "  ВНИМАНИЕ: GLM-5.2 — 744B, ~372 ГБ даже в AWQ INT4 (нужно ~4×H200). На малом железе не загрузится." ;;
    7) printf "HF-идентификатор: "; read -r _cm || _cm=""
       [ -n "${_cm}" ] && VLLM_MODEL="${_cm}"
       echo "  Очень новым моделям может понадобиться свежее образа vLLM (VLLM_IMAGE в .env)." ;;
    *) : ;;
  esac
  if [ "${_gpu_cnt}" -ge 2 ]; then
    printf "Сколько карт задействовать (tensor-parallel) [%s из %s]: " "${VLLM_TP}" "${_gpu_cnt}"; read -r _tp || _tp=""
    [ -n "${_tp}" ] && VLLM_TP="${_tp}"
  fi
  log "Выбрано: ${VLLM_MODEL}, карт (TP)=${VLLM_TP}"
fi

# ----- 1. системные пакеты + Docker + NVIDIA toolkit ------------------------
log "Системные пакеты..."
apt-get update -y
# системный python3 (3.10–3.12 подходят) + venv + pip; версия-специфичный пакет не требуется
apt-get install -y python3 python3-venv python3-pip ffmpeg curl ca-certificates gnupg git
apt-get install -y libgl1 libglib2.0-0 2>/dev/null || true   # зависимости OpenCV/pymupdf/rawpy (загрузка изображений)
apt-get install -y espeak-ng 2>/dev/null || true   # синтез речи для голосовых ответов (TTS)
apt-get install -y libredwg-tools 2>/dev/null || true   # dwg2dxf: конвертация DWG (необязательно)
# критичные пакеты извлечения контента: OCR + .doc + архивы + PDF→картинки.
# Ставим НЕ молча (с повтором), затем проверяем и явно предупреждаем, если не встало.
_content_pkgs="tesseract-ocr tesseract-ocr-rus antiword p7zip-full unar poppler-utils"
apt-get install -y ${_content_pkgs} || { apt-get update -y; apt-get install -y ${_content_pkgs} || true; }
for _b in tesseract antiword 7z unar pdftoppm; do
  command -v "${_b}" >/dev/null 2>&1 || echo "[!] системный пакет для '${_b}' не установился — доставьте: sudo apt install -y ${_content_pkgs}"
done
# ODA File Converter (запасной конвертер DWG→DXF) из локального дистрибутива vendor/oda/*.deb + xvfb
bash "${ROOT_DIR}/scripts/install_oda.sh" "${ROOT_DIR}" || true
# Нужен Python 3.10–3.13 (под 3.14+ ещё НЕТ колёс PyTorch). Если системный слишком новый —
# доустанавливаем python3.12.
_pick_py(){ for v in python3.12 python3.11 python3.13 python3.10; do command -v "$v" >/dev/null 2>&1 && { echo "$v"; return 0; }; done; return 1; }
PYBIN="$(_pick_py || true)"
if [ -z "${PYBIN}" ]; then
  log "Совместимый Python (3.10–3.13) не найден — ставлю python3.12 (для PyTorch)..."
  apt-get install -y python3.12 python3.12-venv python3.12-dev 2>/dev/null || {
    apt-get install -y software-properties-common 2>/dev/null || true
    add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
    apt-get update -y
    apt-get install -y python3.12 python3.12-venv python3.12-dev 2>/dev/null || true
  }
  PYBIN="$(_pick_py || true)"
fi
[ -n "${PYBIN}" ] || { echo "Не удалось получить Python 3.10–3.13. Поставьте вручную: apt install python3.12 python3.12-venv."; exit 1; }
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
upd VLLM_MAX_LEN "${VLLM_MAX_LEN}"; upd VLLM_TP "${VLLM_TP}"; upd VLLM_IMAGE "${VLLM_IMAGE}"
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
# пересоздать venv, если собран другой версией Python (напр. остался на 3.14 без torch)
if [ -d .venv ]; then
  _cur="$(.venv/bin/python -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo none)"
  _want="$(${PYBIN} -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
  [ "${_cur}" != "${_want}" ] && { log "Пересоздаю .venv (${_cur} → ${_want})"; rm -rf .venv; }
fi
sudo -u "${RUN_USER}" "${PYBIN}" -m venv .venv
sudo -u "${RUN_USER}" ./.venv/bin/pip install --upgrade pip wheel
# torch без жёсткой версии — pip подберёт совместимый с вашим Python и CUDA-каналом
sudo -u "${RUN_USER}" ./.venv/bin/pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" || {
  echo "Не удалось поставить torch из канала ${TORCH_CUDA}.";
  echo "Попробуйте другой CUDA-канал: TORCH_CUDA=cu121 (или cu126) повторно запустите скрипт,";
  echo "или используйте Python 3.10–3.12 (для самых новых версий Python колёс может не быть).";
  exit 1; }
sudo -u "${RUN_USER}" ./.venv/bin/pip install -r "${PROJECT_DIR}/requirements-gpu.txt"
sudo -u "${RUN_USER}" ./.venv/bin/pip install -q ezdxf rawpy pytesseract Pillow matplotlib extract-msg py7zr rarfile psutil || true   # DWG/DXF + OCR + Outlook .msg + архивы + метрики
# headless-браузер для парсинга JS-сайтов: OS-зависимости ставим от root (apt),
# сам браузер — в кэш пользователя сервиса (иначе приложение его не найдёт)
"${ROOT_DIR}/.venv/bin/python" -m playwright install-deps chromium 2>/dev/null || true
sudo -u "${RUN_USER}" ./.venv/bin/python -m playwright install chromium 2>/dev/null || true
# XTTS (клонирование голоса) — ОТДЕЛЬНЫЙ venv .venv-xtts + микросервис (systemd rag-xtts ниже),
# чтобы coqui-tts (transformers>=4.57) не конфликтовал с ядром (.venv, transformers==4.44.2).
# Приложение обращается к сервису по HTTP (XTTS_URL). Отключить: INSTALL_XTTS=0.
XTTS_PORT="${XTTS_PORT:-8020}"
INSTALL_XTTS="${INSTALL_XTTS:-1}"
if [ "${INSTALL_XTTS}" = "1" ]; then
  log "Ставлю XTTS в отдельное окружение .venv-xtts (изолированно от ядра)..."
  if sudo -u "${RUN_USER}" "${PYBIN}" -m venv .venv-xtts \
      && sudo -u "${RUN_USER}" ./.venv-xtts/bin/pip install -q --upgrade pip wheel \
      && sudo -u "${RUN_USER}" ./.venv-xtts/bin/pip install -q torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
      && sudo -u "${RUN_USER}" ./.venv-xtts/bin/pip install -q "coqui-tts>=0.24.0" "fastapi" "uvicorn[standard]"; then
    upd XTTS_URL "http://127.0.0.1:${XTTS_PORT}"
    log "XTTS-окружение готово (сервис будет на 127.0.0.1:${XTTS_PORT})."
  else
    log "[!] XTTS-окружение не собралось — голос-клон будет недоступен (остальное работает)."
    INSTALL_XTTS=0
  fi
fi
chmod +x "${PROJECT_DIR}/apply_llm.sh"

# ----- 5. systemd-сервис API (автозапуск + Restart=always) ------------------
log "Регистрирую systemd-сервис rag-api..."
sed -e "s|__USER__|${RUN_USER}|g" -e "s|__ROOT__|${ROOT_DIR}|g" -e "s|__PORT__|${API_PORT}|g" \
    "${PROJECT_DIR}/rag-api.service.tpl" > /etc/systemd/system/rag-api.service
systemctl daemon-reload
systemctl enable --now rag-api

# микросервис XTTS (клонирование голоса) — отдельный юнит rag-xtts из .venv-xtts
if [ "${INSTALL_XTTS}" = "1" ] && [ -x "${ROOT_DIR}/.venv-xtts/bin/python" ]; then
  log "Регистрирую systemd-сервис rag-xtts (клонирование голоса)..."
  sed -e "s|__USER__|${RUN_USER}|g" -e "s|__ROOT__|${ROOT_DIR}|g" \
      -e "s|__PORT__|${XTTS_PORT}|g" -e "s|__GPU__|1|g" \
      "${PROJECT_DIR}/rag-xtts.service.tpl" > /etc/systemd/system/rag-xtts.service
  systemctl daemon-reload
  systemctl enable --now rag-xtts
fi

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

# ----- 6. полная проверка установки (пакеты и компоненты) -------------------
sudo -u "${RUN_USER}" bash "${ROOT_DIR}/scripts/checklist.sh" "${ROOT_DIR}" || true
