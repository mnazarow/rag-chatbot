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
VLLM_TP="${VLLM_TP:-2}"                                      # число GPU для tensor-parallel (по умолч. 2)
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.19.0}"        # версия vLLM (нужна ≥0.19 для Qwen3.6/GLM-5.2)
TORCH_CUDA="${TORCH_CUDA:-cu124}"                            # cu121 / cu124 ...
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"                  # код приложения в корне

log()  { printf "\033[1;32m[setup-gpu]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[setup-gpu]\033[0m %s\n" "$*"; }   # использовалась, но не была объявлена

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

# ----- 0a. проверка свободного места (torch + образ vLLM + веса модели: 40+ ГБ) --
_free_gb="$(df -PBG / 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4);print $4}')"
if [ -n "${_free_gb}" ] && [ "${_free_gb}" -lt 40 ]; then
  echo "[!] На корне (/) свободно ~${_free_gb} ГБ. Нужно 40+ ГБ (torch, образ vLLM, веса модели)."
  echo "    Если это LVM с маленьким корнем — расширьте том на свободное место группы:"
  echo "      sudo lvextend -l +100%FREE /dev/mapper/ubuntu--vg-ubuntu--lv && sudo resize2fs /dev/mapper/ubuntu--vg-ubuntu--lv"
  echo "    (df -h /  и  sudo vgs  покажут текущее и свободное место)."
  printf "Продолжить всё равно? [y/N]: "; read -r _go || _go=""
  case "${_go}" in y|Y|да|Да) : ;; *) echo "Прервано — освободите/расширьте диск и повторите."; exit 1 ;; esac
fi

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
  echo "  Qwen3.6 — МУЛЬТИМОДАЛЬНЫЕ (описывают картинки, vision ✅; базовые, полная точность):"
  echo "   1) Qwen/Qwen3.6-35B-A3B            MoE 35B/3B · vision · нужно 2×48 ГБ (TP=2)"
  echo "   2) Qwen/Qwen3.6-27B                плотная 27B · vision · нужно 2×48 ГБ (TP=2)"
  echo "  Qwen3.6 — квантованные (компактные, обычно ТОЛЬКО текст):"
  echo "   3) QuantTrio/Qwen3.6-35B-A3B-AWQ    ~20 ГБ  — MoE 35B/3B (24 ГБ+; реком. для текста)"
  echo "   4) QuantTrio/Qwen3.6-27B-AWQ        ~15 ГБ  — плотная 27B (24–48 ГБ)"
  echo "   5) Qwen/Qwen3.6-27B-FP8             ~28 ГБ  — выше точность (нужно 48 ГБ)"
  echo "  GLM (Zhipu):"
  echo "   6) QuantTrio/GLM-4.7-Flash-AWQ      ~18 ГБ  — MoE 30B/3B актив., быстрая (24–48 ГБ)"
  echo "   7) QuantTrio/GLM-4.6-AWQ            ~176 ГБ — 357B MoE (нужно ~4×48 ГБ)"
  echo "   8) cyankiwi/GLM-5.2-AWQ-INT4        ~372 ГБ — 744B MoE (нужно ~4×H200/5×A100)"
  echo "  Прочее:"
  echo "   9) Ввести свою модель (HF-идентификатор)"
  echo "   0) Рекомендованную (Enter)"
  echo "------------------------------------------------------------"
  echo "  Для описания картинок при индексации выбирайте 1 или 2 (vision)."
  echo "============================================================"
  printf "Выбор [0-9]: "; read -r _ans || _ans=""
  # для мультимодальных базовых моделей нужен tensor-parallel по 2 карты (не влезают в 48 ГБ)
  _mm_tp=1; [ "${_gpu_cnt}" -ge 2 ] && _mm_tp=2
  case "${_ans}" in
    1) VLLM_MODEL="Qwen/Qwen3.6-35B-A3B";          VLLM_TP="${_mm_tp}"
       echo "  Мультимодальная (vision). Полная точность — нужно 2×48 ГБ (TP=2), образ vLLM ≥0.19 и --trust-remote-code (уже включён)."
       [ "${_gpu_cnt}" -lt 2 ] && echo "  ВНИМАНИЕ: одна карта 48 ГБ — базовая 35B-A3B, скорее всего, не влезет. Возьмите AWQ (п.3) или добавьте GPU." ;;
    2) VLLM_MODEL="Qwen/Qwen3.6-27B";              VLLM_TP="${_mm_tp}"
       echo "  Мультимодальная (vision). Плотная 27B в полной точности — нужно 2×48 ГБ (TP=2)."
       [ "${_gpu_cnt}" -lt 2 ] && echo "  ВНИМАНИЕ: одна карта 48 ГБ — базовая 27B не влезет. Возьмите AWQ (п.4) или добавьте GPU." ;;
    3) VLLM_MODEL="QuantTrio/Qwen3.6-35B-A3B-AWQ"; if [ "${_gpu_cnt}" -ge 2 ]; then VLLM_TP=2; else VLLM_TP=1; fi ;;
    4) VLLM_MODEL="QuantTrio/Qwen3.6-27B-AWQ";     VLLM_TP=1 ;;
    5) VLLM_MODEL="Qwen/Qwen3.6-27B-FP8";          VLLM_TP=1 ;;
    6) VLLM_MODEL="QuantTrio/GLM-4.7-Flash-AWQ";   VLLM_TP=1
       echo "  GLM-4.7-Flash — 30B/3B MoE (~18 ГБ), нужен образ vLLM ≥0.14 (у нас ${VLLM_IMAGE##*:})." ;;
    7) VLLM_MODEL="QuantTrio/GLM-4.6-AWQ";         VLLM_TP="${_gpu_cnt}"
       echo "  ВНИМАНИЕ: GLM-4.6 — 357B, ~176 ГБ в AWQ. Нужно ~4×48 ГБ (192 ГБ). На 3×48 может не влезть." ;;
    8) VLLM_MODEL="cyankiwi/GLM-5.2-AWQ-INT4";     VLLM_TP="${_gpu_cnt}"
       echo "  ВНИМАНИЕ: GLM-5.2 — 744B, ~372 ГБ даже в AWQ INT4 (нужно ~4×H200). На малом железе не загрузится." ;;
    9) printf "HF-идентификатор: "; read -r _cm || _cm=""
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
# критичные пакеты извлечения контента: OCR + .doc + архивы + PDF→картинки.
# Ставим НЕ молча (с повтором), затем проверяем бинарники и явно предупреждаем, если
# что-то не встало (напр. когда-то не хватило места) — чтобы это не оставалось незамеченным.
_content_pkgs="tesseract-ocr tesseract-ocr-rus antiword p7zip-full unar libarchive-tools poppler-utils"
apt-get install -y ${_content_pkgs} || { apt-get update -y; apt-get install -y ${_content_pkgs} || true; }
apt-get install -y libredwg-tools 2>/dev/null || true   # DWG→DXF (может отсутствовать в репо — необязательно)
for _b in tesseract antiword 7z unar bsdtar pdftoppm; do
  command -v "${_b}" >/dev/null 2>&1 || echo "[!] системный пакет для '${_b}' не установился — доставьте вручную: sudo apt install -y ${_content_pkgs}"
done
# ODA File Converter (запасной конвертер DWG→DXF) из локального дистрибутива vendor/oda/*.deb + xvfb
bash "${ROOT_DIR}/scripts/install_oda.sh" "${ROOT_DIR}" || true
# Python для приложения: нужен 3.10–3.13 (под 3.14+ ещё НЕТ колёс PyTorch). Системный
# python3 может быть слишком новым (напр. 3.14) — тогда доустанавливаем 3.12.
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
[ -n "${PYBIN}" ] || { echo "Не удалось получить Python 3.10–3.13. Поставьте вручную: sudo apt install python3.12 python3.12-venv, затем повторите."; exit 1; }
log "Python для приложения: ${PYBIN} ($(${PYBIN} --version 2>&1))"

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

# ----- Профиль настроек по умолчанию (при установке/переустановке) -----------
# Записываются в .env. Правки из админки (runtime_config.json) имеют приоритет, поэтому
# после ручной настройки в панели эти значения не мешают. Отключить: APPLY_DEFAULTS=0.
if [ "${APPLY_DEFAULTS:-1}" = "1" ]; then
  upd TOP_K_RETRIEVE 60;            upd MIN_SCORE 0.15;            upd AUTO_FILTER true
  upd NO_ANSWER_FALLBACK true;      upd HIDE_SOURCES_IF_NO_ANSWER true
  upd INLINE_CITATIONS true;        upd DIALOG_REWRITE true;      upd LLM_THINK true
  upd LLM_QUEUE_TIMEOUT 600
  upd CHUNK_SIZE 1800;             upd CHUNK_OVERLAP 360;         upd INGEST_MAX_CHUNKS 2000000
  upd STRUCTURE_CHUNK true;         upd INDEX_CONTEXTUAL true
  upd INDEX_PARENT_CONTEXT true;    upd INDEX_DEDUP true
  upd TELEGRAM_PROXY "socks5h://10.0.0.2:1080"
  upd WEB_CRAWL_DEPTH 14;           upd WEB_MAX_PAGES 200000;      upd WEB_MAX_FILES 500000
  upd WEB_JS_WAIT load;             upd WEB_RESPECT_CRAWL_DELAY false; upd WEB_SITE_CONCURRENCY 3
  log "Профиль настроек по умолчанию записан в .env."
fi

# ----- 4b. Redis (кэш агрегатов + семантический кэш) — ставим и включаем по умолчанию -----
# Отключить: INSTALL_REDIS=0.
if [ "${INSTALL_REDIS:-1}" = "1" ]; then
  log "Устанавливаю и запускаю Redis (кэш)..."
  apt-get install -y -o DPkg::Lock::Timeout=120 redis-server 2>/dev/null \
    || { apt-get update -y; apt-get install -y redis-server 2>/dev/null || true; }
  systemctl enable --now redis-server 2>/dev/null || systemctl enable --now redis 2>/dev/null || true
  if redis-cli ping >/dev/null 2>&1; then
    upd REDIS_ENABLED true; upd REDIS_HOST 127.0.0.1; upd REDIS_PORT 6379
    log "Redis работает — REDIS_ENABLED=true (кэш агрегатов, семантический кэш, общий учёт vision)."
  else
    warn "Redis не поднялся — приложение работает и без него (кэш в памяти / через rag_logs.db)."
  fi
fi

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
# пересоздать venv, если он собран другой версией Python (напр. остался на 3.14)
if [ -d .venv ]; then
  _cur="$(.venv/bin/python -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo none)"
  _want="$(${PYBIN} -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
  [ "${_cur}" != "${_want}" ] && { log "Пересоздаю .venv (${_cur} → ${_want})"; rm -rf .venv; }
fi
"${PYBIN}" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
pip install -r "${PROJECT_DIR}/requirements-gpu.txt"
# headless-браузер для парсинга JS-сайтов (браузер + системные зависимости)
python -m playwright install --with-deps chromium 2>/dev/null || python -m playwright install chromium 2>/dev/null || true
# XTTS (клонирование голоса) — ОТДЕЛЬНЫЙ venv .venv-xtts + микросервис (systemd rag-xtts ниже).
# Так coqui-tts (transformers>=4.57) не конфликтует с ядром RAG (.venv, transformers==4.44.2).
# Приложение общается с сервисом по HTTP (XTTS_URL). Отключить: INSTALL_XTTS=0.
XTTS_PORT="${XTTS_PORT:-8020}"
INSTALL_XTTS="${INSTALL_XTTS:-1}"
if [ "${INSTALL_XTTS}" = "1" ]; then
  log "Ставлю XTTS в отдельное окружение .venv-xtts (изолированно от ядра)..."
  if ( cd "${ROOT_DIR}" \
        && "${PYBIN}" -m venv .venv-xtts \
        && .venv-xtts/bin/pip install -q --upgrade pip wheel \
        && .venv-xtts/bin/pip install -q torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
        && .venv-xtts/bin/pip install -q "coqui-tts>=0.24.0" "fastapi" "uvicorn[standard]" ); then
    log "XTTS-окружение готово (сервис будет на 127.0.0.1:${XTTS_PORT})."
  else
    warn "XTTS-окружение не собралось — голос-клон будет недоступен (остальное работает)."
    INSTALL_XTTS=0
  fi
fi
# приложение читает .env из текущей папки — кладём симлинк на gpu-конфиг
ln -sf "${PROJECT_DIR}/.env" "${ROOT_DIR}/.env"
# адрес сервиса XTTS в .env (приложение направит синтез голоса-клона на него по HTTP)
if [ "${INSTALL_XTTS}" = "1" ]; then
  if grep -qE '^XTTS_URL=' "${PROJECT_DIR}/.env" 2>/dev/null; then
    sed -i "s|^XTTS_URL=.*|XTTS_URL=http://127.0.0.1:${XTTS_PORT}|" "${PROJECT_DIR}/.env"
  else
    echo "XTTS_URL=http://127.0.0.1:${XTTS_PORT}" >> "${PROJECT_DIR}/.env"
  fi
fi

# ----- 7. прогрев эмбеддера/реранкера на CUDA ------------------------------
log "Прогреваю эмбеддинги и реранк на GPU..."
DEVICE=cuda python - <<'PY'
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
SentenceTransformer("BAAI/bge-m3", device="cuda")
FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
print("OK")
PY

# ----- 7b. запуск приложения как systemd-сервиса (автозапуск + Restart=always) --
API_PORT="${API_PORT:-8000}"
SVC_USER="${SUDO_USER:-root}"
# Сервис-пользователь ДОЛЖЕН иметь доступ к рабочей папке, иначе systemd падает на шаге
# CHDIR (status=200/CHDIR) и бесконечно перезапускается. Частый случай: проект в /root
# (домашняя root, права 700), а SUDO_USER — обычный пользователь → доступа в /root нет.
# Проверяем реальный доступ; если его нет — запускаем сервис от root.
if [ "${SVC_USER}" != "root" ] && ! sudo -u "${SVC_USER}" test -x "${ROOT_DIR}" 2>/dev/null; then
  warn "Пользователь ${SVC_USER} не имеет доступа к ${ROOT_DIR} (проект в /root?) — сервис будет запущен от root."
  SVC_USER="root"
fi
# venv в setup_gpu.sh создаётся под root; если сервис под пользователем — отдать права
[ "${SVC_USER}" != "root" ] && chown -R "${SVC_USER}:${SVC_USER}" "${ROOT_DIR}" 2>/dev/null || true
# папка документов из .env (создаём, если нет)
_docs="$(grep -E '^DOCS_DIR=' "${PROJECT_DIR}/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"')"
[ -n "${_docs}" ] && mkdir -p "${_docs}" 2>/dev/null || true
if command -v systemctl >/dev/null 2>&1; then
  log "Регистрирую и запускаю systemd-сервис rag-api (автозапуск)..."
  sed -e "s|__USER__|${SVC_USER}|g" -e "s|__ROOT__|${ROOT_DIR}|g" -e "s|__PORT__|${API_PORT}|g" \
      "${PROJECT_DIR}/rag-api.service.tpl" > /etc/systemd/system/rag-api.service
  # Эмбеддер/реранкер — на самую свободную GPU (vLLM занимает cuda:0/1 при TP; иначе CUDA OOM).
  # Пробрасываем через CUDA_VISIBLE_DEVICES (drop-in); ingest из админки наследует окружение.
  if command -v nvidia-smi >/dev/null 2>&1 && [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -ge 2 ]; then
    _free_gpu="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
                  | sort -t, -k2 -n -r | head -1 | awk -F, '{gsub(/ /,"",$1); print $1}')"
    if [ -n "${_free_gpu}" ]; then
      mkdir -p /etc/systemd/system/rag-api.service.d
      printf '[Service]\nEnvironment=CUDA_VISIBLE_DEVICES=%s\n' "${_free_gpu}" \
        > /etc/systemd/system/rag-api.service.d/10-embed-gpu.conf
      log "Эмбеддер/реранкер → GPU ${_free_gpu} (самая свободная; CUDA_VISIBLE_DEVICES)."
    fi
  fi
  systemctl daemon-reload
  systemctl enable --now rag-api
  log "Жду готовности веб-интерфейса (загрузка эмбеддера/реранка — до ~1–2 мин)..."
  for i in {1..40}; do curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1 && { log "Веб-интерфейс поднялся."; break; }; sleep 3; done
  # микросервис XTTS (клонирование голоса) — отдельный юнит rag-xtts из .venv-xtts
  if [ "${INSTALL_XTTS}" = "1" ] && [ -x "${ROOT_DIR}/.venv-xtts/bin/python" ]; then
    log "Регистрирую и запускаю systemd-сервис rag-xtts (клонирование голоса)..."
    sed -e "s|__USER__|${SVC_USER}|g" -e "s|__ROOT__|${ROOT_DIR}|g" \
        -e "s|__PORT__|${XTTS_PORT}|g" -e "s|__GPU__|1|g" \
        "${PROJECT_DIR}/rag-xtts.service.tpl" > /etc/systemd/system/rag-xtts.service
    systemctl daemon-reload
    systemctl enable --now rag-xtts
  fi
else
  log "systemd не найден — запустите вручную: ${ROOT_DIR}/.venv/bin/uvicorn app:app --host 0.0.0.0 --port ${API_PORT}"
  [ "${INSTALL_XTTS}" = "1" ] && log "И сервис XTTS: ${ROOT_DIR}/.venv-xtts/bin/python ${ROOT_DIR}/xtts_service.py (порт ${XTTS_PORT})"
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
cat <<EOF

============================================================
  Готово! Приложение запущено как сервис rag-api.
  Веб-чат:     http://${IP:-<ip-сервера>}:${API_PORT}    (раздел «Администратор»)
  vLLM API:    http://localhost:8001/v1   (модель ${VLLM_MODEL})

  Папка документов: DOCS_DIR в gpu_variant/.env (по умолч. ${_docs:-/opt/db}) —
    положите туда файлы и нажмите «Переиндексировать» в админке.
  Управление:  systemctl status|restart|stop rag-api
  Логи:        journalctl -u rag-api -f
  После правки .env:  sudo systemctl restart rag-api
============================================================
EOF

# ----- 8. полная проверка установки (пакеты и компоненты) -------------------
bash "${ROOT_DIR}/scripts/checklist.sh" "${ROOT_DIR}" || true
