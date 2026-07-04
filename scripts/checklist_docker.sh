#!/usr/bin/env bash
# =============================================================================
#  Полная проверка Docker-развёртывания RAG (Linux / macOS).
#  Проверяет контейнеры, сервисы (Qdrant / Redis / приложение / Ollama), а также
#  системные инструменты и Python-пакеты ВНУТРИ образа приложения.
#  Печатает цветной чек-лист с итогом [OK] / [~] / [X].
#
#  Запуск (из папки с docker-compose.yml, напр. docker_variant/):
#     bash ../scripts/checklist_docker.sh
#  Настройка через переменные окружения:
#     APP_CONTAINER=rag_app QDRANT_CONTAINER=rag_qdrant REDIS_CONTAINER=rag_redis \
#     APP_URL=http://localhost:8000 VENV_PY=/opt/venv/bin/python
# =============================================================================

APP="${APP_CONTAINER:-rag_app}"
QDR="${QDRANT_CONTAINER:-rag_qdrant}"
RED="${REDIS_CONTAINER:-rag_redis}"
APP_URL="${APP_URL:-http://localhost:8000}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
VPY="${VENV_PY:-/opt/venv/bin/python}"

if [[ -t 1 ]]; then C_G=$'\033[1;32m'; C_Y=$'\033[1;33m'; C_R=$'\033[1;31m'; C_C=$'\033[1;36m'; C_D=$'\033[0;90m'; C_0=$'\033[0m'
else C_G=""; C_Y=""; C_R=""; C_C=""; C_D=""; C_0=""; fi
FAILS=0; WARNS=0; OKS=0
ok()   { printf "  ${C_G}[OK]${C_0} %s${C_D}%s${C_0}\n" "$1" "${2:+  — $2}"; OKS=$((OKS+1)); }
warn() { printf "  ${C_Y}[~]${C_0}  %s${C_D}%s${C_0}\n" "$1" "${2:+  — $2}"; WARNS=$((WARNS+1)); }
fail() { printf "  ${C_R}[X]${C_0}  %s${C_D}%s${C_0}\n" "$1" "${2:+  — $2}"; FAILS=$((FAILS+1)); }
sec()  { printf "\n${C_C}%s${C_0}\n" "$1"; }

http_ok() { curl -sf -m "${2:-5}" "$1" >/dev/null 2>&1; }
cup()  { [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == "true" ]]; }
cexists() { docker inspect "$1" >/dev/null 2>&1; }
inapp() { docker exec "$APP" sh -lc "command -v $1" >/dev/null 2>&1; }

printf "${C_C}============================================================${C_0}\n"
printf "${C_C}  Чек-лист Docker-установки RAG (контейнеры и образ)${C_0}\n"
printf "${C_C}============================================================${C_0}\n"

if ! command -v docker >/dev/null 2>&1; then
  fail "docker не найден в PATH" "проверьте установку Docker"
  printf "${C_R}  Дальнейшие проверки невозможны.${C_0}\n"; exit 1
fi

# ============================ 1. Контейнеры =================================
sec "1. Контейнеры"
if cup "$QDR"; then ok "Qdrant ($QDR) работает"; else fail "контейнер Qdrant ($QDR) не запущен"; fi
if cexists "$RED"; then
  if cup "$RED"; then ok "Redis ($RED) работает"; else fail "контейнер Redis ($RED) не запущен"; fi
fi
APP_UP=0
if cup "$APP"; then ok "Приложение ($APP) работает"; APP_UP=1; else fail "контейнер приложения ($APP) не запущен"; fi

# ============================ 2. Сервисы ====================================
sec "2. Сервисы и подключения"
http_ok "${QDRANT_URL%/}/collections" 5 && ok "Qdrant отвечает" "$QDRANT_URL" || warn "Qdrant не отвечает" "$QDRANT_URL — возможно, ещё стартует"
if cexists "$RED"; then
  if [[ "$(docker exec "$RED" redis-cli ping 2>/dev/null | tr -d '\r')" == "PONG" ]]; then ok "Redis отвечает (PONG)"; else warn "Redis не отвечает на PING" "возможно, ещё стартует"; fi
fi
http_ok "${APP_URL%/}/health" 5 && ok "Веб-интерфейс отвечает" "$APP_URL" || warn "Веб-интерфейс не отвечает" "${APP_URL}/health — сервис ещё поднимается (первый старт качает модели)"

# приложение видит Qdrant/Redis (через /api/system)
if http_ok "${APP_URL%/}/health" 4; then
  SYS="$(curl -sf -m 6 "${APP_URL%/}/api/system" 2>/dev/null)"
  echo "$SYS" | grep -Eq '"online"[[:space:]]*:[[:space:]]*true' && ok "Приложение видит Qdrant" || warn "Приложение пока не видит Qdrant" "проверьте QDRANT_URL"
  if echo "$SYS" | grep -Eq '"enabled"[[:space:]]*:[[:space:]]*true'; then
    echo "$SYS" | grep -Eq '"reachable"[[:space:]]*:[[:space:]]*true' && ok "Кэш Redis подключён" || warn "Кэш включён, но Redis недоступен приложению"
  fi
fi

# Ollama на хосте + модель
if http_ok "${OLLAMA_URL%/}/api/tags" 4; then
  MODEL=""; [[ -f .env.docker ]] && MODEL="$(grep -E '^LLM_MODEL=' .env.docker 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"')"
  if [[ -n "$MODEL" ]]; then
    base="${MODEL%%:*}"
    curl -sf -m 5 "${OLLAMA_URL%/}/api/tags" 2>/dev/null | grep -q "$base" && ok "Ollama на хосте, модель загружена" "$MODEL" || warn "Ollama доступна, но модель не найдена" "ollama pull $MODEL"
  else ok "Ollama на хосте доступна" "$OLLAMA_URL"; fi
else warn "Ollama на хосте недоступна" "$OLLAMA_URL — установите/запустите Ollama"; fi

# ============================ 3. Инструменты в образе =======================
if [[ "$APP_UP" == "1" ]]; then
  sec "3. Системные инструменты в образе ($APP)"
  inapp ffmpeg    && ok "ffmpeg (аудио/видео, TTS)" || warn "ffmpeg отсутствует в образе" "кадры/транскрибация/голос ограничены"
  inapp tesseract && ok "tesseract OCR" || warn "tesseract отсутствует" "OCR картинок отключён"
  inapp dwg2dxf   && ok "DWG-конвертер (dwg2dxf)" || warn "dwg2dxf отсутствует" "чертежи .dwg не индексируются (сборка libredwg — best-effort)"
  { inapp 7z || inapp 7za; } && ok "7-Zip (архивы)" || warn "7z отсутствует" "распаковка .7z ограничена"
  inapp unar      && ok "unar (RAR и др.)" || warn "unar отсутствует" "часть архивов не распакуется"

  # ============================ 4. Python-пакеты в образе ===================
  sec "4. Python-пакеты в образе ($APP)"
  REPORT="$(docker exec "$APP" "$VPY" - <<'PY' 2>/dev/null
import importlib, importlib.util
req = ["fastapi","uvicorn","qdrant_client","sentence_transformers","FlagEmbedding",
       "torch","transformers","rank_bm25","fitz","docx","pptx","openpyxl","faster_whisper","redis"]
opt = ["xlrd","ezdxf","rawpy","pytesseract","extract_msg","py7zr","rarfile",
       "paramiko","multipart","playwright","pyVoIP","TTS","networkx","lightrag"]
miss_req=[m for m in req if importlib.util.find_spec(m) is None]
miss_opt=[m for m in opt if importlib.util.find_spec(m) is None]
print("REQ_MISS:"+",".join(miss_req))
print("OPT_MISS:"+",".join(miss_opt))
PY
)"
  if [[ -z "$REPORT" ]]; then
    warn "не удалось проверить Python-пакеты в образе" "docker exec недоступен?"
  else
    rmiss="$(echo "$REPORT" | grep '^REQ_MISS:' | cut -d: -f2-)"
    omiss="$(echo "$REPORT" | grep '^OPT_MISS:' | cut -d: -f2-)"
    [[ -z "$rmiss" ]] && ok "Обязательные Python-пакеты на месте" || fail "не хватает обязательных Python-пакетов" "$rmiss"
    [[ -z "$omiss" ]] && ok "Опциональные Python-пакеты на месте" || warn "нет опциональных Python-пакетов" "$omiss"
  fi
fi

# ============================ Итог ==========================================
printf "\n${C_C}============================================================${C_0}\n"
if [[ $FAILS -eq 0 && $WARNS -eq 0 ]]; then
  printf "  ${C_G}ИТОГ: всё на месте ✓  (успешно: %s)${C_0}\n" "$OKS"
elif [[ $FAILS -eq 0 ]]; then
  printf "  ${C_Y}ИТОГ: работает, предупреждений: %s (опциональное/ещё стартует). Успешно: %s${C_0}\n" "$WARNS" "$OKS"
else
  printf "  ${C_R}ИТОГ: есть ошибки: %s. Предупреждений: %s. Успешно: %s${C_0}\n" "$FAILS" "$WARNS" "$OKS"
  printf "  ${C_D}Логи: docker compose logs -f app${C_0}\n"
fi
printf "${C_C}============================================================${C_0}\n"
[[ $FAILS -eq 0 ]] && exit 0 || exit 1
