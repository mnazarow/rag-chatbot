#!/usr/bin/env bash
# =============================================================================
#  Перезапуск ВСЕГО сервиса одной командой (GPU-вариант).
#  Порядок: сначала зависимости (Qdrant, vLLM, Redis), затем приложение
#  (rag-api = веб + Telegram) и голосовой сервис rag-xtts.
#
#  Запуск:   sudo bash gpu_variant/restart_all.sh
#  Ключи:
#     --app     перезапустить только приложение (rag-api[/rag-xtts]) — быстро,
#               без Docker/Redis (например после правки .env)
#     --no-wait не ждать готовности (не проверять /health)
#
#  Замечания:
#   • vLLM после рестарта заново грузит модель — это долго (десятки секунд–минуты).
#   • ingest.py — это НЕ сервис (ручной процесс), скрипт его не трогает.
# =============================================================================
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"       # gpu_variant
COMPOSE="${DIR}/docker-compose.gpu.yml"
ENVF="${DIR}/.env"

APP_ONLY=0; WAIT=1
for a in "$@"; do
  case "$a" in
    --app) APP_ONLY=1 ;;
    --no-wait) WAIT=0 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
  esac
done

# порт приложения из .env (по умолчанию 8000); vLLM — 8001
API_PORT="$(grep -E '^API_PORT=' "${ENVF}" 2>/dev/null | head -1 | cut -d= -f2 | tr -d '"' )"
API_PORT="${API_PORT:-8000}"

G="\033[1;32m"; Y="\033[1;33m"; R="\033[1;31m"; C="\033[1;36m"; Z="\033[0m"
log()  { printf "${C}[restart]${Z} %s\n" "$*"; }
ok()   { printf "  ${G}[OK]${Z}  %s\n" "$*"; }
warn() { printf "  ${Y}[~]${Z}  %s\n" "$*"; }
bad()  { printf "  ${R}[X]${Z}  %s\n" "$*"; }

# sudo-обёртка (если запущено не под root)
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

if [ "${APP_ONLY}" -eq 0 ]; then
  # ---- 1. Векторная БД + LLM-движок (Docker) ----
  if command -v docker >/dev/null 2>&1 && [ -f "${COMPOSE}" ]; then
    # Qdrant: up -d — поднять/обновить (пересоздаст, если изменился compose, напр. ulimit)
    log "Поднимаю/обновляю Qdrant (Docker)..."
    ( cd "${DIR}" && docker compose --env-file .env -f docker-compose.gpu.yml up -d qdrant ) 2>/dev/null \
      || docker restart rag_qdrant 2>/dev/null || warn "Qdrant не обновился"
    # vLLM: --force-recreate — ПЕРЕСОЗДАТЬ, чтобы применились изменения .env (VLLM_GPU_UTIL,
    # модель, TP, длина контекста); обычный restart их не подхватывает.
    log "Пересоздаю vLLM с текущим .env (--force-recreate)..."
    ( cd "${DIR}" && docker compose --env-file .env -f docker-compose.gpu.yml up -d --force-recreate vllm ) 2>/dev/null \
      || docker restart rag_vllm 2>/dev/null || warn "vLLM не пересоздался"
  elif command -v docker >/dev/null 2>&1; then
    log "Перезапускаю контейнеры (compose-файл не найден)..."
    docker restart rag_qdrant rag_vllm 2>/dev/null || warn "контейнеры не найдены"
  else
    warn "docker не найден — пропускаю Qdrant/vLLM"
  fi

  # ---- 2. Redis (кэш) — перезапуск и systemd, и docker-контейнера (best-effort) ----
  log "Перезапускаю Redis..."
  _redis_done=0
  if command -v systemctl >/dev/null 2>&1; then
    if ${SUDO} systemctl restart redis-server 2>/dev/null; then _redis_done=1; ok "redis-server (systemd)"
    elif ${SUDO} systemctl restart redis 2>/dev/null; then _redis_done=1; ok "redis (systemd)"; fi
  fi
  if command -v docker >/dev/null 2>&1 && docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^rag_redis$'; then
    docker restart rag_redis >/dev/null 2>&1 && { _redis_done=1; ok "rag_redis (docker)"; }
  fi
  if [ "${_redis_done}" -eq 1 ]; then
    for i in $(seq 1 10); do redis-cli ping >/dev/null 2>&1 && { ok "redis-cli ping: PONG"; break; }; sleep 1; done
  else
    warn "Redis не найден (ни systemd redis-server/redis, ни docker rag_redis) — пропуск"
  fi
fi

# ---- 3. Приложение (веб + Telegram) ----
log "Перезапускаю rag-api (веб + Telegram)..."
${SUDO} systemctl restart rag-api && ok "rag-api перезапущен" || bad "rag-api не перезапустился (см. journalctl -u rag-api)"

# ---- 4. Сервис клонирования голоса (если установлен) ----
if systemctl list-unit-files 2>/dev/null | grep -q '^rag-xtts\.service'; then
  log "Перезапускаю rag-xtts (клонирование голоса)..."
  ${SUDO} systemctl restart rag-xtts && ok "rag-xtts перезапущен" || warn "rag-xtts не перезапустился"
fi

# ---- 5. Ожидание готовности ----
if [ "${WAIT}" -eq 1 ]; then
  if [ "${APP_ONLY}" -eq 0 ]; then
    log "Жду готовности vLLM (загрузка модели — до ~2 мин)..."
    _vok=0
    for i in $(seq 1 40); do
      curl -sf http://localhost:8001/health >/dev/null 2>&1 && { _vok=1; break; }; sleep 3
    done
    [ "${_vok}" -eq 1 ] && ok "vLLM отвечает (:8001)" || warn "vLLM пока не отвечает — возможно, ещё грузит модель (docker logs -f rag_vllm)"
  fi
  log "Жду готовности веб-интерфейса (прогрев эмбеддера/реранка — до ~2 мин)..."
  _aok=0
  for i in $(seq 1 40); do
    curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1 && { _aok=1; break; }; sleep 3
  done
  [ "${_aok}" -eq 1 ] && ok "веб-интерфейс отвечает (:${API_PORT})" || warn "веб-интерфейс пока не отвечает (journalctl -u rag-api -f)"
fi

# ---- 6. Итоговый статус ----
printf "\n${C}==== Статус ====${Z}\n"
for s in rag-api rag-xtts redis-server; do
  if systemctl list-unit-files 2>/dev/null | grep -q "^${s}\.service"; then
    st="$(systemctl is-active "${s}" 2>/dev/null)"
    [ "${st}" = "active" ] && ok "${s}: ${st}" || bad "${s}: ${st}"
  fi
done
if command -v docker >/dev/null 2>&1; then
  docker ps --format '{{.Names}}: {{.Status}}' 2>/dev/null | grep -E 'rag_(qdrant|vllm|redis)' \
    | while read -r l; do ok "$l"; done
fi
redis-cli ping >/dev/null 2>&1 && ok "redis-cli ping: PONG" || true
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
printf "\n${C}Веб-панель:${Z} http://%s:%s\n" "${IP:-<ip>}" "${API_PORT}"
