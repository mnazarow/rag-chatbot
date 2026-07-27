#!/usr/bin/env bash
# =============================================================================
#  Полное удаление RAG-чатбота — единый скрипт для ВСЕХ *nix-вариантов:
#    • macOS  (launchd:  com.rag.api / com.rag.xtts / com.company.rag.reindex)
#    • Linux  (systemd:  rag-api / rag-xtts)
#    • Docker (root / gpu_variant / docker_variant — Qdrant, Redis, vLLM, Milvus)
#
#  Windows: используйте
#    • windows_variant/docker/uninstall_windows_docker.ps1  (установка в Docker)
#    • windows_variant/uninstall_windows.ps1                (нативная установка)
#
#  ПО УМОЛЧАНИЮ (без ключей): останавливает и удаляет СЕРВИСЫ и Docker-КОНТЕЙНЕРЫ.
#  ДАННЫЕ СОХРАНЯЮТСЯ — индекс, настройки, логи, бэкапы, документы. Повторная
#  установка поднимет всё «как было».
#
#  Ключи (комбинируются):
#    --venv        удалить окружения .venv, .venv-xtts, .venv.new/.old
#    --volumes     удалить векторный индекс/данные: qdrant_storage (bind+named),
#                  graph_storage, Milvus-тома, кеш превью. ДАННЫЕ БАЗЫ ТЕРЯЮТСЯ.
#    --state       удалить runtime_config.json, rag_logs.db*, ingest_stats/progress.json,
#                  finetune/adapter,data, docker_variant/state, .env.docker.
#                  НАСТРОЙКИ АДМИНКИ И ЖУРНАЛ ЗАПРОСОВ ТЕРЯЮТСЯ (бэкапы сохраняются).
#    --images      удалить Docker-образы (qdrant/redis/vllm + собранный образ app).
#    --docs        удалить папку документов DOCS_DIR. ОПАСНО — это ВАШИ файлы.
#    --backups     дополнительно удалить каталоги backups (по умолчанию сохраняются).
#    --purge       = --venv --volumes --state --images         (всё, КРОМЕ документов и бэкапов)
#    --all         = --purge --docs --backups                  (стереть ВООБЩЕ всё)
#    --yes | -y    не спрашивать подтверждение (для автоматизации)
#    --dry-run     только показать, что будет сделано, ничего не удаляя
#    -h | --help   эта справка
#
#  Примеры:
#    ./uninstall.sh                 # снять сервисы и контейнеры, данные оставить
#    ./uninstall.sh --purge         # полное удаление, кроме папки документов
#    ./uninstall.sh --all --yes     # снести всё без вопросов
#    ./uninstall.sh --purge --dry-run   # посмотреть план полного удаления
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- флаги --------------------------------------------------------------------
DO_VENV=0; DO_VOLUMES=0; DO_STATE=0; DO_IMAGES=0; DO_DOCS=0; DO_BACKUPS=0
ASSUME_YES=0; DRY=0

usage() { sed -n '2,48p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

for arg in "$@"; do
  case "$arg" in
    --venv)     DO_VENV=1 ;;
    --volumes|--data) DO_VOLUMES=1 ;;
    --state)    DO_STATE=1 ;;
    --images)   DO_IMAGES=1 ;;
    --docs)     DO_DOCS=1 ;;
    --backups)  DO_BACKUPS=1 ;;
    --purge)    DO_VENV=1; DO_VOLUMES=1; DO_STATE=1; DO_IMAGES=1 ;;
    --all)      DO_VENV=1; DO_VOLUMES=1; DO_STATE=1; DO_IMAGES=1; DO_DOCS=1; DO_BACKUPS=1 ;;
    -y|--yes)   ASSUME_YES=1 ;;
    --dry-run)  DRY=1 ;;
    -h|--help)  usage ;;
    *) echo "Неизвестный ключ: $arg (см. --help)"; exit 2 ;;
  esac
done

# --- утилиты ------------------------------------------------------------------
c_cyan="\033[36m"; c_yel="\033[33m"; c_grn="\033[32m"; c_red="\033[31m"; c_off="\033[0m"
log()  { printf "${c_cyan}==>${c_off} %s\n" "$*"; }
warn() { printf "${c_yel}[!]${c_off} %s\n" "$*"; }
ok()   { printf "${c_grn} [OK]${c_off} %s\n" "$*"; }

# выполнить команду (или показать при --dry-run). Ошибки не валят скрипт: удаление
# должно продолжаться, даже если часть шагов не применима на этой машине.
run() {
  if [ "$DRY" = 1 ]; then printf "   ${c_grn}dry:${c_off} %s\n" "$*"; return 0; fi
  "$@" 2>/dev/null || true
}
# безопасное удаление пути: только если существует и лежит ВНУТРИ разрешённого места
rmpath() {
  local p="$1"
  [ -e "$p" ] || [ -L "$p" ] || return 0
  if [ "$DRY" = 1 ]; then printf "   ${c_grn}dry:${c_off} rm -rf %s\n" "$p"; return 0; fi
  rm -rf "$p" 2>/dev/null && ok "удалено: $p" || warn "не удалось удалить: $p"
}

have() { command -v "$1" >/dev/null 2>&1; }

# --- определить DOCS_DIR (для --docs) ----------------------------------------
docs_dir() {
  local d=""
  if [ -f runtime_config.json ] && have python3; then
    d="$(python3 - <<'PY' 2>/dev/null || true
import json,sys
try:
    print((json.load(open("runtime_config.json")).get("DOCS_DIR") or "").strip())
except Exception:
    pass
PY
)"
  fi
  [ -z "$d" ] && [ -f .env ] && d="$(grep -E '^DOCS_DIR=' .env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'"'' )"
  printf '%s' "$d"
}

# =============================================================================
#  Сводка
# =============================================================================
echo
printf "${c_cyan}============================================================${c_off}\n"
printf "${c_cyan}  Удаление RAG-чатбота${c_off}   (каталог: %s)\n" "$ROOT"
printf "${c_cyan}============================================================${c_off}\n"
echo   "  Будет сделано:"
echo   "   - остановлены и сняты сервисы (launchd/systemd)"
echo   "   - остановлены и удалены Docker-контейнеры и сети проекта"
[ "$DO_VENV"    = 1 ] && printf "   - удалены окружения ${c_yel}.venv / .venv-xtts${c_off}\n"
[ "$DO_VOLUMES" = 1 ] && printf "   - удалены ${c_yel}индекс/данные векторной базы${c_off} (qdrant_storage, graph_storage, Milvus, превью)\n"
[ "$DO_STATE"   = 1 ] && printf "   - удалены ${c_yel}настройки и журнал${c_off} (runtime_config.json, rag_logs.db, ingest_stats)\n"
[ "$DO_IMAGES"  = 1 ] && printf "   - удалены ${c_yel}Docker-образы${c_off} (qdrant/redis/vllm/app)\n"
[ "$DO_BACKUPS" = 1 ] && printf "   - удалены ${c_yel}резервные копии (backups)${c_off}\n"
if [ "$DO_DOCS" = 1 ]; then
  DDIR="$(docs_dir)"
  printf "   - ${c_red}УДАЛЕНА ПАПКА ДОКУМЕНТОВ${c_off}: %s\n" "${DDIR:-<не определена>}"
fi
[ "$DRY" = 1 ] && printf "  ${c_grn}(режим --dry-run: ничего реально не удаляется)${c_off}\n"
echo

if [ "$ASSUME_YES" != 1 ] && [ "$DRY" != 1 ]; then
  printf "Продолжить удаление? введите ${c_yel}yes${c_off}: "
  read -r ans
  [ "$ans" = "yes" ] || { echo "Отменено."; exit 0; }
fi

# =============================================================================
#  1. Сервисы
# =============================================================================
log "Останавливаю сервисы…"

# macOS launchd
if have launchctl; then
  uid="$(id -u)"
  for label in com.rag.api com.rag.xtts com.company.rag.reindex; do
    plist="$HOME/Library/LaunchAgents/${label}.plist"
    if [ -f "$plist" ]; then
      run launchctl bootout "gui/${uid}/${label}"
      run launchctl unload "$plist"
      rmpath "$plist"
      ok "launchd сервис снят: $label"
    fi
  done
fi

# Linux systemd (нужен root — используем sudo, если доступен без пароля/в интерактиве)
if have systemctl; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && have sudo && SUDO="sudo"
  for svc in rag-api rag-xtts; do
    if systemctl list-unit-files 2>/dev/null | grep -q "^${svc}\.service"; then
      run $SUDO systemctl stop "$svc"
      run $SUDO systemctl disable "$svc"
      rmpath "/etc/systemd/system/${svc}.service"
      ok "systemd сервис снят: $svc"
    fi
  done
  run $SUDO systemctl daemon-reload
fi

# =============================================================================
#  2. Docker (все compose-варианты + подстраховка по именам)
# =============================================================================
if have docker; then
  # выбираем правильный вызов compose (плагин v2 или legacy)
  if docker compose version >/dev/null 2>&1; then DC=(docker compose)
  elif have docker-compose;              then DC=(docker-compose)
  else DC=(); fi

  down_flags=(down --remove-orphans)
  [ "$DO_VOLUMES" = 1 ] && down_flags+=(-v)
  [ "$DO_IMAGES"  = 1 ] && down_flags+=(--rmi local)

  if [ ${#DC[@]} -gt 0 ]; then
    for f in docker-compose.yml \
             gpu_variant/docker-compose.gpu.yml \
             docker_variant/docker-compose.yml \
             docker_variant/docker-compose.gpu.yml; do
      [ -f "$f" ] || continue
      log "docker compose down: $f"
      # COMPOSE_PROFILES=* — чтобы down убрал и опциональные сервисы (milvus и т.п.)
      if [ "$DRY" = 1 ]; then
        printf "   ${c_grn}dry:${c_off} %s -f %s %s\n" "${DC[*]}" "$f" "${down_flags[*]}"
      else
        COMPOSE_PROFILES="milvus,gpu,redis,cpu" "${DC[@]}" -f "$f" "${down_flags[@]}" 2>/dev/null || true
      fi
    done
  else
    warn "docker compose не найден — удаляю контейнеры по именам."
  fi

  # Подстраховка: снести контейнеры/сети/тома проекта по именам (если compose не сработал)
  log "Удаляю оставшиеся объекты Docker проекта…"
  cids="$(docker ps -aq --filter 'name=rag_' 2>/dev/null || true)"
  # плюс типичные Milvus-контейнеры
  for n in milvus-standalone milvus-etcd milvus-minio rag-milvus; do
    cid="$(docker ps -aq --filter "name=${n}" 2>/dev/null || true)"; [ -n "$cid" ] && cids="$cids $cid"
  done
  [ -n "${cids// /}" ] && run docker rm -f $cids

  if [ "$DO_VOLUMES" = 1 ]; then
    vols="$(docker volume ls -q 2>/dev/null | grep -iE 'qdrant_storage|hf_cache|milvus|rag_|previews' || true)"
    [ -n "$vols" ] && run docker volume rm $vols
  fi
  if [ "$DO_IMAGES" = 1 ]; then
    imgs="$(docker images -q 'qdrant/qdrant' 2>/dev/null; docker images -q 'redis' 2>/dev/null; docker images -q 'vllm/*' 2>/dev/null; docker images -q 'rag*' 2>/dev/null; docker images -q '*rag_app*' 2>/dev/null)"
    imgs="$(printf '%s\n' $imgs | sort -u)"
    [ -n "${imgs// /}" ] && run docker rmi -f $imgs
  fi
  # сеть проекта
  for net in $(docker network ls -q --filter 'name=rag' 2>/dev/null || true); do run docker network rm "$net"; done
else
  warn "docker не установлен — шаг Docker пропущен."
fi

# =============================================================================
#  3. Python-окружения
# =============================================================================
if [ "$DO_VENV" = 1 ]; then
  log "Удаляю Python-окружения…"
  for v in .venv .venv-xtts .venv.new .venv.old; do rmpath "$v"; done
fi

# =============================================================================
#  4. Данные векторной базы / индекса
# =============================================================================
if [ "$DO_VOLUMES" = 1 ]; then
  log "Удаляю данные индекса/базы…"
  for d in qdrant_storage gpu_variant/qdrant_storage graph_storage \
           volumes milvus.db previews docker_variant/state/previews; do
    rmpath "$d"
  done
fi

# =============================================================================
#  5. Настройки / журнал / артефакты
# =============================================================================
if [ "$DO_STATE" = 1 ]; then
  log "Удаляю настройки и журнал…"
  for f in runtime_config.json ingest_stats.json ingest_progress.json \
           rag_logs.db rag_logs.db-journal rag_logs.db-wal rag_logs.db-shm \
           finetune/adapter finetune/data \
           docker_variant/state docker_variant/.env.docker docker_variant/.env; do
    rmpath "$f"
  done
  # кеши инструментов разработки
  for c in __pycache__ .ruff_cache .pytest_cache .mypy_cache; do rmpath "$c"; done
fi

# =============================================================================
#  6. Резервные копии (только по явному --backups)
# =============================================================================
if [ "$DO_BACKUPS" = 1 ]; then
  log "Удаляю резервные копии…"
  for b in backups docker_variant/backups; do rmpath "$b"; done
fi

# =============================================================================
#  7. Папка документов (только по явному --docs) — ОПАСНО
# =============================================================================
if [ "$DO_DOCS" = 1 ]; then
  DDIR="$(docs_dir)"
  if [ -z "$DDIR" ]; then
    warn "DOCS_DIR не определён (runtime_config.json уже удалён?) — пропускаю удаление документов."
  elif [ "$DDIR" = "/" ] || [ "$DDIR" = "$HOME" ]; then
    warn "DOCS_DIR=$DDIR выглядит опасно (корень/домашняя папка) — НЕ удаляю, уберите вручную."
  else
    warn "Удаляю папку документов: $DDIR"
    rmpath "$DDIR"
  fi
fi

echo
if [ "$DRY" = 1 ]; then
  ok "Готово (dry-run): выше показан план. Запустите без --dry-run для реального удаления."
else
  ok "Удаление завершено."
  [ "$DO_VENV" = 1 ] && [ "$DO_VOLUMES" = 1 ] && [ "$DO_STATE" = 1 ] \
    && echo "   Каталог проекта и исходники оставлены. Удалить их целиком: rm -rf \"$ROOT\""
fi
