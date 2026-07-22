#!/usr/bin/env bash
# =============================================================================
#  ПОЛНАЯ переустановка GPU-сервера с нуля (DESTRUCTIVE).
#  Останавливает сервис и контейнеры, удаляет окружение И ДАННЫЕ
#  (индекс Qdrant, граф, адаптер, журнал, рантайм-настройки), затем заново
#  выполняет run_gpu.sh. Файл .env (с ADMIN_TOKEN и т.п.) СОХРАНЯЕТСЯ.
#
#  Запуск:  sudo bash reinstall_server.sh
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ $EUID -eq 0 ]] || { echo "Запустите через sudo."; exit 1; }

# CONFIRM=yes пропускает интерактивный вопрос (используется при запуске из админки)
if [[ "${CONFIRM:-}" != "yes" ]]; then
  read -r -p "Это удалит окружение и ВСЕ данные (индекс, граф, адаптер, журнал). Продолжить? [y/N] " ans
  [[ "${ans:-}" =~ ^[Yy]$ ]] || { echo "Отменено."; exit 0; }
  # второе подтверждение — деструктивная операция необратима
  read -r -p "Подтвердите ещё раз: введите DELETE заглавными для удаления данных: " ans2
  [[ "${ans2:-}" == "DELETE" ]] || { echo "Отменено."; exit 0; }
else
  echo "[reinstall-server] CONFIRM=yes — деструктивная переустановка без интерактива."
fi

# ----- бэкап ПЕРЕД удалением (авто-снимок; SKIP_BACKUP=1 — пропустить осознанно) -----
if [[ "${SKIP_BACKUP:-0}" != "1" ]]; then
  TS="$(date +%Y%m%d-%H%M%S)"
  BK_DIR="${ROOT}/backups"
  mkdir -p "$BK_DIR"
  echo "[reinstall-server] Резервная копия ПЕРЕД удалением → ${BK_DIR}"
  # 1) штатный механизм backup.py (согласованный снимок настроек/служебных данных/графа)
  PYBIN="${ROOT}/.venv/bin/python"; [ -x "$PYBIN" ] || PYBIN="$(command -v python3 || true)"
  BK_OK=0
  if [ -n "$PYBIN" ]; then
    "$PYBIN" -c "import sys,backup; r=backup.create('service'); print('[backup]', r.get('msg') or r); sys.exit(0 if r.get('ok') else 1)" \
      && BK_OK=1 || echo "[reinstall-server] backup.py не сработал — использую tar-снимок."
  fi
  # 2) tar-снимок ровно тех путей, что будут удалены (в т.ч. Qdrant-индекс, вне scope backup.py)
  FALLBACK_TAR="${BK_DIR}/pre-reinstall-${TS}.tar.gz"
  _bkpaths=()
  for d in gpu_variant/qdrant_storage graph_storage finetune/adapter finetune/data \
           runtime_config.json ingest_stats.json rag_logs.db; do
    [ -e "$d" ] && _bkpaths+=("$d")
  done
  if [ "${#_bkpaths[@]}" -gt 0 ]; then
    tar czf "$FALLBACK_TAR" "${_bkpaths[@]}" 2>/dev/null \
      && echo "[reinstall-server] tar-снимок: ${FALLBACK_TAR}" \
      || echo "[reinstall-server] ПРЕДУПРЕЖДЕНИЕ: не удалось создать tar-снимок."
  fi
  # если ни один способ не дал копии — прерываемся (данные важнее «чистой» переустановки)
  if [ "$BK_OK" != "1" ] && [ ! -s "$FALLBACK_TAR" ]; then
    echo "[reinstall-server] ОШИБКА: резервная копия не создана — прерываю. SKIP_BACKUP=1 чтобы пропустить осознанно."
    exit 1
  fi
else
  echo "[reinstall-server] SKIP_BACKUP=1 — резервная копия ПРОПУЩЕНА (данные будут потеряны безвозвратно)."
fi

echo "[reinstall-server] Останавливаю сервисы и контейнеры..."
systemctl stop rag-api 2>/dev/null || true
systemctl stop rag-xtts 2>/dev/null || true
docker compose -f gpu_variant/docker-compose.gpu.yml down 2>/dev/null || true

echo "[reinstall-server] Удаляю окружение и данные..."
# .venv-xtts — отдельное окружение микросервиса XTTS (пересоздаст run_gpu.sh при INSTALL_XTTS=1)
rm -rf .venv .venv-xtts graph_storage finetune/adapter finetune/data \
       runtime_config.json ingest_stats.json rag_logs.db rag_logs.db-journal \
       gpu_variant/qdrant_storage

echo "[reinstall-server] Запускаю установку заново..."
bash gpu_variant/run_gpu.sh
echo "[reinstall-server] Готово."
