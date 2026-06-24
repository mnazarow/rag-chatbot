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
fi

echo "[reinstall-server] Останавливаю сервис и контейнеры..."
systemctl stop rag-api 2>/dev/null || true
docker compose -f gpu_variant/docker-compose.gpu.yml down 2>/dev/null || true

echo "[reinstall-server] Удаляю окружение и данные..."
rm -rf .venv graph_storage finetune/adapter finetune/data \
       runtime_config.json ingest_stats.json rag_logs.db rag_logs.db-journal \
       gpu_variant/qdrant_storage

echo "[reinstall-server] Запускаю установку заново..."
bash gpu_variant/run_gpu.sh
echo "[reinstall-server] Готово."
