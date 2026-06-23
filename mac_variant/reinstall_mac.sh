#!/usr/bin/env bash
# =============================================================================
#  ПОЛНАЯ переустановка на Mac с нуля (DESTRUCTIVE).
#  Останавливает launchd-агент и Qdrant, удаляет окружение И ДАННЫЕ, затем
#  заново выполняет setup.sh. Файл .env сохраняется.
#
#  Запуск:  bash mac_variant/reinstall_mac.sh
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# CONFIRM=yes пропускает интерактивный вопрос (используется при запуске из админки)
if [[ "${CONFIRM:-}" != "yes" ]]; then
  read -r -p "Удалить окружение и ВСЕ данные (индекс, граф, адаптер, журнал)? [y/N] " ans
  [[ "${ans:-}" =~ ^[Yy]$ ]] || { echo "Отменено."; exit 0; }
fi

echo "[reinstall-mac] Останавливаю сервис и контейнеры..."
launchctl unload "$HOME/Library/LaunchAgents/com.rag.api.plist" 2>/dev/null || true
docker compose -f docker-compose.yml down 2>/dev/null || true

echo "[reinstall-mac] Удаляю окружение и данные..."
rm -rf .venv graph_storage finetune/adapter finetune/data \
       runtime_config.json rag_logs.db rag_logs.db-journal qdrant_storage

echo "[reinstall-mac] Запускаю установку заново..."
./setup.sh
launchctl load "$HOME/Library/LaunchAgents/com.rag.api.plist" 2>/dev/null || true
echo "[reinstall-mac] Готово."
