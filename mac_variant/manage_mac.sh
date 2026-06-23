#!/usr/bin/env bash
# Управление сервисом API на Mac (launchd).
# Использование: bash manage_mac.sh {status|logs|restart|stop|start}
set -euo pipefail
PLIST="$HOME/Library/LaunchAgents/com.rag.api.plist"
UID_="$(id -u)"

case "${1:-status}" in
  status)
    launchctl list | grep com.rag.api || echo "сервис не загружен"
    echo "---- контейнеры ----"
    docker compose -f "$(dirname "$0")/../docker-compose.yml" ps 2>/dev/null || true ;;
  logs)    tail -f /tmp/rag_api.log /tmp/rag_api.err ;;
  restart) launchctl kickstart -k "gui/${UID_}/com.rag.api" 2>/dev/null \
             || { launchctl unload "$PLIST"; launchctl load "$PLIST"; }
           echo "перезапущено" ;;
  stop)    launchctl unload "$PLIST"; echo "остановлено" ;;
  start)   launchctl load "$PLIST"; echo "запущено" ;;
  *) echo "Команды: status|logs|restart|stop|start"; exit 1 ;;
esac
