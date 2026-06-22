#!/usr/bin/env bash
# Управление сервисами GPU-варианта.
# Использование: bash manage.sh {status|logs|restart|stop|start|vllm-logs}
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-status}" in
  status)
    systemctl --no-pager status rag-api || true
    echo "---- контейнеры ----"
    docker compose -f "${DIR}/docker-compose.gpu.yml" ps ;;
  logs)      journalctl -u rag-api -f -n 200 ;;
  restart)   sudo systemctl restart rag-api && echo "API перезапущен" ;;
  stop)
    sudo systemctl stop rag-api
    docker compose -f "${DIR}/docker-compose.gpu.yml" down
    echo "Остановлено" ;;
  start)
    docker compose --env-file "${DIR}/.env" -f "${DIR}/docker-compose.gpu.yml" up -d
    sudo systemctl start rag-api
    echo "Запущено" ;;
  vllm-logs) docker logs -f rag_vllm ;;
  *) echo "Команды: status|logs|restart|stop|start|vllm-logs"; exit 1 ;;
esac
