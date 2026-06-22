#!/usr/bin/env bash
# Перезапуск контейнера vLLM с моделью из gpu_variant/.env (VLLM_MODEL и т.д.).
# Вызывается из админ-панели (кнопка «Применить модель LLM»).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
docker compose --env-file .env -f docker-compose.gpu.yml up -d vllm
echo "vLLM перезапускается с моделью: ${VLLM_MODEL:-$(grep ^VLLM_MODEL= .env | cut -d= -f2)}"
