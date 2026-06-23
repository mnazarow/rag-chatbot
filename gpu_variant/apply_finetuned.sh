#!/usr/bin/env bash
# Перезапуск vLLM с подключённым дообученным LoRA-адаптером.
# Вызывается из админ-панели (кнопка «Применить дообученную модель»).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -d ../finetune/adapter ]]; then
  echo "Адаптер finetune/adapter не найден — сначала запустите дообучение."; exit 1
fi
docker compose --env-file .env -f docker-compose.gpu.yml -f docker-compose.lora.yml up -d vllm
echo "vLLM перезапущен с LoRA-адаптером: ${FINETUNED_MODEL:-company-lora}"
echo "Включите «Использовать дообученную модель» в админке, чтобы запросы шли на адаптер."
