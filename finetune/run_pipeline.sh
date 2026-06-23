#!/usr/bin/env bash
# Полный пайплайн дообучения: датасет -> LoRA-обучение.
# Запускается из админки (кнопка «Запустить дообучение») или вручную.
#   bash finetune/run_pipeline.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

[[ -x ./.venv/bin/python ]] || { echo "Нет .venv — сначала выполните установку (run_gpu.sh)."; exit 1; }

echo "[finetune] Устанавливаю зависимости дообучения..."
./.venv/bin/pip install -q -r finetune/requirements-finetune.txt

echo "[finetune] Собираю датасет из проиндексированных документов..."
./.venv/bin/python finetune/build_dataset.py

echo "[finetune] Запускаю QLoRA-обучение (долго, грузит GPU)..."
./.venv/bin/python finetune/train_lora.py

echo "[finetune] Готово. Адаптер: finetune/adapter"
echo "Примените его кнопкой «Применить дообученную модель» в админ-панели."
