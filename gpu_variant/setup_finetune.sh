#!/usr/bin/env bash
# Установка зависимостей дообучения и запуск полного пайплайна (датасет + LoRA).
# Тонкая обёртка над finetune/run_pipeline.sh для единообразия с setup_hybrid.sh.
#   sudo bash gpu_variant/setup_finetune.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(whoami)}"
[[ -x "${ROOT}/.venv/bin/python" ]] || { echo "Нет .venv — сначала run_gpu.sh."; exit 1; }
sudo -u "${RUN_USER}" bash "${ROOT}/finetune/run_pipeline.sh"
echo "Готово. В админке: «Применить дообученную модель», затем выберите «Дообученная»."
