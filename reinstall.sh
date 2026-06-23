#!/usr/bin/env bash
# =============================================================================
#  Переустановка Python-окружения и зависимостей (без потери данных и настроек).
#  Пересоздаёт .venv, переставляет зависимости и перезапускает сервис.
#  Данные (индекс, граф, адаптер, журнал, .env) сохраняются.
#
#  Запуск:  bash reinstall.sh           (на GPU-сервере лучше: sudo bash reinstall.sh)
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

log(){ printf "\033[1;33m[reinstall]\033[0m %s\n" "$*"; }

PYBIN="$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)"
log "Пересоздаю окружение (${PYBIN})..."
rm -rf .venv
"$PYBIN" -m venv .venv
./.venv/bin/pip install -U pip wheel

if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU: ставлю torch (${TORCH_CUDA:-cu124}) + gpu-зависимости..."
  ./.venv/bin/pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA:-cu124}" \
    || ./.venv/bin/pip install torch --index-url "https://download.pytorch.org/whl/cu126" || true
  ./.venv/bin/pip install -r gpu_variant/requirements-gpu.txt
else
  log "CPU/Apple: ставлю базовые зависимости..."
  ./.venv/bin/pip install -r requirements.txt
fi

# перезапуск сервиса (systemd или launchd)
if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files 2>/dev/null | grep -q '^rag-api'; then
  log "Перезапуск systemd-сервиса rag-api..."
  (sudo systemctl restart rag-api 2>/dev/null || systemctl restart rag-api 2>/dev/null) || \
    log "Не удалось перезапустить автоматически — выполните: sudo systemctl restart rag-api"
elif [ -f "$HOME/Library/LaunchAgents/com.rag.api.plist" ]; then
  log "Перезапуск launchd-агента..."
  launchctl kickstart -k "gui/$(id -u)/com.rag.api" 2>/dev/null || true
fi

log "Готово. Окружение переустановлено."
