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

PYBIN="$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3 || true)"
if [ -z "${PYBIN}" ]; then
  echo "[reinstall] ОШИБКА: не найден интерпретатор python3 (нужен python3.10+). Установите Python и повторите."
  exit 1
fi
# Собираем НОВОЕ окружение рядом (.venv.new) и заменяем старое только после успеха —
# чтобы неудачная сборка не оставила систему без рабочего .venv.
log "Собираю новое окружение (${PYBIN}) в .venv.new..."
rm -rf .venv.new
"$PYBIN" -m venv .venv.new
./.venv.new/bin/pip install -U pip wheel

if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU: ставлю torch (${TORCH_CUDA:-cu124}) + gpu-зависимости..."
  ./.venv.new/bin/pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA:-cu124}" \
    || ./.venv.new/bin/pip install torch --index-url "https://download.pytorch.org/whl/cu126" || true
  ./.venv.new/bin/pip install -r gpu_variant/requirements-gpu.txt
else
  log "CPU/Apple: ставлю базовые зависимости..."
  ./.venv.new/bin/pip install -r requirements.txt
fi

# Сборка удалась — атомарно заменяем старое окружение новым.
log "Активирую новое окружение (замена .venv)..."
rm -rf .venv.old
[ -d .venv ] && mv .venv .venv.old
mv .venv.new .venv
rm -rf .venv.old

# XTTS (клонирование голоса) живёт в отдельном окружении .venv-xtts и не зависит от ядра —
# пересоздание .venv его не трогает. Если окружение уже есть, просто перезапустим сервис ниже.
# (Полная переустановка XTTS выполняется установщиком setup_gpu.sh/run_gpu.sh/setup.sh.)

# перезапуск сервисов (systemd или launchd)
if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files 2>/dev/null | grep -q '^rag-api'; then
  log "Перезапуск systemd-сервиса rag-api..."
  (sudo systemctl restart rag-api 2>/dev/null || systemctl restart rag-api 2>/dev/null) || \
    log "Не удалось перезапустить автоматически — выполните: sudo systemctl restart rag-api"
  if systemctl list-unit-files 2>/dev/null | grep -q '^rag-xtts'; then
    log "Перезапуск systemd-сервиса rag-xtts (клонирование голоса)..."
    (sudo systemctl restart rag-xtts 2>/dev/null || systemctl restart rag-xtts 2>/dev/null) || true
  fi
elif [ -f "$HOME/Library/LaunchAgents/com.rag.api.plist" ]; then
  log "Перезапуск launchd-агента..."
  launchctl kickstart -k "gui/$(id -u)/com.rag.api" 2>/dev/null || true
  [ -f "$HOME/Library/LaunchAgents/com.rag.xtts.plist" ] && \
    launchctl kickstart -k "gui/$(id -u)/com.rag.xtts" 2>/dev/null || true
fi

log "Готово. Окружение переустановлено."

# полная проверка пакетов и компонентов
bash "${ROOT}/scripts/checklist.sh" "${ROOT}" || true
