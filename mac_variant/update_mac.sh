#!/usr/bin/env bash
# Обновление развёрнутого Mac-сервера из GitHub: git pull + зависимости + рестарт.
#   bash mac_variant/update_mac.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="${BRANCH:-main}"
cd "$ROOT"

[[ -x ./.venv/bin/pip ]] || { echo "Нет .venv — сначала выполните mac_variant/deploy_mac.sh"; exit 1; }

OLD="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
git fetch --all -q && git reset --hard "origin/${BRANCH}"
NEW="$(git rev-parse --short HEAD)"

./.venv/bin/pip install -q -r requirements.txt || true
./.venv/bin/pip install -q ezdxf xlrd python-multipart paramiko || true   # новые зависимости (DWG/XLS/загрузка/SSH)
bash mac_variant/manage_mac.sh restart

echo "Обновлено: ${OLD} → ${NEW}"
