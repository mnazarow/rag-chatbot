#!/usr/bin/env bash
# =============================================================================
#  Установка ODA File Converter (запасной конвертер DWG→DXF) из локального
#  дистрибутива vendor/oda/*.deb + xvfb и Qt/X-библиотеки, нужные для headless-
#  запуска (ODA — Qt-приложение, на сервере без дисплея работает через xvfb-run).
#  Идемпотентно. Только для apt-систем (Debian/Ubuntu, в т.ч. контейнер приложения).
#
#  Использование:  bash scripts/install_oda.sh [ROOT_проекта]
#  Скрипт НЕ прерывает работу инсталлятора при неудаче (DWG тогда — через libredwg).
# =============================================================================
set -uo pipefail
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)}"
log(){ printf '\033[1;36m[oda]\033[0m %s\n' "$*"; }

command -v apt-get >/dev/null 2>&1 || { log "не apt-система — установка ODA пропущена."; exit 0; }
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"

# Библиотеки среды выполнения для Qt/GL и виртуальный X-сервер (xvfb)
log "Ставлю xvfb и Qt/X-библиотеки для headless-запуска..."
$SUDO apt-get update -y 2>/dev/null || true
$SUDO apt-get install -y --no-install-recommends \
    xvfb libgl1 libglib2.0-0 libfontconfig1 libxkbcommon0 libdbus-1-3 \
    libx11-6 libxext6 libxrender1 libxcb1 libegl1 2>/dev/null || true

if command -v ODAFileConverter >/dev/null 2>&1; then
    log "ODAFileConverter уже установлен: $(command -v ODAFileConverter)"
    exit 0
fi

DEB="$(ls "${ROOT}"/vendor/oda/*.deb 2>/dev/null | head -1)"
if [ -z "$DEB" ]; then
    log "Дистрибутив vendor/oda/*.deb не найден — пропуск (DWG будет конвертироваться через libredwg, если установлен)."
    exit 0
fi

log "Устанавливаю ODA File Converter из $(basename "$DEB")..."
# у пакета нет объявленных зависимостей, поэтому dpkg -i достаточно; -f install на всякий случай
$SUDO dpkg -i "$DEB" 2>/dev/null || { $SUDO apt-get -f install -y 2>/dev/null || true; $SUDO dpkg -i "$DEB" 2>/dev/null || true; }

if command -v ODAFileConverter >/dev/null 2>&1; then
    log "Готово: $(command -v ODAFileConverter)"
else
    log "Не удалось установить ODA File Converter — DWG будет конвертироваться через libredwg (если установлен)."
fi
exit 0
