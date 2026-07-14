#!/usr/bin/env bash
# =============================================================================
#  Полная проверка установки RAG-чатбота (Linux / macOS).
#  Проверяет системные пакеты, Python-зависимости (обязательные и опциональные)
#  и сервисы (Qdrant / vLLM / Ollama / веб-интерфейс). Печатает цветной чек-лист
#  с итогом [OK] / [~] предупреждение / [X] ошибка.
#
#  Вызывается в конце всех инсталляторов, но можно запускать и отдельно:
#     bash scripts/checklist.sh                 # автоопределение корня и .venv
#     ROOT=/opt/rag bash scripts/checklist.sh   # явно указать корень проекта
#
#  Скрипт НЕ прерывает работу инсталлятора: коды возврата информативны
#  (0 — ошибок нет, 1 — есть проваленные обязательные пункты).
# =============================================================================

# намеренно без `set -e`: один проваленный пункт не должен обрывать весь чек-лист
ROOT="${ROOT:-${1:-}}"
if [[ -z "$ROOT" ]]; then
  # корень проекта = родитель папки scripts/
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
fi
ENVF="${ROOT}/.env"

# ----- вывод -----------------------------------------------------------------
if [[ -t 1 ]]; then C_G=$'\033[1;32m'; C_Y=$'\033[1;33m'; C_R=$'\033[1;31m'; C_C=$'\033[1;36m'; C_D=$'\033[0;90m'; C_0=$'\033[0m'
else C_G=""; C_Y=""; C_R=""; C_C=""; C_D=""; C_0=""; fi
FAILS=0; WARNS=0; OKS=0
ok()   { printf "  ${C_G}[OK]${C_0} %s${C_D}%s${C_0}\n" "$1" "${2:+  — $2}"; OKS=$((OKS+1)); }
warn() { printf "  ${C_Y}[~]${C_0}  %s${C_D}%s${C_0}\n" "$1" "${2:+  — $2}"; WARNS=$((WARNS+1)); }
fail() { printf "  ${C_R}[X]${C_0}  %s${C_D}%s${C_0}\n" "$1" "${2:+  — $2}"; FAILS=$((FAILS+1)); }
sec() { printf "\n${C_C}%s${C_0}\n" "$1"; }   # заголовок раздела (не 'head' — иначе перекрывает команду head)

has() { command -v "$1" >/dev/null 2>&1; }
envval() { [[ -f "$ENVF" ]] && grep -E "^$1=" "$ENVF" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'"; }
http_ok() { curl -sf -m "${2:-5}" "$1" >/dev/null 2>&1; }

# ----- определяем venv и платформу ------------------------------------------
VPY=""
for c in "${ROOT}/.venv/bin/python" "${ROOT}/.venv/bin/python3"; do
  [[ -x "$c" ]] && { VPY="$c"; break; }
done
[[ -z "$VPY" ]] && VPY="$(command -v python3 || true)"
UNAME="$(uname -s 2>/dev/null || echo unknown)"

printf "${C_C}============================================================${C_0}\n"
printf "${C_C}  Чек-лист установки RAG — проверка пакетов и компонентов${C_0}\n"
printf "${C_C}  Проект: %s${C_0}\n" "$ROOT"
printf "${C_C}============================================================${C_0}\n"

# ============================ 1. Системные пакеты ============================
sec "1. Системные пакеты"
if has python3; then ok "Python 3" "$(python3 --version 2>&1)"; else fail "Python 3 не найден"; fi
if has ffmpeg; then ok "ffmpeg (аудио/видео, TTS, VoIP)"; else fail "ffmpeg не найден" "нужен для транскрибации, кадров видео и голосового вывода"; fi

# OCR
if has tesseract; then
  if tesseract --list-langs 2>/dev/null | grep -qiE '^rus$'; then ok "tesseract OCR (+ русский)"; \
  else warn "tesseract есть, но без русского языка" "поставьте tesseract-ocr-rus / tesseract-lang"; fi
else warn "tesseract не найден" "OCR картинок и фото-документов отключён"; fi

# .doc (старый Word)
if has antiword || has libreoffice || has soffice; then ok "чтение старого .doc" "$(has antiword && echo antiword || echo libreoffice)"; \
else warn ".doc-конвертер не найден" "старые .doc (Word 97-2003) не читаются; поставьте antiword/libreoffice"; fi

# архивы
if has 7z || has 7za || has 7zr; then ok "7-Zip (.7z и др.)"; else warn "7z не найден" "распаковка .7z отключена (p7zip-full / p7zip)"; fi
if has unar || has unrar || has bsdtar; then ok "распаковка RAR"; else warn "unar/unrar не найдены" "архивы .rar не распакуются"; fi

# чертежи DWG/DXF
if has dwg2dxf; then ok "DWG-конвертер (dwg2dxf)"; \
elif [[ -n "$(command -v ODAFileConverter 2>/dev/null)" ]] || [[ -d "/Applications/ODAFileConverter.app" ]]; then ok "ODA File Converter (DWG→DXF)"; \
else warn "DWG-конвертер не найден" "чертежи .dwg не индексируются (libredwg-tools или ODA File Converter)"; fi

# PDF-растеризация (poppler)
if has pdftoppm; then ok "poppler (pdftoppm, растеризация PDF)"; else warn "poppler не найден" "рендер страниц PDF в картинки ограничен"; fi
# libmagic (определение типов)
if has file; then ok "libmagic (file)"; else warn "утилита file не найдена"; fi
# git
if has git; then ok "git"; else warn "git не найден" "обновление через git pull недоступно"; fi

# GPU (информативно)
if has nvidia-smi; then ok "NVIDIA GPU" "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"; \
elif [[ "$UNAME" == "Darwin" ]]; then ok "Apple Silicon (Metal/MPS)"; \
else warn "GPU не обнаружен" "эмбеддинги/реранк пойдут на CPU (медленнее)"; fi

# Docker / Ollama наличие
if has docker; then
  if docker info >/dev/null 2>&1; then ok "Docker (демон запущен)"; else warn "Docker установлен, но демон не отвечает" "запустите Docker/Docker Desktop"; fi
else warn "Docker не найден" "Qdrant обычно запускается в Docker"; fi
if has ollama; then ok "Ollama (клиент)"; else warn "Ollama не найдена" "нужна для бэкенда генерации ollama"; fi
has redis-server && ok "redis-server" || true   # опционально

# ============================ 2. Python-зависимости =========================
sec "2. Python-пакеты (окружение: ${VPY:-нет})"
if [[ -z "$VPY" || ! -x "$VPY" ]]; then
  fail "Python-окружение (.venv) не найдено" "создайте venv и поставьте requirements"
else
  # единый прогон: печатаем "имя<TAB>ok|missing<TAB>версия" по каждому модулю
  IMPORT_REPORT="$("$VPY" - <<'PY' 2>/dev/null
import importlib, importlib.metadata as md
req = ["fastapi","uvicorn","qdrant_client","sentence_transformers","FlagEmbedding",
       "torch","transformers","rank_bm25","fitz","docx","pptx","openpyxl",
       "bs4","lxml","charset_normalizer","PIL","psutil","requests","numpy"]
opt = ["xlrd","ezdxf","rawpy","pytesseract","extract_msg","py7zr","rarfile",
       "paramiko","multipart","playwright","pyVoIP","redis","TTS","networkx","lightrag"]
stt = ["faster_whisper","mlx_whisper"]
def ver(m):
    for name in (m, m.replace('_','-')):
        try: return md.version(name)
        except Exception: pass
    return ""
def check(mods, kind):
    for m in mods:
        try:
            importlib.import_module(m); print(f"{kind}\t{m}\tok\t{ver(m)}")
        except Exception:
            print(f"{kind}\t{m}\tmissing\t")
check(req,"req"); check(opt,"opt")
# STT: достаточно одного
present=[m for m in stt if importlib.util.find_spec(m)]
print("stt\t"+("/".join(present) if present else "none")+"\t"+("ok" if present else "missing")+"\t")
PY
)"
  if [[ -z "$IMPORT_REPORT" ]]; then
    fail "не удалось запустить проверку импортов" "venv повреждён?"
  else
    # красивые подписи для ключевых модулей
    while IFS=$'\t' read -r kind mod status ver; do
      [[ -z "$mod" ]] && continue
      label="$mod"
      case "$mod" in
        qdrant_client) label="qdrant-client (векторная БД)";;
        sentence_transformers) label="sentence-transformers (эмбеддинги)";;
        FlagEmbedding) label="FlagEmbedding (реранкер)";;
        rank_bm25) label="rank-bm25 (лексический поиск)";;
        fitz) label="PyMuPDF (fitz, чтение PDF)";;
        docx) label="python-docx";;
        pptx) label="python-pptx";;
        bs4) label="beautifulsoup4 (парсинг HTML)";;
        PIL) label="Pillow (изображения)";;
        multipart) label="python-multipart (загрузка файлов)";;
        pyVoIP) label="pyVoIP (SIP-регистрация, опц.)";;
        TTS) label="coqui-tts / XTTS (клонирование голоса, опц.)";;
        redis) label="redis (общий кэш, опц.)";;
        lightrag) label="LightRAG (граф-RAG, опц.)";;
        networkx) label="networkx (граф, опц.)";;
      esac
      if [[ "$kind" == "req" ]]; then
        [[ "$status" == "ok" ]] && ok "$label" "$ver" || fail "$label — не установлен"
      elif [[ "$kind" == "stt" ]]; then
        [[ "$status" == "ok" ]] && ok "Whisper STT (транскрибация)" "$mod" || warn "Whisper (STT) не установлен" "faster-whisper или mlx-whisper — распознавание голоса отключено"
      else
        [[ "$status" == "ok" ]] && ok "$label" "$ver" || warn "$label — не установлен (опционально)"
      fi
    done <<< "$IMPORT_REPORT"
  fi
fi

# ============================ 3. Сервисы ====================================
sec "3. Сервисы и подключения"
QURL="$(envval QDRANT_URL)"; QURL="${QURL:-http://localhost:6333}"
if http_ok "${QURL%/}/collections" 5; then ok "Qdrant отвечает" "$QURL"; \
else warn "Qdrant не отвечает" "$QURL — возможно, ещё стартует или не запущен"; fi

# vLLM (если применимо): порт 8001
if http_ok "http://localhost:8001/health" 4; then ok "vLLM отвечает" "http://localhost:8001"; fi

# Ollama (если бэкенд ollama или клиент установлен)
BACKEND="$(envval LLM_BACKEND)"
if [[ "$BACKEND" == "ollama" || -z "$BACKEND" ]] && (has ollama || http_ok "http://localhost:11434/api/tags" 3); then
  if http_ok "http://localhost:11434/api/tags" 4; then
    MODEL="$(envval LLM_MODEL)"
    if [[ -n "$MODEL" ]]; then
      base="${MODEL%%:*}"
      if curl -sf -m 5 "http://localhost:11434/api/tags" 2>/dev/null | grep -q "$base"; then ok "Ollama, модель загружена" "$MODEL"; \
      else warn "Ollama работает, но модель не найдена" "выполните: ollama pull $MODEL"; fi
    else ok "Ollama отвечает" "http://localhost:11434"; fi
  else warn "Ollama недоступна" "http://localhost:11434 — запустите ollama serve"; fi
fi

# Веб-интерфейс приложения
PORT="$(envval API_PORT)"; PORT="${PORT:-8000}"
if http_ok "http://localhost:${PORT}/health" 4; then ok "Веб-интерфейс отвечает" "http://localhost:${PORT}"; \
else warn "Веб-интерфейс не отвечает" "http://localhost:${PORT}/health — сервис ещё стартует или не запущен"; fi

# ============================ Итог ==========================================
printf "\n${C_C}============================================================${C_0}\n"
if [[ $FAILS -eq 0 && $WARNS -eq 0 ]]; then
  printf "  ${C_G}ИТОГ: всё на месте ✓  (успешно: %s)${C_0}\n" "$OKS"
elif [[ $FAILS -eq 0 ]]; then
  printf "  ${C_Y}ИТОГ: работает, есть предупреждения: %s (опциональные компоненты). Успешно: %s${C_0}\n" "$WARNS" "$OKS"
else
  printf "  ${C_R}ИТОГ: есть ошибки: %s. Предупреждения: %s. Успешно: %s${C_0}\n" "$FAILS" "$WARNS" "$OKS"
  printf "  ${C_D}Проваленные пункты [X] выше — обязательные компоненты; устраните их.${C_0}\n"
fi
printf "${C_C}============================================================${C_0}\n"
[[ $FAILS -eq 0 ]] && exit 0 || exit 1
