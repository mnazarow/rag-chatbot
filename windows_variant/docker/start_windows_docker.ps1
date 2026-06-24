# Запуск RAG в Docker на Windows.
# Проще всего — двойной клик по start.cmd (он чинит кодировку и зовёт этот скрипт).
# Либо напрямую:
#   powershell -ExecutionPolicy Bypass -File start_windows_docker.ps1 `
#       -DocsDir "C:\path\to\BD" -AdminToken "ваш-пароль"
# Требуется: Docker Desktop и (для генерации) Ollama, установленные на Windows.
param(
    [string]$DocsDir = "",                              # папка с документами (Windows-путь)
    [string]$LlmModel = "qwen2.5:7b-instruct-q4_K_M",   # модель Ollama для генерации
    [string]$AdminToken = ""                            # пароль админ-панели (пусто = не менять)
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

function Log($m){ Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "[!] $m" -ForegroundColor Yellow }

# ----- 1. Проверки окружения -----
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker не найден. Установите Docker Desktop: winget install -e --id Docker.DockerDesktop"
}
try { docker info *> $null } catch { throw "Docker Desktop не запущен — запустите его и повторите." }

# ----- 2. Ollama на хосте (генерация) -----
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Log "Скачиваю модель Ollama: $LlmModel (при первом запуске долго)..."
    try { ollama pull $LlmModel } catch { Warn "Не удалось скачать модель — проверьте, запущен ли Ollama." }
} else {
    Warn "Ollama не найдена на хосте. Установите: winget install -e --id Ollama.Ollama"
    Warn "Без неё контейнер поднимется, но отвечать на вопросы не сможет."
}

# ----- 3. Конфиг .env.docker -----
if (-not (Test-Path ".env.docker")) {
    Copy-Item ".env.docker.example" ".env.docker"
    Log "Создан .env.docker (из примера). При желании отредактируйте."
}
# прописываем выбранную модель
(Get-Content ".env.docker") -replace '^LLM_MODEL=.*', "LLM_MODEL=$LlmModel" | Set-Content ".env.docker"
# пароль админ-панели (если задан параметром) — иначе оставляем как есть
if ($AdminToken -ne "") {
    (Get-Content ".env.docker") -replace '^ADMIN_TOKEN=.*', "ADMIN_TOKEN=$AdminToken" | Set-Content ".env.docker"
    Log "Пароль админ-панели задан."
}

# ----- 4. Папка с документами -----
if (-not $DocsDir) {
    $DocsDir = Read-Host "Укажите полный путь к папке с документами (например C:\rag\BD)"
}
if (-not (Test-Path $DocsDir)) { throw "Папка не найдена: $DocsDir" }
"DOCS_DIR_HOST=$DocsDir" | Set-Content ".env"   # compose читает .env для подстановки пути

# ----- 5. Файлы состояния (настройки + логи) -----
New-Item -ItemType Directory -Force -Path "state" | Out-Null
if (-not (Test-Path "state\runtime_config.json")) { "{}" | Set-Content "state\runtime_config.json" }
if (-not (Test-Path "state\ingest_stats.json"))   { "{}" | Set-Content "state\ingest_stats.json" }
if (-not (Test-Path "state\rag_logs.db"))         { New-Item -ItemType File -Force -Path "state\rag_logs.db" | Out-Null }
New-Item -ItemType Directory -Force -Path "backups" | Out-Null   # резервные копии (том)

# ----- 6. Сборка и запуск -----
Log "Собираю и запускаю контейнеры (первый раз — долго: качаются образы и модели)..."
docker compose -f docker-compose.windows.yml up -d --build

Log "Готово. Веб-интерфейс: http://localhost:8000"
Log "Логи приложения:   docker compose -f docker-compose.windows.yml logs -f app"
Log "Остановить:        docker compose -f docker-compose.windows.yml down"
Write-Host ""
Log "Дальше: откройте http://localhost:8000 -> вкладка «Админ» -> «Переиндексировать»,"
Log "чтобы проиндексировать документы из указанной папки."
