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
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path","User")
}
# Пункт чеклиста: status = ok | fail | warn
function Item($status, $label, $detail) {
    switch ($status) {
        "ok"   { $mark = "[OK]  "; $col = "Green"  }
        "fail" { $mark = "[X]   "; $col = "Red"    }
        default{ $mark = "[~]   "; $col = "Yellow" }
    }
    $line = " $mark $label"
    if ($detail) { $line += "  - $detail" }
    Write-Host $line -ForegroundColor $col
}
# HTTP-проверка: вернёт $true при коде 200
function HttpOk($url, $timeoutSec = 5) {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec $timeoutSec -ErrorAction Stop
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}
# Контейнер запущен?
function ContainerUp($name) {
    try { return ((docker inspect -f '{{.State.Running}}' $name 2>$null) -eq 'true') } catch { return $false }
}
# Наличие команды внутри контейнера приложения
function InAppHas($cmd) {
    try { docker exec rag_app sh -lc "command -v $cmd" *> $null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}

# ----- 1. Docker: установка (winget) + запуск + ожидание движка -----
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Warn "Docker Desktop не установлен."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Log "Устанавливаю Docker Desktop (winget; может потребоваться подтверждение прав)..."
        winget install -e --id Docker.DockerDesktop --silent --accept-source-agreements --accept-package-agreements
        Refresh-Path
    } else {
        Warn "winget не найден. Установите Docker Desktop вручную:"
        Warn "  https://www.docker.com/products/docker-desktop/  затем повторите start.cmd."
        exit 1
    }
}
docker info *> $null
if (-not $?) {
    $dd = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Log "Запускаю Docker Desktop..."; Start-Process $dd }
    Log "Жду запуска движка Docker (до ~5 минут)..."
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) { docker info *> $null; if ($?) { $ready = $true; break }; Start-Sleep -Seconds 5 }
    if (-not $ready) {
        Warn "Docker ещё не запустился. Если это первая установка — нужна ПЕРЕЗАГРУЗКА (WSL2/Hyper-V)."
        Warn "Перезагрузите Windows и снова запустите start.cmd — всё уже установлено, он просто продолжит."
        exit 1
    }
}
Log "Docker готов."

# ----- 2. Ollama: установка (winget) + модель -----
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Log "Устанавливаю Ollama (winget)..."
        winget install -e --id Ollama.Ollama --silent --accept-source-agreements --accept-package-agreements
        Refresh-Path
        Start-Sleep -Seconds 5
    }
}
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Log "Скачиваю модель Ollama: $LlmModel (при первом запуске долго)..."
    $pulled = $false
    for ($i = 0; $i -lt 6; $i++) {
        try { ollama pull $LlmModel; $pulled = $true; break } catch { Start-Sleep -Seconds 5 }
    }
    if (-not $pulled) { Warn "Модель не скачалась. Запустите Ollama (значок в трее) и выполните: ollama pull $LlmModel" }
} else {
    Warn "Ollama не установлена — контейнер поднимется, но отвечать на вопросы не сможет."
    Warn "Установите вручную: winget install -e --id Ollama.Ollama"
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
$composeOk = ($LASTEXITCODE -eq 0)

# ----- 7. Чеклист после сборки -----
Write-Host ""
Log "Жду готовности приложения (загрузка моделей эмбеддингов может занять до ~2 минут)..."
$appOk = $false
for ($i = 0; $i -lt 40; $i++) {
    if (HttpOk "http://localhost:8000/health" 4) { $appOk = $true; break }
    Start-Sleep -Seconds 3
}

$ollamaModel = (Select-String -Path ".env.docker" -Pattern '^LLM_MODEL=(.*)$' | ForEach-Object { $_.Matches[0].Groups[1].Value }) 2>$null
$ollamaUp = HttpOk "http://localhost:11434/api/tags" 4
$ollamaHasModel = $false
if ($ollamaUp -and $ollamaModel) {
    try { $tags = (Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:11434/api/tags" -TimeoutSec 5).Content
          $ollamaHasModel = ($tags -match [Regex]::Escape($ollamaModel.Split(':')[0])) } catch {}
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Чеклист сборки и запуска" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if ($composeOk) { Item ok "Сборка образа и запуск docker compose" } else { Item fail "Сборка/запуск docker compose" "см. вывод выше" }
if (ContainerUp "rag_qdrant") { Item ok "Контейнер Qdrant (rag_qdrant) работает" } else { Item fail "Контейнер Qdrant не запущен" }
if (ContainerUp "rag_app")    { Item ok "Контейнер приложения (rag_app) работает" } else { Item fail "Контейнер приложения не запущен" }
if (HttpOk "http://localhost:6333/collections" 4) { Item ok "Qdrant отвечает (порт 6333)" } else { Item warn "Qdrant пока не отвечает" "возможно, ещё стартует" }
if ($appOk) { Item ok "Веб-интерфейс отвечает" "http://localhost:8000" } else { Item warn "Веб-интерфейс ещё не готов" "подождите и обновите страницу; логи: docker compose -f docker-compose.windows.yml logs -f app" }

if ($ollamaUp) {
    if ($ollamaHasModel) { Item ok "Ollama на хосте, модель загружена" $ollamaModel }
    else { Item warn "Ollama доступна, но модель не найдена" "выполните: ollama pull $ollamaModel" }
} else { Item fail "Ollama на хосте недоступна" "ответы работать не будут; winget install -e --id Ollama.Ollama" }

# Опциональные инструменты внутри образа (информативно)
if (ContainerUp "rag_app") {
    if (InAppHas "dwg2dxf")   { Item ok   "DWG-конвертер (dwg2dxf) в образе" } else { Item warn "DWG-конвертер недоступен" "DWG-чертежи не индексируются" }
    if (InAppHas "tesseract") { Item ok   "OCR (tesseract) в образе" }        else { Item warn "tesseract недоступен" "OCR картинок отключён" }
    if (InAppHas "ffmpeg")    { Item ok   "ffmpeg (видео/аудио) в образе" }    else { Item warn "ffmpeg недоступен" "кадры/транскрибация отключены" }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Log "Веб-интерфейс: http://localhost:8000   (раздел «Администратор»)"
Log "Логи:          docker compose -f docker-compose.windows.yml logs -f app"
Log "Остановить:    docker compose -f docker-compose.windows.yml down"
Log "Дальше: откройте панель -> «Администратор» -> «Переиндексировать»."
