# Автоматическая установка RAG в Docker с Redis (Windows) — одной командой.
# Проще всего: двойной клик по start.cmd (он чинит кодировку и зовёт этот скрипт).
# Либо напрямую:
#   powershell -ExecutionPolicy Bypass -File install.ps1 [-DocsDir "C:\rag\BD"] [-AdminToken "пароль"]
# Скрипт сам ставит Docker Desktop и Ollama (winget), скачивает модель, поднимает
# контейнеры qdrant + redis + app (Redis включён) и печатает чеклист.
param(
    [string]$DocsDir = "",                            # папка с документами (по умолчанию .\docs)
    [string]$LlmModel = "qwen3.6:35b-a3b-q4_K_M",     # модель Ollama для генерации
    [string]$AdminToken = ""                          # пароль админ-панели (пусто = не менять)
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
function Item($status, $label, $detail) {
    switch ($status) {
        "ok"   { $mark = "[OK]  "; $col = "Green"  }
        "fail" { $mark = "[X]   "; $col = "Red"    }
        default{ $mark = "[~]   "; $col = "Yellow" }
    }
    $line = " $mark $label"; if ($detail) { $line += "  - $detail" }
    Write-Host $line -ForegroundColor $col
}
function HttpOk($url, $timeoutSec = 5) {
    try { $r = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec $timeoutSec -ErrorAction Stop
          return ($r.StatusCode -eq 200) } catch { return $false }
}
function ContainerUp($name) {
    try { return ((docker inspect -f '{{.State.Running}}' $name 2>$null) -eq 'true') } catch { return $false }
}
function ShowLog($label, [scriptblock]$action) {
    Write-Host ""; Write-Host "     --- подробный лог: $label ---" -ForegroundColor DarkYellow
    try { $out = & $action 2>&1
          if ($out) { $out | ForEach-Object { Write-Host "     $_" -ForegroundColor Gray } } else { Write-Host "     (пусто)" -ForegroundColor Gray } }
    catch { Write-Host "     (не удалось получить лог: $($_.Exception.Message))" -ForegroundColor Gray }
    Write-Host "     --- конец лога ---" -ForegroundColor DarkYellow; Write-Host ""
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
        Refresh-Path; Start-Sleep -Seconds 5
    }
}
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Log "Скачиваю модель Ollama: $LlmModel (при первом запуске долго)..."
    $pulled = $false
    for ($i = 0; $i -lt 6; $i++) { try { ollama pull $LlmModel; $pulled = $true; break } catch { Start-Sleep -Seconds 5 } }
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
(Get-Content ".env.docker") -replace '^LLM_MODEL=.*', "LLM_MODEL=$LlmModel" | Set-Content ".env.docker"
if ($AdminToken -ne "") {
    (Get-Content ".env.docker") -replace '^ADMIN_TOKEN=.*', "ADMIN_TOKEN=$AdminToken" | Set-Content ".env.docker"
    Log "Пароль админ-панели задан."
}

# ----- 4. Папка с документами (по умолчанию .\docs — без вопросов) -----
if (-not $DocsDir) {
    $DocsDir = Join-Path $Here "docs"
    New-Item -ItemType Directory -Force -Path $DocsDir | Out-Null
    Log "Папка документов по умолчанию: $DocsDir (положите туда файлы и переиндексируйте)."
}
if (-not (Test-Path $DocsDir)) { throw "Папка не найдена: $DocsDir" }
"DOCS_DIR_HOST=$DocsDir" | Set-Content ".env"

# ----- 5. Файлы состояния -----
New-Item -ItemType Directory -Force -Path "state" | Out-Null
if (-not (Test-Path "state\runtime_config.json")) { "{}" | Set-Content "state\runtime_config.json" }
if (-not (Test-Path "state\ingest_stats.json"))   { "{}" | Set-Content "state\ingest_stats.json" }
if (-not (Test-Path "state\rag_logs.db"))         { New-Item -ItemType File -Force -Path "state\rag_logs.db" | Out-Null }
New-Item -ItemType Directory -Force -Path "backups" | Out-Null

# ----- 6. Сборка и запуск (qdrant + redis + app) -----
Log "Собираю и запускаю контейнеры (первый раз — долго: качаются образы и модели)..."
docker compose up -d --build
$composeOk = ($LASTEXITCODE -eq 0)

# ----- 7. Чеклист -----
Write-Host ""
Log "Жду готовности приложения (загрузка моделей эмбеддингов может занять до ~2 минут)..."
$appOk = $false
for ($i = 0; $i -lt 40; $i++) { if (HttpOk "http://localhost:8000/health" 4) { $appOk = $true; break }; Start-Sleep -Seconds 3 }

$fails = 0; $warns = 0
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Чеклист установки (Docker + Redis)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if ($composeOk) { Item ok "Сборка образа и запуск docker compose" }
else { Item fail "Сборка/запуск docker compose"; $fails++
       ShowLog "docker compose ps + логи" { docker compose ps; docker compose logs --tail 40 } }

if (ContainerUp "rag_qdrant") { Item ok "Контейнер Qdrant (rag_qdrant) работает" }
else { Item fail "Контейнер Qdrant не запущен"; $fails++; ShowLog "docker logs rag_qdrant" { docker logs --tail 40 rag_qdrant } }

if (ContainerUp "rag_redis") { Item ok "Контейнер Redis (rag_redis) работает" }
else { Item fail "Контейнер Redis не запущен"; $fails++; ShowLog "docker logs rag_redis" { docker logs --tail 40 rag_redis } }

# Redis отвечает PONG
$redisPong = $false
try { $redisPong = ((docker exec rag_redis redis-cli ping 2>$null) -match 'PONG') } catch {}
if ($redisPong) { Item ok "Redis отвечает (PONG)" }
else { Item warn "Redis пока не отвечает" "возможно, ещё стартует"; $warns++ }

$appUp = ContainerUp "rag_app"
if ($appUp) { Item ok "Контейнер приложения (rag_app) работает" }
else { Item fail "Контейнер приложения не запущен"; $fails++; ShowLog "docker logs rag_app" { docker logs --tail 60 rag_app } }

if (HttpOk "http://localhost:6333/collections" 4) { Item ok "Qdrant отвечает (порт 6333)" }
else { Item warn "Qdrant пока не отвечает" "возможно, ещё стартует"; $warns++ }

if ($appOk) { Item ok "Веб-интерфейс отвечает" "http://localhost:8000" }
else { Item fail "Веб-интерфейс не отвечает (/health)"; $fails++
       if ($appUp) { ShowLog "docker logs rag_app (60 строк)" { docker logs --tail 60 rag_app } } }

# Приложение видит Qdrant и Redis (из /api/system)
if ($appOk) {
    $qOnline = $false; $cacheOn = $false; $cacheReach = $false
    try {
        $sys = (Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8000/api/system" -TimeoutSec 6).Content | ConvertFrom-Json
        $qOnline = [bool]$sys.qdrant.online
        $cacheOn = [bool]$sys.cache.enabled
        $cacheReach = [bool]$sys.cache.reachable
    } catch {}
    if ($qOnline) { Item ok "Приложение видит Qdrant" "QDRANT_URL=http://qdrant:6333" }
    else { Item fail "Приложение НЕ видит Qdrant"; $fails++
           ShowLog "QDRANT_URL" { docker exec rag_app printenv QDRANT_URL } }
    if ($cacheOn -and $cacheReach) { Item ok "Приложение видит Redis (кэш включён)" "REDIS_HOST=redis" }
    elseif ($cacheOn) { Item warn "Кэш включён, но Redis недоступен приложению" "проверьте контейнер redis"; $warns++ }
    else { Item warn "Кэш Redis выключен в приложении" "ожидался REDIS_ENABLED=1"; $warns++
           ShowLog "REDIS_* окружение" { docker exec rag_app sh -lc "printenv | grep -i redis" } }
}

# Ollama на хосте
$ollamaUp = HttpOk "http://localhost:11434/api/tags" 4
if ($ollamaUp) { Item ok "Ollama на хосте доступна" }
else { Item warn "Ollama на хосте недоступна (http://localhost:11434)" "установите/запустите Ollama"; $warns++ }

Write-Host "============================================================" -ForegroundColor Cyan
if ($fails -eq 0 -and $warns -eq 0) { Write-Host "  ИТОГ: всё успешно" -ForegroundColor Green }
elseif ($fails -eq 0) { Write-Host "  ИТОГ: запущено, есть предупреждения ($warns)." -ForegroundColor Yellow }
else { Write-Host "  ИТОГ: есть ошибки ($fails). Логи — выше." -ForegroundColor Red }
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Log "Веб-интерфейс: http://localhost:8000"
Log "Кэш Redis виден в разделе «Система» -> «Кэш Redis»."
Log "Логи:       docker compose logs -f app"
Log "Остановить: docker compose down"
if ($appOk) { Start-Process "http://localhost:8000" }
