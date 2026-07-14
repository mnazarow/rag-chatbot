# =============================================================================
#  ПЕРЕУСТАНОВКА RAG в Docker на Windows — чистая пересборка «с нуля».
#  В отличие от update (инкрементальная пересборка), reinstall удаляет контейнеры
#  проекта и пересобирает образ приложения БЕЗ КЕША (--no-cache), затем поднимает
#  всё заново. По умолчанию ДАННЫЕ СОХРАНЯЮТСЯ: том индекса Qdrant (qdrant_storage),
#  кеш моделей (hf_cache), настройки/логи (папка state), документы и бэкапы.
#
#  Запуск (проще всего — двойной клик по reinstall.cmd), либо:
#     powershell -ExecutionPolicy Bypass -File reinstall_windows_docker.ps1 [ключи]
#
#  Ключи (можно комбинировать):
#     -Cpu       пересобрать БЕЗ GPU (по умолчанию — с поддержкой GPU NVIDIA)
#     -Data      ДОПОЛНИТЕЛЬНО удалить тома данных (индекс Qdrant + кеш моделей hf_cache
#                + Milvus). Векторная база и кеш моделей ТЕРЯЮТСЯ (модели скачаются заново).
#     -State     удалить локальные настройки/логи (папка state): runtime_config.json,
#                журнал запросов, статистику. Настройки админки сбрасываются (бэкапы целы).
#     -NoPull    не тянуть код из GitHub, пересобрать из текущих файлов.
#     -Branch    ветка git (по умолчанию main).
#     -Yes       не спрашивать подтверждение (для автоматизации).
# =============================================================================
param(
    [switch]$Cpu,
    [switch]$Data,
    [switch]$State,
    [switch]$NoPull,
    [string]$Branch = "main",
    [switch]$Yes
)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here
$RepoRoot = (Resolve-Path (Join-Path $Here "..\..")).Path

function Log($m){ Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "[!] $m" -ForegroundColor Yellow }
function Item($status,$label,$detail=""){
    switch ($status) { "ok"{$mark="[OK]  ";$col="Green"} "fail"{$mark="[X]   ";$col="Red"} default{$mark="[~]   ";$col="Yellow"} }
    $line=" $mark $label"; if ($detail){ $line += "  - $detail" }; Write-Host $line -ForegroundColor $col
}
function HttpOk($url,$sec=5){ try{ (Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec $sec).StatusCode -eq 200 }catch{ $false } }
function ContainerUp($n){ try{ ((docker inspect -f '{{.State.Running}}' $n 2>$null) -eq 'true') }catch{ $false } }

# Файлы compose: базовый + GPU-override (по умолчанию). Ключ -Cpu отключает GPU.
$Compose = @("-f","docker-compose.windows.yml")
if (-not $Cpu) { $Compose += @("-f","docker-compose.gpu.yml") }

# ----- подтверждение (reinstall пересобирает образ с нуля; при -Data/-State — теряются данные) -----
if (-not $Yes) {
    $what = "Полная пересборка образа приложения без кеша (это дольше обычного update)."
    if ($Data)  { $what += "`n    + УДАЛЕНИЕ томов: индекс Qdrant и кеш моделей будут потеряны." }
    if ($State) { $what += "`n    + УДАЛЕНИЕ настроек/логов (папка state): настройки админки сбросятся." }
    Write-Host $what -ForegroundColor Yellow
    $ans = Read-Host "Продолжить переустановку? [y/N]"
    if ($ans -notmatch '^[Yy]$') { Log "Отменено."; exit 0 }
}

# ----- 0. Docker доступен? -----
docker info *> $null
if (-not $?) {
    $dd = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Log "Запускаю Docker Desktop..."; Start-Process $dd }
    Log "Жду запуска движка Docker (до ~3 минут)..."
    $ready=$false; for($i=0;$i -lt 36;$i++){ docker info *> $null; if($?){ $ready=$true; break }; Start-Sleep 5 }
    if (-not $ready) { Warn "Движок Docker не запущен. Запустите Docker Desktop и повторите."; exit 1 }
}

# ----- 1. Обновление кода из GitHub -----
$oldRev = ""; $newRev = ""
if (-not $NoPull) {
    if ((Test-Path (Join-Path $RepoRoot ".git")) -and (Get-Command git -ErrorAction SilentlyContinue)) {
        Log "Обновляю код из GitHub (origin/$Branch)..."
        try {
            $oldRev = (git -C $RepoRoot rev-parse --short HEAD 2>$null)
            git -C $RepoRoot fetch --all -q
            git -C $RepoRoot reset --hard "origin/$Branch"
            $newRev = (git -C $RepoRoot rev-parse --short HEAD 2>$null)
            if ($oldRev -eq $newRev) { Log "Код уже актуален ($newRev)." } else { Log "Код обновлён: $oldRev -> $newRev" }
        } catch { Warn "git pull не удался: $($_.Exception.Message). Продолжаю с текущим кодом." }
    } else {
        Warn "Это не git-репозиторий (или git не установлен) — пересобираю из текущих файлов."
    }
}

# ----- 2. Останавливаю и удаляю контейнеры проекта -----
$eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
Log "Останавливаю и удаляю контейнеры проекта..."
if ($Data) {
    Warn "Ключ -Data: удаляю также тома данных (индекс Qdrant, кеш моделей)."
    docker compose @Compose down -v
} else {
    docker compose @Compose down
}
# одиночный rag_redis из старого обходного запуска — убрать, чтобы не мешал
try { if (docker ps -aq -f "name=^rag_redis$") { docker rm -f rag_redis *> $null } } catch {}

# ----- 2b. Опциональная очистка локального состояния (settings/logs) -----
if ($State) {
    Warn "Ключ -State: удаляю папку state (настройки/логи)."
    if (Test-Path "state") { Remove-Item -Recurse -Force "state" }
}

# ----- 2c. Гарантируем .env.docker и state-файлы (bind-mount'ам нужны существующие файлы) -----
if (-not (Test-Path ".env.docker")) {
    if (Test-Path ".env.docker.example") { Copy-Item ".env.docker.example" ".env.docker"; Log "Создан .env.docker из примера." }
    else { "" | Set-Content ".env.docker"; Warn ".env.docker.example не найден — создан пустой .env.docker." }
}
New-Item -ItemType Directory -Force -Path "state" | Out-Null
if (-not (Test-Path "state\runtime_config.json")) { "{}" | Set-Content "state\runtime_config.json" }
if (-not (Test-Path "state\ingest_stats.json"))   { "{}" | Set-Content "state\ingest_stats.json" }
if (-not (Test-Path "state\rag_logs.db"))          { New-Item -ItemType File -Force -Path "state\rag_logs.db" | Out-Null }
New-Item -ItemType Directory -Force -Path "backups" | Out-Null

# ----- 3. Чистая пересборка образа (--no-cache) -----
if (-not $Cpu) { Log "Режим GPU (по умолчанию): пересобираю CUDA-образ. Для CPU — ключ -Cpu." }
Log "Пересобираю образ приложения с нуля (--no-cache) — это дольше обычного update..."
$buildOk = $false
for ($try = 1; $try -le 3; $try++) {
    if ($try -gt 1) { Log "Сетевой сбой при сборке. Повтор $try из 3 через 6 c..."; Start-Sleep 6 }
    docker compose @Compose build --no-cache
    if ($LASTEXITCODE -eq 0) { $buildOk = $true; break }
}

# ----- 4. Запуск заново -----
$upOk = $false
if ($buildOk) {
    Log "Поднимаю контейнеры..."
    docker compose @Compose up -d
    if ($LASTEXITCODE -eq 0) { $upOk = $true }
}
$ErrorActionPreference = $eap

# ----- 5. Чеклист -----
Write-Host ""
Log "Жду готовности приложения (прогрев моделей — до ~2 минут)..."
$appOk = $false
for ($i=0;$i -lt 40;$i++){ if (HttpOk "http://localhost:8000/health" 4){ $appOk=$true; break }; Start-Sleep 3 }

$fails=0; $warns=0
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Чек-лист переустановки (Docker на Windows)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
if ($newRev) { Item ok "Код из GitHub" $(if($oldRev -eq $newRev){"актуален ($newRev)"}else{"$oldRev -> $newRev"}) }
if ($buildOk) { Item ok "Чистая пересборка образа (--no-cache)" } else { Item fail "Пересборка образа не удалась"; $fails++ }
if ($upOk) { Item ok "Контейнеры подняты (docker compose up)" } else { Item fail "Запуск контейнеров не удался"; $fails++ }
if (ContainerUp "rag_qdrant") { Item ok "Контейнер Qdrant (rag_qdrant) работает" } else { Item fail "Контейнер Qdrant не запущен"; $fails++ }
$appUp = ContainerUp "rag_app"
if ($appUp) { Item ok "Контейнер приложения (rag_app) работает" } else { Item fail "Контейнер приложения не запущен"; $fails++ }
if (HttpOk "http://localhost:6333/collections" 4) { Item ok "Qdrant отвечает (порт 6333)" } else { Item warn "Qdrant пока не отвечает" "возможно, ещё стартует"; $warns++ }
if ($appOk) { Item ok "Веб-интерфейс отвечает" "http://localhost:8000" } else { Item fail "Веб-интерфейс не отвечает (/health)"; $fails++ }

if ($appUp) {
    try {
        $cu = (docker exec rag_app /opt/venv/bin/python -c "import torch; a=torch.cuda.is_available(); print('AVAIL', a); print('COUNT', torch.cuda.device_count())" 2>&1 | Out-String)
        if ($cu -match "AVAIL\s+True") {
            $cnt = if ($cu -match "COUNT\s+(\d+)"){$Matches[1]}else{"0"}
            Item ok "CUDA доступна в контейнере (эмбеддинги/реранк на GPU)" "устройств: $cnt"
        } elseif (-not $Cpu) {
            Item fail "Ожидался GPU (режим по умолчанию), но CUDA в контейнере НЕ доступна"; $fails++
            Write-Host "     Если GPU нет — переустановите на CPU: reinstall.cmd -Cpu" -ForegroundColor Gray
        } else {
            Item ok "Запущено на CPU (ключ -Cpu)" "для GPU переустановите без -Cpu"
        }
    } catch { Item warn "Не удалось проверить CUDA в контейнере" "$($_.Exception.Message)"; $warns++ }
}

Write-Host "============================================================" -ForegroundColor Cyan
if ($fails -eq 0 -and $warns -eq 0) { Write-Host "  ИТОГ: переустановка успешна" -ForegroundColor Green }
elseif ($fails -eq 0) { Write-Host "  ИТОГ: переустановлено, есть предупреждения ($warns)." -ForegroundColor Yellow }
else { Write-Host "  ИТОГ: есть ошибки ($fails). Логи: docker compose $($Compose -join ' ') logs -f app" -ForegroundColor Red }
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Log "Веб-интерфейс: http://localhost:8000"
Log "Логи:          docker compose $($Compose -join ' ') logs -f app"
if ($appOk) { Start-Process "http://localhost:8000" }
