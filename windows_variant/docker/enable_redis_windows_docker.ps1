# =============================================================================
#  Включение (или выключение) кэша Redis для RAG в Docker на Windows — одной командой.
#  Поднимает контейнер redis, указывает приложению хост redis и включает кэш,
#  затем проверяет доступность. Данные приложения не тронутся.
#
#  Запуск (проще всего — двойной клик по redis.cmd), либо:
#     powershell -ExecutionPolicy Bypass -File enable_redis_windows_docker.ps1
#  Параметры:
#     -Cpu    поднимать стек БЕЗ GPU (по умолчанию — с GPU)
#     -Off    наоборот, ВЫКЛЮЧИТЬ кэш Redis (уберёт «включён, недоступен»)
# =============================================================================
param([switch]$Cpu, [switch]$Off)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

function Log($m){ Write-Host "==> $m" -ForegroundColor Cyan }
function Item($status,$label,$detail=""){
    switch ($status) { "ok"{$mark="[OK]  ";$col="Green"} "fail"{$mark="[X]   ";$col="Red"} default{$mark="[~]   ";$col="Yellow"} }
    $line=" $mark $label"; if ($detail){ $line += "  - $detail" }; Write-Host $line -ForegroundColor $col
}
function HttpOk($url,$sec=5){ try{ (Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec $sec).StatusCode -eq 200 }catch{ $false } }

# ----- вариант «выключить» -----
if ($Off) {
    Log "Выключаю кэш Redis в приложении..."
    docker exec rag_app /opt/venv/bin/python -c "import settings; settings.update({'REDIS_ENABLED':False}); print('REDIS_ENABLED =', settings.get('REDIS_ENABLED'))"
    Item ok "Кэш Redis выключен" "предупреждение «включён, недоступен» исчезнет"
    exit 0
}

# ----- вариант «включить» -----
$Compose = @("-f","docker-compose.windows.yml")
if (-not $Cpu) { $Compose += @("-f","docker-compose.gpu.yml") }

Log "Поднимаю стек с контейнером Redis (данные сохраняются)..."
$eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
# Убрать «одиночный» контейнер rag_redis (из старого обходного запуска), иначе compose
# не сможет создать свой сервис redis — конфликт имени контейнера rag_redis.
try {
    if (docker ps -aq -f "name=^rag_redis$") {
        $proj = (docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' rag_redis 2>$null)
        if (-not $proj) { Log "Удаляю одиночный контейнер rag_redis (конфликт имени с compose)..."; docker rm -f rag_redis *> $null }
    }
} catch {}
docker compose @Compose up -d --build
$composeOk = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $eap
if (-not $composeOk) { Item fail "docker compose up не выполнился"; exit 1 }

Log "Жду готовности приложения..."
for ($i=0; $i -lt 40; $i++){ if (HttpOk "http://localhost:8000/health" 4){ break }; Start-Sleep 3 }

Log "Указываю приложению хост redis и включаю кэш..."
docker exec rag_app /opt/venv/bin/python -c "import settings; settings.update({'REDIS_ENABLED':True,'REDIS_HOST':'redis','REDIS_PORT':6379,'REDIS_DB':0}); print('REDIS:', settings.get('REDIS_HOST'), settings.get('REDIS_PORT'), 'enabled=', settings.get('REDIS_ENABLED'))"

# ----- проверка -----
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Кэш Redis" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
$pong = $false
try { $pong = ((docker exec rag_redis redis-cli ping 2>$null) -match 'PONG') } catch {}
if ($pong) { Item ok "Redis отвечает (PONG)" "контейнер rag_redis" } else { Item warn "Redis пока не ответил" "возможно, ещё стартует"; }
$reach = $false
try {
    $sys = (Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8000/api/system" -TimeoutSec 6).Content | ConvertFrom-Json
    $reach = [bool]$sys.cache.reachable
} catch {}
if ($reach) { Item ok "Приложение видит Redis (кэш доступен)" "REDIS_HOST=redis" }
else { Item warn "Приложение пока не подтвердило доступность кэша" "обновите раздел «Система» через несколько секунд"; }
Write-Host "============================================================" -ForegroundColor Cyan
Log "Готово. Раздел «Система» -> «Кэш Redis» покажет статистику."
