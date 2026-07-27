# =============================================================================
#  Полное удаление RAG (НАТИВНАЯ установка на Windows — Scheduled Task + Qdrant в Docker).
#  Для установки в Docker используйте windows_variant\docker\uninstall_windows_docker.ps1
#
#  Запуск:
#     powershell -ExecutionPolicy Bypass -File uninstall_windows.ps1 [ключи]
#
#  ПО УМОЛЧАНИЮ (без ключей): снимает автозапуск (задачу RagApi) и останавливает/удаляет
#  контейнер Qdrant. ДАННЫЕ СОХРАНЯЮТСЯ (индекс, настройки, логи, документы).
#
#  Ключи (комбинируются):
#     -Venv      удалить окружение .venv
#     -Volumes   удалить индекс Qdrant (том qdrant_storage) и graph_storage. ДАННЫЕ БАЗЫ ТЕРЯЮТСЯ.
#     -State     удалить runtime_config.json, rag_logs.db*, ingest_stats.json, .env.
#                НАСТРОЙКИ АДМИНКИ И ЖУРНАЛ ТЕРЯЮТСЯ (бэкапы сохраняются).
#     -Images    удалить Docker-образ qdrant/qdrant.
#     -Docs      удалить папку документов DOCS_DIR (из .env, по умолчанию C:\db). ОПАСНО.
#     -Backups   дополнительно удалить папку backups.
#     -Purge     = -Venv -Volumes -State -Images   (всё, КРОМЕ документов и бэкапов)
#     -All       = -Purge -Docs -Backups           (стереть всё)
#     -Yes       не спрашивать подтверждение
#     -DryRun    показать план, ничего не удаляя
# =============================================================================
param(
    [switch]$Venv,
    [switch]$Volumes,
    [switch]$State,
    [switch]$Images,
    [switch]$Docs,
    [switch]$Backups,
    [switch]$Purge,
    [switch]$All,
    [switch]$Yes,
    [switch]$DryRun
)
$ErrorActionPreference = "Continue"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $Here          # каталог проекта = родитель windows_variant\
Set-Location $Root

if ($Purge) { $Venv = $true; $Volumes = $true; $State = $true; $Images = $true }
if ($All)   { $Venv = $true; $Volumes = $true; $State = $true; $Images = $true; $Docs = $true; $Backups = $true }

function Log($m){ Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "[!] $m" -ForegroundColor Yellow }
function Done($m){ Write-Host " [OK] $m" -ForegroundColor Green }
function Do($desc, [scriptblock]$act){
    if ($DryRun){ Write-Host "   dry: $desc" -ForegroundColor Green; return }
    try { & $act | Out-Null; Done $desc } catch { Warn "не удалось: $desc" }
}
function RmPath($p){
    if (-not $p) { return }
    if (Test-Path $p){
        if ($DryRun){ Write-Host "   dry: rm $p" -ForegroundColor Green; return }
        try { Remove-Item -Recurse -Force -LiteralPath $p; Done "удалено: $p" } catch { Warn "не удалось удалить: $p" }
    }
}
function DocsDir(){
    $rc = Join-Path $Root "runtime_config.json"
    if (Test-Path $rc){ try { $d = (Get-Content $rc -Raw | ConvertFrom-Json).DOCS_DIR; if ($d){ return $d } } catch {} }
    $env = Join-Path $Root ".env"
    if (Test-Path $env){ $m = Select-String -Path $env -Pattern '^DOCS_DIR=(.*)$'; if ($m){ return ($m.Matches[0].Groups[1].Value.Trim('"',"'")) } }
    return $null
}

# ----- Сводка -----
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Удаление RAG (нативная установка Windows)   $Root" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Будет сделано:"
Write-Host "   - снята задача автозапуска RagApi"
Write-Host "   - остановлен и удалён контейнер Qdrant (rag_qdrant)"
if ($Venv)    { Write-Host "   - удалено окружение .venv" -ForegroundColor Yellow }
if ($Volumes) { Write-Host "   - удалён индекс Qdrant (том qdrant_storage) и graph_storage — ДАННЫЕ ТЕРЯЮТСЯ" -ForegroundColor Yellow }
if ($State)   { Write-Host "   - удалены настройки и журнал (runtime_config.json, rag_logs.db, .env)" -ForegroundColor Yellow }
if ($Images)  { Write-Host "   - удалён образ qdrant/qdrant" -ForegroundColor Yellow }
if ($Backups) { Write-Host "   - удалена папка backups" -ForegroundColor Yellow }
if ($Docs)    { Write-Host "   - УДАЛЕНА ПАПКА ДОКУМЕНТОВ: $(DocsDir)" -ForegroundColor Red }
if ($DryRun)  { Write-Host "  (режим -DryRun: ничего реально не удаляется)" -ForegroundColor Green }
Write-Host ""

if (-not $Yes -and -not $DryRun){
    $ans = Read-Host "Продолжить удаление? введите yes"
    if ($ans -ne "yes"){ Write-Host "Отменено."; exit 0 }
}

# ----- 1. Автозапуск (Scheduled Task RagApi) -----
Log "Снимаю задачу автозапуска RagApi…"
Do "schtasks /End RagApi"    { schtasks /End /TN "RagApi" 2>$null }
Do "schtasks /Delete RagApi" { schtasks /Delete /TN "RagApi" /F 2>$null }
RmPath (Join-Path $Here "start_app.cmd")

# ----- 2. Docker (Qdrant) -----
$hasDocker = $null -ne (Get-Command docker -ErrorAction SilentlyContinue)
if ($hasDocker){
    Log "Останавливаю и удаляю контейнер Qdrant…"
    Do "docker rm -f rag_qdrant" { docker rm -f rag_qdrant 2>$null }
    if ($Volumes){ Do "docker volume rm qdrant_storage" { docker volume rm qdrant_storage 2>$null } }
    if ($Images) { Do "docker rmi qdrant/qdrant:v1.12.4" { docker rmi -f qdrant/qdrant:v1.12.4 2>$null } }
} else {
    Warn "docker не найден — контейнер/том Qdrant удалите вручную через Docker Desktop."
}

# ----- 3. Python-окружение -----
if ($Venv){ Log "Удаляю .venv…"; RmPath (Join-Path $Root ".venv") }

# ----- 4. Данные индекса -----
if ($Volumes){ Log "Удаляю graph_storage…"; RmPath (Join-Path $Root "graph_storage") }

# ----- 5. Настройки / журнал -----
if ($State){
    Log "Удаляю настройки и журнал…"
    foreach($f in @("runtime_config.json","ingest_stats.json","ingest_progress.json",
                    "rag_logs.db","rag_logs.db-journal","rag_logs.db-wal","rag_logs.db-shm",".env")){
        RmPath (Join-Path $Root $f)
    }
    foreach($d in @("finetune\adapter","finetune\data")){ RmPath (Join-Path $Root $d) }
}

# ----- 6. Бэкапы -----
if ($Backups){ Log "Удаляю backups…"; RmPath (Join-Path $Root "backups") }

# ----- 7. Документы (ОПАСНО) -----
if ($Docs){
    $dd = DocsDir
    if (-not $dd){ Warn "DOCS_DIR не определён — пропускаю." }
    elseif ($dd -eq "C:\" -or $dd -eq $env:USERPROFILE){ Warn "DOCS_DIR=$dd выглядит опасно — НЕ удаляю, уберите вручную." }
    else { Warn "Удаляю папку документов: $dd"; RmPath $dd }
}

Write-Host ""
if ($DryRun){ Done "Готово (dry-run): показан план. Запустите без -DryRun для реального удаления." }
else { Done "Удаление завершено. Каталог проекта и исходники оставлены." }
