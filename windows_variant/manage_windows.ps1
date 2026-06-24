# Управление сервисом RAG на Windows (Scheduled Task 'RagApi').
#   powershell -File windows_variant\manage_windows.ps1 status|start|stop|restart|logs
param([string]$Action = "status")
$ErrorActionPreference = "SilentlyContinue"

switch ($Action) {
  "status" {
    schtasks /Query /TN "RagApi" /V /FO LIST 2>$null
    Write-Host "---- контейнеры ----"
    docker ps --filter "name=rag_qdrant"
  }
  "start"   { schtasks /Run /TN "RagApi"; Write-Host "запущено" }
  "stop"    {
    schtasks /End /TN "RagApi" 2>$null
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
      Where-Object { $_.CommandLine -like "*uvicorn*app:app*" } |
      ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Write-Host "остановлено"
  }
  "restart" {
    & $PSCommandPath stop | Out-Null
    Start-Sleep -Seconds 1
    schtasks /Run /TN "RagApi"
    Write-Host "перезапущено"
  }
  "logs"    { Get-Content "$env:TEMP\rag_api.log" -Tail 200 -Wait }
  default   { Write-Host "Команды: status | start | stop | restart | logs" }
}
