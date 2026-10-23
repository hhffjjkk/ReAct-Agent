# 一键停止：导入服务(:8000) + 查询服务(:8001)，可选停止 Docker 基础设施
# 用法：powershell -ExecutionPolicy Bypass -File .\stop_all.ps1
#       powershell -ExecutionPolicy Bypass -File .\stop_all.ps1 -Infra   # 同时停止容器

param(
    [switch]$Infra
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RunDir = Join-Path $Root ".run"

Write-Host "[1/3] 停止 Python 服务..." -ForegroundColor Cyan

# 先按命令行匹配（可覆盖 uv 派生的 python 子进程），再按 pid 文件兜底
$targets = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'app\.(import_process|query_process)\.api\.(import|query)_server' }

foreach ($p in $targets) {
    Write-Host "  停止 python 进程 PID=$($p.ProcessId)" -ForegroundColor Yellow
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}

if (Test-Path $RunDir) {
    foreach ($name in @("import", "query")) {
        $pidFile = Join-Path $RunDir "$name.pid"
        if (Test-Path $pidFile) {
            $savedPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
            if ($savedPid) {
                Stop-Process -Id ([int]$savedPid) -Force -ErrorAction SilentlyContinue
            }
            Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host "[2/3] 确认端口已释放..." -ForegroundColor Cyan
$deadline = (Get-Date).AddSeconds(20)
while ((Get-Date) -lt $deadline) {
    $busy8000 = [bool](Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue)
    $busy8001 = [bool](Get-NetTCPConnection -State Listen -LocalPort 8001 -ErrorAction SilentlyContinue)
    if (-not $busy8000 -and -not $busy8001) { break }
    Start-Sleep -Seconds 1
}

if ($busy8000) { Write-Host "  警告：端口 8000 仍被占用（PID: $((Get-NetTCPConnection -State Listen -LocalPort 8000).OwningProcess)）" -ForegroundColor Yellow }
if ($busy8001) { Write-Host "  警告：端口 8001 仍被占用（PID: $((Get-NetTCPConnection -State Listen -LocalPort 8001).OwningProcess)）" -ForegroundColor Yellow }
if (-not $busy8000 -and -not $busy8001) { Write-Host "  端口 8000 / 8001 已释放。" -ForegroundColor Green }

if ($Infra) {
    Write-Host "[3/3] 停止 Docker 基础设施..." -ForegroundColor Cyan
    docker compose down
    docker stop mongo 2>$null | Out-Null
} else {
    Write-Host "[3/3] 保留 Docker 基础设施（如需停止：.\stop_all.ps1 -Infra）。" -ForegroundColor DarkGray
}
