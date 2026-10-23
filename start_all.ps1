# 一键启动：Docker 基础设施 + 导入服务(:8000) + 查询服务(:8001)
# 用法：powershell -ExecutionPolicy Bypass -File .\start_all.ps1
# 说明：全程只使用本地已有镜像（docker compose --pull never），不会下载新镜像。

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$LogDir = Join-Path $Root "logs"
$RunDir = Join-Path $Root ".run"
New-Item -ItemType Directory -Force -Path $LogDir, $RunDir | Out-Null

function Test-PortBusy([int]$Port) {
    return [bool](Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
}

function Wait-Port([int]$Port, [int]$TimeoutSec, [string]$Name) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-PortBusy $Port) { return $true }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Wait-ContainerHealthy([string]$Name, [int]$TimeoutSec) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $status = docker inspect --format "{{.State.Health.Status}}" $Name 2>$null
        if ($status -eq "healthy") { return $true }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Show-LogTail([string]$Path, [int]$Lines = 30) {
    if (Test-Path $Path) {
        Write-Host "----- $Path (最后 $Lines 行) -----" -ForegroundColor Yellow
        Get-Content $Path -Tail $Lines
        Write-Host "----------------------------------" -ForegroundColor Yellow
    }
}

Write-Host "[1/5] 检查 Docker..." -ForegroundColor Cyan
docker info 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker 未运行，请先启动 Docker Desktop。"
}

$requiredImages = @(
    "milvusdb/milvus:v2.4.11",
    "quay.io/coreos/etcd:v3.5.5",
    "minio/minio:RELEASE.2023-03-20T20-16-18Z",
    "zilliz/attu:v2.4.0",
    "mongo:latest"
)
foreach ($img in $requiredImages) {
    docker image inspect $img 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "本地缺少镜像 $img，本脚本不会联网下载，请先手动准备该镜像。"
    }
}

Write-Host "[2/5] 启动基础设施（Milvus / etcd / MinIO / MongoDB）..." -ForegroundColor Cyan
docker compose up -d --pull never
if ($LASTEXITCODE -ne 0) { throw "docker compose 启动失败。" }

# Attu 默认占用宿主 8000，与导入服务冲突；本项目不需要 Attu，启动后将其停止。
docker stop milvus-attu 2>$null | Out-Null
$attuDeadline = (Get-Date).AddSeconds(20)
while ((Get-Date) -lt $attuDeadline -and (Test-PortBusy 8000)) { Start-Sleep -Milliseconds 500 }

$mongoExists = docker ps -a --filter "name=^/mongo$" --format "{{.Names}}"
if ($mongoExists -eq "mongo") {
    docker start mongo 2>$null | Out-Null
} else {
    docker run -d --name mongo --restart unless-stopped -p 27017:27017 -v mongo-data:/data/db mongo:latest | Out-Null
}
if ($LASTEXITCODE -ne 0) { throw "MongoDB 启动失败。" }

if (-not (Wait-ContainerHealthy "milvus-standalone" 180)) { throw "Milvus 未在 180 秒内变为 healthy，请执行 docker logs milvus-standalone 排查。" }
if (-not (Wait-ContainerHealthy "milvus-minio" 60)) { throw "MinIO 未在 60 秒内变为 healthy，请执行 docker logs milvus-minio 排查。" }

Write-Host "[3/5] 检查 Python 环境..." -ForegroundColor Cyan
uv sync
if ($LASTEXITCODE -ne 0) { throw "uv sync 失败，请检查依赖环境。" }

if (Test-PortBusy 8000) { throw "端口 8000 已被占用，请先运行 .\stop_all.ps1 或释放该端口。" }
if (Test-PortBusy 8001) { throw "端口 8001 已被占用，请先运行 .\stop_all.ps1 或释放该端口。" }

Write-Host "[4/5] 启动导入服务(:8000)与查询服务(:8001)..." -ForegroundColor Cyan
$env:PYTHONPATH = $Root

$importOut = Join-Path $LogDir "import_server.log"
$importErr = Join-Path $LogDir "import_server.err.log"
$queryOut = Join-Path $LogDir "query_server.log"
$queryErr = Join-Path $LogDir "query_server.err.log"

$importProc = Start-Process -FilePath "uv" `
    -ArgumentList @("run", "python", "-m", "app.import_process.api.import_server") `
    -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $importOut -RedirectStandardError $importErr

if (-not (Wait-Port 8000 180 "导入服务")) {
    Show-LogTail $importErr
    Show-LogTail $importOut
    throw "导入服务未能在 180 秒内监听 8000 端口。"
}
$importProc.Id | Set-Content (Join-Path $RunDir "import.pid")

$queryProc = Start-Process -FilePath "uv" `
    -ArgumentList @("run", "python", "-m", "app.query_process.api.query_server") `
    -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $queryOut -RedirectStandardError $queryErr

if (-not (Wait-Port 8001 180 "查询服务")) {
    Show-LogTail $queryErr
    Show-LogTail $queryOut
    throw "查询服务未能在 180 秒内监听 8001 端口。"
}
$queryProc.Id | Set-Content (Join-Path $RunDir "query.pid")

Write-Host "[5/5] 启动完成。" -ForegroundColor Green
Write-Host ""
Write-Host "  导入页    http://127.0.0.1:8000/import" -ForegroundColor Green
Write-Host "  问答页    http://127.0.0.1:8001/chat.html" -ForegroundColor Green
Write-Host "  监控页    http://127.0.0.1:8001/monitor.html" -ForegroundColor Green
Write-Host "  健康检查  http://127.0.0.1:8001/health" -ForegroundColor Green
Write-Host ""
Write-Host "  日志目录  $LogDir" -ForegroundColor DarkGray
Write-Host "  停止服务  powershell -ExecutionPolicy Bypass -File .\stop_all.ps1" -ForegroundColor DarkGray
