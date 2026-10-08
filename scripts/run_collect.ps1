# 작업 스케줄러가 3시간마다 실행하는 래퍼: 예보 수집 -> 새 파일이 있으면 커밋·푸시
# 키는 저장소 루트의 .env (gitignore 됨)에서 읽는다:  KMA_APIHUB_KEY=...
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = Join-Path $root "data\collect.log"

Get-Content (Join-Path $root ".env") -Encoding UTF8 | ForEach-Object {
    if ($_ -match '^\s*([A-Z_]+)\s*=\s*(.+?)\s*$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
}

$py = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$out = (& $py scripts/collect_kma_forecast.py 2>&1 | Out-String).Trim()
Add-Content $log "[$stamp] $($out.Split("`n")[-1].Trim())" -Encoding UTF8

git add data/raw/kma_forecast
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -q -m "collect forecast $stamp"
    git push -q
    if ($LASTEXITCODE -ne 0) { git pull -q --rebase --autostash; git push -q }
}
