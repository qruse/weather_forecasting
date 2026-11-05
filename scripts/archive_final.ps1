# One last scoring after the service stopped (2026-11-05 issuance): official observations for its last days, then the record goes up.
# No GPU and no new forecast. Run by hand once on or after 2026-11-08: powershell -File scripts\archive_final.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$log = "data\service\logs\archive_final.log"
& { python scripts\download_asos.py 2026-11-06 2026-11-07 --all; python scripts\service.py archive; python scripts\service.py upload } *>&1 | ForEach-Object { "$_" } | Select-Object -Last 30 | Out-File $log -Encoding utf8
