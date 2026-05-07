# PowerShell wrapper around build.py so the qmark MSIX orchestrator can
# call all three children (qmark, OpenName, OpenCrop) with the same
# command shape. Also normalises Nuitka's app.dist -> OpenCrop.dist so
# downstream staging code doesn't need to special-case OpenCrop.

[CmdletBinding()]
param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

if ($Clean -and (Test-Path "$root\dist")) {
    Write-Host "Removing existing dist/ ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "$root\dist"
}

& python build.py
if ($LASTEXITCODE -ne 0) {
    throw "OpenCrop Nuitka build failed (exit $LASTEXITCODE)."
}

$nuitkaOut = "$root\dist\app.dist"
$wantDir = "$root\dist\OpenCrop.dist"
if (Test-Path $nuitkaOut) {
    if (Test-Path $wantDir) { Remove-Item -Recurse -Force $wantDir }
    Rename-Item $nuitkaOut $wantDir
}

$exePath = "$wantDir\OpenCrop.exe"
if (-not (Test-Path $exePath)) {
    throw "Build claimed success but $exePath is missing."
}

if (Test-Path "$root\LICENSE") {
    Copy-Item "$root\LICENSE" -Destination (Join-Path $wantDir "LICENSE.txt") -Force
}

Write-Host ""
Write-Host "OpenCrop build complete." -ForegroundColor Green
Write-Host "  Executable: $exePath" -ForegroundColor Green
