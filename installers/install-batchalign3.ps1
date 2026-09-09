# install-batchalign3.ps1: install or update the batchalign3 CLI from the latest
# GitHub release on Windows. Bootstraps uv if absent and installs into an
# isolated uv tool environment using a uv-managed Python (default 3.13).
#
#   irm https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.ps1 | iex
#
# Re-running upgrades an existing installation. Override the managed Python with
# the BATCHALIGN3_PYTHON environment variable (for example 3.13). There is no
# PyPI package; distribution is via GitHub releases.
$ErrorActionPreference = "Stop"
$Repo = "FranklinChen/talkbank-tools"
$PythonVersion = if ($env:BATCHALIGN3_PYTHON) { $env:BATCHALIGN3_PYTHON } else { "3.13" }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "install-batchalign3: uv not found; installing uv"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

$plat = "win_amd64"
$api = "https://api.github.com/repos/$Repo/releases/latest"
$release = Invoke-RestMethod -Uri $api
$asset = $release.assets | Where-Object { $_.name -like "*-abi3-$plat.whl" } | Select-Object -First 1
if (-not $asset) { throw "no abi3 wheel for $plat in the latest $Repo release" }

Write-Host "install-batchalign3: installing $($asset.name) (Python $PythonVersion)"
uv tool install --force --python $PythonVersion $asset.browser_download_url
uv tool update-shell 2>$null
Write-Host "install-batchalign3: done. Open a new terminal and run: batchalign3 --help"
