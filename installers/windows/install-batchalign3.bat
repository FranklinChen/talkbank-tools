@echo off
REM install-batchalign3.bat: one-click Batchalign3 installer for Windows.
REM
REM Double-click in Explorer. This runs the canonical GitHub-release installer,
REM which installs the uv package manager if needed and then installs the
REM batchalign3 CLI. There is no PyPI package; distribution is via GitHub
REM releases. The first install downloads large ML dependencies.
REM
REM After installation, open a new PowerShell or Command Prompt and run:
REM   batchalign3 --help

echo ============================================
echo   Batchalign3 Installer for Windows
echo ============================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.ps1 | iex"

echo.
echo ============================================
echo   Open a NEW terminal and run:
echo     batchalign3 --help
echo ============================================
echo.
if not "%CI%"=="true" pause
