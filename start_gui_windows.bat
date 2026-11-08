@echo off
rem Uncomment the next line to use the pywebview desktop window.
set "BILI_USE_PYWEBVIEW=1"
chcp 65001 >nul
cd /d %~dp0
rem Run the backend in a separate cmd /c so this batch exits immediately.
rem Ctrl+C then terminates only the backend, without the "Terminate batch job (Y/N)" prompt.
start /b "" cmd /c "python -m app.desktop"
exit /b 0
