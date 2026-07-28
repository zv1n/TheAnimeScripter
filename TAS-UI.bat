@echo off
setlocal
cd /d "%~dp0"
title TAS Web UI
echo.
echo   Starting TAS Web UI...  (a browser tab will open shortly)
echo   Close this window to stop the server.
echo.
".\python.exe" "webui\server.py"
echo.
echo   Server stopped.
pause
