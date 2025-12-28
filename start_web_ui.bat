@echo off
cd /d "%~dp0"
set WEB_PORT=7870
start "3D Server" python tools\web_server.py
timeout /t 2 >nul
start "" "http://127.0.0.1:%WEB_PORT%"
