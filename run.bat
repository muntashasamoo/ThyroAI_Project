@echo off
cd /d %~dp0
start http://127.0.0.1:8000/ui
uvicorn api:app --reload --port 8000
pause