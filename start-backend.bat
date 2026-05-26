@echo off
chcp 65001 >nul
title DocMind-Backend

cd /d %~dp0backend
..\.venv\Scripts\activate && python -m uvicorn app.main:app --reload --port 8000 --host 0.0.0.0
pause
