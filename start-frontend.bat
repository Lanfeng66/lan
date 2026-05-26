@echo off
chcp 65001 >nul
title DocMind-Frontend

cd /d %~dp0frontend
npm run dev
pause
