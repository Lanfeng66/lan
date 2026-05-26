@echo off
chcp 65001 >nul
title DocMind-Frontend

cd /d %~dp0frontend
npx next build && npx next start -p 3000
pause
