@echo off
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is not installed. Install it from https://nodejs.org and run this again.
  pause
  exit /b 1
)
call npx --yes vercel@latest link --yes --project harman-one-neuron
call npx --yes vercel@latest deploy --prod --yes
pause
