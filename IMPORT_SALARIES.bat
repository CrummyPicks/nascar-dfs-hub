@echo off
echo.
echo ==========================================
echo   NASCAR DFS - Import Salaries and Odds
echo ==========================================
echo.
echo Import DraftKings/FanDuel salary CSVs
echo and/or paste Bovada odds from the website.
echo.

cd /d "%~dp0"
python import_salaries.py

echo.
pause
