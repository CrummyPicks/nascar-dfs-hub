@echo off
echo.
echo ==========================================
echo   NASCAR DFS - Import DraftKings Salaries
echo ==========================================
echo.
echo Download DKSalaries CSVs from DraftKings,
echo then run this to import all series at once.
echo.

cd /d "%~dp0"
python import_salaries.py

echo.
pause
