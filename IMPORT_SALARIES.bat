@echo off
echo.
echo ==============================
echo   NASCAR DFS - Import Salaries
echo ==============================
echo.
echo Drop your DKSalaries.csv in Downloads or this folder,
echo then this script will find it and import it.
echo.

cd /d "%~dp0"
python import_salaries.py --push

echo.
pause
