@echo off
echo.
echo ==========================================
echo   NASCAR DFS - Import DraftKings Salaries
echo ==========================================
echo.
echo Download DKSalaries CSV from DraftKings for each series.
echo This script will find the most recent CSV in Downloads.
echo.

cd /d "%~dp0"

echo Which series? (you can run this multiple times for each)
echo   [1] Cup
echo   [2] Xfinity
echo   [3] Truck
echo.
set /p SERIES="Select series (1/2/3): "

if "%SERIES%"=="1" set SNAME=cup
if "%SERIES%"=="2" set SNAME=xfinity
if "%SERIES%"=="3" set SNAME=truck
if "%SNAME%"=="" set SNAME=cup

echo.
python import_salaries.py --series %SNAME% --push

echo.
echo ==========================================
echo   Run again for another series? (close to exit)
echo ==========================================
echo.
pause
