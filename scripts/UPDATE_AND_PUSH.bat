@echo off
echo ============================================================
echo   NASCAR DFS Hub — Update Data + Push to GitHub
echo ============================================================
echo.

cd /d "%~dp0.."

:: Step 1: Refresh data for all three series
echo [1/3] Fetching latest race data...
echo.
python scripts\refresh_data.py --series cup
python scripts\refresh_data.py --series xfinity
python scripts\refresh_data.py --series truck

:: Step 2: Stage and commit
echo.
echo [2/3] Committing to git...
echo.
git add nascar.db
git commit -m "Data update %date%"

:: Step 3: Push to GitHub (triggers auto-deploy)
echo.
echo [3/3] Pushing to GitHub...
echo.
git push

echo.
echo ============================================================
echo   Done! App will auto-redeploy in ~60 seconds.
echo ============================================================
echo.
pause
