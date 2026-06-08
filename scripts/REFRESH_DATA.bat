@echo off
echo ============================================================
echo   NASCAR DFS Hub — Data Refresh
echo ============================================================
echo.
echo NOTE: data refresh runs AUTOMATICALLY every day via GitHub
echo Actions. You normally do NOT need this. Use it only for a
echo manual one-off pull.
echo.
echo Usage:
echo   - Just press Enter to fetch all new Cup 2026 races
echo   - Or close this and run manually with flags:
echo       python scripts\refresh_data.py --series xfinity
echo       python scripts\refresh_data.py --year 2025
echo       python scripts\refresh_data.py --all
echo       python scripts\refresh_data.py --race 5596
echo.
echo ============================================================
echo.
cd /d "%~dp0.."
python scripts\refresh_data.py
echo.
pause
