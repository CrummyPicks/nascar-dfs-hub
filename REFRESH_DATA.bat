@echo off
echo ============================================================
echo   NASCAR DFS Hub — Data Refresh
echo ============================================================
echo.
echo Usage:
echo   - Just press Enter to fetch all new Cup 2026 races
echo   - Or close this and run manually with flags:
echo       python refresh_data.py --series xfinity
echo       python refresh_data.py --year 2025
echo       python refresh_data.py --all
echo       python refresh_data.py --race 5596
echo.
echo ============================================================
echo.
python refresh_data.py
echo.
pause
