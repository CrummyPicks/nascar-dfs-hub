@echo off
echo.
echo ==========================================
echo   NASCAR DFS Hub - LOCAL app
echo ==========================================
echo.
echo Starting the app... your browser will open at
echo http://localhost:8502 in a few seconds.
echo.
echo LEAVE THIS WINDOW OPEN while you use the app.
echo Close it (or press Ctrl+C) to stop the app.
echo.
cd /d "%~dp0"
python -m streamlit run nascar_dfs_app.py --server.port 8502
echo.
echo App stopped.
pause
