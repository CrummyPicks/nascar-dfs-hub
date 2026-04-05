@echo off
echo ============================================
echo   NASCAR DFS Hub - Starting...
echo ============================================
echo.
echo Opening in your browser at http://localhost:8501
echo Press Ctrl+C to stop the app.
echo.

cd /d "%~dp0"
python -m streamlit run nascar_dfs_app.py --server.port 8501
pause
