@echo off
echo ============================================
echo   NASCAR DFS Hub - First Time Setup
echo ============================================
echo.
echo Installing required Python packages...
echo.

python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy plotly requests beautifulsoup4 lxml openpyxl

echo.
echo ============================================
echo   Setup complete!
echo   Run START_APP.bat to launch the app.
echo ============================================
pause
