@echo off
echo ============================================
echo   NASCAR DFS Hub - First Time Setup
echo ============================================
echo.
echo Installing required Python packages...
echo.

cd /d "%~dp0.."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ============================================
echo   Setup complete!
echo   Run scripts\START_APP.bat to launch the app.
echo ============================================
pause
