@echo off
echo.
echo ==============================
echo   NASCAR DFS - Push Code + DB
echo ==============================
echo.
echo This pushes all code changes AND the database
echo to ensure salary data is never lost on deploy.
echo.

cd /d "%~dp0"

git add -A
git add nascar.db
git status
echo.
set /p MSG="Commit message: "
git commit -m "%MSG%"
git push

echo.
echo Done! Code + DB deployed to Streamlit Cloud.
echo.
pause
