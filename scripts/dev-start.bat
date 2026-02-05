@echo off
echo.
echo ============================================
echo   OCTA Vessel Analyzer - Development Mode
echo ============================================
echo.

:: Start backend in new window
echo Starting backend server...
cd /d "%~dp0..\backend"
start "Backend Server" cmd /k "python -m api.main"

:: Wait for backend
echo Waiting for backend...
timeout /t 3 /nobreak >nul

:: Start frontend dev server in new window
echo Starting frontend dev server...
cd /d "%~dp0..\frontend"
start "Frontend Dev" cmd /k "npm run dev"

echo.
echo ============================================
echo   Development servers started!
echo ============================================
echo.
echo Backend API: http://localhost:8000
echo Frontend:    http://localhost:5173
echo API Docs:    http://localhost:8000/docs
echo.
echo Press any key to exit this window...
echo (Servers will continue running in their own windows)
echo.
pause
