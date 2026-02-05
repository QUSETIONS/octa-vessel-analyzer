@echo off
chcp 437 >nul 2>&1
setlocal EnableDelayedExpansion

echo.
echo ============================================
echo   OCTA Vessel Analyzer - Smart Launcher
echo ============================================
echo.

:: Set paths
set "ROOT_DIR=%~dp0.."
set "BACKEND_DIR=%ROOT_DIR%\backend"
set "FRONTEND_DIR=%ROOT_DIR%\frontend"
set "VENV_PYTHON=%BACKEND_DIR%\.venv\Scripts\python.exe"

:: ============================================
:: Step 1: Check virtual environment
:: ============================================
echo [1/6] Checking virtual environment...

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment not found!
    echo Please run install.bat first.
    echo.
    pause
    exit /b 1
)
echo [OK] Virtual environment found

:: ============================================
:: Step 2: Clean up old processes
:: ============================================
echo [2/6] Cleaning up old processes...

:: Kill processes on port 8000
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: Kill processes on port 5173
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr :5173 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo [OK] Ports cleared

:: ============================================
:: Step 3: Start backend server
:: ============================================
echo [3/6] Starting backend server...

cd /d "%BACKEND_DIR%"
start "OCTA Backend" cmd /c ""%VENV_PYTHON%" -m api.main"

:: ============================================
:: Step 4: Wait for backend
:: ============================================
echo [4/6] Waiting for backend to start...

set "BACKEND_READY=0"
set "MAX_WAIT=30"
set "WAIT_COUNT=0"

:wait_backend
if %WAIT_COUNT% GEQ %MAX_WAIT% (
    echo [ERROR] Backend failed to start after %MAX_WAIT% seconds
    pause
    exit /b 1
)

curl -s http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "BACKEND_READY=1"
    goto backend_ready
)

set /a WAIT_COUNT+=1
echo     Waiting... (%WAIT_COUNT%/%MAX_WAIT%)
timeout /t 1 /nobreak >nul
goto wait_backend

:backend_ready
echo [OK] Backend is running on http://localhost:8000

:: ============================================
:: Step 5: Start frontend (Vite + Electron)
:: ============================================
echo [5/6] Starting frontend...

cd /d "%FRONTEND_DIR%"
start "OCTA Frontend" cmd /c "npm run dev"

:: ============================================
:: Step 6: Wait for Vite and start Electron
:: ============================================
echo [6/6] Waiting for Vite to start...

set "VITE_READY=0"
set "WAIT_COUNT=0"

:wait_vite
if %WAIT_COUNT% GEQ %MAX_WAIT% (
    echo [ERROR] Vite failed to start after %MAX_WAIT% seconds
    pause
    exit /b 1
)

curl -s http://localhost:5173 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "VITE_READY=1"
    goto vite_ready
)

set /a WAIT_COUNT+=1
echo     Waiting... (%WAIT_COUNT%/%MAX_WAIT%)
timeout /t 1 /nobreak >nul
goto wait_vite

:vite_ready
echo [OK] Vite is running on http://localhost:5173

:: Start Electron
echo.
echo Starting Electron app...
cd /d "%FRONTEND_DIR%"
start "OCTA Electron" cmd /c "npx electron ."

echo.
echo ============================================
echo   Application Started Successfully!
echo ============================================
echo.
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo   API Docs: http://localhost:8000/docs
echo.
echo   You can also open the frontend in browser:
echo   http://localhost:5173
echo.
echo   Press any key to exit this launcher...
echo   (Application windows will keep running)
echo.
pause >nul
