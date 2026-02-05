@echo off
echo.
echo ============================================
echo   OCTA Vessel Analyzer - Build for Windows
echo ============================================
echo.

:: Check if npm is available
npm --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm not found. Please install Node.js first.
    pause
    exit /b 1
)

:: Build frontend
echo [1/3] Building frontend...
cd /d "%~dp0..\frontend"
call npm run build
if errorlevel 1 (
    echo [ERROR] Frontend build failed!
    pause
    exit /b 1
)
echo [OK] Frontend built successfully

:: Package with Electron Builder
echo [2/3] Packaging with Electron Builder...
call npm run electron:build:win
if errorlevel 1 (
    echo [ERROR] Electron packaging failed!
    pause
    exit /b 1
)
echo [OK] Electron package created

:: Copy backend files
echo [3/3] Preparing backend files...
cd /d "%~dp0.."
if not exist "dist-electron\win-unpacked\resources\backend" (
    mkdir "dist-electron\win-unpacked\resources\backend"
)
xcopy /E /Y "backend\*" "dist-electron\win-unpacked\resources\backend\"
echo [OK] Backend files copied

echo.
echo ============================================
echo   Build Complete!
echo ============================================
echo.
echo Output directory: dist-electron\
echo.
echo The installer is located at:
echo   dist-electron\OCTA Vessel Analyzer Setup.exe
echo.
pause
