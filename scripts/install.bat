@echo off
echo.
echo ============================================
echo   OCTA Vessel Analyzer - Installation
echo ============================================
echo.

:: Check Python
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python is installed

:: Check Node.js
echo [2/4] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Please install Node.js 18+
    echo Download: https://nodejs.org/
    pause
    exit /b 1
)
echo [OK] Node.js is installed

:: Install Python dependencies
echo [3/4] Installing Python dependencies...
cd /d "%~dp0..\backend"
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some Python packages may not be installed correctly
)
echo [OK] Python dependencies installed

:: Install Node.js dependencies
echo [4/4] Installing Node.js dependencies...
cd /d "%~dp0..\frontend"
call npm install
if errorlevel 1 (
    echo [WARNING] Some Node.js packages may not be installed correctly
)
echo [OK] Node.js dependencies installed

:: Create data directories
echo.
echo Creating data directories...
cd /d "%~dp0..\backend"
if not exist "data" mkdir "data"
if not exist "data\uploads" mkdir "data\uploads"
if not exist "data\processed" mkdir "data\processed"
if not exist "data\annotations" mkdir "data\annotations"
if not exist "data\models" mkdir "data\models"
if not exist "data\results" mkdir "data\results"
echo [OK] Data directories created

echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo Usage:
echo   1. Run start.bat to launch the application
echo   2. Or run dev-start.bat for development mode
echo.
echo See INSTALL.md for more information
echo.
pause
