@echo off
REM setup.bat — Audio-to-Score setup for Windows

echo ======================================
echo   Audio-to-Score Setup
echo ======================================
echo.

REM Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found. Install from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo Using: %PYVER%
echo.

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ======================================
echo   Optional dependencies
echo ======================================

echo.
set /p INSTALL_CAIRO="Install cairosvg for better PDF rendering? (y/N): "
if /i "%INSTALL_CAIRO%"=="y" (
    pip install cairosvg
)

echo.
set /p INSTALL_DEMUCS="Install demucs for source separation? (y/N): "
if /i "%INSTALL_DEMUCS%"=="y" (
    pip install demucs
)

echo.
echo ======================================
echo   Setup complete!
echo ======================================
echo.
echo Activate the environment with:
echo   venv\Scripts\activate.bat
echo.
echo Usage:
echo   python transcribe.py song.mp3
echo   python transcribe.py song.wav --instrument piano
echo   python transcribe.py *.mp3 --output .\scores
echo.
pause
