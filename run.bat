@echo off
title AI Email Spam Classifier
color 0A

echo.
echo  ================================================
echo   AI Email Spam Classifier - Starting Up...
echo  ================================================
echo.

:: Move to script's directory (so it works from anywhere)
cd /d "%~dp0"

:: Step 1 - Create virtual environment if it doesn't exist
if not exist "venv\Scripts\activate.bat" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo         Make sure Python 3.10+ is installed.
        pause
        exit /b 1
    )
    echo       Done.
) else (
    echo [1/3] Virtual environment already exists. Skipping.
)

:: Step 2 - Install/update dependencies
echo [2/3] Installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo       Done.

:: Step 3 - Train model if not already trained
echo [3/3] Checking model...
if not exist "model\spam_model.pkl" (
    echo       Model not found. Training now (this may take a minute)...
    python train_model.py
    if errorlevel 1 (
        echo [ERROR] Model training failed.
        pause
        exit /b 1
    )
    echo       Model trained successfully.
) else (
    echo       Model already trained. Skipping.
)

:: Launch
echo.
echo  ================================================
echo   Launching app at http://127.0.0.1:5000
echo   Press CTRL+C to stop the server.
echo  ================================================
echo.

python app.py

pause
