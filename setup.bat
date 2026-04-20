@echo off
echo ==============================================================
echo DSAI Navigation Assistant - Windows Quick Setup
echo ==============================================================

if not exist .env (
    echo [INFO] Creating .env from .env.example...
    copy .env.example .env
)

if not exist env\ (
    echo [INFO] Creating Python virtual environment...
    python -m venv env
)

echo [INFO] Activating virtual environment and installing dependencies...
call env\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-hf.txt

echo.
echo [SUCCESS] Setup complete!
echo ==============================================================
python run_public_ui.py
