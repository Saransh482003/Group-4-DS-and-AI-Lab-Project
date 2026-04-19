#!/bin/bash
echo "=============================================================="
echo "DSAI Navigation Assistant - Linux/Mac Quick Setup"
echo "=============================================================="

if [ ! -f .env ]; then
    echo "[INFO] Creating .env from .env.example..."
    cp .env.example .env
fi

if [ ! -d "env" ]; then
    echo "[INFO] Creating Python virtual environment..."
    python3 -m venv env
fi

echo "[INFO] Activating virtual environment and installing dependencies..."
source env/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements-hf.txt

echo ""
echo "[SUCCESS] Setup complete! Starting Streamlit app..."
echo "=============================================================="
streamlit run app/streamlit_app.py
