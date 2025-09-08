@echo off
echo 🌊 Lancement de l'application UWSN PPO Streamlit...
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python n'est pas installé ou pas dans le PATH
    pause
    exit /b 1
)

REM Vérifier si Streamlit est installé
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installation de Streamlit...
    pip install streamlit
)

REM Lancer l'application
echo 🚀 Lancement de l'application...
streamlit run app/streamlit_app.py

pause
