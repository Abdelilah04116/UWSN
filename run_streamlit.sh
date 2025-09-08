#!/bin/bash

echo "🌊 Lancement de l'application UWSN PPO Streamlit..."
echo

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé"
    exit 1
fi

# Vérifier si Streamlit est installé
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "📦 Installation de Streamlit..."
    pip3 install streamlit
fi

# Lancer l'application
echo "🚀 Lancement de l'application..."
streamlit run app/streamlit_app.py
