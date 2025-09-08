### Fichier: app/streamlit_app.py

"""
Application Streamlit pour l'optimisation de routage UWSN avec PPO
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import sys
from typing import List, Dict, Any, Tuple
import time

# Ajout du chemin src pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.env_gym import UWSNRoutingEnv
from src.utils_network import create_sample_network, Node, AcousticPropagation, EnergyModel
from src.ppo_train import UWSNTrainer

# Configuration de la page
st.set_page_config(
    page_title="UWSN Routing Optimization",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        color: #28a745;
        font-weight: bold;
    }
    .warning-metric {
        color: #ffc107;
        font-weight: bold;
    }
    .danger-metric {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def load_model(model_path: str):
    """Charge un modèle PPO entraîné"""
    try:
        from stable_baselines3 import PPO
        
        # Vérifier si le fichier existe
        if not os.path.exists(model_path):
            st.error(f"Fichier modèle non trouvé: {model_path}")
            return None
        
        # Charger le modèle
        model = PPO.load(model_path)
        
        # Afficher les informations du modèle
        st.success(f"✅ Modèle chargé: {model_path}")
        st.info(f"📊 Modèle PPO avec {model.policy} et {model.learning_rate} learning rate")
        
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

def create_network_visualization(nodes: List[Node], path: List[int] = None, 
                                source: int = None, destination: int = None) -> go.Figure:
    """Crée une visualisation 3D du réseau UWSN"""
    
    # Préparation des données
    node_ids = [node.id for node in nodes]
    x_coords = [node.x for node in nodes]
    y_coords = [node.y for node in nodes]
    z_coords = [node.z for node in nodes]
    energies = [node.energy for node in nodes]
    temperatures = [node.temperature for node in nodes]
    
    # Couleurs basées sur l'énergie
    colors = []
    for energy in energies:
        if energy > 700:
            colors.append('green')
        elif energy > 400:
            colors.append('orange')
        else:
            colors.append('red')
    
    # Taille des nœuds basée sur l'énergie
    sizes = [max(5, energy / 50) for energy in energies]
    
    # Création du graphique 3D
    fig = go.Figure()
    
    # Ajout des nœuds
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8,
            line=dict(width=2, color='black')
        ),
        text=[f"Nœud {i}<br>Énergie: {e:.1f}J<br>Temp: {t:.1f}°C" 
              for i, e, t in zip(node_ids, energies, temperatures)],
        hovertemplate='%{text}<extra></extra>',
        name='Nœuds'
    ))
    
    # Mise en évidence du nœud source
    if source is not None:
        fig.add_trace(go.Scatter3d(
            x=[nodes[source].x],
            y=[nodes[source].y],
            z=[nodes[source].z],
            mode='markers',
            marker=dict(
                size=15,
                color='blue',
                symbol='diamond'
            ),
            name='Source',
            hovertemplate=f'Source: Nœud {source}<extra></extra>'
        ))
    
    # Mise en évidence du nœud destination
    if destination is not None:
        fig.add_trace(go.Scatter3d(
            x=[nodes[destination].x],
            y=[nodes[destination].y],
            z=[nodes[destination].z],
            mode='markers',
            marker=dict(
                size=15,
                color='purple',
                symbol='diamond'
            ),
            name='Destination',
            hovertemplate=f'Destination: Nœud {destination}<extra></extra>'
        ))
    
    # Ajout du chemin optimal
    if path and len(path) > 1:
        path_x = [nodes[i].x for i in path]
        path_y = [nodes[i].y for i in path]
        path_z = [nodes[i].z for i in path]
        
        fig.add_trace(go.Scatter3d(
            x=path_x,
            y=path_y,
            z=path_z,
            mode='lines+markers',
            line=dict(color='red', width=6),
            marker=dict(size=8, color='red'),
            name='Chemin optimal',
            hovertemplate='Chemin optimal<extra></extra>'
        ))
    
    # Configuration du layout
    fig.update_layout(
        title="Réseau de Capteurs Sous-marins (UWSN)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Profondeur (m)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600
    )
    
    return fig

def create_2d_visualization(nodes: List[Node], path: List[int] = None,
                           source: int = None, destination: int = None) -> go.Figure:
    """Crée une visualisation 2D du réseau UWSN (vue de dessus)"""
    
    # Préparation des données
    x_coords = [node.x for node in nodes]
    y_coords = [node.y for node in nodes]
    energies = [node.energy for node in nodes]
    node_ids = [node.id for node in nodes]
    
    # Couleurs basées sur l'énergie
    colors = []
    for energy in energies:
        if energy > 700:
            colors.append('green')
        elif energy > 400:
            colors.append('orange')
        else:
            colors.append('red')
    
    # Taille des nœuds
    sizes = [max(10, energy / 30) for energy in energies]
    
    # Création du graphique 2D
    fig = go.Figure()
    
    # Ajout des nœuds
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8,
            line=dict(width=2, color='black')
        ),
        text=[f"Nœud {i}<br>Énergie: {e:.1f}J" 
              for i, e in zip(node_ids, energies)],
        hovertemplate='%{text}<extra></extra>',
        name='Nœuds'
    ))
    
    # Mise en évidence du nœud source
    if source is not None:
        fig.add_trace(go.Scatter(
            x=[nodes[source].x],
            y=[nodes[source].y],
            mode='markers',
            marker=dict(
                size=20,
                color='blue',
                symbol='diamond'
            ),
            name='Source',
            hovertemplate=f'Source: Nœud {source}<extra></extra>'
        ))
    
    # Mise en évidence du nœud destination
    if destination is not None:
        fig.add_trace(go.Scatter(
            x=[nodes[destination].x],
            y=[nodes[destination].y],
            mode='markers',
            marker=dict(
                size=20,
                color='purple',
                symbol='star'
            ),
            name='Destination',
            hovertemplate=f'Destination: Nœud {destination}<extra></extra>'
        ))
    
    # Ajout du chemin optimal
    if path and len(path) > 1:
        path_x = [nodes[i].x for i in path]
        path_y = [nodes[i].y for i in path]
        
        fig.add_trace(go.Scatter(
            x=path_x,
            y=path_y,
            mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=10, color='red'),
            name='Chemin optimal',
            hovertemplate='Chemin optimal<extra></extra>'
        ))
    
    # Configuration du layout
    fig.update_layout(
        title="Vue de dessus du réseau UWSN",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        width=800,
        height=600,
        showlegend=True
    )
    
    return fig

def calculate_network_metrics_detailed(nodes: List[Node], path: List[int], 
                                     data_size: int) -> Dict[str, Any]:
    """Calcule des métriques détaillées pour un chemin"""
    if len(path) < 2:
        return {
            'total_energy': 0,
            'total_distance': 0,
            'num_hops': 0,
            'latency': 0,
            'success_rate': 0,
            'energy_per_hop': [],
            'distances': [],
            'acoustic_losses': []
        }
    
    acoustic = AcousticPropagation()
    energy_model = EnergyModel()
    
    total_energy = 0
    total_distance = 0
    total_latency = 0
    energy_per_hop = []
    distances = []
    acoustic_losses = []
    
    for i in range(len(path) - 1):
        current_node = nodes[path[i]]
        next_node = nodes[path[i + 1]]
        
        # Distance
        distance = current_node.distance_to(next_node)
        distances.append(distance)
        total_distance += distance
        
        # Énergie
        tx_energy = energy_model.transmission_energy(
            data_size, distance, current_node.transmission_power
        )
        rx_energy = energy_model.reception_energy(data_size)
        hop_energy = tx_energy + rx_energy
        energy_per_hop.append(hop_energy)
        total_energy += hop_energy
        
        # Latence
        sound_speed = acoustic.sound_speed(
            current_node.temperature, current_node.salinity, current_node.z
        )
        latency = distance / sound_speed
        total_latency += latency
        
        # Perte acoustique
        acoustic_loss = acoustic.path_loss(
            distance, current_node.frequency, current_node.temperature,
            current_node.salinity, current_node.z
        )
        acoustic_losses.append(acoustic_loss)
    
    # Taux de succès basé sur l'énergie
    success_rate = 1.0
    for node_id in path:
        if nodes[node_id].energy < 100:
            success_rate *= 0.5
    
    return {
        'total_energy': total_energy,
        'total_distance': total_distance,
        'num_hops': len(path) - 1,
        'latency': total_latency,
        'success_rate': success_rate,
        'energy_per_hop': energy_per_hop,
        'distances': distances,
        'acoustic_losses': acoustic_losses
    }

def main():
    """Fonction principale de l'application Streamlit"""
    
    # En-tête principal
    st.markdown('<h1 class="main-header">🌊 Optimisation de Routage UWSN avec PPO</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar pour la configuration
    st.sidebar.header("⚙️ Configuration du Réseau")
    
    # Paramètres du réseau
    num_nodes = st.sidebar.slider("Nombre de nœuds", 5, 30, 15)
    area_size = st.sidebar.slider("Taille de la zone (m)", 500, 2000, 1000)
    depth_min = st.sidebar.slider("Profondeur minimale (m)", -200, -50, -100)
    depth_max = st.sidebar.slider("Profondeur maximale (m)", -50, -5, -10)
    
    # Paramètres de transmission
    st.sidebar.header("📡 Paramètres de Transmission")
    data_size = st.sidebar.slider("Taille des données (bits)", 100, 5000, 1000)
    frequency = st.sidebar.slider("Fréquence acoustique (kHz)", 10, 50, 25)
    
    # Bouton pour générer un nouveau réseau
    if st.sidebar.button("🔄 Générer un nouveau réseau"):
        st.session_state.network_generated = False
    
    # Génération du réseau
    if 'network_generated' not in st.session_state or not st.session_state.network_generated:
        with st.spinner("Génération du réseau UWSN..."):
            nodes = create_sample_network(
                num_nodes=num_nodes,
                area_size=area_size,
                depth_range=(depth_min, depth_max)
            )
            
            # Mise à jour des fréquences
            for node in nodes:
                node.frequency = frequency
            
            st.session_state.nodes = nodes
            st.session_state.network_generated = True
    
    nodes = st.session_state.nodes
    
    # Sélection source et destination
    st.sidebar.header("🎯 Sélection du Chemin")
    node_ids = [node.id for node in nodes]
    source = st.sidebar.selectbox("Nœud source", node_ids, index=0)
    destination = st.sidebar.selectbox("Nœud destination", node_ids, index=min(1, len(node_ids)-1))
    
    # Chargement du modèle
    st.sidebar.header("🤖 Modèle PPO")
    
    # Détection automatique du modèle
    model_candidates = [
        "models/ppo_uwsn_final.zip",
        "models/ppo_uwsn_colab.zip", 
        "models/ppo_uwsn.zip"
    ]
    
    available_models = [path for path in model_candidates if os.path.exists(path)]
    
    if available_models:
        model_path = st.sidebar.selectbox(
            "Modèle disponible", 
            available_models,
            index=0
        )
        st.sidebar.success(f"✅ {len(available_models)} modèle(s) trouvé(s)")
    else:
        model_path = st.sidebar.text_input("Chemin du modèle", "models/ppo_uwsn.zip")
        st.sidebar.warning("⚠️ Aucun modèle trouvé dans le dossier models/")
    
    if st.sidebar.button("📥 Charger le modèle"):
        model = load_model(model_path)
        if model:
            st.session_state.model = model
            st.sidebar.success("Modèle chargé avec succès!")
        else:
            st.sidebar.error("Erreur lors du chargement du modèle")
    
    # Chargement automatique du premier modèle disponible
    if 'model' not in st.session_state and available_models:
        with st.spinner("Chargement automatique du modèle..."):
            model = load_model(available_models[0])
            if model:
                st.session_state.model = model
                st.success(f"✅ Modèle chargé automatiquement: {available_models[0]}")
    
    # Interface principale
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 Visualisation", "🤖 Prédiction PPO", "📊 Métriques", "⚙️ Configuration Avancée"])
    
    with tab1:
        st.header("Visualisation du Réseau")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vue 3D")
            fig_3d = create_network_visualization(nodes, source=source, destination=destination)
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            st.subheader("Vue 2D (dessus)")
            fig_2d = create_network_visualization(nodes, source=source, destination=destination)
            st.plotly_chart(fig_2d, use_container_width=True)
        
        # Informations sur les nœuds
        st.subheader("📋 Informations des Nœuds")
        node_data = []
        for node in nodes:
            node_data.append({
                'ID': node.id,
                'Position X (m)': f"{node.x:.1f}",
                'Position Y (m)': f"{node.y:.1f}",
                'Profondeur (m)': f"{node.z:.1f}",
                'Énergie (J)': f"{node.energy:.1f}",
                'Température (°C)': f"{node.temperature:.1f}",
                'Salinité (PSU)': f"{node.salinity:.1f}",
                'Fréquence (kHz)': f"{node.frequency:.1f}"
            })
        
        df_nodes = pd.DataFrame(node_data)
        st.dataframe(df_nodes, use_container_width=True)
    
    with tab2:
        st.header("Prédiction avec PPO")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord charger un modèle PPO dans la sidebar")
        else:
            model = st.session_state.model
            
            if st.button("🚀 Lancer la prédiction PPO"):
                with st.spinner("Calcul du chemin optimal..."):
                    # Création de l'environnement
                    env = UWSNRoutingEnv(nodes=nodes, max_steps=50)
                    env.source = source
                    env.destination = destination
                    env.data_size = data_size
                    
                    # Simulation avec le modèle PPO
                    obs = env.reset()
                    done = False
                    path = [source]
                    step_count = 0
                    
                    while not done and step_count < 50:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, info = env.step(action)
                        path.append(action)
                        step_count += 1
                    
                    st.session_state.ppo_path = path
                    st.session_state.ppo_metrics = calculate_network_metrics_detailed(nodes, path, data_size)
            
            if 'ppo_path' in st.session_state:
                st.success("✅ Prédiction PPO terminée!")
                
                # Affichage du chemin
                st.subheader("🛤️ Chemin PPO")
                path_str = " → ".join([f"Nœud {i}" for i in st.session_state.ppo_path])
                st.write(f"**Chemin:** {path_str}")
                
                # Visualisation du chemin
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_3d_path = create_network_visualization(
                        nodes, path=st.session_state.ppo_path, 
                        source=source, destination=destination
                    )
                    st.plotly_chart(fig_3d_path, use_container_width=True)
                
                with col2:
                    fig_2d_path = create_network_visualization(
                        nodes, path=st.session_state.ppo_path,
                        source=source, destination=destination
                    )
                    st.plotly_chart(fig_2d_path, use_container_width=True)
    
    with tab3:
        st.header("Métriques et Analyses")
        
        if 'ppo_metrics' in st.session_state:
            metrics = st.session_state.ppo_metrics
            
            # Métriques principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Énergie Totale",
                    f"{metrics['total_energy']:.2f} J",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Distance Totale",
                    f"{metrics['total_distance']:.1f} m",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Nombre de Sauts",
                    f"{metrics['num_hops']}",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "Latence",
                    f"{metrics['latency']:.3f} s",
                    delta=None
                )
            
            # Graphiques détaillés
            st.subheader("📈 Analyse Détaillée")
            
            if metrics['energy_per_hop']:
                # Énergie par saut
                fig_energy = go.Figure()
                fig_energy.add_trace(go.Bar(
                    x=[f"Saut {i+1}" for i in range(len(metrics['energy_per_hop']))],
                    y=metrics['energy_per_hop'],
                    name="Énergie par saut",
                    marker_color='lightblue'
                ))
                fig_energy.update_layout(
                    title="Énergie consommée par saut",
                    xaxis_title="Saut",
                    yaxis_title="Énergie (J)"
                )
                st.plotly_chart(fig_energy, use_container_width=True)
            
            if metrics['distances']:
                # Distance par saut
                fig_distance = go.Figure()
                fig_distance.add_trace(go.Bar(
                    x=[f"Saut {i+1}" for i in range(len(metrics['distances']))],
                    y=metrics['distances'],
                    name="Distance par saut",
                    marker_color='lightgreen'
                ))
                fig_distance.update_layout(
                    title="Distance parcourue par saut",
                    xaxis_title="Saut",
                    yaxis_title="Distance (m)"
                )
                st.plotly_chart(fig_distance, use_container_width=True)
            
            if metrics['acoustic_losses']:
                # Perte acoustique par saut
                fig_acoustic = go.Figure()
                fig_acoustic.add_trace(go.Bar(
                    x=[f"Saut {i+1}" for i in range(len(metrics['acoustic_losses']))],
                    y=metrics['acoustic_losses'],
                    name="Perte acoustique par saut",
                    marker_color='lightcoral'
                ))
                fig_acoustic.update_layout(
                    title="Perte acoustique par saut",
                    xaxis_title="Saut",
                    yaxis_title="Perte (dB)"
                )
                st.plotly_chart(fig_acoustic, use_container_width=True)
        
        else:
            st.info("ℹ️ Lancez d'abord une prédiction PPO pour voir les métriques")
    
    with tab4:
        st.header("Configuration Avancée")
        
        st.subheader("🔧 Paramètres Physiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Modèle de Propagation Acoustique**")
            st.write("- Vitesse du son: 1500 m/s (base)")
            st.write("- Équation de Mackenzie pour la vitesse")
            st.write("- Modèle Francois & Garrison pour l'absorption")
            st.write("- Perte de trajet: 20*log10(d) + α*d/1000")
        
        with col2:
            st.write("**Modèle Énergétique**")
            st.write("- Énergie électronique: 50 nJ/bit")
            st.write("- Énergie d'amplification: 1 pJ/bit/m²")
            st.write("- Énergie de réception: 50 nJ/bit")
            st.write("- Énergie en veille: 1 μJ/s")
        
        st.subheader("🎯 Fonction de Récompense PPO")
        st.write("La récompense est calculée comme suit:")
        st.write("- **Énergie**: -énergie_totale/1000 (minimiser)")
        st.write("- **Distance**: -distance/1000 (pénalité)")
        st.write("- **Succès**: +100 si destination atteinte")
        st.write("- **Boucles**: -5 pour nœuds déjà visités")
        st.write("- **Énergie faible**: -10 pour nœuds < 200J")
        st.write("- **Proximité**: +proximité_amélioration/100")
        
        st.subheader("📊 Métriques de Performance")
        st.write("- **Taux de succès**: % d'épisodes réussis")
        st.write("- **Consommation énergétique**: Énergie totale (J)")
        st.write("- **Latence**: Temps total de transmission (s)")
        st.write("- **Efficacité**: Énergie par bit transmis")

if __name__ == "__main__":
    main()
