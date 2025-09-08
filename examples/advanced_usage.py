### Fichier: examples/advanced_usage.py

"""
Exemples d'utilisation avancée du projet UWSN PPO
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.utils_network import create_sample_network, AcousticPropagation, EnergyModel
from src.env_gym import UWSNRoutingEnv
from src.ppo_train import UWSNTrainer

def example_custom_network():
    """Exemple de création d'un réseau personnalisé"""
    print("🔧 Exemple: Réseau personnalisé")
    
    # Créer un réseau en grille
    nodes = []
    grid_size = 4
    spacing = 100.0
    
    for i in range(grid_size):
        for j in range(grid_size):
            node = Node(
                id=i * grid_size + j,
                x=i * spacing,
                y=j * spacing,
                z=-50.0,  # Profondeur fixe
                energy=1000.0,
                temperature=15.0,
                salinity=35.0,
                frequency=25.0
            )
            nodes.append(node)
    
    print(f"✅ Réseau en grille créé: {len(nodes)} nœuds")
    return nodes

def example_custom_reward():
    """Exemple de fonction de récompense personnalisée"""
    print("🎯 Exemple: Fonction de récompense personnalisée")
    
    class CustomUWSNEnv(UWSNRoutingEnv):
        def _calculate_reward(self, next_node: int) -> float:
            """Fonction de récompense personnalisée"""
            current_node = self.state.current_node
            destination = self.state.destination
            
            # Distance et énergie
            distance = self.nodes[current_node].distance_to(self.nodes[next_node])
            
            # Énergie de transmission
            tx_energy = self.energy_model.transmission_energy(
                self.data_size, distance, self.nodes[current_node].transmission_power
            )
            
            # Énergie de réception
            rx_energy = self.energy_model.reception_energy(self.data_size)
            total_energy = tx_energy + rx_energy
            
            # Récompense personnalisée
            reward = (
                -total_energy / 1000.0 +  # Énergie (minimiser)
                -distance / 1000.0 +      # Distance (minimiser)
                100.0 if next_node == destination else 0.0 +  # Succès
                -10.0 if next_node in self.state.visited_nodes[:-1] else 0.0 +  # Boucles
                -20.0 if self.nodes[next_node].energy < 300 else 0.0 +  # Énergie critique
                self._calculate_qos_reward(next_node)  # QoS personnalisé
            )
            
            return reward
        
        def _calculate_qos_reward(self, next_node: int) -> float:
            """Calcul de récompense basé sur la QoS"""
            # Récompense basée sur la latence
            current_node = self.state.current_node
            distance = self.nodes[current_node].distance_to(self.nodes[next_node])
            
            # Latence estimée
            sound_speed = self.acoustic.sound_speed(
                self.nodes[current_node].temperature,
                self.nodes[current_node].salinity,
                self.nodes[current_node].z
            )
            latency = distance / sound_speed
            
            # Récompense de latence (préférer les faibles latences)
            latency_reward = -latency * 10.0
            
            return latency_reward
    
    return CustomUWSNEnv

def example_advanced_training():
    """Exemple d'entraînement avancé avec callbacks personnalisés"""
    print("🏋️ Exemple: Entraînement avancé")
    
    from stable_baselines3.common.callbacks import BaseCallback
    import matplotlib.pyplot as plt
    
    class TrainingCallback(BaseCallback):
        """Callback personnalisé pour l'entraînement"""
        
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            self.energy_consumptions = []
        
        def _on_step(self) -> bool:
            # Collecter les métriques
            if 'episode' in self.locals['infos'][0]:
                info = self.locals['infos'][0]['episode']
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])
                
                if 'episode_stats' in self.locals['infos'][0]:
                    stats = self.locals['infos'][0]['episode_stats']
                    self.energy_consumptions.append(stats['total_energy'])
            
            return True
        
        def _on_training_end(self) -> None:
            # Créer des graphiques de monitoring
            self._plot_training_curves()
        
        def _plot_training_curves(self):
            """Affiche les courbes d'entraînement"""
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Récompenses
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title('Récompenses par épisode')
            axes[0, 0].set_xlabel('Épisode')
            axes[0, 0].set_ylabel('Récompense')
            
            # Longueurs
            axes[0, 1].plot(self.episode_lengths)
            axes[0, 1].set_title('Longueur des épisodes')
            axes[0, 1].set_xlabel('Épisode')
            axes[0, 1].set_ylabel('Longueur')
            
            # Énergie
            if self.energy_consumptions:
                axes[1, 0].plot(self.energy_consumptions)
                axes[1, 0].set_title('Consommation énergétique')
                axes[1, 0].set_xlabel('Épisode')
                axes[1, 0].set_ylabel('Énergie (J)')
            
            # Taux de succès (approximation)
            if len(self.episode_rewards) > 100:
                window_size = 100
                success_rates = []
                for i in range(window_size, len(self.episode_rewards)):
                    recent_rewards = self.episode_rewards[i-window_size:i]
                    success_rate = sum(1 for r in recent_rewards if r > 50) / len(recent_rewards)
                    success_rates.append(success_rate)
                
                axes[1, 1].plot(success_rates)
                axes[1, 1].set_title('Taux de succès (fenêtre glissante)')
                axes[1, 1].set_xlabel('Épisode')
                axes[1, 1].set_ylabel('Taux de succès')
            
            plt.tight_layout()
            plt.savefig('advanced_training_curves.png', dpi=150, bbox_inches='tight')
            print("📊 Graphiques d'entraînement sauvegardés: advanced_training_curves.png")
    
    # Utilisation du callback
    nodes = create_sample_network(num_nodes=12, area_size=800.0)
    trainer = UWSNTrainer(num_nodes=12, area_size=800.0)
    
    # Entraînement avec callback
    callback = TrainingCallback()
    model = trainer.train(total_timesteps=50000, callback=callback)
    
    return model

def example_network_analysis():
    """Exemple d'analyse approfondie du réseau"""
    print("📊 Exemple: Analyse du réseau")
    
    # Créer un réseau
    nodes = create_sample_network(num_nodes=20, area_size=1000.0)
    
    # Analyser les propriétés du réseau
    distances = []
    energies = []
    temperatures = []
    salinities = []
    
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i != j:
                distances.append(node1.distance_to(node2))
        
        energies.append(node1.energy)
        temperatures.append(node1.temperature)
        salinities.append(node1.salinity)
    
    # Statistiques
    print(f"📈 Statistiques du réseau:")
    print(f"   Distance moyenne: {np.mean(distances):.2f} m")
    print(f"   Distance max: {np.max(distances):.2f} m")
    print(f"   Énergie moyenne: {np.mean(energies):.2f} J")
    print(f"   Température moyenne: {np.mean(temperatures):.2f} °C")
    print(f"   Salinité moyenne: {np.mean(salinities):.2f} PSU")
    
    # Créer des graphiques d'analyse
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribution des distances
    axes[0, 0].hist(distances, bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('Distribution des distances')
    axes[0, 0].set_xlabel('Distance (m)')
    axes[0, 0].set_ylabel('Fréquence')
    
    # Distribution de l'énergie
    axes[0, 1].hist(energies, bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('Distribution de l\'énergie')
    axes[0, 1].set_xlabel('Énergie (J)')
    axes[0, 1].set_ylabel('Fréquence')
    
    # Température vs Salinité
    scatter = axes[1, 0].scatter(temperatures, salinities, c=energies, s=100, alpha=0.7, cmap='viridis')
    axes[1, 0].set_title('Température vs Salinité')
    axes[1, 0].set_xlabel('Température (°C)')
    axes[1, 0].set_ylabel('Salinité (PSU)')
    plt.colorbar(scatter, ax=axes[1, 0], label='Énergie (J)')
    
    # Topologie du réseau
    x_coords = [node.x for node in nodes]
    y_coords = [node.y for node in nodes]
    energies = [node.energy for node in nodes]
    
    scatter = axes[1, 1].scatter(x_coords, y_coords, c=energies, s=100, alpha=0.7, cmap='viridis')
    axes[1, 1].set_title('Topologie du réseau')
    axes[1, 1].set_xlabel('X (m)')
    axes[1, 1].set_ylabel('Y (m)')
    plt.colorbar(scatter, ax=axes[1, 1], label='Énergie (J)')
    
    plt.tight_layout()
    plt.savefig('network_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Analyse du réseau sauvegardée: network_analysis.png")

def example_visualization_3d():
    """Exemple de visualisation 3D avancée"""
    print("🎨 Exemple: Visualisation 3D")
    
    # Créer un réseau
    nodes = create_sample_network(num_nodes=15, area_size=800.0)
    
    # Créer la visualisation 3D
    fig = go.Figure()
    
    # Nœuds
    x_coords = [node.x for node in nodes]
    y_coords = [node.y for node in nodes]
    z_coords = [node.z for node in nodes]
    energies = [node.energy for node in nodes]
    temperatures = [node.temperature for node in nodes]
    
    # Nœuds colorés par énergie
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(
            size=8,
            color=energies,
            colorscale='viridis',
            opacity=0.8,
            colorbar=dict(title="Énergie (J)")
        ),
        text=[f"Nœud {i}<br>Énergie: {e:.1f}J<br>Temp: {t:.1f}°C" 
              for i, e, t in zip(range(len(nodes)), energies, temperatures)],
        hovertemplate='%{text}<extra></extra>',
        name='Nœuds'
    ))
    
    # Connexions (exemple)
    for i in range(0, len(nodes), 3):
        for j in range(i+1, min(i+4, len(nodes))):
            fig.add_trace(go.Scatter3d(
                x=[nodes[i].x, nodes[j].x],
                y=[nodes[i].y, nodes[j].y],
                z=[nodes[i].z, nodes[j].z],
                mode='lines',
                line=dict(color='gray', width=2),
                opacity=0.3,
                showlegend=False
            ))
    
    # Configuration du layout
    fig.update_layout(
        title="Réseau UWSN 3D - Visualisation Avancée",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Profondeur (m)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800
    )
    
    # Sauvegarder
    fig.write_html("uwsn_3d_advanced.html")
    print("🎨 Visualisation 3D sauvegardée: uwsn_3d_advanced.html")

def main():
    """Fonction principale pour les exemples avancés"""
    print("🚀 Exemples d'utilisation avancée UWSN PPO")
    print("=" * 50)
    
    # Exemple 1: Réseau personnalisé
    print("\n1️⃣ Réseau personnalisé")
    custom_nodes = example_custom_network()
    
    # Exemple 2: Fonction de récompense personnalisée
    print("\n2️⃣ Fonction de récompense personnalisée")
    CustomEnv = example_custom_reward()
    
    # Exemple 3: Entraînement avancé
    print("\n3️⃣ Entraînement avancé")
    try:
        model = example_advanced_training()
    except Exception as e:
        print(f"⚠️ Erreur lors de l'entraînement: {e}")
    
    # Exemple 4: Analyse du réseau
    print("\n4️⃣ Analyse du réseau")
    example_network_analysis()
    
    # Exemple 5: Visualisation 3D
    print("\n5️⃣ Visualisation 3D")
    example_visualization_3d()
    
    print("\n🎉 Exemples avancés terminés!")
    print("\n📁 Fichiers générés:")
    print("   - advanced_training_curves.png")
    print("   - network_analysis.png")
    print("   - uwsn_3d_advanced.html")

if __name__ == "__main__":
    main()
