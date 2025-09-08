### Fichier: demo.py

"""
Script de démonstration pour le projet UWSN PPO
"""

import sys
import os
sys.path.append('src')

from src.utils_network import create_sample_network, AcousticPropagation, EnergyModel
from src.env_gym import UWSNRoutingEnv
import numpy as np

def main():
    """Démonstration du projet UWSN PPO"""
    
    print("🌊 Démonstration UWSN PPO - Optimisation de Routage")
    print("=" * 60)
    
    # 1. Création d'un réseau de test
    print("\n1️⃣ Création du réseau de test...")
    nodes = create_sample_network(num_nodes=8, area_size=500.0)
    print(f"✅ Réseau créé avec {len(nodes)} nœuds")
    
    # Affichage des nœuds
    print("\n📋 Informations des nœuds:")
    for node in nodes:
        print(f"   Nœud {node.id}: Pos({node.x:.1f}, {node.y:.1f}, {node.z:.1f}) "
              f"Énergie: {node.energy:.1f}J Temp: {node.temperature:.1f}°C")
    
    # 2. Test des modèles physiques
    print("\n2️⃣ Test des modèles physiques...")
    acoustic = AcousticPropagation()
    energy_model = EnergyModel()
    
    # Test entre deux nœuds
    node1, node2 = nodes[0], nodes[1]
    distance = node1.distance_to(node2)
    
    print(f"   Distance nœud {node1.id} → nœud {node2.id}: {distance:.2f} m")
    
    # Vitesse du son
    sound_speed = acoustic.sound_speed(node1.temperature, node1.salinity, node1.z)
    print(f"   Vitesse du son: {sound_speed:.2f} m/s")
    
    # Perte acoustique
    path_loss = acoustic.path_loss(distance, node1.frequency, node1.temperature, 
                                  node1.salinity, node1.z)
    print(f"   Perte acoustique: {path_loss:.2f} dB")
    
    # Consommation énergétique
    data_size = 1000
    tx_energy = energy_model.transmission_energy(data_size, distance, node1.transmission_power)
    rx_energy = energy_model.reception_energy(data_size)
    total_energy = tx_energy + rx_energy
    
    print(f"   Énergie transmission: {tx_energy*1e6:.2f} μJ")
    print(f"   Énergie réception: {rx_energy*1e6:.2f} μJ")
    print(f"   Énergie totale: {total_energy*1e6:.2f} μJ")
    
    # 3. Test de l'environnement Gym
    print("\n3️⃣ Test de l'environnement Gym...")
    env = UWSNRoutingEnv(nodes=nodes, max_steps=20)
    
    print(f"   Espace d'observation: {env.observation_space.shape}")
    print(f"   Espace d'action: {env.action_space.n}")
    
    # Simulation d'un épisode
    obs = env.reset()
    print(f"   Source: {env.source}, Destination: {env.destination}")
    
    done = False
    step = 0
    total_reward = 0
    
    print("   Simulation de l'épisode:")
    while not done and step < 10:
        action = env.action_space.sample()  # Action aléatoire
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        
        print(f"     Étape {step}: Action {action}, Récompense {reward:.2f}")
        
        if done:
            print(f"     ✅ Épisode terminé! Succès: {action == env.destination}")
            break
    
    print(f"   Récompense totale: {total_reward:.2f}")
    
    # 4. Test de métriques
    print("\n4️⃣ Test des métriques...")
    if 'episode_stats' in info:
        stats = info['episode_stats']
        print(f"   Énergie totale: {stats['total_energy']:.2f} J")
        print(f"   Distance totale: {stats['total_distance']:.2f} m")
        print(f"   Nombre de sauts: {stats['num_hops']}")
        print(f"   Chemin: {stats['path']}")
    
    # 5. Test de visualisation simple
    print("\n5️⃣ Test de visualisation...")
    try:
        import matplotlib.pyplot as plt
        
        # Graphique 2D simple
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Nœuds
        x_coords = [node.x for node in nodes]
        y_coords = [node.y for node in nodes]
        energies = [node.energy for node in nodes]
        
        scatter = ax.scatter(x_coords, y_coords, c=energies, s=100, alpha=0.7, cmap='viridis')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Réseau UWSN - Vue de dessus')
        
        # Couleur bar
        plt.colorbar(scatter, label='Énergie (J)')
        
        # Annotations
        for i, node in enumerate(nodes):
            ax.annotate(f'N{i}', (node.x, node.y), xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig('uwsn_network_demo.png', dpi=150, bbox_inches='tight')
        print("   ✅ Graphique sauvegardé: uwsn_network_demo.png")
        
    except ImportError:
        print("   ⚠️ Matplotlib non disponible, visualisation ignorée")
    
    print("\n🎉 Démonstration terminée avec succès!")
    print("\n📝 Prochaines étapes:")
    print("   1. Lancer l'entraînement: python src/ppo_train.py")
    print("   2. Interface Streamlit: streamlit run app/streamlit_app.py")
    print("   3. Notebook Colab: notebooks/uwsn_ppo_colab.ipynb")

if __name__ == "__main__":
    main()
