### Fichier: examples/simple_demo.py

"""
Démonstration simple du projet UWSN PPO
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils_network import create_sample_network, AcousticPropagation, EnergyModel
from src.env_gym import UWSNRoutingEnv
import matplotlib.pyplot as plt
import numpy as np

def main():
    """Démonstration simple"""
    
    print("🌊 Démonstration Simple UWSN PPO")
    print("=" * 40)
    
    # 1. Création du réseau
    print("\n1️⃣ Création du réseau...")
    nodes = create_sample_network(num_nodes=6, area_size=300.0)
    print(f"✅ Réseau créé avec {len(nodes)} nœuds")
    
    # 2. Test des modèles physiques
    print("\n2️⃣ Test des modèles physiques...")
    acoustic = AcousticPropagation()
    energy_model = EnergyModel()
    
    # Test entre deux nœuds
    node1, node2 = nodes[0], nodes[1]
    distance = node1.distance_to(node2)
    
    print(f"   Distance: {distance:.1f} m")
    
    # Vitesse du son
    sound_speed = acoustic.sound_speed(
        node1.temperature, node1.salinity, node1.z
    )
    print(f"   Vitesse du son: {sound_speed:.1f} m/s")
    
    # Consommation énergétique
    data_size = 1000
    tx_energy = energy_model.transmission_energy(
        data_size, distance, node1.transmission_power
    )
    rx_energy = energy_model.reception_energy(data_size)
    total_energy = tx_energy + rx_energy
    
    print(f"   Énergie totale: {total_energy*1e6:.2f} μJ")
    
    # 3. Simulation d'un épisode
    print("\n3️⃣ Simulation d'un épisode...")
    env = UWSNRoutingEnv(nodes=nodes, max_steps=20)
    
    obs = env.reset()
    print(f"   Source: {env.source}, Destination: {env.destination}")
    
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < 15:
        action = env.action_space.sample()  # Action aléatoire
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        
        print(f"     Étape {step}: Action {action}, Récompense {reward:.2f}")
        
        if done:
            success = action == env.destination
            print(f"     ✅ Épisode terminé! Succès: {success}")
            break
    
    print(f"   Récompense totale: {total_reward:.2f}")
    
    # 4. Visualisation simple
    print("\n4️⃣ Création de la visualisation...")
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Nœuds
        x_coords = [node.x for node in nodes]
        y_coords = [node.y for node in nodes]
        energies = [node.energy for node in nodes]
        
        scatter = ax.scatter(x_coords, y_coords, c=energies, s=150, alpha=0.7, cmap='viridis')
        
        # Annotations
        for i, node in enumerate(nodes):
            ax.annotate(f'N{i}', (node.x, node.y), xytext=(5, 5), textcoords='offset points')
        
        # Source et destination
        ax.scatter(nodes[env.source].x, nodes[env.source].y, s=200, c='blue', marker='s', label='Source')
        ax.scatter(nodes[env.destination].x, nodes[env.destination].y, s=200, c='red', marker='^', label='Destination')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Réseau UWSN - Vue de dessus')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Couleur bar
        plt.colorbar(scatter, label='Énergie (J)')
        
        plt.tight_layout()
        plt.savefig('uwsn_simple_demo.png', dpi=150, bbox_inches='tight')
        print("   ✅ Graphique sauvegardé: uwsn_simple_demo.png")
        
    except Exception as e:
        print(f"   ⚠️ Erreur visualisation: {e}")
    
    print("\n🎉 Démonstration terminée!")
    print("\n📝 Prochaines étapes:")
    print("   - python train_quick.py (entraînement rapide)")
    print("   - streamlit run app/streamlit_app.py (interface web)")
    print("   - notebooks/uwsn_ppo_colab.ipynb (notebook complet)")

if __name__ == "__main__":
    main()
