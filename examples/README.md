# 📚 Exemples UWSN PPO

Ce dossier contient des exemples d'utilisation du projet UWSN PPO.

## 🚀 Démarrage rapide

### 1. Test d'installation
```bash
python test_installation.py
```

### 2. Démonstration basique
```bash
python demo.py
```

### 3. Entraînement rapide
```bash
python train_quick.py
```

### 4. Interface web
```bash
streamlit run app/streamlit_app.py
```

## 📊 Exemples de code

### Création d'un réseau simple
```python
from src.utils_network import create_sample_network

# Réseau de 10 nœuds dans une zone de 500m
nodes = create_sample_network(num_nodes=10, area_size=500.0)

# Affichage des nœuds
for node in nodes:
    print(f"Nœud {node.id}: Pos({node.x:.1f}, {node.y:.1f}, {node.z:.1f}) "
          f"Énergie: {node.energy:.1f}J")
```

### Test des modèles physiques
```python
from src.utils_network import AcousticPropagation, EnergyModel

acoustic = AcousticPropagation()
energy_model = EnergyModel()

# Calcul pour deux nœuds
node1, node2 = nodes[0], nodes[1]
distance = node1.distance_to(node2)

# Vitesse du son
sound_speed = acoustic.sound_speed(
    node1.temperature, node1.salinity, node1.z
)

# Consommation énergétique
data_size = 1000  # bits
tx_energy = energy_model.transmission_energy(
    data_size, distance, node1.transmission_power
)
```

### Simulation d'un épisode
```python
from src.env_gym import UWSNRoutingEnv

# Créer l'environnement
env = UWSNRoutingEnv(nodes=nodes, max_steps=50)

# Simulation
obs = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # Action aléatoire
    obs, reward, done, info = env.step(action)
    total_reward += reward

print(f"Récompense totale: {total_reward:.2f}")
print(f"Chemin: {info['episode_stats']['path']}")
```

### Entraînement PPO
```python
from src.ppo_train import UWSNTrainer

# Créer le trainer
trainer = UWSNTrainer(
    num_nodes=15,
    area_size=1000.0,
    depth_range=(-100, -10)
)

# Entraîner
model = trainer.train(total_timesteps=100000)

# Évaluer
metrics = trainer.evaluate(num_episodes=100)
print(f"Taux de succès: {metrics['success_rate']:.2%}")
```

## 🎯 Cas d'usage

### 1. Optimisation de routage
- Minimiser la consommation énergétique
- Respecter les contraintes acoustiques
- Adapter aux conditions environnementales

### 2. Simulation de réseaux
- Tester différentes topologies
- Analyser les performances
- Comparer les algorithmes

### 3. Recherche et développement
- Développer de nouveaux algorithmes
- Valider des modèles physiques
- Optimiser les paramètres

## 🔧 Personnalisation

### Modifier les paramètres physiques
```python
# Dans config.py
PHYSICS_CONFIG = {
    'electronic_energy': 30e-9,  # Réduire la consommation
    'amplification_energy': 0.5e-12,  # Amplificateur plus efficace
    'frequency_range': (15, 35),  # Plage de fréquences différente
}
```

### Adapter la fonction de récompense
```python
# Dans src/env_gym.py, méthode _calculate_reward
def _calculate_reward(self, next_node: int) -> float:
    # Votre logique personnalisée
    reward = (
        -total_energy / 1000.0 +  # Énergie
        -distance / 1000.0 +      # Distance
        100.0 if success else 0.0 # Succès
        # Ajouter vos propres termes...
    )
    return reward
```

### Créer des réseaux personnalisés
```python
# Réseau en grille
def create_grid_network(width, height, spacing):
    nodes = []
    for i in range(width):
        for j in range(height):
            node = Node(
                id=i * height + j,
                x=i * spacing,
                y=j * spacing,
                z=-50.0,  # Profondeur fixe
                energy=1000.0
            )
            nodes.append(node)
    return nodes
```

## 📈 Visualisation

### Graphique 2D simple
```python
import matplotlib.pyplot as plt

# Créer le graphique
fig, ax = plt.subplots(figsize=(10, 8))

# Nœuds
x_coords = [node.x for node in nodes]
y_coords = [node.y for node in nodes]
energies = [node.energy for node in nodes]

scatter = ax.scatter(x_coords, y_coords, c=energies, s=100, alpha=0.7)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Réseau UWSN')

plt.colorbar(scatter, label='Énergie (J)')
plt.show()
```

### Graphique 3D avec Plotly
```python
import plotly.graph_objects as go

fig = go.Figure()

# Ajouter les nœuds
fig.add_trace(go.Scatter3d(
    x=[node.x for node in nodes],
    y=[node.y for node in nodes],
    z=[node.z for node in nodes],
    mode='markers',
    marker=dict(
        size=10,
        color=[node.energy for node in nodes],
        colorscale='viridis'
    )
))

fig.update_layout(
    title="Réseau UWSN 3D",
    scene=dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Profondeur (m)"
    )
)

fig.show()
```

## 🐛 Dépannage

### Problèmes courants

1. **ImportError** : Vérifier que tous les modules sont installés
2. **CUDA** : Utiliser PyTorch CPU-only si pas de GPU
3. **Mémoire** : Réduire la taille du réseau ou du batch
4. **Convergence** : Ajuster les hyperparamètres PPO

### Logs de débogage
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Votre code ici...
```

## 📚 Ressources

- [Documentation Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Documentation Gym](https://gym.openai.com/)
- [Documentation Plotly](https://plotly.com/python/)
- [Documentation Streamlit](https://docs.streamlit.io/)

---

*Pour plus d'exemples, consultez le notebook Google Colab : `notebooks/uwsn_ppo_colab.ipynb`*
