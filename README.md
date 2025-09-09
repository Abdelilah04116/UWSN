# 🌊 Optimisation de Routage UWSN avec PPO (Deep Reinforcement Learning)

Ce projet implémente une solution complète d'optimisation de routage dans les réseaux de capteurs sous-marins (UWSN) en utilisant l'algorithme PPO (Proximal Policy Optimization) de Deep Reinforcement Learning.

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Modèles physiques](#modèles-physiques)
- [Exemples](#exemples)
- [Contributions](#contributions)

## 🎯 Vue d'ensemble

Les réseaux de capteurs sous-marins (UWSN) présentent des défis uniques pour l'optimisation de routage :

- **Communication acoustique** : Propagation complexe avec pertes et absorption
- **Consommation énergétique** : Batteries limitées, recharge difficile
- **Environnement dynamique** : Température, salinité, profondeur variables
- **Contraintes physiques** : Latence élevée, bande passante limitée

Ce projet utilise l'apprentissage par renforcement pour trouver des chemins optimaux qui minimisent la consommation énergétique tout en respectant les contraintes acoustiques.

## ✨ Fonctionnalités

### 🔬 Modèles Physiques Réalistes
- **Propagation acoustique** : Équations de Mackenzie et Francois & Garrison
- **Consommation énergétique** : Modèles basés sur la distance et la puissance
- **Environnement dynamique** : Température, salinité, profondeur variables

### 🤖 Deep Reinforcement Learning
- **Algorithme PPO** : Stable-Baselines3 avec PyTorch
- **Environnement Gym** : Interface standardisée pour l'entraînement
- **Fonction de récompense** : Optimisation multi-objectifs

### 📊 Visualisations Interactives
- **Réseau 3D** : Visualisation Plotly interactive
- **Métriques en temps réel** : Consommation, latence, succès
- **Comparaisons** : PPO vs méthodes de baseline

### 🌐 Interface Utilisateur
- **Streamlit** : Application web interactive
- **Google Colab** : Notebook prêt à l'emploi
- **Configuration flexible** : Paramètres réseau et physiques

## 🚀 Installation

### Prérequis
- Python 3.8+
- PyTorch 2.0+
- CUDA (optionnel, pour l'accélération GPU)

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/votre-username/uwsn-ppo-routing.git
cd uwsn-ppo-routing

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales
```
gym==0.21.0
stable-baselines3==2.0.0
torch==2.0.1
numpy==1.24.3
matplotlib==3.7.1
plotly==5.15.0
streamlit==1.25.0
pandas==2.0.3
scipy==1.11.1
```

## 📖 Utilisation

### 1. Notebook Google Colab

Le moyen le plus rapide de commencer :

1. Ouvrir `notebooks/uwsn_ppo_colab.ipynb` dans Google Colab
2. Exécuter toutes les cellules
3. Suivre les instructions pour l'entraînement et l'évaluation

### 2. Application Streamlit

Interface web interactive :

```bash
# Lancer l'application Streamlit
streamlit run app/streamlit_app.py
```

Fonctionnalités de l'interface :
- Configuration du réseau (nombre de nœuds, zone, profondeur)
- Sélection source/destination
- Chargement de modèles PPO
- Visualisation 3D/2D interactive
- Métriques détaillées

### 3. Entraînement personnalisé

```python
from src.ppo_train import UWSNTrainer

# Créer un trainer
trainer = UWSNTrainer(
    num_nodes=15,
    area_size=1000.0,
    depth_range=(-100, -10)
)

# Entraîner le modèle
model = trainer.train(total_timesteps=200000)

# Évaluer
metrics = trainer.evaluate(num_episodes=100)
```

### 4. Utilisation de l'environnement

```python
from src.env_gym import UWSNRoutingEnv
from src.utils_network import create_sample_network

# Créer un réseau
nodes = create_sample_network(num_nodes=10)

# Créer l'environnement
env = UWSNRoutingEnv(nodes=nodes)

# Simulation
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Action aléatoire
    obs, reward, done, info = env.step(action)
```

## 📁 Structure du projet

```
uwsn-ppo-routing/
├── src/                          # Code source principal
│   ├── __init__.py
│   ├── env_gym.py               # Environnement Gym personnalisé
│   ├── ppo_train.py             # Script d'entraînement PPO
│   └── utils_network.py         # Fonctions utilitaires
├── app/                         # Application Streamlit
│   ├── __init__.py
│   └── streamlit_app.py         # Interface web
├── notebooks/                   # Notebooks Jupyter
│   ├── __init__.py
│   └── uwsn_ppo_colab.ipynb    # Notebook Google Colab
├── models/                      # Modèles entraînés
│   └── ppo_uwsn.zip            # Modèle PPO sauvegardé
├── requirements.txt             # Dépendances Python
└── README.md                   # Documentation
```

## 🔬 Modèles physiques

### Propagation Acoustique

#### Vitesse du son (Équation de Mackenzie)
```
c(T,S,D) = 1448.96 + 4.591*T - 5.304×10⁻²*T² + 2.374×10⁻⁴*T³
           + 1.340*(S-35) + 1.630×10⁻²*D + 1.675×10⁻⁷*D²
           - 1.025×10⁻²*T*(S-35) - 7.139×10⁻¹³*T*D³
```

Où :
- T : Température (°C)
- S : Salinité (PSU)
- D : Profondeur (m)

#### Absorption acoustique (Francois & Garrison)
```
α(f,T,S,D) = A₁*P₁*f₁*f²/(f₁² + f²) + A₂*P₂*f₂*f²/(f₂² + f²) + A₃*P₃*f²
```

#### Perte de trajet
```
TL = 20*log₁₀(d) + α*d/1000
```

### Consommation Énergétique

#### Transmission
```
E_tx = (E_elec + E_amp * d²) * k
```

#### Réception
```
E_rx = E_elec * k
```

Où :
- E_elec = 50 nJ/bit (électronique)
- E_amp = 1 pJ/bit/m² (amplification)
- d : Distance (m)
- k : Taille des données (bits)

## 🎮 Fonction de récompense PPO

La récompense est calculée comme suit :

```python
reward = (
    -total_energy / 1000.0 +           # Minimiser l'énergie
    -distance / 1000.0 +               # Pénalité distance
    100.0 if success else 0.0 +        # Récompense succès
    -5.0 if visited_penalty else 0.0 + # Pénalité boucles
    -10.0 if low_energy else 0.0 +     # Pénalité énergie faible
    proximity_improvement / 100.0      # Récompense proximité
)
```

## 📊 Exemples

### Exemple 1 : Réseau simple

```python
# Créer un réseau de 10 nœuds
nodes = create_sample_network(num_nodes=10, area_size=500.0)

# Créer l'environnement
env = UWSNRoutingEnv(nodes=nodes)

# Simulation d'un épisode
obs = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_reward += reward

print(f"Récompense totale: {total_reward:.2f}")
print(f"Chemin: {info['episode_stats']['path']}")
```

### Exemple 2 : Entraînement rapide

```python
# Configuration minimale pour test
trainer = UWSNTrainer(num_nodes=8, area_size=500.0)

# Entraînement rapide (5 minutes)
model = trainer.train(total_timesteps=50000)

# Évaluation
metrics = trainer.evaluate(num_episodes=50)
print(f"Taux de succès: {metrics['success_rate']:.2%}")
```

### Exemple 3 : Visualisation

```python
import plotly.graph_objects as go

# Créer la visualisation 3D
fig = go.Figure()

# Ajouter les nœuds
for node in nodes:
    fig.add_trace(go.Scatter3d(
        x=[node.x], y=[node.y], z=[node.z],
        mode='markers',
        marker=dict(size=10, color=node.energy),
        text=f"Nœud {node.id}<br>Énergie: {node.energy:.1f}J"
    ))

fig.show()
```

## 🎯 Métriques de performance

### Métriques principales
- **Taux de succès** : Pourcentage d'épisodes réussis
- **Consommation énergétique** : Énergie totale (J)
- **Latence** : Temps de transmission (s)
- **Efficacité** : Énergie par bit transmis

### Comparaisons avec baselines
- **Chemin le plus court** : Dijkstra
- **Politique aléatoire** : Actions aléatoires
- **PPO** : Notre méthode

## 🔧 Configuration avancée

### Paramètres d'entraînement PPO

```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5
)
```

### Paramètres du réseau

```python
# Configuration réseau
num_nodes = 15          # Nombre de nœuds
area_size = 1000.0      # Taille de zone (m)
depth_range = (-100, -10)  # Profondeur (m)
data_size = 1000        # Taille données (bits)
frequency = 25.0        # Fréquence acoustique (kHz)
```

## 🐛 Dépannage

### Problèmes courants

1. **Erreur CUDA** : Installer PyTorch CPU-only
2. **Mémoire insuffisante** : Réduire `num_nodes` ou `batch_size`
3. **Entraînement lent** : Utiliser GPU ou réduire `total_timesteps`

### Logs et débogage

```python
# Activer les logs détaillés
import logging
logging.basicConfig(level=logging.DEBUG)

# Vérifier l'environnement
env = UWSNRoutingEnv(nodes=nodes)
print(f"Espace d'observation: {env.observation_space}")
print(f"Espace d'action: {env.action_space}")
```

## 🤝 Contributions

Les contributions sont les bienvenues ! Voici comment contribuer :

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les changements (`git commit -am 'Ajouter nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

### Idées d'améliorations
- [ ] Support de réseaux dynamiques
- [ ] Intégration de contraintes de QoS
- [ ] Optimisation multi-objectifs
- [ ] Support de protocoles de routage existants
- [ ] Tests unitaires complets

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📚 Références

1. **PPO** : Schulman, J., et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).

2. **Propagation acoustique** : Mackenzie, K. V. "Nine-term equation for sound speed in the oceans." The Journal of the Acoustical Society of America 70.3 (1981): 807-812.

3. **Absorption acoustique** : Francois, R. E., and G. R. Garrison. "Sound absorption based on ocean measurements. Part II: Boric acid contribution and equation for total absorption." The Journal of the Acoustical Society of America 72.6 (1982): 1879-1890.

4. **UWSN** : Akyildiz, I. F., et al. "Underwater acoustic sensor networks: research challenges." Ad hoc networks 3.3 (2005): 257-279.

## 📞 Contact

- **Auteur** : Abdelilah ourti
- **Email** : abdelilahourti@gmail.com

---

⭐ N'hésitez pas à donner une étoile si ce projet vous a aidé !
