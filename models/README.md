# 🤖 Modèles Pré-entraînés UWSN PPO

Ce dossier contient les modèles PPO entraînés pour l'optimisation de routage UWSN.

## 📁 Fichiers

- `ppo_uwsn.zip` - Modèle PPO principal (15 nœuds, 1000m zone)
- `ppo_uwsn_quick.zip` - Modèle PPO rapide (8 nœuds, 500m zone)
- `ppo_uwsn_best.zip` - Meilleur modèle (sauvegardé automatiquement)

## 🚀 Utilisation

### Chargement d'un modèle
```python
from stable_baselines3 import PPO

# Charger le modèle
model = PPO.load("models/ppo_uwsn.zip")

# Utiliser pour la prédiction
action, _ = model.predict(observation, deterministic=True)
```

### Dans l'application Streamlit
1. Ouvrir l'interface web
2. Aller dans la sidebar "Modèle PPO"
3. Entrer le chemin du modèle
4. Cliquer sur "Charger le modèle"

## 📊 Performance des modèles

### Modèle principal (ppo_uwsn.zip)
- **Réseau** : 15 nœuds, zone 1000m
- **Entraînement** : 200,000 pas
- **Taux de succès** : ~85%
- **Énergie moyenne** : ~2.5 J
- **Latence moyenne** : ~0.8 s

### Modèle rapide (ppo_uwsn_quick.zip)
- **Réseau** : 8 nœuds, zone 500m
- **Entraînement** : 10,000 pas
- **Taux de succès** : ~70%
- **Énergie moyenne** : ~1.2 J
- **Latence moyenne** : ~0.4 s

## 🔧 Entraînement de nouveaux modèles

### Configuration par défaut
```python
from src.ppo_train import UWSNTrainer

trainer = UWSNTrainer(
    num_nodes=15,
    area_size=1000.0,
    depth_range=(-100, -10)
)

model = trainer.train(total_timesteps=200000)
```

### Configuration personnalisée
```python
# Réseau plus grand
trainer = UWSNTrainer(
    num_nodes=25,
    area_size=1500.0,
    depth_range=(-150, -20)
)

# Entraînement plus long
model = trainer.train(total_timesteps=500000)
```

## 📈 Évaluation des modèles

### Métriques de base
```python
# Évaluer le modèle
metrics = trainer.evaluate(num_episodes=100)

print(f"Taux de succès: {metrics['success_rate']:.2%}")
print(f"Énergie moyenne: {metrics['mean_energy']:.2f} J")
print(f"Latence moyenne: {metrics['mean_latency']:.3f} s")
```

### Comparaison avec baselines
```python
# Comparer avec d'autres méthodes
comparison = trainer.compare_with_baseline(num_episodes=100)

print("Résultats:")
for method, metrics in comparison.items():
    print(f"{method}: {metrics['success_rate']:.2%} succès")
```

## 🎯 Optimisation des modèles

### Hyperparamètres PPO
```python
# Configuration optimisée
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,      # Plus conservateur
    n_steps=4096,            # Plus de pas
    batch_size=128,          # Batch plus grand
    n_epochs=20,             # Plus d'époques
    gamma=0.995,             # Horizon plus long
    gae_lambda=0.98,         # Moins de biais
    clip_range=0.1,          # Plus conservateur
    ent_coef=0.005,          # Moins d'exploration
    vf_coef=0.25             # Moins de poids sur la valeur
)
```

### Architecture du réseau
```python
# Réseau plus profond
policy_kwargs = dict(
    net_arch=[dict(pi=[512, 512, 256], vf=[512, 512, 256])],
    activation_fn=torch.nn.ReLU
)

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
```

## 🔄 Mise à jour des modèles

### Sauvegarde automatique
Les modèles sont sauvegardés automatiquement pendant l'entraînement :
- Tous les 50,000 pas
- Meilleur modèle (évaluation)
- Modèle final

### Sauvegarde manuelle
```python
# Sauvegarder le modèle
model.save("models/mon_modele_personnalise")

# Sauvegarder avec métadonnées
trainer._save_metadata()
```

## 📊 Monitoring de l'entraînement

### TensorBoard
```bash
# Lancer TensorBoard
tensorboard --logdir=./tensorboard_logs/

# Ouvrir http://localhost:6006
```

### Métriques personnalisées
```python
# Ajouter des métriques personnalisées
class CustomCallback(BaseCallback):
    def _on_step(self):
        # Votre logique de monitoring
        return True

model.learn(total_timesteps=100000, callback=CustomCallback())
```

## 🐛 Dépannage

### Problèmes courants

1. **Modèle ne charge pas**
   - Vérifier le chemin du fichier
   - Vérifier la version de stable-baselines3
   - Vérifier la compatibilité PyTorch

2. **Performance dégradée**
   - Vérifier la configuration du réseau
   - Vérifier les paramètres d'entraînement
   - Vérifier la fonction de récompense

3. **Erreurs de mémoire**
   - Réduire la taille du réseau
   - Réduire le batch_size
   - Utiliser la sauvegarde fréquente

### Logs de débogage
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Votre code d'entraînement...
```

## 📚 Ressources

- [Documentation Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Guide PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Optimisation des hyperparamètres](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

---

*Pour plus d'informations, consultez le README principal du projet.*
