### Fichier: train_quick.py

"""
Script d'entraînement rapide pour tester le projet UWSN PPO
"""

import sys
import os
sys.path.append('src')

from src.ppo_train import UWSNTrainer
import time

def main():
    """Entraînement rapide pour test"""
    
    print("🚀 Entraînement rapide UWSN PPO")
    print("=" * 40)
    
    # Configuration minimale pour test rapide
    config = {
        'num_nodes': 8,           # Petit réseau
        'area_size': 500.0,       # Zone réduite
        'depth_range': (-50, -10), # Profondeur réduite
        'total_timesteps': 10000, # Entraînement court
        'learning_rate': 3e-4,
        'n_steps': 512,           # Moins de pas par mise à jour
        'batch_size': 32,         # Batch plus petit
        'n_epochs': 5             # Moins d'époques
    }
    
    print(f"📊 Configuration:")
    print(f"   Nœuds: {config['num_nodes']}")
    print(f"   Zone: {config['area_size']}m")
    print(f"   Pas d'entraînement: {config['total_timesteps']}")
    print(f"   Taux d'apprentissage: {config['learning_rate']}")
    
    # Création du trainer
    print("\n🏗️ Création du trainer...")
    trainer = UWSNTrainer(
        num_nodes=config['num_nodes'],
        area_size=config['area_size'],
        depth_range=config['depth_range'],
        model_save_path="models/ppo_uwsn_quick"
    )
    
    # Entraînement
    print("\n🏋️ Début de l'entraînement...")
    start_time = time.time()
    
    try:
        model = trainer.train(
            total_timesteps=config['total_timesteps'],
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            eval_freq=2000,
            save_freq=5000
        )
        
        training_time = time.time() - start_time
        print(f"✅ Entraînement terminé en {training_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        return False
    
    # Évaluation rapide
    print("\n🔍 Évaluation rapide...")
    try:
        metrics = trainer.evaluate(num_episodes=20)
        
        print(f"📈 Résultats:")
        print(f"   Récompense moyenne: {metrics['mean_reward']:.2f}")
        print(f"   Taux de succès: {metrics['success_rate']:.1%}")
        print(f"   Énergie moyenne: {metrics['mean_energy']:.2f} J")
        print(f"   Longueur moyenne: {metrics['mean_length']:.1f} étapes")
        
    except Exception as e:
        print(f"⚠️ Erreur lors de l'évaluation: {e}")
    
    # Test de prédiction
    print("\n🎯 Test de prédiction...")
    try:
        # Créer un environnement de test
        test_env = trainer.create_environment()
        
        # Test sur quelques épisodes
        for episode in range(3):
            obs = test_env.reset()
            done = False
            step = 0
            total_reward = 0
            
            print(f"   Épisode {episode + 1}:")
            print(f"     Source: {test_env.source}, Destination: {test_env.destination}")
            
            while not done and step < 20:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                total_reward += reward
                step += 1
                
                if step <= 5:  # Afficher les 5 premières étapes
                    print(f"       Étape {step}: Action {action}, Récompense {reward:.2f}")
            
            print(f"     Résultat: {'Succès' if done and action == test_env.destination else 'Échec'}")
            print(f"     Récompense totale: {total_reward:.2f}")
            print(f"     Chemin: {info.get('episode_stats', {}).get('path', [])}")
            print()
        
    except Exception as e:
        print(f"⚠️ Erreur lors du test: {e}")
    
    print("🎉 Test d'entraînement terminé!")
    print("\n📝 Prochaines étapes:")
    print("   1. Entraînement complet: python src/ppo_train.py")
    print("   2. Interface Streamlit: streamlit run app/streamlit_app.py")
    print("   3. Notebook Colab: notebooks/uwsn_ppo_colab.ipynb")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
