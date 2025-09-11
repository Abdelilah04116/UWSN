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
        'num_nodes': 15,           # Taille de réseau réaliste
        'area_size': 800.0,       # Zone un peu plus grande
        'depth_range': (-100, -20), # Plage de profondeur
        'total_timesteps': 100000, # Entraînement plus long pour meilleure perf
        'learning_rate': 3e-4,
        'n_steps': 2048,          # Plus de pas par update (meilleure utilisation GPU)
        'batch_size': 512,        # Batch conséquent pour stabilité
        'n_epochs': 10            # Davantage d'époques par update
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
            eval_freq=5000,  # évalue et log périodiquement
            save_freq=20000  # checkpoints périodiques
        )
        
        training_time = time.time() - start_time
        print(f"✅ Entraînement terminé en {training_time:.1f}s")
        
        # Export du meilleur modèle (évalué) sous un nom pratique
        best_dir = "models/ppo_uwsn_quick_best"
        best_ckpt = os.path.join(best_dir, "best_model.zip")
        export_path = "models/ppo_uwsn_quick_best.zip"
        try:
            if os.path.exists(best_ckpt):
                import shutil
                shutil.copyfile(best_ckpt, export_path)
                print(f"💾 Meilleur modèle copié vers: {export_path}")
            else:
                print("ℹ️ Aucun best_model.zip trouvé (évaluation trop rare ou non atteinte).")
        except Exception as e:
            print(f"⚠️ Impossible de copier le meilleur modèle: {e}")
        
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
    
    print("🎉 Test d'entraînement terminé!")
    print("\n📝 Prochaines étapes:")
    print("   1. Entraînement complet: python -m src.ppo_train")
    print("   2. Interface Streamlit: streamlit run app/streamlit_app.py")
    print("   3. Notebook Colab: notebooks/uwsn_ppo_colab.ipynb")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
