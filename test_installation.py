### Fichier: test_installation.py

"""
Script de test pour vérifier l'installation et les dépendances
"""

import sys
import importlib
import traceback

def test_import(module_name, package_name=None):
    """Test l'import d'un module"""
    try:
        if package_name:
            module = importlib.import_module(module_name, package_name)
        else:
            module = importlib.import_module(module_name)
        print(f"✅ {module_name}: {module.__version__ if hasattr(module, '__version__') else 'OK'}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False
    except Exception as e:
        print(f"⚠️ {module_name}: {e}")
        return False

def test_project_modules():
    """Test les modules du projet"""
    print("\n🔍 Test des modules du projet...")
    
    modules = [
        'src.utils_network',
        'src.env_gym',
        'src.ppo_train'
    ]
    
    success = True
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}: OK")
        except Exception as e:
            print(f"❌ {module}: {e}")
            success = False
    
    return success

def main():
    """Fonction principale de test"""
    print("🧪 Test d'installation UWSN PPO")
    print("=" * 50)
    
    # Test des dépendances principales
    print("\n📦 Test des dépendances principales...")
    
    dependencies = [
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('gym', 'gym'),
        ('stable_baselines3', 'stable_baselines3'),
        ('matplotlib', 'matplotlib'),
        ('plotly', 'plotly'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('seaborn', 'seaborn'),
        ('streamlit', 'streamlit')
    ]
    
    success_count = 0
    for display_name, module_name in dependencies:
        if test_import(module_name):
            success_count += 1
    
    print(f"\n📊 Dépendances: {success_count}/{len(dependencies)} installées")
    
    # Test des modules du projet
    project_success = test_project_modules()
    
    # Test de l'environnement
    print("\n🌊 Test de l'environnement...")
    try:
        from src.utils_network import create_sample_network, AcousticPropagation, EnergyModel
        from src.env_gym import UWSNRoutingEnv
        
        # Créer un petit réseau de test
        nodes = create_sample_network(num_nodes=5, area_size=200.0)
        print(f"✅ Création réseau: {len(nodes)} nœuds")
        
        # Test des modèles physiques
        acoustic = AcousticPropagation()
        energy_model = EnergyModel()
        print("✅ Modèles physiques: OK")
        
        # Test de l'environnement
        env = UWSNRoutingEnv(nodes=nodes, max_steps=10)
        obs = env.reset()
        print(f"✅ Environnement Gym: {env.observation_space.shape}")
        
        # Test d'une action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"✅ Simulation: Récompense {reward:.2f}")
        
    except Exception as e:
        print(f"❌ Test environnement: {e}")
        traceback.print_exc()
        project_success = False
    
    # Résumé
    print("\n" + "=" * 50)
    if success_count == len(dependencies) and project_success:
        print("🎉 Installation réussie! Le projet est prêt à être utilisé.")
        print("\n📝 Prochaines étapes:")
        print("   1. python demo.py - Démonstration rapide")
        print("   2. streamlit run app/streamlit_app.py - Interface web")
        print("   3. python src/ppo_train.py - Entraînement PPO")
    else:
        print("⚠️ Installation incomplète. Veuillez installer les dépendances manquantes.")
        print("\n🔧 Commandes utiles:")
        print("   pip install -r requirements.txt")
        print("   pip install --upgrade stable-baselines3")
    
    return success_count == len(dependencies) and project_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
