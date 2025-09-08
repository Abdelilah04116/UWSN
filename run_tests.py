### Fichier: run_tests.py

"""
Script pour exécuter tous les tests
"""

import sys
import os
import unittest

# Ajouter le chemin src
sys.path.append('src')

def run_tests():
    """Exécute tous les tests"""
    
    print("🧪 Exécution des tests UWSN PPO")
    print("=" * 40)
    
    # Découvrir et exécuter les tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Résumé
    print("\n" + "=" * 40)
    if result.wasSuccessful():
        print("🎉 Tous les tests ont réussi!")
        return True
    else:
        print(f"❌ {len(result.failures)} test(s) ont échoué")
        print(f"❌ {len(result.errors)} erreur(s) détectée(s)")
        return False

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
