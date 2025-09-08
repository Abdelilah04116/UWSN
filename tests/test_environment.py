### Fichier: tests/test_environment.py

"""
Tests unitaires pour l'environnement Gym
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
from src.env_gym import UWSNRoutingEnv
from src.utils_network import create_sample_network

class TestUWSNRoutingEnv(unittest.TestCase):
    """Tests pour l'environnement UWSN"""
    
    def setUp(self):
        self.nodes = create_sample_network(num_nodes=5, area_size=200.0)
        self.env = UWSNRoutingEnv(nodes=self.nodes, max_steps=20)
    
    def test_environment_creation(self):
        """Test de création de l'environnement"""
        self.assertEqual(self.env.num_nodes, 5)
        self.assertEqual(self.env.max_steps, 20)
        self.assertEqual(self.env.action_space.n, 5)
    
    def test_reset(self):
        """Test de la réinitialisation"""
        obs = self.env.reset()
        
        # Vérifier que l'observation a la bonne forme
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        
        # Vérifier que source != destination
        self.assertNotEqual(self.env.source, self.env.destination)
    
    def test_step(self):
        """Test d'une étape"""
        obs = self.env.reset()
        
        # Action valide
        action = 1
        obs, reward, done, info = self.env.step(action)
        
        # Vérifier les types de retour
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
    
    def test_invalid_action(self):
        """Test d'une action invalide"""
        obs = self.env.reset()
        
        # Action invalide (hors limites)
        action = 10  # Plus grand que num_nodes
        obs, reward, done, info = self.env.step(action)
        
        # Devrait retourner une pénalité et terminer
        self.assertLess(reward, 0)
        self.assertTrue(done)
    
    def test_episode_termination(self):
        """Test de la fin d'épisode"""
        obs = self.env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 25:  # Plus que max_steps
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            step_count += 1
        
        # L'épisode devrait se terminer
        self.assertTrue(done)
    
    def test_observation_space(self):
        """Test de l'espace d'observation"""
        obs_space = self.env.observation_space
        
        # Vérifier que c'est un Box
        self.assertEqual(obs_space.__class__.__name__, 'Box')
        
        # Vérifier la forme
        expected_dim = (3 +  # current_node, destination, data_size
                       5 +  # nodes_energy
                       5 * 3 +  # nodes_position
                       5 +  # nodes_temperature
                       5 +  # nodes_salinity
                       5 +  # visited_mask
                       1)  # path_length
        
        self.assertEqual(obs_space.shape[0], expected_dim)
    
    def test_action_space(self):
        """Test de l'espace d'action"""
        action_space = self.env.action_space
        
        # Vérifier que c'est un Discrete
        self.assertEqual(action_space.__class__.__name__, 'Discrete')
        self.assertEqual(action_space.n, 5)

if __name__ == '__main__':
    unittest.main()
