### Fichier: tests/test_physics.py

"""
Tests unitaires pour les modèles physiques
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
from src.utils_network import AcousticPropagation, EnergyModel, Node

class TestAcousticPropagation(unittest.TestCase):
    """Tests pour la propagation acoustique"""
    
    def setUp(self):
        self.acoustic = AcousticPropagation()
    
    def test_sound_speed(self):
        """Test de la vitesse du son"""
        # Conditions typiques
        T, S, D = 15.0, 35.0, 50.0
        
        speed = self.acoustic.sound_speed(T, S, D)
        
        # Vérifier que la vitesse est dans une plage raisonnable
        self.assertGreater(speed, 1400)
        self.assertLess(speed, 1600)
    
    def test_absorption_coefficient(self):
        """Test du coefficient d'absorption"""
        f, T, S, D = 25.0, 15.0, 35.0, 50.0
        
        alpha = self.acoustic.absorption_coefficient(f, T, S, D)
        
        # Vérifier que l'absorption est positive
        self.assertGreater(alpha, 0)
        self.assertLess(alpha, 100)  # Plage raisonnable
    
    def test_path_loss(self):
        """Test de la perte de trajet"""
        d, f, T, S, D = 100.0, 25.0, 15.0, 35.0, 50.0
        
        loss = self.acoustic.path_loss(d, f, T, S, D)
        
        # Vérifier que la perte est positive et croissante avec la distance
        self.assertGreater(loss, 0)
        
        # Test avec distance plus grande
        loss2 = self.acoustic.path_loss(d*2, f, T, S, D)
        self.assertGreater(loss2, loss)

class TestEnergyModel(unittest.TestCase):
    """Tests pour le modèle énergétique"""
    
    def setUp(self):
        self.energy_model = EnergyModel()
    
    def test_transmission_energy(self):
        """Test de l'énergie de transmission"""
        data_size, distance, power = 1000, 100.0, 1.0
        
        energy = self.energy_model.transmission_energy(data_size, distance, power)
        
        # Vérifier que l'énergie est positive
        self.assertGreater(energy, 0)
        
        # Vérifier que l'énergie croît avec la distance
        energy2 = self.energy_model.transmission_energy(data_size, distance*2, power)
        self.assertGreater(energy2, energy)
    
    def test_reception_energy(self):
        """Test de l'énergie de réception"""
        data_size = 1000
        
        energy = self.energy_model.reception_energy(data_size)
        
        # Vérifier que l'énergie est positive
        self.assertGreater(energy, 0)
        
        # Vérifier que l'énergie croît avec la taille des données
        energy2 = self.energy_model.reception_energy(data_size*2)
        self.assertGreater(energy2, energy)

class TestNode(unittest.TestCase):
    """Tests pour la classe Node"""
    
    def test_node_creation(self):
        """Test de création d'un nœud"""
        node = Node(
            id=0,
            x=100.0,
            y=200.0,
            z=-50.0,
            energy=500.0
        )
        
        self.assertEqual(node.id, 0)
        self.assertEqual(node.x, 100.0)
        self.assertEqual(node.y, 200.0)
        self.assertEqual(node.z, -50.0)
        self.assertEqual(node.energy, 500.0)
    
    def test_distance_calculation(self):
        """Test du calcul de distance"""
        node1 = Node(0, 0, 0, 0, 1000)
        node2 = Node(1, 3, 4, 0, 1000)
        
        distance = node1.distance_to(node2)
        
        # Distance 3D: sqrt(3² + 4² + 0²) = 5
        self.assertAlmostEqual(distance, 5.0, places=5)
    
    def test_is_alive(self):
        """Test de la méthode is_alive"""
        node_alive = Node(0, 0, 0, 0, 500.0)
        node_dead = Node(1, 0, 0, 0, 0.0)
        
        self.assertTrue(node_alive.is_alive())
        self.assertFalse(node_dead.is_alive())

if __name__ == '__main__':
    unittest.main()
