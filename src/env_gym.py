### Fichier: src/env_gym.py

"""
Environnement Gym personnalisé pour l'optimisation de routage
dans les réseaux de capteurs sous-marins (UWSN)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import random
from dataclasses import dataclass

from .utils_network import Node, AcousticPropagation, EnergyModel, calculate_network_metrics


@dataclass
class UWSNState:
    """État du réseau UWSN"""
    current_node: int
    destination: int
    data_size: int
    nodes_energy: np.ndarray
    nodes_position: np.ndarray
    nodes_temperature: np.ndarray
    nodes_salinity: np.ndarray
    visited_nodes: List[int]
    path: List[int]


class UWSNRoutingEnv(gym.Env):
    """
    Environnement Gym pour l'optimisation de routage UWSN
    
    L'agent doit trouver le chemin optimal entre un nœud source et un nœud destination
    en minimisant la consommation énergétique et en tenant compte des contraintes acoustiques.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, nodes: List[Node], max_steps: int = 50, 
                 data_size_range: Tuple[int, int] = (500, 2000)):
        """
        Initialise l'environnement UWSN
        
        Args:
            nodes: Liste des nœuds du réseau
            max_steps: Nombre maximum d'étapes par épisode
            data_size_range: Plage de taille de données (bits)
        """
        super(UWSNRoutingEnv, self).__init__()
        
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.max_steps = max_steps
        self.data_size_range = data_size_range
        
        # Modèles physiques
        self.acoustic = AcousticPropagation()
        self.energy_model = EnergyModel()
        
        # Espaces d'observation et d'action
        self.observation_space = self._create_observation_space()
        self.action_space = spaces.Discrete(self.num_nodes)
        
        # État actuel
        self.state: Optional[UWSNState] = None
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Statistiques
        self.episode_stats = {
            'total_energy': 0.0,
            'total_distance': 0.0,
            'num_hops': 0,
            'success': False,
            'path': []
        }
    
    def _create_observation_space(self) -> spaces.Box:
        """Crée l'espace d'observation"""
        # Observation: [current_node, destination, data_size, nodes_energy, 
        #               nodes_position, nodes_temperature, nodes_salinity, 
        #               visited_mask, path_length]
        obs_dim = (
            3 +  # current_node, destination, data_size (normalisés)
            self.num_nodes +  # nodes_energy
            self.num_nodes * 3 +  # nodes_position (x, y, z)
            self.num_nodes +  # nodes_temperature
            self.num_nodes +  # nodes_salinity
            self.num_nodes +  # visited_mask
            1  # path_length
        )
        
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement pour un nouvel épisode"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Sélection aléatoire de source et destination
        self.source = random.randint(0, self.num_nodes - 1)
        self.destination = random.randint(0, self.num_nodes - 1)
        
        # Éviter que source == destination
        while self.destination == self.source:
            self.destination = random.randint(0, self.num_nodes - 1)
        
        # Taille de données aléatoire
        self.data_size = random.randint(*self.data_size_range)
        
        # Initialisation de l'état
        self.state = UWSNState(
            current_node=self.source,
            destination=self.destination,
            data_size=self.data_size,
            nodes_energy=self._get_nodes_energy(),
            nodes_position=self._get_nodes_position(),
            nodes_temperature=self._get_nodes_temperature(),
            nodes_salinity=self._get_nodes_salinity(),
            visited_nodes=[self.source],
            path=[self.source]
        )
        
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Réinitialisation des statistiques
        self.episode_stats = {
            'total_energy': 0.0,
            'total_distance': 0.0,
            'num_hops': 0,
            'success': False,
            'path': [self.source]
        }
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Exécute une action dans l'environnement
        
        Args:
            action: ID du nœud de destination choisi
        
        Returns:
            observation: Nouvel état observé
            reward: Récompense obtenue
            terminated: Si l'épisode est terminé (succès/échec)
            truncated: Si l'épisode est tronqué (limite de temps)
            info: Informations supplémentaires
        """
        if self.state is None:
            raise ValueError("L'environnement doit être réinitialisé avant step()")
        
        self.step_count += 1
        
        # Vérification de la validité de l'action
        if action < 0 or action >= self.num_nodes:
            return self._get_observation(), -100.0, True, False, {"error": "Action invalide"}
        
        if action == self.state.current_node:
            return self._get_observation(), -10.0, False, False, {"error": "Même nœud"}
        
        if not self.nodes[action].is_alive():
            return self._get_observation(), -50.0, False, False, {"error": "Nœud mort"}
        
        # Calcul de la récompense
        reward = self._calculate_reward(action)
        
        # Mise à jour de l'état
        self.state.current_node = action
        self.state.visited_nodes.append(action)
        self.state.path.append(action)
        
        # Vérification de la fin d'épisode
        terminated = self._is_done()
        truncated = self.step_count >= self.max_steps
        
        if terminated:
            self.episode_stats['success'] = (action == self.destination)
            self.episode_stats['path'] = self.state.path.copy()
        
        # Informations supplémentaires
        info = self._get_info()
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _calculate_reward(self, next_node: int) -> float:
        """Calcule la récompense pour l'action choisie"""
        current_node = self.state.current_node
        destination = self.state.destination
        
        # Distance et énergie
        distance = self.nodes[current_node].distance_to(self.nodes[next_node])
        
        # Énergie de transmission
        tx_energy = self.energy_model.transmission_energy(
            self.data_size, distance, self.nodes[current_node].transmission_power
        )
        
        # Énergie de réception
        rx_energy = self.energy_model.reception_energy(self.data_size)
        
        total_energy = tx_energy + rx_energy
        
        # Mise à jour des statistiques
        self.episode_stats['total_energy'] += total_energy
        self.episode_stats['total_distance'] += distance
        self.episode_stats['num_hops'] += 1
        
        # Récompense basée sur l'énergie (négative pour minimiser)
        energy_reward = -total_energy / 1000.0  # Normalisation
        
        # Récompense de distance (pénalité pour les longs trajets)
        distance_penalty = -distance / 1000.0
        
        # Récompense de succès (arrivée à destination)
        success_reward = 100.0 if next_node == destination else 0.0
        
        # Pénalité pour les nœuds déjà visités (éviter les boucles)
        visited_penalty = -5.0 if next_node in self.state.visited_nodes[:-1] else 0.0
        
        # Pénalité pour les nœuds avec peu d'énergie
        energy_penalty = -10.0 if self.nodes[next_node].energy < 200 else 0.0
        
        # Pénalité pour les étapes trop longues
        step_penalty = -1.0 if self.step_count > self.max_steps * 0.8 else 0.0
        
        # Récompense de proximité à la destination
        current_to_dest = self.nodes[current_node].distance_to(self.nodes[destination])
        next_to_dest = self.nodes[next_node].distance_to(self.nodes[destination])
        proximity_reward = (current_to_dest - next_to_dest) / 100.0
        
        total_reward = (energy_reward + distance_penalty + success_reward + 
                       visited_penalty + energy_penalty + step_penalty + proximity_reward)
        
        return total_reward
    
    def _is_done(self) -> bool:
        """Vérifie si l'épisode est terminé (succès/échec)"""
        # Succès : arrivée à destination
        if self.state.current_node == self.destination:
            return True
        
        # Échec : nœud actuel mort
        if not self.nodes[self.state.current_node].is_alive():
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Retourne l'observation actuelle"""
        if self.state is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Normalisation des valeurs
        obs = []
        
        # Nœud actuel et destination (normalisés)
        obs.append(self.state.current_node / self.num_nodes)
        obs.append(self.state.destination / self.num_nodes)
        obs.append(self.state.data_size / max(self.data_size_range))
        
        # Énergie des nœuds (normalisée)
        obs.extend(self.state.nodes_energy / 1000.0)
        
        # Position des nœuds (normalisée)
        pos = self.state.nodes_position
        obs.extend(pos.flatten() / 1000.0)
        
        # Température des nœuds (normalisée)
        obs.extend(self.state.nodes_temperature / 30.0)
        
        # Salinité des nœuds (normalisée)
        obs.extend(self.state.nodes_salinity / 40.0)
        
        # Masque des nœuds visités
        visited_mask = np.zeros(self.num_nodes)
        for node_id in self.state.visited_nodes:
            visited_mask[node_id] = 1.0
        obs.extend(visited_mask)
        
        # Longueur du chemin (normalisée)
        obs.append(len(self.state.path) / self.max_steps)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_nodes_energy(self) -> np.ndarray:
        """Retourne l'énergie de tous les nœuds"""
        return np.array([node.energy for node in self.nodes], dtype=np.float32)
    
    def _get_nodes_position(self) -> np.ndarray:
        """Retourne la position de tous les nœuds"""
        positions = []
        for node in self.nodes:
            positions.append([node.x, node.y, node.z])
        return np.array(positions, dtype=np.float32)
    
    def _get_nodes_temperature(self) -> np.ndarray:
        """Retourne la température de tous les nœuds"""
        return np.array([node.temperature for node in self.nodes], dtype=np.float32)
    
    def _get_nodes_salinity(self) -> np.ndarray:
        """Retourne la salinité de tous les nœuds"""
        return np.array([node.salinity for node in self.nodes], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Retourne les informations supplémentaires"""
        return {
            'step': self.step_count,
            'current_node': self.state.current_node,
            'destination': self.state.destination,
            'path_length': len(self.state.path),
            'visited_nodes': self.state.visited_nodes.copy(),
            'episode_stats': self.episode_stats.copy()
        }
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Rendu de l'environnement"""
        if mode == 'human':
            print(f"Étape {self.step_count}: Nœud actuel {self.state.current_node}, "
                  f"Destination {self.state.destination}, "
                  f"Chemin: {self.state.path}")
        elif mode == 'rgb_array':
            # Pour la visualisation graphique
            return None
    
    def get_network_info(self) -> Dict[str, Any]:
        """Retourne les informations du réseau"""
        return {
            'num_nodes': self.num_nodes,
            'nodes': [
                {
                    'id': node.id,
                    'position': [node.x, node.y, node.z],
                    'energy': node.energy,
                    'temperature': node.temperature,
                    'salinity': node.salinity
                }
                for node in self.nodes
            ]
        }
    
    def get_optimal_path(self) -> List[int]:
        """Retourne le chemin optimal (Dijkstra simplifié)"""
        from .utils_network import find_shortest_path
        return find_shortest_path(self.nodes, self.source, self.destination)
    
    def get_path_metrics(self, path: List[int]) -> Dict[str, float]:
        """Calcule les métriques pour un chemin donné"""
        return calculate_network_metrics(self.nodes, path, self.data_size)
