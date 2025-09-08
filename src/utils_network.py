### Fichier: src/utils_network.py

"""
Fonctions utilitaires pour les calculs de propagation acoustique et énergétique
dans les réseaux de capteurs sous-marins (UWSN)
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Node:
    """Représente un nœud du réseau sous-marin"""
    id: int
    x: float  # Position x (m)
    y: float  # Position y (m)
    z: float  # Profondeur (m, négative)
    energy: float  # Énergie restante (J)
    temperature: float = 15.0  # Température (°C)
    salinity: float = 35.0  # Salinité (PSU)
    max_energy: float = 1000.0  # Énergie maximale (J)
    transmission_power: float = 1.0  # Puissance de transmission (W)
    frequency: float = 25.0  # Fréquence acoustique (kHz)
    
    def is_alive(self) -> bool:
        """Vérifie si le nœud a encore de l'énergie"""
        return self.energy > 0
    
    def distance_to(self, other: 'Node') -> float:
        """Calcule la distance 3D vers un autre nœud"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


class AcousticPropagation:
    """
    Classe pour calculer la propagation acoustique sous-marine
    Basée sur les modèles physiques réalistes
    """
    
    def __init__(self):
        # Constantes physiques
        self.c0 = 1500.0  # Vitesse du son dans l'eau (m/s)
        self.rho0 = 1025.0  # Densité de l'eau (kg/m³)
        
    def sound_speed(self, temperature: float, salinity: float, depth: float) -> float:
        """
        Calcule la vitesse du son selon l'équation de Mackenzie
        c(T,S,D) = 1448.96 + 4.591*T - 5.304*10^-2*T² + 2.374*10^-4*T³
                   + 1.340*(S-35) + 1.630*10^-2*D + 1.675*10^-7*D²
                   - 1.025*10^-2*T*(S-35) - 7.139*10^-13*T*D³
        """
        T = temperature
        S = salinity
        D = abs(depth)  # Profondeur positive
        
        c = (1448.96 + 4.591*T - 5.304e-2*T**2 + 2.374e-4*T**3 +
             1.340*(S-35) + 1.630e-2*D + 1.675e-7*D**2 -
             1.025e-2*T*(S-35) - 7.139e-13*T*D**3)
        
        return c
    
    def absorption_coefficient(self, frequency: float, temperature: float, 
                             salinity: float, depth: float, ph: float = 8.0) -> float:
        """
        Calcule le coefficient d'absorption acoustique selon Francois & Garrison
        α(f,T,S,D,pH) = A1*P1*f1*f²/(f1² + f²) + A2*P2*f2*f²/(f2² + f²) + A3*P3*f²
        
        où:
        - f: fréquence (kHz)
        - T: température (°C)
        - S: salinité (PSU)
        - D: profondeur (m)
        - pH: pH de l'eau
        """
        f = frequency
        T = temperature
        S = salinity
        D = abs(depth)
        
        # Calcul de la pression (Pa)
        P = 1.01325e5 + 1.025e4 * D
        
        # Coefficients pour l'absorption
        A1 = 0.106 * np.exp((T - 26) / 9)
        A2 = 0.52 * (1 + T / 43) * (S / 35)
        A3 = 0.00049 * np.exp(-(T / 27 + D / 17))
        
        f1 = 0.78 * np.sqrt(S / 35) * np.exp(T / 26)
        f2 = 42 * np.exp(T / 17)
        
        P1 = 1
        P2 = 1 - 1.37e-4 * D + 6.2e-9 * D**2
        P3 = 1 - 3.84e-4 * D + 7.57e-8 * D**2
        
        # Coefficient d'absorption (dB/km)
        alpha = (A1 * P1 * f1 * f**2 / (f1**2 + f**2) +
                A2 * P2 * f2 * f**2 / (f2**2 + f**2) +
                A3 * P3 * f**2)
        
        return alpha
    
    def path_loss(self, distance: float, frequency: float, temperature: float,
                  salinity: float, depth: float) -> float:
        """
        Calcule la perte de trajet acoustique (dB)
        TL = 20*log10(d) + α*d/1000
        
        où:
        - d: distance (m)
        - α: coefficient d'absorption (dB/km)
        """
        # Perte de trajet géométrique (dB)
        geometric_loss = 20 * math.log10(distance)
        
        # Perte d'absorption (dB)
        alpha = self.absorption_coefficient(frequency, temperature, salinity, depth)
        absorption_loss = alpha * distance / 1000
        
        return geometric_loss + absorption_loss
    
    def received_power(self, transmitted_power: float, distance: float, 
                      frequency: float, temperature: float, salinity: float, 
                      depth: float) -> float:
        """
        Calcule la puissance reçue (W)
        Pr = Pt * 10^(-TL/10)
        """
        TL = self.path_loss(distance, frequency, temperature, salinity, depth)
        received_power = transmitted_power * 10**(-TL / 10)
        return received_power
    
    def signal_to_noise_ratio(self, received_power: float, noise_power: float = 1e-12) -> float:
        """
        Calcule le rapport signal/bruit (dB)
        SNR = 10*log10(Pr/Pn)
        """
        if noise_power <= 0:
            return float('inf')
        return 10 * math.log10(received_power / noise_power)


class EnergyModel:
    """
    Modèle de consommation énergétique pour les nœuds UWSN
    """
    
    def __init__(self):
        # Paramètres énergétiques
        self.elec = 50e-9  # Énergie par bit pour l'électronique (J/bit)
        self.amp = 1e-12  # Énergie d'amplification (J/bit/m²)
        self.recv = 50e-9  # Énergie de réception (J/bit)
        self.idle = 1e-6  # Énergie en veille (J/s)
        
    def transmission_energy(self, data_size: int, distance: float, 
                           transmission_power: float) -> float:
        """
        Calcule l'énergie de transmission (J)
        Etx = (Eelec + Eamp * d²) * k
        """
        return (self.elec + self.amp * distance**2) * data_size
    
    def reception_energy(self, data_size: int) -> float:
        """
        Calcule l'énergie de réception (J)
        Erx = Eelec * k
        """
        return self.elec * data_size
    
    def idle_energy(self, time: float) -> float:
        """
        Calcule l'énergie en veille (J)
        """
        return self.idle * time


def create_sample_network(num_nodes: int = 10, area_size: float = 1000.0, 
                        depth_range: Tuple[float, float] = (-100, -10)) -> List[Node]:
    """
    Crée un réseau de test avec des nœuds aléatoirement positionnés
    
    Args:
        num_nodes: Nombre de nœuds
        area_size: Taille de la zone (m)
        depth_range: Plage de profondeur (m)
    
    Returns:
        Liste des nœuds du réseau
    """
    np.random.seed(42)  # Pour la reproductibilité
    
    nodes = []
    for i in range(num_nodes):
        node = Node(
            id=i,
            x=np.random.uniform(0, area_size),
            y=np.random.uniform(0, area_size),
            z=np.random.uniform(depth_range[0], depth_range[1]),
            energy=np.random.uniform(500, 1000),
            temperature=np.random.uniform(10, 20),
            salinity=np.random.uniform(33, 37),
            max_energy=1000.0,
            transmission_power=np.random.uniform(0.5, 2.0),
            frequency=np.random.uniform(20, 30)
        )
        nodes.append(node)
    
    return nodes


def calculate_network_metrics(nodes: List[Node], path: List[int], 
                            data_size: int = 1000) -> Dict[str, float]:
    """
    Calcule les métriques du réseau pour un chemin donné
    
    Args:
        nodes: Liste des nœuds
        path: Chemin (liste d'IDs de nœuds)
        data_size: Taille des données (bits)
    
    Returns:
        Dictionnaire des métriques
    """
    if len(path) < 2:
        return {
            'total_energy': 0,
            'total_distance': 0,
            'num_hops': 0,
            'latency': 0,
            'success_rate': 0
        }
    
    acoustic = AcousticPropagation()
    energy_model = EnergyModel()
    
    total_energy = 0
    total_distance = 0
    total_latency = 0
    
    for i in range(len(path) - 1):
        current_node = nodes[path[i]]
        next_node = nodes[path[i + 1]]
        
        # Distance et énergie
        distance = current_node.distance_to(next_node)
        total_distance += distance
        
        # Énergie de transmission
        tx_energy = energy_model.transmission_energy(
            data_size, distance, current_node.transmission_power
        )
        
        # Énergie de réception
        rx_energy = energy_model.reception_energy(data_size)
        
        total_energy += tx_energy + rx_energy
        
        # Latence (vitesse du son)
        sound_speed = acoustic.sound_speed(
            current_node.temperature, current_node.salinity, current_node.z
        )
        latency = distance / sound_speed
        total_latency += latency
    
    # Taux de succès basé sur l'énergie restante
    success_rate = 1.0
    for node_id in path:
        if nodes[node_id].energy < 100:  # Seuil d'énergie critique
            success_rate *= 0.5
    
    return {
        'total_energy': total_energy,
        'total_distance': total_distance,
        'num_hops': len(path) - 1,
        'latency': total_latency,
        'success_rate': success_rate
    }


def find_shortest_path(nodes: List[Node], source: int, destination: int) -> List[int]:
    """
    Trouve le chemin le plus court (Dijkstra simplifié)
    """
    if source == destination:
        return [source]
    
    # Matrice de distances
    n = len(nodes)
    distances = np.full(n, float('inf'))
    distances[source] = 0
    previous = np.full(n, -1)
    visited = np.zeros(n, dtype=bool)
    
    for _ in range(n):
        # Trouve le nœud non visité avec la distance minimale
        u = -1
        min_dist = float('inf')
        for i in range(n):
            if not visited[i] and distances[i] < min_dist:
                min_dist = distances[i]
                u = i
        
        if u == -1 or u == destination:
            break
            
        visited[u] = True
        
        # Met à jour les distances des voisins
        for v in range(n):
            if not visited[v]:
                dist = nodes[u].distance_to(nodes[v])
                if distances[u] + dist < distances[v]:
                    distances[v] = distances[u] + dist
                    previous[v] = u
    
    # Reconstruit le chemin
    path = []
    current = destination
    while current != -1:
        path.append(current)
        current = previous[current]
    
    return path[::-1] if path[0] == source else [source]
