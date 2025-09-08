### Fichier: config.py

"""
Configuration par défaut pour le projet UWSN PPO
"""

# Configuration du réseau
NETWORK_CONFIG = {
    'num_nodes': 15,
    'area_size': 1000.0,  # m
    'depth_range': (-100, -10),  # m
    'data_size_range': (500, 2000),  # bits
    'max_steps': 50
}

# Configuration PPO
PPO_CONFIG = {
    'total_timesteps': 200000,
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5
}

# Configuration physique
PHYSICS_CONFIG = {
    'sound_speed_base': 1500.0,  # m/s
    'water_density': 1025.0,  # kg/m³
    'electronic_energy': 50e-9,  # J/bit
    'amplification_energy': 1e-12,  # J/bit/m²
    'reception_energy': 50e-9,  # J/bit
    'idle_energy': 1e-6,  # J/s
    'frequency_range': (20, 30),  # kHz
    'temperature_range': (10, 20),  # °C
    'salinity_range': (33, 37),  # PSU
    'energy_range': (500, 1000)  # J
}

# Configuration de l'interface
UI_CONFIG = {
    'plot_width': 800,
    'plot_height': 600,
    'color_scheme': 'viridis',
    'marker_size': 10,
    'line_width': 4
}

# Chemins des fichiers
PATHS = {
    'model_save': 'models/ppo_uwsn',
    'tensorboard_logs': './tensorboard_logs/',
    'plots_save': 'plots/',
    'data_save': 'data/'
}

# Configuration de l'évaluation
EVAL_CONFIG = {
    'num_episodes': 100,
    'eval_freq': 10000,
    'save_freq': 50000,
    'deterministic': True
}
