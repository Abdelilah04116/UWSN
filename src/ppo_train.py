### Fichier: src/ppo_train.py

"""
Script d'entraînement PPO pour l'optimisation de routage UWSN
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import pickle
import json
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .env_gym import UWSNRoutingEnv
from .utils_network import create_sample_network, Node


class UWSNTrainer:
    """
    Classe pour l'entraînement du modèle PPO sur l'environnement UWSN
    """
    
    def __init__(self, num_nodes: int = 10, area_size: float = 1000.0, 
                 depth_range: Tuple[float, float] = (-100, -10),
                 model_save_path: str = "models/ppo_uwsn"):
        """
        Initialise le trainer UWSN
        
        Args:
            num_nodes: Nombre de nœuds dans le réseau
            area_size: Taille de la zone (m)
            depth_range: Plage de profondeur (m)
            model_save_path: Chemin de sauvegarde du modèle
        """
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.depth_range = depth_range
        self.model_save_path = model_save_path
        
        # Création du réseau
        self.nodes = create_sample_network(num_nodes, area_size, depth_range)
        
        # Environnement
        self.env = None
        self.model = None
        
        # Historique d'entraînement
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'energy_consumption': [],
            'evaluation_rewards': []
        }
        
        # Création du dossier de sauvegarde
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    def create_environment(self, max_steps: int = 50) -> UWSNRoutingEnv:
        """Crée l'environnement UWSN"""
        return UWSNRoutingEnv(
            nodes=self.nodes,
            max_steps=max_steps,
            data_size_range=(500, 2000)
        )
    
    def train(self, total_timesteps: int = 100000, 
              learning_rate: float = 3e-4,
              n_steps: int = 2048,
              batch_size: int = 64,
              n_epochs: int = 10,
              gamma: float = 0.99,
              gae_lambda: float = 0.95,
              clip_range: float = 0.2,
              ent_coef: float = 0.01,
              vf_coef: float = 0.5,
              max_grad_norm: float = 0.5,
              eval_freq: int = 10000,
              save_freq: int = 50000) -> PPO:
        """
        Entraîne le modèle PPO
        
        Args:
            total_timesteps: Nombre total de pas d'entraînement
            learning_rate: Taux d'apprentissage
            n_steps: Nombre de pas par mise à jour
            batch_size: Taille du batch
            n_epochs: Nombre d'époques par mise à jour
            gamma: Facteur de remise
            gae_lambda: Paramètre GAE
            clip_range: Plage de clipping PPO
            ent_coef: Coefficient d'entropie
            vf_coef: Coefficient de la fonction de valeur
            max_grad_norm: Norme maximale du gradient
            eval_freq: Fréquence d'évaluation
            save_freq: Fréquence de sauvegarde
        
        Returns:
            Modèle PPO entraîné
        """
        print("🚀 Début de l'entraînement PPO pour UWSN...")
        print(f"📊 Configuration: {self.num_nodes} nœuds, {total_timesteps} pas")
        
        # Création de l'environnement
        self.env = self.create_environment()
        
        # Environnement d'évaluation
        eval_env = Monitor(self.create_environment())
        
        # Configuration PPO
        policy_kwargs = dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=torch.nn.ReLU
        )
        
        # Sélection du device (GPU si dispo)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Création du modèle PPO
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )
        
        # Callback d'évaluation
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.model_save_path}_best",
            log_path=f"{self.model_save_path}_logs",
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Entraînement
        print("🏋️ Entraînement en cours...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=False
        )
        
        # Sauvegarde du modèle final
        self.model.save(self.model_save_path)
        print(f"💾 Modèle sauvegardé: {self.model_save_path}")

        # Export convivial du meilleur modèle évalué si disponible
        try:
            import shutil, os
            best_ckpt = os.path.join(f"{self.model_save_path}_best", "best_model.zip")
            export_best = f"{self.model_save_path}_best.zip"
            if os.path.exists(best_ckpt):
                shutil.copyfile(best_ckpt, export_best)
                print(f"🏅 Meilleur modèle exporté vers: {export_best}")
            else:
                print("ℹ️ Aucun best_model.zip détecté (évaluation pas encore produite).")
        except Exception as e:
            print(f"⚠️ Export du meilleur modèle échoué: {e}")
        
        # Sauvegarde des métadonnées
        self._save_metadata()
        
        return self.model
    
    def evaluate(self, num_episodes: int = 100, render: bool = False) -> Dict[str, Any]:
        """
        Évalue le modèle entraîné
        
        Args:
            num_episodes: Nombre d'épisodes d'évaluation
            render: Afficher les épisodes
        
        Returns:
            Dictionnaire des métriques d'évaluation
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant l'évaluation")
        
        print(f"🔍 Évaluation sur {num_episodes} épisodes...")
        
        eval_env = self.create_environment()
        
        episode_rewards = []
        episode_lengths = []
        success_rates = []
        energy_consumptions = []
        paths = []
        
        for episode in range(num_episodes):
            obs, _ = eval_env.reset()
            terminated, truncated = False, False
            episode_reward = 0.0
            step_count = 0
            
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                step_count += 1
                
                if render:
                    eval_env.render()
            
            # Collecte des métriques
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            if 'episode_stats' in info:
                stats = info['episode_stats']
                success_rates.append(1.0 if stats['success'] else 0.0)
                energy_consumptions.append(stats['total_energy'])
                paths.append(stats['path'])
        
        # Calcul des métriques moyennes
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': np.mean(success_rates),
            'mean_energy': np.mean(energy_consumptions),
            'std_energy': np.std(energy_consumptions),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'success_rates': success_rates,
            'energy_consumptions': energy_consumptions,
            'paths': paths
        }
        
        print(f"📈 Résultats d'évaluation:")
        print(f"   Récompense moyenne: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"   Longueur moyenne: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
        print(f"   Taux de succès: {metrics['success_rate']:.2%}")
        print(f"   Énergie moyenne: {metrics['mean_energy']:.2f} ± {metrics['std_energy']:.2f}")
        
        return metrics
    
    def compare_with_baseline(self, num_episodes: int = 100) -> Dict[str, Any]:
        """
        Compare le modèle PPO avec des méthodes de baseline
        
        Args:
            num_episodes: Nombre d'épisodes pour la comparaison
        
        Returns:
            Dictionnaire des résultats de comparaison
        """
        print("🆚 Comparaison avec les méthodes de baseline...")
        
        # Évaluation du modèle PPO
        ppo_metrics = self.evaluate(num_episodes)
        
        # Baseline: Chemin le plus court (Dijkstra)
        eval_env = self.create_environment()
        baseline_metrics = self._evaluate_baseline(eval_env, num_episodes)
        
        # Baseline: Aléatoire
        random_metrics = self._evaluate_random(eval_env, num_episodes)
        
        comparison = {
            'ppo': ppo_metrics,
            'shortest_path': baseline_metrics,
            'random': random_metrics
        }
        
        # Affichage des résultats
        print("\n📊 Comparaison des méthodes:")
        print(f"{'Méthode':<15} {'Récompense':<12} {'Succès':<8} {'Énergie':<10}")
        print("-" * 50)
        print(f"{'PPO':<15} {ppo_metrics['mean_reward']:<12.2f} {ppo_metrics['success_rate']:<8.2%} {ppo_metrics['mean_energy']:<10.2f}")
        print(f"{'Plus court':<15} {baseline_metrics['mean_reward']:<12.2f} {baseline_metrics['success_rate']:<8.2%} {baseline_metrics['mean_energy']:<10.2f}")
        print(f"{'Aléatoire':<15} {random_metrics['mean_reward']:<12.2f} {random_metrics['success_rate']:<8.2%} {random_metrics['mean_energy']:<10.2f}")
        
        return comparison
    
    def _evaluate_baseline(self, env: UWSNRoutingEnv, num_episodes: int) -> Dict[str, Any]:
        """Évalue la méthode du chemin le plus court"""
        episode_rewards = []
        success_rates = []
        energy_consumptions = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            destination = env.destination
            
            # Chemin optimal
            optimal_path = env.get_optimal_path()
            
            if len(optimal_path) < 2:
                episode_rewards.append(-1000)
                success_rates.append(0.0)
                energy_consumptions.append(float('inf'))
                continue
            
            total_reward = 0.0
            info = {}
            for i in range(len(optimal_path) - 1):
                action = optimal_path[i + 1]
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            success_rates.append(1.0 if env.state.current_node == destination else 0.0)
            energy_consumptions.append(info.get('episode_stats', {}).get('total_energy', 0.0))
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': np.mean(success_rates),
            'mean_energy': np.mean(energy_consumptions),
            'std_energy': np.std(energy_consumptions)
        }
    
    def _evaluate_random(self, env: UWSNRoutingEnv, num_episodes: int) -> Dict[str, Any]:
        """Évalue une politique aléatoire"""
        episode_rewards = []
        success_rates = []
        energy_consumptions = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            terminated, truncated = False, False
            total_reward = 0.0
            step_count = 0
            max_steps = 50
            
            while not (terminated or truncated) and step_count < max_steps:
                valid_actions = [i for i in range(env.num_nodes) if env.nodes[i].is_alive()]
                if not valid_actions:
                    break
                action = np.random.choice(valid_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
            
            episode_rewards.append(total_reward)
            success_rates.append(1.0 if env.state.current_node == env.destination else 0.0)
            energy_consumptions.append(info.get('episode_stats', {}).get('total_energy', 0.0))
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': np.mean(success_rates),
            'mean_energy': np.mean(energy_consumptions),
            'std_energy': np.std(energy_consumptions)
        }
    
    def plot_training_curves(self, save_path: str = None):
        """Affiche les courbes d'entraînement"""
        if not self.training_history['episode_rewards']:
            print("Aucune donnée d'entraînement disponible")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Récompenses par épisode
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_title('Récompenses par épisode')
        axes[0, 0].set_xlabel('Épisode')
        axes[0, 0].set_ylabel('Récompense')
        axes[0, 0].grid(True)
        
        # Longueur des épisodes
        axes[0, 1].plot(self.training_history['episode_lengths'])
        axes[0, 1].set_title('Longueur des épisodes')
        axes[0, 1].set_xlabel('Épisode')
        axes[0, 1].set_ylabel('Longueur')
        axes[0, 1].grid(True)
        
        # Taux de succès
        axes[1, 0].plot(self.training_history['success_rates'])
        axes[1, 0].set_title('Taux de succès')
        axes[1, 0].set_xlabel('Épisode')
        axes[1, 0].set_ylabel('Taux de succès')
        axes[1, 0].grid(True)
        
        # Consommation énergétique
        axes[1, 1].plot(self.training_history['energy_consumption'])
        axes[1, 1].set_title('Consommation énergétique')
        axes[1, 1].set_xlabel('Épisode')
        axes[1, 1].set_ylabel('Énergie (J)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Graphiques sauvegardés: {save_path}")
        
        plt.show()
    
    def _save_metadata(self):
        """Sauvegarde les métadonnées du modèle"""
        metadata = {
            'num_nodes': self.num_nodes,
            'area_size': self.area_size,
            'depth_range': self.depth_range,
            'training_date': datetime.now().isoformat(),
            'model_path': self.model_save_path,
            'nodes_info': [
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
        
        metadata_path = f"{self.model_save_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Métadonnées sauvegardées: {metadata_path}")


def main():
    """Fonction principale pour l'entraînement"""
    print("🌊 Entraînement PPO pour l'optimisation de routage UWSN")
    print("=" * 60)
    
    # Configuration
    num_nodes = 15
    area_size = 1000.0
    depth_range = (-100, -10)
    total_timesteps = 20000
    
    # Création du trainer
    trainer = UWSNTrainer(
        num_nodes=num_nodes,
        area_size=area_size,
        depth_range=depth_range,
        model_save_path="models/ppo_uwsn"
    )
    
    # Entraînement
    model = trainer.train(total_timesteps=total_timesteps)
    
    # Évaluation
    print("\n" + "=" * 60)
    print("🔍 Évaluation du modèle entraîné")
    metrics = trainer.evaluate(num_episodes=50)
    
    # Comparaison avec les baselines
    print("\n" + "=" * 60)
    print("🆚 Comparaison avec les méthodes de baseline")
    comparison = trainer.compare_with_baseline(num_episodes=100)
    
    # Affichage des courbes d'entraînement
    trainer.plot_training_curves("training_curves.png")
    
    print("\n✅ Entraînement terminé avec succès!")
    print(f"💾 Modèle sauvegardé: {trainer.model_save_path}")


if __name__ == "__main__":
    main()
