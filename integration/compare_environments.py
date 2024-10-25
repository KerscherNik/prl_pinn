import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from integration.gym_integration import PINNCartPoleEnv
from model.pinn_model import CartpolePINN
from visualization.visualization import create_animation, plot_trajectory
import logging
from tqdm import tqdm
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Output to console
                        logging.FileHandler('app.log')  # Log to a file
                    ])

logger = logging.getLogger(__name__)

def collect_trajectory(env, model, max_steps=500, visualize=False, env_name=None):
    """
    Collect trajectory with enforced step limit.
    """
    logger.debug("Starting trajectory collection")
    obs, _ = env.reset()
    states, actions, rewards = [], [], []
    predicted_forces, predicted_mu_c, predicted_mu_p = [], [], []
    step_count = 0

    animation = create_animation(obs, max_steps, visualize)
    done = False
    truncated = False
    
    # Determine environment type if not explicitly provided
    if env_name is None:
        env_name = "PINN" if isinstance(env, PINNCartPoleEnv) else "Original"

    with tqdm(total=max_steps, desc=f"[{env_name}] Collecting trajectory", leave=False) as pbar:
        while not (done or truncated) and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            states.append(obs)
            actions.append(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            step_count += 1

            # Extract predictions from info dictionary
            if "predicted_force" in info:
                predicted_forces.append(info["predicted_force"])
            if "predicted_mu_c" in info:
                predicted_mu_c.append(info["predicted_mu_c"])
            if "predicted_mu_p" in info:
                predicted_mu_p.append(info["predicted_mu_p"])

            if visualize:
                plot_trajectory(states, actions, rewards, visualize)

            done = terminated
            pbar.update(1)
            pbar.set_postfix({'reward': f"{sum(rewards):.2f}"})

    logger.debug(f"Trajectory collection completed: {len(states)} states, {len(actions)} actions, {len(predicted_forces)} forces.")
    
    # Convert to numpy arrays, handling empty lists
    states_array = np.array(states) if states else np.array([[]])
    actions_array = np.array(actions) if actions else np.array([])
    rewards_array = np.array(rewards) if rewards else np.array([])
    forces_array = np.array(predicted_forces) if predicted_forces else np.array([])
    mu_c_array = np.array(predicted_mu_c) if predicted_mu_c else np.array([])
    mu_p_array = np.array(predicted_mu_p) if predicted_mu_p else np.array([])

    return (states_array, actions_array, rewards_array, 
            forces_array, mu_c_array, mu_p_array)

def plot_episode_timeline(time_steps, forces, mu_c, mu_p, save_path, episode):
    """
    Plot a timeline of predicted forces and friction parameters for a single episode.
    """
    logger.debug(f"Plotting timeline for episode {episode}")
    if len(forces) == 0:
        logger.warning(f"No force data available for plotting timeline in episode {episode}")
        return

    fig, ax1 = plt.subplots(1, 1, figsize=(15, 10), sharex=True)

    if mu_c is not None and mu_p is not None and len(mu_c) > 0 and len(mu_p) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        ax2.plot(time_steps, mu_c, label='Predicted μ_c')
        ax2.plot(time_steps, mu_p, label='Predicted μ_p')
        ax2.set_ylabel('Friction Coefficient')
        ax2.set_title(f'Timeline of Predicted Friction Parameters - Episode {episode}')
        ax2.legend()
        ax2.grid(True)
        
    ax1.plot(time_steps, forces, label='Predicted Force')
    ax1.set_ylabel('Force')
    ax1.set_title(f'Timeline of Predicted Force - Episode {episode}')
    ax1.legend()
    ax1.grid(True)

    plt.xlabel('Time Steps')
    plt.tight_layout()
    plt.savefig(f'{save_path}/timeline_plot_episode_{episode}.png')
    plt.close()
    logger.debug(f"Timeline plot saved for episode {episode}")

def plot_episode_states(original_states, pinn_states, save_path, episode):
    """
    Plot state comparisons for a single episode.
    """
    logger.debug(f"Plotting state comparisons for episode {episode}")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    state_labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']
    for i in range(4):
        axs[i // 2, i % 2].plot(original_states[:, i], label='Original', alpha=0.7)
        if pinn_states.ndim > 1 and pinn_states.shape[0] > 1:
            axs[i // 2, i % 2].plot(pinn_states[:, i], label='PINN', alpha=0.7)
        axs[i // 2, i % 2].set_title(f'{state_labels[i]} - Episode {episode}')
        axs[i // 2, i % 2].set_xlabel('Time Step')
        axs[i // 2, i % 2].set_ylabel('Value')
        axs[i // 2, i % 2].legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}/state_comparison_episode_{episode}.png')
    plt.close()
    logger.debug(f"State comparison plot saved for episode {episode}")

def plot_episode_rewards(original_rewards, pinn_rewards, save_path, episode):
    """
    Plot reward comparison for a single episode.
    """
    logger.debug(f"Plotting reward comparisons for episode {episode}")
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(original_rewards), label='Original', alpha=0.7)
    plt.plot(np.cumsum(pinn_rewards), label='PINN', alpha=0.7)
    plt.title(f'Cumulative Reward Comparison - Episode {episode}')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{save_path}/reward_comparison_episode_{episode}.png')
    plt.close()
    logger.debug(f"Reward comparison plot saved for episode {episode}")

def plot_best_runs_comparison(best_original_data, best_pinn_data, save_path):
    """
    Plot comparison of the best runs from both environments.
    """
    logger.info("Creating comparison plot of best runs")
    # Compare pole angles as the key metric
    plt.figure(figsize=(12, 6))
    plt.plot(best_original_data['states'][:, 2], label='Original Best Run', alpha=0.7)
    plt.plot(best_pinn_data['states'][:, 2], label='PINN Best Run', alpha=0.7)
    plt.title('Pole Angle Comparison of Best Runs')
    plt.xlabel('Time Step')
    plt.ylabel('Pole Angle (radians)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}/best_runs_comparison.png')
    plt.close()
    logger.info("Best runs comparison plot saved")

def compare_environments(pinn_model, params, predict_friction=False, num_episodes=100, max_steps=500, visualize=False, save_folder_name=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn_model = pinn_model.to(device)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_path = f'media/{save_folder_name}/{current_time}'
    os.makedirs(base_save_path, exist_ok=True)
    logger.info(f"Created output directory at {base_save_path}")

    logger.info("Setting up environments for comparison.")
    original_env = Monitor(gym.make('CartPole-v1'))
    pinn_env = Monitor(PINNCartPoleEnv(pinn_model, params))

    model_path = f'integration/ppo_model_{current_time}'
    temp_model_path = f'integration/ppo_model_20241001_235054'

    if os.path.exists("integration/ppo_model_20241001_235054.zip"):
        logger.info(f"Loading existing PPO model from {temp_model_path}.")
        ppo_model = PPO.load(temp_model_path, env=original_env, device=device)
    else:
        total_timesteps_ppo = 50000
        logger.info(f"Training PPO agent for {total_timesteps_ppo} timesteps.")
        ppo_model = PPO('MlpPolicy', original_env, verbose=1, device=device)
        ppo_model.learn(total_timesteps=total_timesteps_ppo)
        logger.info(f"Saving PPO model to {model_path}")
        ppo_model.save(model_path)

    # Initialize storage for best runs and all rewards
    best_original_run = {'reward': float('-inf'), 'states': None, 'actions': None, 'rewards': None}
    best_pinn_run = {'reward': float('-inf'), 'states': None, 'actions': None, 'rewards': None, 
                     'forces': None, 'mu_c': None, 'mu_p': None}
    
    # Lists to store all episode rewards for statistical analysis
    original_episode_rewards = []
    pinn_episode_rewards = []

    # Evaluate episodes
    logger.info(f"Starting evaluation of {num_episodes} episodes")
    with tqdm(total=num_episodes, desc="Evaluating episodes") as pbar:
        for episode in range(num_episodes):
            logger.info(f"Starting episode {episode + 1}/{num_episodes}")
            episode_path = f'{base_save_path}/episode_{episode}'
            os.makedirs(episode_path, exist_ok=True)

            # Collect trajectories for this episode
            logger.debug(f"Collecting trajectories for episode {episode}")
            original_states, original_actions, original_rewards, _, _, _ = collect_trajectory(
                original_env, ppo_model, max_steps, visualize, env_name="Original")
            pinn_states, pinn_actions, pinn_rewards, pinn_forces, pinn_mu_c, pinn_mu_p = collect_trajectory(
                pinn_env, ppo_model, max_steps, visualize, env_name="PINN")

            # Plot episode-specific visualizations
            logger.debug(f"Creating visualizations for episode {episode}")
            plot_episode_timeline(np.arange(len(pinn_forces)), pinn_forces, pinn_mu_c, pinn_mu_p, 
                                episode_path, episode)
            plot_episode_states(original_states, pinn_states, episode_path, episode)
            plot_episode_rewards(original_rewards, pinn_rewards, episode_path, episode)

            # Calculate total rewards for this episode
            original_total_reward = np.sum(original_rewards)
            pinn_total_reward = np.sum(pinn_rewards)
            
            # Store episode rewards for statistical analysis
            original_episode_rewards.append(original_total_reward)
            pinn_episode_rewards.append(pinn_total_reward)

            # Update best runs based on total reward
            if original_total_reward > best_original_run['reward']:
                logger.info(f"New best original run found in episode {episode} with reward {original_total_reward:.2f}")
                best_original_run = {
                    'reward': original_total_reward,
                    'states': original_states,
                    'actions': original_actions,
                    'rewards': original_rewards
                }

            if pinn_total_reward > best_pinn_run['reward']:
                logger.info(f"New best PINN run found in episode {episode} with reward {pinn_total_reward:.2f}")
                best_pinn_run = {
                    'reward': pinn_total_reward,
                    'states': pinn_states,
                    'actions': pinn_actions,
                    'rewards': pinn_rewards,
                    'forces': pinn_forces,
                    'mu_c': pinn_mu_c,
                    'mu_p': pinn_mu_p
                }

            logger.info(f"Results for episode {episode}: Original Reward: {original_total_reward:.2f}, PINN Reward: {pinn_total_reward:.2f}")

            pbar.update(1)
            pbar.set_postfix({
                'Best Original': f"{best_original_run['reward']:.2f}",
                'Best PINN': f"{best_pinn_run['reward']:.2f}"
            })

    # Calculate and log statistical measures
    original_mean = np.mean(original_episode_rewards)
    original_std = np.std(original_episode_rewards)
    pinn_mean = np.mean(pinn_episode_rewards)
    pinn_std = np.std(pinn_episode_rewards)

    logger.info("\nFinal Statistical Results:")
    logger.info(f"Original Environment - Mean Reward: {original_mean:.2f} ± {original_std:.2f}")
    logger.info(f"PINN Environment - Mean Reward: {pinn_mean:.2f} ± {pinn_std:.2f}")
    logger.info(f"Best Original Run Reward: {best_original_run['reward']:.2f}")
    logger.info(f"Best PINN Run Reward: {best_pinn_run['reward']:.2f}")

    logger.info("Plotting final comparison of best runs")
    plot_best_runs_comparison(best_original_run, best_pinn_run, base_save_path)

    # Close environments
    logger.info("Closing environments.")
    original_env.close()
    pinn_env.close()

    return best_original_run, best_pinn_run


if __name__ == "__main__":
    logger.info("Comparing environments with preloaded trained model...")
    # Load your trained PINN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_friction = False  # Set this to True when using a model that predicts friction
    save_folder_name = "with_friction" if predict_friction else "without_friction"
    pinn_model = CartpolePINN(predict_friction=predict_friction, sequence_length=10)
    try:
        # Alternative path with best with friction model: model_archive/with_friction/trained_pinn_model_with_friction_20240926_000026.pth. Change predict_friction to True above.
        pinn_model.load_state_dict(torch.load('model_archive/without_friction/trained_pinn_model_without_friction_20240926_145523.pth', map_location=device))
        #pinn_model.load_state_dict(torch.load('model_archive/with_friction/trained_pinn_model_with_friction_20240926_000026.pth', map_location=device))
        logger.info("Successfully loaded trained model.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.warning("Proceeding with untrained model.")
    pinn_model = pinn_model.to(device)
    pinn_model.eval()

    params = {
        "m_c": 1.0,
        "m_p": 0.1,
        "l": 1.0,
        "g": 9.8,
        "mu_c": 0.0,
        "mu_p": 0.0,
        "force_mag": 10.0,
        "tau" : 0.02
    }
    

    compare_environments(pinn_model, params, predict_friction=True, num_episodes=100, max_steps=500, visualize=False, save_folder_name=save_folder_name)
