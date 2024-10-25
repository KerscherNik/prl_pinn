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

def evaluate_env(env, model, num_episodes=100, max_steps=500):
    """
    Evaluate the policy on a given environment with a progress bar and step limit.
    """
    logger.info(f"Evaluating environment with {num_episodes} episodes, max {max_steps} steps each.")
    rewards = []
    steps_taken = []
    
    with tqdm(total=num_episodes, desc="Evaluating") as pbar:
        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            step_count = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):# and step_count < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                step_count += 1
            
            logger.debug(f"Episode reward: {episode_reward:.2f}, steps taken: {step_count}")
            rewards.append(episode_reward)
            steps_taken.append(step_count)
            pbar.update(1)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_steps = np.mean(steps_taken)
    logger.info(f"Average episode length: {mean_steps:.2f} steps")
    
    # Alternative evaluation method using stable_baselines3
    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_episodes)
    return mean_reward, std_reward

def collect_trajectory(env, model, max_steps=500, visualize=False):
    """
    Collect trajectory with enforced step limit.
    """
    obs, _ = env.reset()
    states, actions, rewards = [], [], []
    predicted_forces, predicted_mu_c, predicted_mu_p = [], [], []
    step_count = 0

    animation = create_animation(obs, max_steps, visualize)
    done = False
    truncated = False

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

    logger.debug(f"Trajectory collected: {len(states)} states, {len(actions)} actions, {len(predicted_forces)} forces.")
    
    # Convert to numpy arrays, handling empty lists
    states_array = np.array(states) if states else np.array([[]])
    actions_array = np.array(actions) if actions else np.array([])
    rewards_array = np.array(rewards) if rewards else np.array([])
    forces_array = np.array(predicted_forces) if predicted_forces else np.array([])
    mu_c_array = np.array(predicted_mu_c) if predicted_mu_c else np.array([])
    mu_p_array = np.array(predicted_mu_p) if predicted_mu_p else np.array([])

    return (states_array, actions_array, rewards_array, 
            forces_array, mu_c_array, mu_p_array)

def plot_timeline(time_steps, forces, mu_c, mu_p, save_folder_name):
    """
    Plot a timeline of predicted forces and friction parameters.
    """
    # Check if we have valid data to plot
    if len(forces) == 0:
        logger.warning("No force data available for plotting timeline.")
        return

    fig, ax1 = plt.subplots(1, 1, figsize=(15, 10), sharex=True)

    # Plot friction parameters only if they exist and have data
    if mu_c is not None and mu_p is not None and len(mu_c) > 0 and len(mu_p) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        ax2.plot(time_steps, mu_c, label='Predicted μ_c')
        ax2.plot(time_steps, mu_p, label='Predicted μ_p')
        ax2.set_ylabel('Friction Coefficient')
        ax2.set_title('Timeline of Predicted Friction Parameters')
        ax2.legend()
        ax2.grid(True)
        
    # Plot forces
    ax1.plot(time_steps, forces, label='Predicted Force')
    ax1.set_ylabel('Force')
    ax1.set_title('Timeline of Predicted Force')
    ax1.legend()
    ax1.grid(True)

    plt.xlabel('Time Steps')
    plt.tight_layout()
    
    # Ensure the media directory exists
    os.makedirs(f'media/{save_folder_name}', exist_ok=True)
    plt.savefig(f'media/{save_folder_name}/timeline_plot.png')
    plt.close()

def compare_environments(pinn_model, params, predict_friction=False, num_episodes=100, max_steps=500, visualize=False, save_folder_name=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn_model = pinn_model.to(device)

    logger.info("Setting up environments for comparison.")
    original_env = Monitor(gym.make('CartPole-v1'))
    pinn_env = Monitor(PINNCartPoleEnv(pinn_model, params))

    # Define the path for saving the PPO model
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'integration/ppo_model_{current_time}'
    temp_model_path = f'integration/ppo_model_20241001_235054'  # temp placeholder with pretrained PPO

    # Check if the model already exists
    if os.path.exists("integration/ppo_model_20241001_235054.zip"):
        logger.info(f"Loading existing PPO model from {temp_model_path}.")
        ppo_model = PPO.load(temp_model_path, env=original_env, device=device)
    else:
        total_timesteps_ppo = 50000
        logger.info(f"Training PPO agent on the original CartPole environment for {total_timesteps_ppo} total timesteps.")
        
        ppo_model = PPO('MlpPolicy', original_env, verbose=1, device=device)
        ppo_model.learn(total_timesteps=total_timesteps_ppo)
        
        logger.info(f"Saving PPO model to {model_path}.")
        ppo_model.save(model_path)

    # Collect trajectories first
    logger.info("Collecting trajectories from both environments.")
    original_states, original_actions, original_rewards, _, _, _ = collect_trajectory(
        original_env, ppo_model, max_steps, visualize)
    pinn_states, pinn_actions, pinn_rewards, pinn_forces, pinn_mu_c, pinn_mu_p = collect_trajectory(
        pinn_env, ppo_model, max_steps, visualize)

    logger.info("Evaluating on PINN environment:")
    pinn_mean, pinn_std = evaluate_env(pinn_env, ppo_model, num_episodes)
    logger.info(f"PINN environment - Mean reward: {pinn_mean:.2f} +/- {pinn_std:.2f}")

    logger.info("Evaluating on original environment:")
    original_mean, original_std = evaluate_env(original_env, ppo_model, num_episodes)
    logger.info(f"Original environment - Mean reward: {original_mean:.2f} +/- {original_std:.2f}")

    logger.info("Generating force, friction, and pole angle comparison table.")
    print(f"{'Time Step':<10}{'Predicted Force':<20}{'Pole Angle (rad)':<20}{'Predicted mu_c':<20}{'Predicted mu_p':<20}")
    
    # Generate the comparison table using collected trajectory data
    for t in range(min(len(pinn_forces), max_steps)):
        pole_angle = pinn_states[t, 2] if t < len(pinn_states) and pinn_states.ndim > 1 else float('nan')
        force = pinn_forces[t] if t < len(pinn_forces) else float('nan')
        mu_c = pinn_mu_c[t] if t < len(pinn_mu_c) else float('nan')
        mu_p = pinn_mu_p[t] if t < len(pinn_mu_p) else float('nan')
        
        print(f"{t:<10}{force:<20.5f}{pole_angle:<20.5f}{mu_c:<20.5f}{mu_p:<20.5f}")

    # Plot force comparison
    if len(pinn_forces) > 0:
        logger.info("Comparing predicted forces from PINN and forces from Gym.")
        plt.figure(figsize=(10, 5))
        plt.plot(original_env.unwrapped.force_mag * np.ones(len(original_actions)), 
                label='Gym Forces', alpha=0.7)
        plt.plot(pinn_forces, label='Predicted Forces (PINN)', alpha=0.7)
        plt.title('Force Comparison: Gym vs PINN Prediction')
        plt.xlabel('Time Step')
        plt.ylabel('Force')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'media/{save_folder_name}/force_comparison.png')
        plt.close()

    # Plot timeline
    if len(pinn_forces) > 0:
        time_steps = np.arange(len(pinn_forces))
        plot_timeline(time_steps, pinn_forces, pinn_mu_c, pinn_mu_p, save_folder_name)
    else:
        logger.warning("No force data available for timeline plot.")

    # Plot state comparisons
    logger.info("Plotting state comparisons.")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    state_labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']
    for i in range(4):
        axs[i // 2, i % 2].plot(original_states[:, i], label='Original', alpha=0.7)
        if pinn_states.ndim > 1 and pinn_states.shape[0] > 1:
            axs[i // 2, i % 2].plot(pinn_states[:, i], label='PINN', alpha=0.7)
        axs[i // 2, i % 2].set_title(state_labels[i])
        axs[i // 2, i % 2].set_xlabel('Time Step')
        axs[i // 2, i % 2].set_ylabel('Value')
        axs[i // 2, i % 2].legend()
    plt.tight_layout()
    plt.savefig(f'media/{save_folder_name}/state_comparison.png')
    plt.close()

    # Plot reward comparison
    logger.info("Plotting reward comparisons.")
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(original_rewards), label='Original', alpha=0.7)
    plt.plot(np.cumsum(pinn_rewards), label='PINN', alpha=0.7)
    plt.title('Cumulative Reward Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'media/{save_folder_name}/reward_comparison.png')
    plt.close()

    # Close environments
    logger.info("Closing environments.")
    original_env.close()
    pinn_env.close()

    return original_rewards, pinn_rewards


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
