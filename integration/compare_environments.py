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

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Output to console
                        logging.FileHandler('app.log')  # Log to a file
                    ])

logger = logging.getLogger(__name__)

def evaluate_env(env, model, num_episodes=100):
    """
    Evaluate the policy on a given environment with a progress bar.
    """
    logger.info(f"Evaluating environment with {num_episodes} episodes.")
    rewards = []
    with tqdm(total=num_episodes, desc="Evaluating") as pbar:
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            rewards.append(episode_reward)
            pbar.update(1)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return mean_reward, std_reward

def collect_trajectory(env, model, max_steps=500, visualize=False):
    """
    Collect trajectory (states, actions, rewards) of the policy in a given environment.
    """
    obs, _ = env.reset()
    states, actions, rewards, predicted_forces = [], [], [], []
    predicted_mu_c, predicted_mu_p = [], []

    animation = create_animation(obs, max_steps, visualize)

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        states.append(obs)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if "predicted_force" in info:
            predicted_forces.append(info["predicted_force"])
        if "predicted_mu_c" in info:
            predicted_mu_c.append(info["predicted_mu_c"])
        if "predicted_mu_p" in info:
            predicted_mu_p.append(info["predicted_mu_p"])

        plot_trajectory(states, actions, rewards, visualize)

        if terminated or truncated:
            logger.info(f"Trajectory collection terminated after {step + 1} steps.")
            break

    logger.debug(f"Trajectory collected: {len(states)} states, {len(actions)} actions.")
    return (np.array(states), np.array(actions), np.array(rewards), 
            np.array(predicted_forces) if predicted_forces else None,
            np.array(predicted_mu_c) if predicted_mu_c else None,
            np.array(predicted_mu_p) if predicted_mu_p else None)

def plot_timeline(time_steps, forces, mu_c, mu_p, save_folder_name):
    """
    Plot a timeline of predicted forces and friction parameters.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot forces
    ax1.plot(time_steps, forces, label='Predicted Force')
    ax1.set_ylabel('Force')
    ax1.set_title('Timeline of Predicted Force')
    ax1.legend()
    ax1.grid(True)

    # Plot friction parameters
    if mu_c is not None and mu_p is not None:
        ax2.plot(time_steps, mu_c, label='Predicted μ_c')
        ax2.plot(time_steps, mu_p, label='Predicted μ_p')
        ax2.set_ylabel('Friction Coefficient')
        ax2.set_title('Timeline of Predicted Friction Parameters')
        ax2.legend()
        ax2.grid(True)

    plt.xlabel('Time Steps')
    plt.tight_layout()
    plt.savefig(f'media/{save_folder_name}/timeline_plot.png')
    plt.close()

def compare_environments(pinn_model, params, predict_friction=False, num_episodes=100, max_steps=500, visualize=False, save_folder_name=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn_model = pinn_model.to(device)

    logger.info("Setting up environments for comparison.")
    original_env = Monitor(gym.make('CartPole-v1'))
    pinn_env = Monitor(PINNCartPoleEnv(pinn_model, params))

    total_timesteps_ppo = 10000
    logger.info(f"Training PPO agent on the original CartPole environment for {total_timesteps_ppo} total timesteps.")
    
    ppo_model = PPO('MlpPolicy', original_env, verbose=1, device=device)
    ppo_model.learn(total_timesteps=total_timesteps_ppo)

    # Evaluate on both environments with progress bars
    logger.info("Evaluating on original environment:")
    original_mean, original_std = evaluate_env(original_env, ppo_model, num_episodes)
    logger.info(f"Original environment - Mean reward: {original_mean:.2f} +/- {original_std:.2f}")

    logger.info("Evaluating on PINN environment:")
    pinn_mean, pinn_std = evaluate_env(pinn_env, ppo_model, num_episodes)
    logger.info(f"PINN environment - Mean reward: {pinn_mean:.2f} +/- {pinn_std:.2f}")

    # Collect trajectories
    logger.info("Collecting trajectories from both environments.")
    original_states, original_actions, original_rewards, _, _, _ = collect_trajectory(original_env, ppo_model, max_steps, visualize)
    pinn_states, pinn_actions, pinn_rewards, pinn_forces, pinn_mu_c, pinn_mu_p = collect_trajectory(pinn_env, ppo_model, max_steps, visualize)

    # Plot timeline
    time_steps = np.arange(len(pinn_forces))
    plot_timeline(time_steps, pinn_forces, pinn_mu_c, pinn_mu_p, save_folder_name)

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

    # Compare predicted vs actual forces
    if pinn_forces is not None:
        logger.info("Comparing predicted forces from PINN and forces from Gym.")
        plt.figure(figsize=(10, 5))
        plt.plot(original_env.force_mag * np.ones(len(original_actions)), label='Gym Forces', alpha=0.7)
        plt.plot(pinn_forces, label='Predicted Forces (PINN)', alpha=0.7)
        plt.title('Force Comparison: Gym vs PINN Prediction')
        plt.xlabel('Time Step')
        plt.ylabel('Force')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'media/{save_folder_name}/force_comparison.png')
        plt.close()

    # Create a comparison table for predicted forces, friction coefficients, and pole angles
    logger.info("Generating force, friction, and pole angle comparison table.")
    print(f"{'Time Step':<10}{'Predicted Force':<20}{'Pole Angle (rad)':<20}{'Predicted mu_c':<20}{'Predicted mu_p':<20}")
    for t in range(len(pinn_forces)):
        predicted_force = pinn_forces[t] if t < len(pinn_forces) and isinstance(pinn_forces[t], (int, float)) else float('nan')
        pole_angle = pinn_states[t, 2] if t < len(pinn_states) and isinstance(pinn_states[t, 2], (int, float)) else float('nan')
        mu_c = pinn_env.unwrapped.info.get("predicted_mu_c", float('nan')) if isinstance(pinn_env.unwrapped.info.get("predicted_mu_c"), (int, float)) else float('nan')
        mu_p = pinn_env.unwrapped.info.get("predicted_mu_p", float('nan')) if isinstance(pinn_env.unwrapped.info.get("predicted_mu_p"), (int, float)) else float('nan')
        
        print(f"{t:<10}{predicted_force:<20.5f}{pole_angle:<20.5f}{mu_c:<20.5f}{mu_p:<20.5f}")


    # Optional: Plot actions taken in the original environment
    logger.info("Plotting actions taken in the original environment.")
    plt.figure(figsize=(10, 5))
    plt.plot(original_actions, label='Original Actions', alpha=0.7)
    plt.title('Actions Taken in Original Environment')
    plt.xlabel('Time Step')
    plt.ylabel('Action')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'media/{save_folder_name}/actions_comparison.png')
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
        pinn_model.load_state_dict(torch.load('model_archive/without_friction/trained_pinn_model_without_friction_20240926_145523.pth', map_location=device))
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
