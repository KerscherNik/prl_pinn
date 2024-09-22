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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Output to console
                        logging.FileHandler('app.log')  # Log to a file
                    ])

logger = logging.getLogger(__name__)

def evaluate_env(env, model, num_episodes=100):
    """
    Evaluate the policy on a given environment.
    
    Args:
    env (gym.Env): The environment to evaluate on.
    model (stable_baselines3.BaseAlgorithm): The trained model to evaluate.
    num_episodes (int): Number of episodes to run for evaluation.
    
    Returns:
    tuple: Mean and standard deviation of rewards.
    """
    logger.info(f"Evaluating environment with {num_episodes} episodes.")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_episodes)
    return mean_reward, std_reward

def collect_trajectory(env, model, max_steps=500, visualize=False):
    """
    Collect trajectory (states, actions, rewards) of the policy in a given environment.
    
    Args:
    env (gym.Env): The environment to collect trajectory from.
    model (stable_baselines3.BaseAlgorithm): The trained model to use for action prediction.
    max_steps (int): Maximum number of steps to run the environment.
    visualize (bool): Whether to visualize the trajectory.
    
    Returns:
    tuple: Numpy arrays of states, actions, and rewards.
    """
    obs, _ = env.reset()
    states, actions, rewards, predicted_forces = [], [], [], []

    animation = create_animation(obs, max_steps, visualize)

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        states.append(obs)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if "predicted_force" in info:
            predicted_forces.append(info["predicted_force"])

        plot_trajectory(states, actions, rewards, visualize)

        if terminated or truncated:
            logger.info(f"Trajectory collection terminated after {step + 1} steps.")
            break

    logger.debug(f"Trajectory collected: {len(states)} states, {len(actions)} actions.")
    return (np.array(states), np.array(actions), np.array(rewards), np.array(predicted_forces) if predicted_forces else None)

def compare_environments(pinn_model, params, predict_friction=False, num_episodes=100, max_steps=500, visualize=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn_model = pinn_model.to(device)

    logger.info("Setting up environments for comparison.")
    # Create environments with Monitor wrapper
    original_env = Monitor(gym.make('CartPole-v1'))
    pinn_env = Monitor(PINNCartPoleEnv(pinn_model, params))

    # Train a policy on the original environment
    logger.info("Training PPO agent on the original CartPole environment.")
    ppo_model = PPO('MlpPolicy', original_env, verbose=1, device=device)
    ppo_model.learn(total_timesteps=20000)

    # Evaluate on both environments
    original_mean, original_std = evaluate_env(original_env, ppo_model, num_episodes)
    logger.info(f"Original environment - Mean reward: {original_mean:.2f} +/- {original_std:.2f}")

    pinn_mean, pinn_std = evaluate_env(pinn_env, ppo_model, num_episodes)
    logger.info(f"PINN environment - Mean reward: {pinn_mean:.2f} +/- {pinn_std:.2f}")

    # Collect trajectories
    logger.info("Collecting trajectories from both environments.")
    original_states, original_actions, original_rewards, _ = collect_trajectory(original_env, ppo_model, max_steps, visualize)
    pinn_states, pinn_actions, pinn_rewards, pinn_forces = collect_trajectory(pinn_env, ppo_model, max_steps, visualize)

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
    plt.savefig('media/state_comparison.png')
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
    plt.savefig('media/reward_comparison.png')
    plt.close()

    # Compare predicted vs actual forces
    if pinn_forces is not None:
        logger.info("Comparing predicted forces from PINN and forces from Gym.")
        plt.figure(figsize=(10, 5))
        plt.plot(original_env.force_mag * np.ones(len(original_actions)), label='Gym Forces', alpha=0.7)  # Constant force used in Gym
        plt.plot(pinn_forces, label='Predicted Forces (PINN)', alpha=0.7)
        plt.title('Force Comparison: Gym vs PINN Prediction')
        plt.xlabel('Time Step')
        plt.ylabel('Force')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('media/force_comparison.png')
        plt.close()

    # Create a comparison table for predicted forces and pole angles
    logger.info("Generating force and pole angle comparison table.")
    print(f"{'Time Step':<10}{'Predicted Force':<20}{'Pole Angle (rad)':<20}")
    for t in range(len(pinn_forces)):
        predicted_force = pinn_forces[t] if t < len(pinn_forces) else "N/A"
        pole_angle = pinn_states[t, 2] if t < len(pinn_states) else "N/A"
        print(f"{t:<10}{predicted_force:<20.5f}{pole_angle:<20.5f}")

    # Optional: Plot actions taken in the original environment
    logger.info("Plotting actions taken in the original environment.")
    plt.figure(figsize=(10, 5))
    plt.plot(original_actions, label='Original Actions', alpha=0.7)
    plt.title('Actions Taken in Original Environment')
    plt.xlabel('Time Step')
    plt.ylabel('Action')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('media/actions_comparison.png')
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
    pinn_model = CartpolePINN(predict_friction=False, sequence_length=10)  # Adjust sequence_length as needed
    try:
        pinn_model.load_state_dict(torch.load('../model_archive/trained_pinn_model_without_friction_20240919_092248.pth', map_location=device))
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
        "force_mag": 10.0
    }

    compare_environments(pinn_model, params, visualize=True)
