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
    states, actions, rewards = [], [], []

    animation = create_animation(obs, max_steps, visualize)

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        states.append(obs)
        actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

        plot_trajectory(states, actions, rewards, visualize)

        if terminated or truncated:
            break

    return np.array(states), np.array(actions), np.array(rewards)

def compare_environments(pinn_model, params, predict_friction=False, num_episodes=100, max_steps=500, visualize=False):
    """
    Compare performance of policy in original CartPole env and in PINN-based env.
    Trains a policy on gym env and compares its performance on both environments.
    
    Args:
    pinn_model (CartpolePINN): The trained PINN model.
    params (dict): Parameters for the CartPole environment.
    predict_friction (bool): Whether the PINN model predicts friction.
    num_episodes (int): Number of episodes for evaluation.
    max_steps (int): Maximum steps per episode.
    visualize (bool): Whether to visualize the trajectories.
    
    Returns:
    tuple: Lists of rewards for original and PINN environments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn_model = pinn_model.to(device)
    
    # Create environments with Monitor wrapper
    original_env = Monitor(gym.make('CartPole-v1'))
    pinn_env = Monitor(PINNCartPoleEnv(pinn_model, params))

    # Train a policy on the original environment
    print("Training PPO agent on original CartPole environment...")
    ppo_model = PPO('MlpPolicy', original_env, verbose=1, device=device)
    ppo_model.learn(total_timesteps=5) # TODO: Adjust total_timesteps as needed

    # Evaluate on both environments
    print("Evaluating on original environment:")
    original_mean, original_std = evaluate_env(original_env, ppo_model, num_episodes)
    print(f"Mean reward: {original_mean:.2f} +/- {original_std:.2f}")

    print("\nEvaluating on PINN-based environment:")
    pinn_mean, pinn_std = evaluate_env(pinn_env, ppo_model, num_episodes)
    print(f"Mean reward: {pinn_mean:.2f} +/- {pinn_std:.2f}")

    # Collect trajectories
    original_states, original_actions, original_rewards = collect_trajectory(original_env, ppo_model, max_steps, visualize)
    pinn_states, pinn_actions, pinn_rewards = collect_trajectory(pinn_env, ppo_model, max_steps, visualize)

    # Plot state comparisons
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

    # Close environments
    original_env.close()
    pinn_env.close()

    return original_rewards, pinn_rewards

if __name__ == "__main__":
    print("Comparing environments with preloaded trained model...")
    # Load your trained PINN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn_model = CartpolePINN(predict_friction=False, sequence_length=10)  # Adjust sequence_length as needed
    try:
        pinn_model.load_state_dict(torch.load('../model_archive/270824_pinnModel.pth', map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Proceeding with untrained model.")
    pinn_model = pinn_model.to(device)
    pinn_model.eval()

    params = {
        "m_c": 0.466,
        "m_p": 0.06,
        "l": 0.201,
        "g": 9.81,
        "mu_c": 0.1,
        "mu_p": 0.01,
        "force_mag": 10.0
    }

    compare_environments(pinn_model, params, visualize=True)