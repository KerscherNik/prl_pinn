import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gym_integration import PINNCartPoleEnv
from pinn_model import CartpolePINN

# evaluate policy on given environment by mean and standard deviation of the rewards
def evaluate_env(env, model, num_episodes=100):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_episodes)
    return mean_reward, std_reward

# Collects trajectories (states, actions, rewards) of the policy in a given environment
# Runs the env for max_steps, predicts next actions and returns the states, actions and rewards
def collect_trajectory(env, model, max_steps=500):
    obs, _ = env.reset()  # Unpack both observation and info, discard info
    states, actions, rewards = [], [], []
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        states.append(obs)
        actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    return np.array(states), np.array(actions), np.array(rewards)


# Compares performance of policy in original CartPole env and in PINN-based env
# Trains a policy on gym env to compare performance of this policy on gym and pinn env
# and collects trajectories for that
def compare_environments(pinn_model, params, predict_friction, num_episodes=100, max_steps=500):
    # Create environments with Monitor wrapper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn_model = pinn_model.to(device)
    original_env = Monitor(gym.make('CartPole-v1'))
    pinn_env = Monitor(PINNCartPoleEnv(pinn_model, params, predict_friction))

    # Train a policy on the original environment (MLP policy and verbose=1 to enable logging of training progress
    print("Training PPO agent on original CartPole environment...")
    ppo_model = PPO('MlpPolicy', original_env, verbose=1, device=device)
    ppo_model.learn(total_timesteps=50000)

    # Evaluate on both environments
    print("Evaluating on original environment:")
    original_mean, original_std = evaluate_env(original_env, ppo_model)
    print(f"Mean reward: {original_mean:.2f} +/- {original_std:.2f}")

    print("\nEvaluating on PINN-based environment:")
    pinn_mean, pinn_std = evaluate_env(pinn_env, ppo_model)
    print(f"Mean reward: {pinn_mean:.2f} +/- {pinn_std:.2f}")

    # Collect trajectories
    original_states, original_actions, original_rewards = collect_trajectory(original_env, ppo_model)
    pinn_states, pinn_actions, pinn_rewards = collect_trajectory(pinn_env, ppo_model)

    # Plot state comparisons
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    state_labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']
    for i in range(4):
        axs[i // 2, i % 2].plot(original_states[:, i], label='Original')
        axs[i // 2, i % 2].plot(pinn_states[:, i], label='PINN')
        axs[i // 2, i % 2].set_title(state_labels[i])
        axs[i // 2, i % 2].legend()
    plt.tight_layout()
    plt.savefig('state_comparison.png')
    plt.close()

    # Plot reward comparison
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(original_rewards), label='Original')
    plt.plot(np.cumsum(pinn_rewards), label='PINN')
    plt.title('Cumulative Reward')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.savefig('reward_comparison.png')
    plt.close()

    # Close environments
    original_env.close()
    pinn_env.close()

if __name__ == "__main__":
    print("Comparing environments with preloaded trained model...")
    # Load your trained PINN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn_model = CartpolePINN(predict_friction=False)
    pinn_model.load_state_dict(torch.load('trained_pinn_model.pth', weights_only=True, map_location=device))
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

    compare_environments(pinn_model, params)