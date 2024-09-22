import gymnasium as gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from collections import deque
import io

from integration.gym_integration import PINNCartPoleEnv
from model.pinn_model import CartpolePINN
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('app.log')
                    ])

logger = logging.getLogger(__name__)
class InputBuffer:
    def __init__(self, buffer_size=5):
        self.buffer = deque(maxlen=buffer_size)
        self.last_execution_time = time.time()

    def add_input(self, action):
        self.buffer.append((action, time.time()))

    def get_action(self, delay=0.1):
        current_time = time.time()
        if self.buffer and current_time - self.last_execution_time >= delay:
            self.last_execution_time = current_time
            return self.buffer.popleft()[0]
        return None

def handle_input(input_buffer):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                input_buffer.add_input(0)
            elif event.key == pygame.K_RIGHT:
                input_buffer.add_input(1)
            elif event.key == pygame.K_q:
                return True
    return False

def create_side_by_side_env(env1, env2):
    class SideBySideEnv(gym.Env):
        def __init__(self, env1, env2):
            self.env1 = env1
            self.env2 = env2
            self.action_space = env1.action_space
            self.observation_space = gym.spaces.Dict({
                'env1': env1.observation_space,
                'env2': env2.observation_space
            })

        def reset(self):
            obs1, _ = self.env1.reset()
            obs2, _ = self.env2.reset()
            return {'env1': obs1, 'env2': obs2}, {}

        def step(self, action):
            obs1, reward1, done1, truncated1, info1 = self.env1.step(action)
            obs2, reward2, done2, truncated2, info2 = self.env2.step(action)
            return ({'env1': obs1, 'env2': obs2}, 
                    {'env1': reward1, 'env2': reward2},  # Return rewards for both environments
                    done1 or done2,
                    truncated1 or truncated2,
                    {'env1': info1, 'env2': info2})

        def render(self):
            # Rendering is handled externally
            pass

    return SideBySideEnv(env1, env2)

def visualize_side_by_side(env1, env2, ppo_model=None, max_steps=500, slow_motion_factor=0.1):
    pygame.init()
    screen = pygame.display.set_mode((1000, 600))
    pygame.display.set_caption('CartPole Environment Comparison')

    clock = pygame.time.Clock()

    side_by_side_env = create_side_by_side_env(env1, env2)
    obs, _ = side_by_side_env.reset()
    
    step = 0
    done = False
    use_ppo = ppo_model is not None

    font = pygame.font.Font(None, 24)

    env1_rewards = []
    env2_rewards = []
    current_env1_reward = 0
    current_env2_reward = 0
    termination_reason = "Maximum Steps Reached"

    input_buffer = InputBuffer()

    while not done and step < max_steps:
        screen.fill((255, 255, 255))

        done = handle_input(input_buffer)

        if use_ppo:
            action, _ = ppo_model.predict(obs['env1'], deterministic=True)
        else:
            if step == 0:
                action = env1.action_space.sample()
            else:
                action = input_buffer.get_action(delay=slow_motion_factor)
                if action is None:
                    continue  # Skip this frame if no action is available

        prev_obs = obs
        obs, rewards, done, truncated, info = side_by_side_env.step(action)
        current_env1_reward += rewards['env1']
        current_env2_reward += rewards['env2']

        # Render both environments
        env1_array = env1.render()
        env2_array = env2.render()

        # Convert numpy arrays to pygame surfaces
        env1_surface = pygame.surfarray.make_surface(env1_array.swapaxes(0, 1))
        env2_surface = pygame.surfarray.make_surface(env2_array.swapaxes(0, 1))

        # Scale surfaces if necessary
        env1_surface = pygame.transform.scale(env1_surface, (400, 400))
        env2_surface = pygame.transform.scale(env2_surface, (400, 400))

        screen.blit(env1_surface, (0, 0))
        screen.blit(env2_surface, (500, 0))

        # Add labels
        text1 = font.render("Original Env", True, (0, 0, 0))
        text2 = font.render("PINN Env", True, (0, 0, 0))
        screen.blit(text1, (10, 10))
        screen.blit(text2, (510, 10))

        # Add real-time info
        info_text = [
            f"Step: {step}",
            f"Action: {'Left' if action == 0 else 'Right'}",
            f"Env1 Reward: {rewards['env1']:.2f}",
            f"Env2 Reward: {rewards['env2']:.2f}",
            f"Cumulative Env1 Reward: {current_env1_reward:.2f}",
            f"Cumulative Env2 Reward: {current_env2_reward:.2f}",
            f"Original Env State: {obs['env1']}",
            f"PINN Env State: {obs['env2']}",
        ]
        for i, text in enumerate(info_text):
            info_surface = font.render(text, True, (0, 0, 0))
            screen.blit(info_surface, (10, 420 + i * 20))

        # Check for significant differences between environments
        state_diff = np.abs(obs['env1'] - obs['env2'])
        if np.any(state_diff > 0.5):  # Threshold for significant difference
            diff_text = font.render("Significant state difference detected!", True, (255, 0, 0))
            screen.blit(diff_text, (10, 560))

        # Add controls legend
        legend_text = [
            "Controls:",
            "Left Arrow: Push Left",
            "Right Arrow: Push Right",
            "Q: Quit"
        ]
        for i, text in enumerate(legend_text):
            legend_surface = font.render(text, True, (0, 0, 0))
            screen.blit(legend_surface, (800, 420 + i * 20))

        pygame.display.flip()
        clock.tick(int(60 * slow_motion_factor))

        step += 1

        logger.info(f"Step: {step}, Action: {action}, Env1 Reward: {rewards['env1']}, Env2 Reward: {rewards['env2']}")
        logger.info(f"Original Env State: {obs['env1']}")
        logger.info(f"PINN Env State: {obs['env2']}")
        if isinstance(env2, PINNCartPoleEnv):
            logger.info(f"PINN Predicted Force: {info['env2']['predicted_force']}")
            logger.info(f"PINN Scaled Force: {info['env2']['scaled_force']}")

        env1_rewards.append(rewards['env1'])
        env2_rewards.append(rewards['env2'])

        # Check termination conditions
        if done:
            if abs(obs['env1'][0]) > env1.x_threshold or abs(obs['env2'][0]) > env2.x_threshold:
                termination_reason = f"{'Env1 (Original)' if abs(obs['env1'][0]) > env1.x_threshold else 'Env2 (PINN)'} Cart Position Limit Exceeded"
            elif abs(obs['env1'][2]) > env1.theta_threshold_radians or abs(obs['env2'][2]) > env2.theta_threshold_radians:
                termination_reason = f"{'Env1 (Original)' if abs(obs['env1'][2]) > env1.theta_threshold_radians else 'Env2 (PINN)'} Pole Angle Limit Exceeded"
        elif step >= max_steps:
            termination_reason = "Maximum Steps Reached"

    env1.close()
    env2.close()
    pygame.quit()

    # Display end-of-episode statistics
    print(f"\nEpisode ended. Reason: {termination_reason}")
    print(f"Total Steps: {step}")
    print(f"Final Env1 (Original) Reward: {current_env1_reward:.2f}")
    print(f"Final Env2 (PINN) Reward: {current_env2_reward:.2f}")

    # Create a plot comparing both environments
    plt.figure(figsize=(10, 5))
    plt.plot(env1_rewards, label='Original Env')
    plt.plot(env2_rewards, label='PINN Env')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title('Reward Comparison: Original vs PINN Environment')
    plt.legend()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Display the plot using pygame
    plot_surface = pygame.image.load(buf)
    plot_surface = pygame.transform.scale(plot_surface, (800, 400))
    
    screen = pygame.display.set_mode((800, 400))
    pygame.display.set_caption('Reward Comparison Plot')
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((255, 255, 255))
        screen.blit(plot_surface, (0, 0))
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    logger.info("Starting side-by-side CartPole visualization.")

    sequence_length = 5
    params = {
        "m_c": 1.0,
        "m_p": 0.1,
        "l": 0.5,
        "g": 9.8,
        "mu_c": 0.0,
        "mu_p": 0.0,
        "force_mag": 10.0,
        "tau": 0.02
    }

    saved_model_path = 'model_archive/trained_pinn_model_without_friction_20240922_183400.pth'
    loaded_model = CartpolePINN(sequence_length, predict_friction=False)
    loaded_model.load_state_dict(torch.load(saved_model_path))

    original_env = gym.make('CartPole-v1', render_mode="rgb_array")
    pinn_env = PINNCartPoleEnv(loaded_model, params, render_mode="rgb_array")

    # Train a PPO model on the original environment
    ppo_model = PPO("MlpPolicy", Monitor(original_env), verbose=1)
    ppo_model.learn(total_timesteps=5000)

    # Visualize with PPO model control
    visualize_side_by_side(original_env, pinn_env, ppo_model=ppo_model, slow_motion_factor=0.2)

    # Uncomment to visualize with keyboard control instead
    #visualize_side_by_side(original_env, pinn_env, slow_motion_factor=0.6)