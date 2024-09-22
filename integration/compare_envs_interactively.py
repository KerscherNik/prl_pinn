import gymnasium as gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
import time
from stable_baselines3.common.monitor import Monitor

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

def visualize_interactive(env1, env2, max_steps=500, visualize=True, slow_motion_factor=0.5):
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption('CartPole Environment Interaction')

    pygame.key.set_repeat(50, 50)

    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    action = 0

    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        update_plot1 = create_animation(fig, ax1, "pinn", limits=(-2.4, 2.4))
        update_plot2 = create_animation(fig, ax2, "ori", limits=(-2.4, 2.4))

    clock = pygame.time.Clock()

    for step in range(max_steps):
        obs1, reward1, terminated1, truncated1, _ = env1.step(action)
        obs2, reward2, terminated2, truncated2, _ = env2.step(action)
        logger.info(f"Step: {step + 1}, Action: {action}, Reward1: {reward1}, Reward2: {reward2}")

        if visualize:
            update_plot1(obs1)
            update_plot2(obs2)
            plt.pause(0.01)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                logger.warning("Quit event detected, closing environment.")
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    action = 0
                    logger.debug("Key pressed: Left (a), moving cart left.")
                elif event.key == pygame.K_d:
                    action = 1
                    logger.debug("Key pressed: Right (d), moving cart right.")
        
        if terminated1 or truncated1 or terminated2 or truncated2:
            logger.info(f"Environment finished after {step + 1} steps")
            break

        time.sleep(slow_motion_factor)
        clock.tick(30)

    env1.close()
    env2.close()
    pygame.quit()
    if visualize:
        plt.show()

if __name__ == "__main__":
    logger.info("Starting interactive CartPole visualization.")
    sequence_length = 5
    params = {
        "m_c": 1.0,
        "m_p": 0.1,
        "l": 1.0,
        "g": 9.8,
        "mu_c": 0.0,
        "mu_p": 0.0,
        "force_mag": 10.0
    }

    saved_model_path = f'model_archive/trained_pinn_model_without_friction_20240919_092248.pth'
    loaded_model = CartpolePINN(sequence_length, predict_friction=False)
    loaded_model.load_state_dict(torch.load(saved_model_path))

    original_env = Monitor(gym.make('CartPole-v1'))
    pinn_env = Monitor(PINNCartPoleEnv(loaded_model, params))
    visualize_interactive(pinn_env, original_env, max_steps=500, visualize=True, slow_motion_factor=0.1)
