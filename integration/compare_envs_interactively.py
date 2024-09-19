import gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
import time
from stable_baselines3.common.monitor import Monitor

from integration.gym_integration import PINNCartPoleEnv
from model.pinn_model import CartpolePINN


def create_animation(fig, ax, limits=(-2.4, 2.4)):
    """
    Create the initial figure and elements for the animation.
    """
    cart_plot, = ax.plot([], [], 's-', markersize=20)  # Cart
    pend_plot, = ax.plot([], [], 'o-', markersize=10)  # Pendulum
    ax.set_xlim(limits)  # CartPole cart limits
    ax.set_ylim(-1.5, 1.5)
    ax.set_title('CartPole Visualization')
    ax.set_xlabel('Cart Position')
    ax.set_ylabel('Pendulum Position')

    def update(obs):
        x = obs[0]  # Cart position
        theta = obs[2] - np.pi  # Pendulum angle

        cart_plot.set_data([x], [0])
        pend_x = x + np.sin(theta)
        pend_y = -np.cos(theta)
        pend_plot.set_data([x, pend_x], [0, pend_y])
        return cart_plot, pend_plot

    return update

def visualize_interactive(env1, env2, max_steps=500, visualize=True, slow_motion_factor=0.5):
    """
    Visualize and control two CartPole environments interactively using 'a' and 'd' keys.

    Args:
    env1 (gym.Env): First CartPole environment.
    env2 (gym.Env): Second CartPole environment.
    max_steps (int): Maximum number of steps.
    visualize (bool): Whether to visualize the environments.
    slow_motion_factor (float): Factor to slow down the environments (0.5 is half speed).
    """
    # Initialize pygame for interactive control
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption('CartPole Environment Interaction')

    # Enable key repeat for continuous movement
    pygame.key.set_repeat(50, 50)  # (delay, interval)

    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    action = 0  # Default action

    # Create the animation with matplotlib
    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Two subplots for two envs
        update_plot1 = create_animation(fig, ax1, limits=(-2.4, 2.4))
        update_plot2 = create_animation(fig, ax2, limits=(-2.4, 2.4))

    clock = pygame.time.Clock()

    for step in range(max_steps):
        # Step both environments
        obs1, reward1, terminated1, truncated1, _ = env1.step(action)
        obs2, reward2, terminated2, truncated2, _ = env2.step(action)
        print(f"Step: {step + 1}, Action: {action}, Reward1: {reward1}, Reward2: {reward2}")

        # Render matplotlib visualizations for both environments
        if visualize:
            update_plot1(obs1)
            update_plot2(obs2)
            plt.pause(0.01)  # Pause to allow animation update

        # Handle user input (a and d keys)
        for event in pygame.event.get():
            print("Event detected:", event.type)
            if event.type == pygame.QUIT:
                pygame.quit()
                print("Quit event detected.")
                return
            elif event.type == pygame.KEYDOWN:
                print(f"Key pressed: {event.key}")
                if event.key == pygame.K_a:  # Press 'a' for moving left
                    action = 0  # Move cart to the left
                    print("Moving Left")
                elif event.key == pygame.K_d:  # Press 'd' for moving right
                    action = 1  # Move cart to the right
                    print("Moving Right")
            elif event.type == pygame.KEYUP:
                print(f"Key released: {event.key}")

        # Check for termination in either environment
        if terminated1 or truncated1 or terminated2 or truncated2:
            print(f"Environment finished after {step + 1} steps")
            break

        # Slow down the loop
        time.sleep(slow_motion_factor)
        clock.tick(30)  # Control the speed of pygame loop

    # Close environments and pygame
    env1.close()
    env2.close()
    pygame.quit()
    if visualize:
        plt.show()

if __name__ == "__main__":
    sequence_length = 5  # Adjust this value as needed

    params = {
        "m_c": 0.466,
        "m_p": 0.06,
        "l": 0.201,
        "g": 9.81,
        "mu_c": 0.1,
        "mu_p": 0.01,
        "force_mag": 10.0
    }

    # Load the already saved model
    saved_model_path = f'model_archive/trained_pinn_model_without_friction_20240919_092248.pth'
    loaded_model = CartpolePINN(sequence_length, predict_friction=False)
    loaded_model.load_state_dict(torch.load(saved_model_path))

    # visualize interactively: uncomment only this block and the block above to load the pinn model
    original_env = Monitor(gym.make('CartPole-v1'))
    pinn_env = Monitor(PINNCartPoleEnv(loaded_model, params))
    visualize_interactive(pinn_env, original_env, max_steps=500, visualize=True, slow_motion_factor=0.1)
