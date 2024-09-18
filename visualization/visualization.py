import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd

def create_animation(obs, max_steps, visualize):
    if not visualize:
        return None

    fig, ax = plt.subplots()
    cart_plot, = plt.plot([], [], 's-', markersize=20)  # Cart
    pend_plot, = plt.plot([], [], 'o-', markersize=10)  # Pendulum
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title('CartPole Visualization')
    ax.set_xlabel('Cart Position')
    ax.set_ylabel('Pendulum Position')

    def init():
        cart_plot.set_data([], [])
        pend_plot.set_data([], [])
        return cart_plot, pend_plot

    def update(frame):
        x = obs[0]  # Cart position
        theta = obs[2] - np.pi # Pendulum angle

        cart_plot.set_data([x], [0])
        pend_x = x + np.sin(theta)
        pend_y = -np.cos(theta)
        pend_plot.set_data([x, pend_x], [0, pend_y])
        return cart_plot, pend_plot

    ani = FuncAnimation(fig, update, frames=max_steps, init_func=init, blit=True)

    return ani

def plot_trajectory(states, actions, rewards, visualize):
    if visualize:
        plt.pause(0.05)  # Pause for the animation to update

    return states, actions, rewards


def visualize_csv_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    cart_pos = df['cartPos'].values
    pend_pos = df['pendPos'].values

    cart_width = 0.2
    cart_height = 0.1
    pole_length = 1.0

    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.5, 1.5)

    cart_plot, = plt.plot([], [], 's-', markersize=20)  # Cart
    pend_plot, = plt.plot([], [], 'o-', markersize=10)  # Pendulum

    def init():
        cart_plot.set_data([], [])
        pend_plot.set_data([], [])
        return cart_plot, pend_plot

    def update(frame):
        x = cart_pos[frame]
        theta = pend_pos[frame] - np.pi

        cart_plot.set_data([x], [0])

        pend_x = x + pole_length * np.sin(theta)
        pend_y = -pole_length * np.cos(theta)

        pend_plot.set_data([x, pend_x], [0, pend_y])

        return cart_plot, pend_plot

    ani = FuncAnimation(fig, update, frames=len(cart_pos), init_func=init, blit=True, interval=50)
    plt.show()

# Example usage for visualization from CSV file
if __name__ == "__main__":
    # visualize_csv_data('../data/cartpole_data.csv')
    visualize_csv_data('../data/cartpole_swingup_data.csv')