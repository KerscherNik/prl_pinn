import gym
import gym_cartpole_swingup
import pandas as pd
import numpy as np
from datetime import datetime

# Create the CartPoleSwingUp environment
env = gym.make('CartPoleSwingUp-v0')
data = []

num_episodes = 1

for episode in range(num_episodes):
    observation = env.reset()

    # Set random initial state similar to your approach
    new_init_state = np.array([
        np.random.random() * 2 - 1,  # cartPos
        np.random.random() * 12 - 6,  # cartVel
        np.random.random() * 2 * np.pi,  # pendPos
        np.random.random() * 40 - 20  # pendVel
    ]).astype(np.float32)

    env.state = new_init_state
    observation = new_init_state

    done = False

    while not done:
        action = env.action_space.sample()  # Random action

        next_observation, reward, done, truncated = env.step(action)

        done = done or truncated

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        # Save the observation and action data
        data.append([current_time, observation[0], observation[1], observation[2], observation[3], action])

        observation = next_observation

# Save the data to a CSV file
df = pd.DataFrame(data, columns=['datetime', 'cartPos', 'cartVel', 'pendPos', 'pendVel', 'action'])
df.to_csv('cartpole_swingup_data.csv', index=False, sep=';')

print("Data saved in 'cartpole_swingup_data.csv'.")
