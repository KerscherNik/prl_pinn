import gymnasium as gym
import pandas as pd
import numpy as np
from datetime import datetime

env = gym.make('CartPole-v1')
data = []

num_episodes = 5000

for episode in range(num_episodes):
    # reset environment to start new episode
    observation, info = env.reset()

    # random init state
    new_init_state = np.array([
        np.random.random() * 2 - 1,  # cartPos
        np.random.random() * 12 - 6,  # cartVel
        np.random.random() * 2 * np.pi,  # pendPos
        np.random.random() * 40 - 20  # pendVel
    ]).astype(np.float32)

    env.env.state = new_init_state

    # set observation to new random init state manually
    observation = new_init_state

    done = False

    while not done:
        action = env.action_space.sample()

        next_observation, reward, done, truncated, info = env.step(action)

        done = done or truncated

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        #if not done:
        data.append([current_time, observation[0], observation[1], observation[2], observation[3], action])

        observation = next_observation

df = pd.DataFrame(data, columns=['datetime', 'cartPos', 'cartVel', 'pendPos', 'pendVel', 'action'])
df.to_csv('cartpole_data.csv', index=False, sep=';')

print("Data saved in 'cartpole_data.csv'.")
