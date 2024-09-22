import gymnasium as gym
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Output to console
                        logging.FileHandler('app.log')  # Optionally log to a file
                    ])

logger = logging.getLogger(__name__)

env = gym.make('CartPole-v1')
data = []

num_episodes = 500

logger.info("Starting data collection for %d episodes of CartPole.", num_episodes)

for episode in range(num_episodes):
    observation, info = env.reset()

    new_init_state = np.array([
        np.random.random() * 2 - 1,  # cartPos
        np.random.random() * 12 - 6,  # cartVel
        np.random.random() * 2 * np.pi,  # pendPos (terminates if angle not in range (-.2095, .2095))
        np.random.random() * 40 - 20  # pendVel
    ]).astype(np.float32)

    env.state = new_init_state
    observation = new_init_state

    done = False
    logger.debug("Episode %d started with initial state: %s", episode + 1, new_init_state)

    while not done:
        action = env.action_space.sample()
        next_observation, reward, done, truncated, info = env.step(action)
        done = done or truncated

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        data.append([current_time, observation[0], observation[1], observation[2], observation[3], action])

        observation = next_observation

    logger.debug("Episode %d finished.", episode + 1)

df = pd.DataFrame(data, columns=['datetime', 'cartPos', 'cartVel', 'pendPos', 'pendVel', 'action'])
df.to_csv('data/cartpole_data.csv', index=False, sep=';')

logger.info("Data saved in 'data/cartpole_data.csv'.")
