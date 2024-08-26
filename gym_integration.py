import gymnasium as gym
import torch
import numpy as np

class PINNCartPoleEnv(gym.Env):
    def __init__(self, pinn_model, params):
        super().__init__()
        self.pinn_model = pinn_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pinn_model.to(device)
        
        self.params = params
        
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        
        self.state = None
        self.steps_beyond_done = None
        
        # Define thresholds
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * np.pi / 360

    def step(self, action):
        assert self.action_space.contains(action), f"{action} is an invalid action"
        
        x, x_dot, theta, theta_dot = self.state
        
        # Convert action to force
        force = self.params['force_mag'] if action == 1 else -self.params['force_mag']
        
        # Prepare input for PINN model
        t = torch.tensor([0.0])
        state_tensor = torch.tensor([x, x_dot, theta, theta_dot], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            if self.pinn_model.predict_friction:
                F, _, _ = self.pinn_model(t, state_tensor[:, 0], state_tensor[:, 1], state_tensor[:, 2], state_tensor[:, 3])
            else:
                F = self.pinn_model(t, state_tensor[:, 0], state_tensor[:, 1], state_tensor[:, 2], state_tensor[:, 3])
        
        # Use F to update the state (simple Euler integration)
        dt = 0.02  # time step
        x_ddot = F.item()
        theta_ddot = self.params['g'] * np.sin(theta) / self.params['l']
        
        x += x_dot * dt
        x_dot += x_ddot * dt
        theta += theta_dot * dt
        theta_dot += theta_ddot * dt
        
        self.state = (x, x_dot, theta, theta_dot)

        # Determine if the episode is terminated
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        # For simplicity, let's say the episode is never truncated
        truncated = False

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32), {}
