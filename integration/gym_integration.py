import gymnasium as gym
import torch
import numpy as np

class PINNCartPoleEnv(gym.Env):
    def __init__(self, pinn_model, params):
        super(PINNCartPoleEnv, self).__init__()
        self.pinn_model = pinn_model
        self.params = params
        
        self.gravity = 9.8
        self.masscart = params['m_c']
        self.masspole = params['m_p']
        self.total_mass = self.masspole + self.masscart
        self.length = params['l']
        self.polemass_length = self.masspole * self.length
        self.force_mag = params['force_mag']
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        # Define threshold for cart position, cart velocity, pole angle, pole angular velocity
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.state = None
        self.steps_beyond_done = None
        self.sequence_length = self.pinn_model.sequence_length
        self.sequence_buffer = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state

        # Update sequence buffer
        new_state = torch.tensor([x, x_dot, theta, theta_dot, float(action)], 
                                 dtype=torch.float32, 
                                 device=self.pinn_model.device).unsqueeze(0).unsqueeze(0)
        if self.sequence_buffer is None:
            self.sequence_buffer = new_state.repeat(1, self.sequence_length, 1)
        else:
            self.sequence_buffer = torch.cat([self.sequence_buffer[:, 1:, :], new_state], dim=1)

        # Get force from PINN
        with torch.no_grad():
            force = self.pinn_model(self.sequence_buffer)
            force = force.item() * self.params['force_mag']

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.sequence_buffer = None
        return np.array(self.state, dtype=np.float32), {}