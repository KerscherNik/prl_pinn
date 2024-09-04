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

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state

        # Get force from PINN
        with torch.no_grad():
            t = torch.zeros(1)
            force = self.pinn_model(t, torch.tensor([x]), torch.tensor([x_dot]), 
                                    torch.tensor([theta]), torch.tensor([theta_dot]))
            force = force.item() * self.force_mag

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # Use equations from the original CartPole environment
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        truncated = False # We don't truncate episodes in this environment

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.00

        return np.array(self.state, dtype=np.float32), reward, done, truncated, {}


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32), {}
