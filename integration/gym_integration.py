import gymnasium as gym
import torch
import numpy as np
from scipy.integrate import solve_ivp
from model.pinn_model import CartpolePINN
import logging
import traceback

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('app.log')
                    ])

logger = logging.getLogger(__name__)

class PINNCartPoleEnv(gym.Env):
    def __init__(self, pinn_model, params, render_mode=None):
        super(PINNCartPoleEnv, self).__init__()
        logger.info("Initializing the PINN CartPole environment.")

        self.pinn_model = pinn_model
        self.params = params
        self.predict_friction = isinstance(self.pinn_model, CartpolePINN) and self.pinn_model.predict_friction
        
        self.gravity = 9.8
        self.masscart = params['m_c']
        self.masspole = params['m_p']
        self.total_mass = self.masspole + self.masscart
        self.length = params['l']
        self.polemass_length = self.masspole * self.length
        self.force_mag = params['force_mag']
        self.tau = params['tau']

        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

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

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True
        self.info = {}

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        logger.debug(f"Step called with action: {action}")

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        logger.debug(f"Current state: x={x}, x_dot={x_dot}, theta={theta}, theta_dot={theta_dot}")
        logger.debug(f"Applied force: {force}")

        # Prepare input for PINN model
        new_state = torch.tensor([x, x_dot, theta, theta_dot, float(action)], 
                                 dtype=torch.float32, 
                                 device=self.pinn_model.device).unsqueeze(0).unsqueeze(0)
        logger.debug(f"Prepared new state tensor: {new_state}")

        if self.sequence_buffer is None:
            self.sequence_buffer = new_state.repeat(1, self.sequence_length, 1)
            logger.debug("Initialized sequence buffer")
        else:
            self.sequence_buffer = torch.cat([self.sequence_buffer[:, 1:, :], new_state], dim=1)
            logger.debug("Updated sequence buffer")

        logger.debug(f"Sequence buffer shape: {self.sequence_buffer.shape}")

        # Predict force using PINN
        try:
            with torch.no_grad():
                if self.predict_friction:
                    predicted_force, mu_c, mu_p = self.pinn_model(self.sequence_buffer)
                    predicted_force = predicted_force.item()
                    mu_c = mu_c.item()
                    mu_p = mu_p.item()
                    logger.debug(f"PINN prediction: force={predicted_force}, mu_c={mu_c}, mu_p={mu_p}")
                else:
                    predicted_force = self.pinn_model(self.sequence_buffer).item()
                    mu_c, mu_p = self.params['mu_c'], self.params['mu_p']
                    logger.debug(f"PINN prediction: force={predicted_force}")
        except Exception as e:
            logger.error(f"Error during PINN prediction: {e}")
            logger.error(traceback.format_exc())
            raise

        # Scale the predicted force to match the original environment
        scaled_force = predicted_force * self.force_mag
        logger.debug(f"Scaled force: {scaled_force}")

        # Update params with predicted friction if applicable
        current_params = self.params.copy()
        if self.predict_friction:
            current_params['mu_c'] = mu_c
            current_params['mu_p'] = mu_p
            logger.debug(f"Updated params with predicted friction: mu_c={mu_c}, mu_p={mu_p}")

        # Use scaled force for state estimation
        def cartpole_ode(t, y):
            x, x_dot, theta, theta_dot = y
            costheta = np.cos(theta)
            sintheta = np.sin(theta)

            temp = (scaled_force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

            return [x_dot, xacc, theta_dot, thetaacc]

        # Solve ODE to get next state
        try:
            sol = solve_ivp(cartpole_ode, [0, self.tau], [x, x_dot, theta, theta_dot], method='RK45')
            self.state = sol.y[:, -1]
            logger.debug(f"ODE solution: {sol.y[:, -1]}")
        except Exception as e:
            logger.error(f"Error during ODE solution: {e}")
            logger.error(traceback.format_exc())
            raise

        x, x_dot, theta, theta_dot = self.state
        logger.debug(f"New state: x={x}, x_dot={x_dot}, theta={theta}, theta_dot={theta_dot}")

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        logger.debug(f"Done: {done}")

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
            logger.debug("Pole just fell! Steps beyond done set to 0.")
        else:
            if self.steps_beyond_done == 0:
                logger.warning("Calling 'step()' even though environment is done.")
            self.steps_beyond_done += 1
            reward = 0.0

        logger.debug(f"Reward: {reward}")

        if self.render_mode == "human":
            self.render()
        
        info = {
            "predicted_force": predicted_force,
            "scaled_force": scaled_force,
            "reward": reward
        }
        if self.predict_friction:
            info["predicted_mu_c"] = mu_c
            info["predicted_mu_p"] = mu_p
        
        self.info = info
        logger.debug(f"Step completed. Info: {info}")
        return np.array(self.state, dtype=np.float32), reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.sequence_buffer = None
        logger.debug("Environment reset.")
        
        if self.render_mode == "human":
            self.render()
        
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            logger.warn("You are calling render method without specifying any render mode. "
                        "You can specify the render_mode at initialization, "
                        "e.g. PINNCartPoleEnv(..., render_mode='rgb_array')")
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install pygame`"
            )

        screen_width = 600
        screen_height = 400

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:  # rgb_array
                self.screen = pygame.Surface((screen_width, screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(50)
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False