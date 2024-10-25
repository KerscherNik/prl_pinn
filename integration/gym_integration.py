import gymnasium as gym
import torch
import numpy as np
from scipy.integrate import solve_ivp
from model.pinn_model import CartpolePINN
import logging
import traceback
import time
from typing import Optional

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('gym_integration.log')
                    ])

logger = logging.getLogger(__name__)

class PINNCartPoleEnv(gym.Env):
    """
    CartPole environment with PINN integration for force prediction.
    Closely follows the original CartPole implementation while adding PINN capabilities.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, pinn_model, params, render_mode: Optional[str] = None):
        logger.debug("Initializing PINN CartPole environment")
        super(PINNCartPoleEnv, self).__init__()
        
        self.pinn_model = pinn_model
        self.render_mode = render_mode
        
        # Step counting
        self.current_step = 0
        self.max_episode_steps = 500  # Explicit step limit
        
        # Physical parameters
        self.gravity = 9.8
        self.masscart = params.get('m_c', 1.0)
        self.masspole = params.get('m_p', 0.1)
        self.total_mass = self.masspole + self.masscart
        self.length = params.get('l', 0.5)  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = params.get('force_mag', 10.0)
        self.tau = params.get('tau', 0.02)  # seconds between state updates
        
        # Thresholds for episode termination
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        # Spaces
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max,
        ], dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # PINN-specific attributes
        self.sequence_length = self.pinn_model.sequence_length
        self.sequence_buffer = None
        self.predict_friction = hasattr(self.pinn_model, 'predict_friction') and self.pinn_model.predict_friction

        # Rendering setup
        self.screen = None
        self.clock = None
        self.isopen = True

        # Episode tracking
        self.steps_beyond_terminated = None
        self.state = None

    def step(self, action):
        logger.debug(f"Step {self.current_step} called with action: {action}")
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        # Increment step counter
        self.current_step += 1
        
        # Check for step limit before processing action
        if self.current_step >= self.max_episode_steps:
            logger.debug(f"Episode truncated after {self.current_step} steps")
            return (
                np.array(self.state, dtype=np.float32),
                1.0,  # Reward for truncated episode
                False,  # terminated
                True,  # truncated
                {"episode_length": self.current_step}
            )

        # Prepare state for PINN
        x, x_dot, theta, theta_dot = self.state
        try:
            new_state = torch.tensor([x, x_dot, theta, theta_dot, float(action)], 
                                   dtype=torch.float32, 
                                   device=self.pinn_model.device).unsqueeze(0).unsqueeze(0)

            # Update sequence buffer
            if self.sequence_buffer is None:
                self.sequence_buffer = new_state.repeat(1, self.sequence_length, 1)
            else:
                self.sequence_buffer = torch.cat([self.sequence_buffer[:, 1:, :], new_state], dim=1)

            # Get PINN predictions
            with torch.no_grad():
                if self.predict_friction:
                    predicted_force, mu_c, mu_p = self.pinn_model(self.sequence_buffer)
                    force = predicted_force.item() * self.force_mag
                    logger.debug(f"PINN predictions: force={force:.4f}, mu_c={mu_c.item():.4f}, mu_p={mu_p.item():.4f}")
                else:
                    predicted_force = self.pinn_model(self.sequence_buffer)
                    force = predicted_force.item() * self.force_mag
                    logger.debug(f"PINN prediction: force={force:.4f}")
        except Exception as e:
            logger.error(f"Error during PINN prediction: {str(e)}")
            # Return terminal state in case of PINN error
            return (
                np.array(self.state, dtype=np.float32),
                0.0,
                True,
                False,
                {"error": "PINN prediction failed"}
            )

        # Physics calculations
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        # Detailed termination checking
        termination_reason = None
        if x < -self.x_threshold:
            termination_reason = "x_position_left_threshold"
            logger.debug(f"Episode TERMINATED: Cart position ({x:.3f}) < left threshold (-{self.x_threshold})")
        elif x > self.x_threshold:
            termination_reason = "x_position_right_threshold"
            logger.debug(f"Episode TERMINATED: Cart position ({x:.3f}) > right threshold ({self.x_threshold})")
        elif theta < -self.theta_threshold_radians:
            termination_reason = "theta_below_threshold"
            logger.debug(f"Episode TERMINATED: Pole angle ({theta:.3f} rad) < min threshold (-{self.theta_threshold_radians})")
        elif theta > self.theta_threshold_radians:
            termination_reason = "theta_above_threshold"
            logger.debug(f"Episode TERMINATED: Pole angle ({theta:.3f} rad) > max threshold ({self.theta_threshold_radians})")
        elif not np.isfinite(x):
            termination_reason = "x_position_infinite"
            logger.debug("Episode TERMINATED: Cart position is infinite")
        elif not np.isfinite(x_dot):
            termination_reason = "x_velocity_infinite"
            logger.debug("Episode TERMINATED: Cart velocity is infinite")
        elif not np.isfinite(theta):
            termination_reason = "theta_infinite"
            logger.debug("Episode TERMINATED: Pole angle is infinite")
        elif not np.isfinite(theta_dot):
            termination_reason = "theta_dot_infinite"
            logger.debug("Episode TERMINATED: Pole angular velocity is infinite")

        terminated = termination_reason is not None

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned "
                    "terminated = True. You should always call 'reset()' once you receive "
                    "'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        # Prepare info dict
        info = {
            "predicted_force": predicted_force.item(),
            "scaled_force": force,
            "current_step": self.current_step
        }
        if self.predict_friction:
            info.update({
                "friction_cart": mu_c.item(),
                "friction_pole": mu_p.item()
            })

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        logger.debug("Resetting environment")
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Reset state
        low, high = -0.05, 0.05  # default reset bounds
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None
        self.sequence_buffer = None

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