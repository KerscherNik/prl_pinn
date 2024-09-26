import torch
import torch.nn as nn
from model.physics_helpers import calculate_theta_ddot, calculate_x_ddot
from torchdiffeq import odeint
import logging
import traceback

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Output to console
                        logging.FileHandler('app.log')  # Optionally log to a file
                    ])

logger = logging.getLogger(__name__)

class CartpolePINN(nn.Module):
    def __init__(self, sequence_length, predict_friction=False):
        super().__init__()
        self.predict_friction = predict_friction
        self.sequence_length = sequence_length
        input_size = 5 * sequence_length  # (x, x_dot, theta, theta_dot, action) * sequence_length
        output_size = 3 if predict_friction else 1

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(input_size=5, hidden_size=128, num_layers=2, batch_first=True)

        logger.info("LSTM initialized with input size %d and hidden size 128.", input_size)

        # Main network with residual connections
        self.network = nn.Sequential(
            nn.Linear(128, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),  # Dropout to prevent overfitting
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128)
        )
        
        # Output layer with two branches if predicting friction
        if self.predict_friction:
            self.force_output = nn.Linear(128, 1)  # Predicting force
            self.friction_output = nn.Linear(128, 2)  # Predicting mu_c and mu_p (friction coefficients)
            logger.info("Model configured to predict friction.")
        else:
            self.output_layer = nn.Linear(128, 1)  # Only predict force
            logger.info("Model configured to predict force only.")
        
        # Initialize weights
        self.network.apply(self.init_weights)
        logger.info("Network weights initialized.")

    def forward(self, sequence):
        batch_size, seq_len, features = sequence.shape
        logger.debug(f"Forward pass input: batch_size={batch_size}, seq_len={seq_len}, features={features}")
        logger.debug(f"Input sequence: {sequence}")
        
        # Process the sequence with LSTM
        try:
            lstm_out, _ = self.lstm(sequence)
            logger.debug(f"LSTM output shape: {lstm_out.shape}")
            logger.debug(f"LSTM output: {lstm_out}")
        except Exception as e:
            logger.error(f"Error in LSTM processing: {e}")
            logger.error(traceback.format_exc())
            raise
        
        # Take the final output from the LSTM (many-to-one)
        lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, 128)
        logger.debug(f"Final LSTM output: {lstm_out}")
        
        # Pass through the feedforward network
        try:
            residual = lstm_out
            x = self.network(lstm_out)
            x += residual  # Add skip connection for better gradient flow
            logger.debug(f"Feedforward network output: {x}")
        except Exception as e:
            logger.error(f"Error in feedforward network: {e}")
            logger.error(traceback.format_exc())
            raise
        
        if self.predict_friction:
            try:
                force = self.force_output(x).squeeze(-1)
                friction_params = self.friction_output(x)
                mu_c, mu_p = friction_params[:, 0], friction_params[:, 1]
                logger.debug(f"Predicted force: {force}, mu_c: {mu_c}, mu_p: {mu_p}")
                return force, mu_c, mu_p
            except Exception as e:
                logger.error(f"Error in friction prediction: {e}")
                logger.error(traceback.format_exc())
                raise
        else:
            try:
                force = self.output_layer(x).squeeze(-1)
                logger.debug(f"Predicted force: {force}")
                return force
            except Exception as e:
                logger.error(f"Error in force prediction: {e}")
                logger.error(traceback.format_exc())
                raise

    def dynamics(self, t, state, F, params):
        x, x_dot, theta, theta_dot = state

        # Compute the accelerations using the current state and force
        theta_ddot = calculate_theta_ddot(F, x_dot, theta, theta_dot, params['mu_c'], params['mu_p'], params)
        x_ddot = calculate_x_ddot(F, x_dot, theta, theta_dot, params['mu_c'], params['mu_p'], params)
        
        logger.debug("Dynamics computed: theta_ddot=%s, x_ddot=%s", theta_ddot, x_ddot)
        return torch.stack([x_dot, x_ddot, theta_dot, theta_ddot])

    def simulate_next_state(self, initial_state, F, params, callback=None):
        t = torch.tensor([0, params['tau']], device=self.device)

        # Wrapper for ODE dynamics
        def wrapped_dynamics(t, y):
            if callback:
                callback(t, y)
            return self.dynamics(t, y, F, params)
        
        # Integrate using RK4
        trajectory = odeint(wrapped_dynamics, initial_state, t, method='rk4')
        return trajectory[-1]  # Return the last state after integration

    def simulate(self, t_span, initial_state, params, callback=None):
        logger.info("Simulating over time span %s", t_span)
        # Simulate for 20 steps over the given time span
        t = torch.linspace(0, t_span, 20, device=self.device)
        initial_state = initial_state.to(self.device)
        
        # Dynamics wrapper for the ODE solver
        def wrapped_dynamics(t, state):
            if callback:
                callback(t, state)
            return self.dynamics(t, state, params)
        
        # Integrate the ODE
        trajectory = odeint(wrapped_dynamics, initial_state, t, method='rk4')
        
        logger.info("Simulation complete. Trajectory shape: %s", trajectory.shape)
        return trajectory

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Kaiming initialization for stability
            nn.init.constant_(m.bias, 0)
        logger.debug("Weights initialized for layer: %s", m)
