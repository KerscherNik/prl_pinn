import torch
import torch.nn as nn
from model.physics_helpers import calculate_theta_ddot, calculate_x_ddot
from torchdiffeq import odeint

class CartpolePINN(nn.Module):
    def __init__(self, sequence_length, predict_friction=False):
        super().__init__()
        self.predict_friction = predict_friction
        self.sequence_length = sequence_length
        input_size = 5 * sequence_length  # (x, x_dot, theta, theta_dot, action) * sequence_length
        output_size = 3 if predict_friction else 1

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_size)
        )
        self.network.apply(init_weights)

    def forward(self, sequence):
        batch_size, seq_len, features = sequence.shape
        inputs = sequence.view(batch_size, -1)
        outputs = self.network(inputs)
        
        if self.predict_friction:
            F, mu_c, mu_p = outputs[:, 0], outputs[:, 1], outputs[:, 2]
            return F, mu_c, mu_p
        else:
            return outputs.squeeze()

    def dynamics(self, t, state):
        x, x_dot, theta, theta_dot = state
        
        # Create a sequence using the current state
        sequence = torch.stack([x.repeat(self.sequence_length), 
                                x_dot.repeat(self.sequence_length), 
                                theta.repeat(self.sequence_length), 
                                theta_dot.repeat(self.sequence_length),
                                torch.zeros(self.sequence_length, device=self.device)], dim=0).unsqueeze(0)
        
        F = self.forward(sequence)

        params = {
            "m_c": 1.0,
            "m_p": 0.1,
            "l": 1.0,
            "g": 9.8,
            "mu_c": 0.0,
            "mu_p": 0.0,
            "force_mag": 10.0
        }

        theta_ddot = calculate_theta_ddot(F, x_dot, theta, theta_dot, params['mu_c'], params['mu_p'], params)
        x_ddot = calculate_x_ddot(F, x_dot, theta, theta_dot, params['mu_c'], params['mu_p'], params)

        return torch.stack([x_dot, x_ddot, theta_dot, theta_ddot])

    def simulate(self, t_span, initial_state, params, callback=None):
        t = torch.linspace(0, t_span, 20, device=self.device)
        initial_state = initial_state.to(self.device)
        
        def wrapped_dynamics(t, state):
            if callback:
                callback(t, state)
            return self.dynamics(t, state)
        
        trajectory = odeint(wrapped_dynamics, initial_state, t, method='rk4')
        
        return trajectory

    @property
    def device(self):
        return next(self.parameters()).device

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)