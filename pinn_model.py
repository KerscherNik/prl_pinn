import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class CartpolePINN(nn.Module):
    def __init__(self, predict_friction=False):
        super().__init__()
        self.predict_friction = predict_friction
        
        input_size = 5  # t, x, x_dot, theta, theta_dot
        output_size = 3 if predict_friction else 1
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_size)
        )
        
        self.network.apply(init_weights)

    def forward(self, t, x, x_dot, theta, theta_dot):
        device = next(self.parameters()).device
        inputs = torch.stack([t, x, x_dot, theta, theta_dot], dim=1).to(device)
        outputs = self.network(inputs)
        
        if self.predict_friction:
            F, mu_c, mu_p = outputs[:, 0], outputs[:, 1], outputs[:, 2]
            return F, mu_c, mu_p
        else:
            return outputs.squeeze()