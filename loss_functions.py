import torch

import torch

def pinn_loss(model, x, x_dot, theta, theta_dot, action, params, physics_weight=1.0):
    t = torch.zeros_like(x)
    
    if model.predict_friction:
        F, mu_c, mu_p = model(t, x, x_dot, theta, theta_dot)
    else:
        F = model(t, x, x_dot, theta, theta_dot)
        mu_c, mu_p = params['mu_c'], params['mu_p']

    # Calculate derivatives
    theta_ddot = calculate_theta_ddot(F, x_dot, theta, theta_dot, mu_c, mu_p, params)
    x_ddot = calculate_x_ddot(F, x_dot, theta, theta_dot, mu_c, mu_p, params)

    # Calculate losses
    mse_loss = torch.nn.functional.mse_loss(F, action)
    physics_loss = torch.mean(
        (x_ddot - calculate_x_ddot(action, x_dot, theta, theta_dot, mu_c, mu_p, params))**2 +
        (theta_ddot - calculate_theta_ddot(action, x_dot, theta, theta_dot, mu_c, mu_p, params))**2
    )
    
    # Check for invalid values
    if torch.isnan(mse_loss) or torch.isinf(mse_loss):
        mse_loss = torch.tensor(1e6, device=mse_loss.device)
    
    if torch.isnan(physics_loss) or torch.isinf(physics_loss):
        physics_loss = torch.tensor(1e6, device=physics_loss.device)
    
    total_loss = mse_loss + physics_weight * physics_loss
    
    return total_loss, mse_loss, physics_loss

def calculate_theta_ddot(F, x_dot, theta, theta_dot, mu_c, mu_p, params):
    m_c, m_p, l, g = params['m_c'], params['m_p'], params['l'], params['g']
    
    numerator = g * torch.sin(theta) + torch.cos(theta) * (
        -F - m_p * l * theta_dot**2 * torch.sin(theta) + mu_c * torch.sign(x_dot)
    ) / (m_c + m_p) - (mu_p * theta_dot) / (m_p * l)
    
    denominator = l * (4/3 - (m_p * torch.cos(theta)**2) / (m_c + m_p) + 1e-8)
    
    return numerator / denominator

def calculate_x_ddot(F, x_dot, theta, theta_dot, mu_c, mu_p, params):
    m_c, m_p, l = params['m_c'], params['m_p'], params['l']
    
    return (F + m_p * l * (theta_dot**2 * torch.sin(theta) - calculate_theta_ddot(F, x_dot, theta, theta_dot, mu_c, mu_p, params) * torch.cos(theta)) - mu_c * torch.sign(x_dot)) / (m_c + m_p + 1e-8)