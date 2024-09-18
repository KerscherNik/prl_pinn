import torch

def calculate_theta_ddot(F, x_dot, theta, theta_dot, mu_c, mu_p, params):
    g = params['g']
    m_c = params['m_c']
    m_p = params['m_p']
    l = params['l']

    numerator = g * torch.sin(theta) + torch.cos(theta) * (
        -F - m_p * l * theta_dot**2 * torch.sin(theta) + mu_c * torch.sign(x_dot)
    ) / (m_c + m_p)
    denominator = l * (4/3 - m_p * torch.cos(theta)**2 / (m_c + m_p))
    return numerator / denominator - mu_p * theta_dot / (m_p * l**2)

def calculate_x_ddot(F, x_dot, theta, theta_dot, mu_c, mu_p, params):
    m_c = params['m_c']
    m_p = params['m_p']
    l = params['l']

    return (F + m_p * l * (theta_dot**2 * torch.sin(theta) - calculate_theta_ddot(F, x_dot, theta, theta_dot, mu_c, mu_p, params) * torch.cos(theta)) - mu_c * torch.sign(x_dot)) / (m_c + m_p)