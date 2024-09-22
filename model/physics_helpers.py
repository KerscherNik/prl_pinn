import torch
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Output to console
                        logging.FileHandler('app.log')  # Optionally log to a file
                    ])

logger = logging.getLogger(__name__)

def calculate_theta_ddot(F, x_dot, theta, theta_dot, mu_c, mu_p, params):
    """
    Calculates the angular acceleration of the pendulum (θ̈) in the CartPole system.
    """
    g = params['g']
    m_c = params['m_c']
    m_p = params['m_p']
    l = params['l']

    numerator = g * torch.sin(theta) + torch.cos(theta) * (
        -F - m_p * l * theta_dot**2 * torch.sin(theta) + mu_c * torch.sign(x_dot)
    ) / (m_c + m_p)
    denominator = l * (4/3 - m_p * torch.cos(theta)**2 / (m_c + m_p))
    
    theta_ddot = numerator / denominator - mu_p * theta_dot / (m_p * l**2)
    logger.debug("Calculated theta_ddot: %s", theta_ddot)
    return theta_ddot

def calculate_x_ddot(F, x_dot, theta, theta_dot, mu_c, mu_p, params):
    """
    Calculates the horizontal acceleration of the cart (ẍ) in the CartPole system.
    """
    m_c = params['m_c']
    m_p = params['m_p']
    l = params['l']

    x_ddot = (F + m_p * l * (theta_dot**2 * torch.sin(theta) - calculate_theta_ddot(F, x_dot, theta, theta_dot, mu_c, mu_p, params) * torch.cos(theta)) - mu_c * torch.sign(x_dot)) / (m_c + m_p)
    logger.debug("Calculated x_ddot: %s", x_ddot)
    return x_ddot
