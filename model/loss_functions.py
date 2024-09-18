import torch
from model.physics_helpers import calculate_theta_ddot, calculate_x_ddot

def pinn_loss(model, sequences, targets, params, physics_weight=1.0, t_span=1.0, simulation_callback=None, loss_calculation_callback=None):
    batch_size = sequences.shape[0]
    device = sequences.device
    
    F = model(sequences)
    mu_c, mu_p = params['mu_c'], params['mu_p']

    total_mse_loss = 0
    total_physics_loss = 0

    for i in range(batch_size):
        initial_state = sequences[i, -1, :4]  # Use the last state in the sequence
        target = targets[i]
        
        trajectory = model.simulate(t_span, initial_state, params, callback=simulation_callback)
        
        # MSE loss between predicted force and target force
        mse_loss = torch.nn.functional.mse_loss(F[i], target[-1])  # Compare with the last action in the target
        
        # Physics loss
        physics_loss = 0
        for j, state in enumerate(trajectory):
            x, x_dot, theta, theta_dot = state
            x_ddot = calculate_x_ddot(F[i], x_dot, theta, theta_dot, mu_c, mu_p, params)
            theta_ddot = calculate_theta_ddot(F[i], x_dot, theta, theta_dot, mu_c, mu_p, params)
            
            physics_loss += (x_ddot - calculate_x_ddot(target[-1], x_dot, theta, theta_dot, mu_c, mu_p, params))**2
            physics_loss += (theta_ddot - calculate_theta_ddot(target[-1], x_dot, theta, theta_dot, mu_c, mu_p, params))**2
            
            if loss_calculation_callback:
                loss_calculation_callback((i * len(trajectory) + j + 1) / (batch_size * len(trajectory)))
        
        total_mse_loss += mse_loss
        total_physics_loss += physics_loss / len(trajectory)

    avg_mse_loss = total_mse_loss / batch_size
    avg_physics_loss = total_physics_loss / batch_size
    total_loss = avg_mse_loss + physics_weight * avg_physics_loss

    return total_loss, avg_mse_loss, avg_physics_loss
