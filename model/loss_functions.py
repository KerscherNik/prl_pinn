import torch
from model.physics_helpers import calculate_theta_ddot, calculate_x_ddot

def pinn_loss(model, sequences, targets, params, physics_weight=1.0, t_span=1.0, reg_weight = 1e-5, predict_friction = False, simulation_callback=None, loss_calculation_callback=None):
    batch_size = sequences.shape[0]
    device = sequences.device
    
    if predict_friction:
        F, mu_c, mu_p = model(sequences)
    else:
        F = model(sequences)
        mu_c, mu_p = params['mu_c'], params['mu_p']

    total_mse_loss = 0
    total_physics_loss = 0

    for i in range(batch_size):
        initial_state = sequences[i, -1, :4]  # Last state in sequence (x, x_dot, theta, theta_dot)
        target_state = targets[i, :4]  # Next true state (no action)
        
        # Simulate next state using the predicted force and the current state
        predicted_next_state = model.simulate_next_state(initial_state, F[i], params, callback=simulation_callback)
        
        # MSE loss between the predicted next state and the true target state
        mse_loss = torch.nn.functional.mse_loss(predicted_next_state, target_state)
        
        # Physics loss by ensuring the accelerations (x_ddot, theta_ddot) follow the ODEs
        x, x_dot, theta, theta_dot = predicted_next_state
        x_ddot_pred = calculate_x_ddot(F[i], x_dot, theta, theta_dot, params['mu_c'], params['mu_p'], params)
        theta_ddot_pred = calculate_theta_ddot(F[i], x_dot, theta, theta_dot, params['mu_c'], params['mu_p'], params)

        x_ddot_true = calculate_x_ddot(target_state, x_dot, theta, theta_dot, params['mu_c'], params['mu_p'], params)
        theta_ddot_true = calculate_theta_ddot(target_state, x_dot, theta, theta_dot, params['mu_c'], params['mu_p'], params)
        
        physics_loss = (x_ddot_pred - x_ddot_true).pow(2).mean() + (theta_ddot_pred - theta_ddot_true).pow(2).mean()

        if loss_calculation_callback:
            loss_calculation_callback((i + 1) / batch_size)

        total_mse_loss += mse_loss
        total_physics_loss += physics_loss

    avg_mse_loss = total_mse_loss / batch_size
    avg_physics_loss = total_physics_loss / batch_size

    # L2 regularization
    l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
    
    # Combine losses: MSE + weighted physics loss + regularization
    total_loss = avg_mse_loss + physics_weight * avg_physics_loss + reg_weight * l2_reg

    return total_loss, avg_mse_loss, avg_physics_loss
