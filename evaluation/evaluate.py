import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from model.physics_helpers import calculate_x_ddot, calculate_theta_ddot
from model.loss_functions import pinn_loss
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Output to console
                        logging.FileHandler('app.log')  # Optionally log to a file
                    ])

logger = logging.getLogger(__name__)

def evaluate_pinn(model, dataloader, params, scaler, predict_friction=False, save_folder_name=""):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    predicted_forces = []
    scaled_forces = []
    states = []
    next_states = []
    total_mse_loss = 0
    total_physics_loss = 0
    predicted_mu_c = []
    predicted_mu_p = []

    logger.info("Starting evaluation on device: %s", device)

    with torch.no_grad():
        for sequences, targets in tqdm(dataloader, desc="Evaluating", leave=True):
            sequences, targets = sequences.to(device), targets.to(device)

            if predict_friction:
                predicted_force, mu_c, mu_p = model(sequences)
                predicted_mu_c.extend(mu_c.cpu().numpy())
                predicted_mu_p.extend(mu_p.cpu().numpy())
            else:
                predicted_force = model(sequences)
            
            scaled_force = predicted_force * params['force_mag']
            
            loss, mse, physics_loss = pinn_loss(model, sequences, targets, params, predict_friction=predict_friction)
            total_mse_loss += mse.item()
            total_physics_loss += physics_loss.item()
            
            current_state = sequences[:, -1, :4]
            next_state = targets[:, :4]

            predicted_forces.extend(predicted_force.cpu().numpy())
            scaled_forces.extend(scaled_force.cpu().numpy())
            states.extend(current_state.cpu().numpy())
            next_states.extend(next_state.cpu().numpy())

    predicted_forces = np.array(predicted_forces)
    scaled_forces = np.array(scaled_forces)
    states = np.array(states)
    next_states = np.array(next_states)

    states_inv = scaler.inverse_transform(states)
    next_states_inv = scaler.inverse_transform(next_states)

    logger.debug("Head of states (inversed):\n%s", states_inv[:5])
    logger.debug("Head of next states (inversed):\n%s", next_states_inv[:5])

    # Calculate implied forces based on state transitions
    implied_forces = calculate_implied_forces(states_inv, next_states_inv, params)

    # Evaluate the model's predictions
    mse = mean_squared_error(implied_forces, scaled_forces)
    r2 = r2_score(implied_forces, scaled_forces)
    relative_error = np.abs(implied_forces - scaled_forces) / (np.abs(implied_forces) + 1e-8)
    mean_relative_error = np.mean(relative_error)

    # Calculate average losses
    avg_mse_loss = total_mse_loss / len(dataloader)
    avg_physics_loss = total_physics_loss / len(dataloader)

    logger.info(f"Mean Squared Error: {mse:.4f}")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"Average MSE Loss: {avg_mse_loss:.4f}")
    logger.info(f"Average Physics Loss: {avg_physics_loss:.4f}")
    logger.info(f"Mean Relative Error: {mean_relative_error:.4f}")

    # Plotting
    plot_true_vs_predicted(implied_forces, scaled_forces, save_folder_name)
    plot_error_distribution(implied_forces, scaled_forces, save_folder_name)
    plot_forces_vs_states(states_inv, implied_forces, scaled_forces, save_folder_name)

    if predict_friction:
        plot_friction_coefficients(predicted_mu_c, predicted_mu_p)

    return mse, r2, avg_mse_loss, avg_physics_loss, mean_relative_error

def calculate_implied_forces(states, next_states, params):
    x, x_dot, theta, theta_dot = states.T
    next_x, next_x_dot, next_theta, next_theta_dot = next_states.T
    
    x_ddot = (next_x_dot - x_dot) / params['tau']
    theta_ddot = (next_theta_dot - theta_dot) / params['tau']
    
    # Solve for F using the equations of motion
    numerator = (params['m_c'] + params['m_p']) * x_ddot + params['m_p'] * params['l'] * np.sin(theta) * (theta_dot**2) - params['m_p'] * params['l'] * np.cos(theta) * theta_ddot
    denominator = 1 + params['m_p'] * np.sin(theta)**2 / (params['m_c'] + params['m_p'])
    
    return numerator / denominator

def plot_friction_coefficients(mu_c, mu_p, save_folder_name=""):
    plt.figure(figsize=(10, 5))
    plt.scatter(mu_c, mu_p, alpha=0.5)
    plt.xlabel("Coulomb Friction Coefficient (μ_c)")
    plt.ylabel("Viscous Friction Coefficient (μ_p)")
    plt.title("Predicted Friction Coefficients")
    plt.grid(True)
    plt.savefig(f'media/{save_folder_name}/friction_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot: media/{save_folder_name}/friction_coefficients.png")

def plot_true_vs_predicted(implied_forces, predicted_forces, save_folder_name=""):
    logger.info("Plotting Implied vs Predicted Forces...")
    plt.figure(figsize=(10, 6))
    plt.scatter(implied_forces, predicted_forces, alpha=0.5)
    min_val = min(min(implied_forces), min(predicted_forces))
    max_val = max(max(implied_forces), max(predicted_forces))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel("Implied Forces (N)")
    plt.ylabel("Predicted Forces (N)")
    plt.title("Implied vs Predicted Forces")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'media/{save_folder_name}/implied_vs_predicted_forces.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot: media/{save_folder_name}/implied_vs_predicted_forces.png")

def plot_error_distribution(implied_forces, predicted_forces, save_folder_name=""):
    logger.info("Plotting Force Prediction Error Distribution...")
    errors = implied_forces - predicted_forces
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black')
    plt.xlabel("Prediction Error (N)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Force Prediction Errors")
    plt.grid(True)
    plt.savefig(f'media/{save_folder_name}/force_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot: media/{save_folder_name}/force_error_distribution.png")

def plot_forces_vs_states(states, implied_forces, predicted_forces, save_folder_name=""):
    logger.info("Plotting Forces vs States...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    state_labels = ['Cart Position (m)', 'Cart Velocity (m/s)', 'Pole Angle (rad)', 'Pole Angular Velocity (rad/s)']
    for i in range(4):
        axs[i // 2, i % 2].scatter(states[:, i], implied_forces, alpha=0.5, label='Implied', s=10)
        axs[i // 2, i % 2].scatter(states[:, i], predicted_forces, alpha=0.5, label='Predicted', s=10)
        axs[i // 2, i % 2].set_xlabel(state_labels[i])
        axs[i // 2, i % 2].set_ylabel('Force (N)')
        axs[i // 2, i % 2].legend()
        axs[i // 2, i % 2].grid(True)
    plt.tight_layout()
    plt.savefig(f'media/{save_folder_name}/forces_vs_states.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot: media/{save_folder_name}/forces_vs_states.png")
