import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from model.loss_functions import pinn_loss

def evaluate_pinn(model, dataloader, params, scaler):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    true_actions = []
    predicted_actions = []
    states = []
    total_mse_loss = 0
    total_physics_loss = 0

    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences, targets = sequences.to(device), targets.to(device)
            batch_size, seq_len, features = sequences.shape

            predicted_action = model(sequences)

            # Use the entire sequence for loss calculation
            loss, mse, physics_loss = pinn_loss(model, sequences, targets, params)
            total_mse_loss += mse.item()
            total_physics_loss += physics_loss.item()

            # Extract the last action from targets for comparison
            true_action = targets[:, -1]

            true_actions.extend(true_action.cpu().numpy())
            predicted_actions.extend(predicted_action.cpu().numpy())
            states.extend(sequences[:, -1, :].cpu().numpy())  # Use the last state in the sequence

    true_actions = np.array(true_actions)
    predicted_actions = np.array(predicted_actions)
    states = np.array(states)

    # Inverse transform the states only
    true_states_inv = scaler.inverse_transform(states)
    predicted_states_inv = scaler.inverse_transform(states)

    # Combine with true and predicted actions
    true_combined_inv = np.column_stack((true_states_inv, true_actions))
    predicted_combined_inv = np.column_stack((predicted_states_inv, predicted_actions))

    # Extract the inverse transformed actions
    true_actions = true_combined_inv[:, -1]
    predicted_actions = predicted_combined_inv[:, -1]
    states = true_combined_inv[:, :-1]  # Use true states for plotting

    # Remove NaN values
    mask = ~np.isnan(true_actions) & ~np.isnan(predicted_actions)
    true_actions = true_actions[mask]
    predicted_actions = predicted_actions[mask]
    states = states[mask]

    if len(true_actions) == 0:
        print("Warning: All values are NaN. Cannot calculate metrics.")
        return float('inf'), float('inf'), float('inf'), float('inf'), float('inf')

    # Calculate metrics
    mse = mean_squared_error(true_actions, predicted_actions)
    r2 = r2_score(true_actions, predicted_actions)
    avg_mse_loss = total_mse_loss / len(dataloader)
    avg_physics_loss = total_physics_loss / len(dataloader)
    relative_error = np.abs(true_actions - predicted_actions) / (np.abs(true_actions) + 1e-8)
    mean_relative_error = np.mean(relative_error)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Average MSE Loss: {avg_mse_loss:.4f}")
    print(f"Average Physics Loss: {avg_physics_loss:.4f}")
    print(f"Mean Relative Error: {mean_relative_error:.4f}")

    # Plotting
    plot_residuals(predicted_actions, true_actions)
    plot_true_vs_predicted(true_actions, predicted_actions)
    plot_error_distribution(true_actions, predicted_actions)
    plot_actions_vs_states(states, true_actions, predicted_actions)

    return mse, r2, avg_mse_loss, avg_physics_loss, mean_relative_error

def plot_residuals(predicted_actions, true_actions):
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted_actions, true_actions - predicted_actions, alpha=0.5)
    plt.xlabel("Predicted Actions (N)")
    plt.ylabel("Residuals (N)")
    plt.title("Residual Plot: Difference between True and Predicted Actions")
    plt.grid(True)
    plt.savefig('media/residual_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_true_vs_predicted(true_actions, predicted_actions):
    plt.figure(figsize=(10, 6))
    plt.scatter(true_actions, predicted_actions, alpha=0.5)
    min_val = min(min(true_actions), min(predicted_actions))
    max_val = max(max(true_actions), max(predicted_actions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel("True Actions (N)")
    plt.ylabel("Predicted Actions (N)")
    plt.title("True vs Predicted Actions")
    plt.legend()
    plt.grid(True)
    plt.savefig('media/true_vs_predicted_actions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(true_actions, predicted_actions):
    errors = true_actions - predicted_actions
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black')
    plt.xlabel("Prediction Error (N)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.grid(True)
    plt.savefig('media/error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_actions_vs_states(states, true_actions, predicted_actions):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    state_labels = ['Cart Position (m)', 'Cart Velocity (m/s)', 'Pole Angle (rad)', 'Pole Angular Velocity (rad/s)']
    for i in range(4):
        axs[i // 2, i % 2].scatter(states[:, i], true_actions, alpha=0.5, label='True', s=10)
        axs[i // 2, i % 2].scatter(states[:, i], predicted_actions, alpha=0.5, label='Predicted', s=10)
        axs[i // 2, i % 2].set_xlabel(state_labels[i])
        axs[i // 2, i % 2].set_ylabel('Action (N)')
        axs[i // 2, i % 2].legend()
        axs[i // 2, i % 2].grid(True)
    plt.tight_layout()
    plt.savefig('media/actions_vs_states.png', dpi=300, bbox_inches='tight')
    plt.close()