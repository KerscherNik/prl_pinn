import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from loss_functions import pinn_loss

# Evaluate the PINN model on a test dataset and generate metrics and plots
# Move data in batches to device, compute mse & physics loss, mean relative error
def evaluate_pinn(model, dataloader, params):
    model.eval() # Sets the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    true_actions = []
    predicted_actions = []
    states = []
    total_mse_loss = 0
    total_physics_loss = 0

    with torch.no_grad(): # Disables gradient calc (reduce memory & speed up computation) - no training
        for batch in dataloader:
            x, x_dot, theta, theta_dot, action = [b.to(device) for b in batch]

            t = torch.zeros_like(x)
            if model.predict_friction:
                F, _, _ = model(t, x, x_dot, theta, theta_dot)
            else:
                F = model(t, x, x_dot, theta, theta_dot)

            true_actions.extend(action.cpu().numpy())
            predicted_actions.extend(F.cpu().numpy())
            states.extend(torch.stack([x, x_dot, theta, theta_dot], dim=1).cpu().numpy())

            mse, phys = pinn_loss(model, x, x_dot, theta, theta_dot, action, params)
            total_mse_loss += mse.item()
            total_physics_loss += phys.item()

    true_actions = np.array(true_actions)
    predicted_actions = np.array(predicted_actions)
    states = np.array(states)
    
    relative_error = np.abs(true_actions - predicted_actions) / np.abs(true_actions)
    mean_relative_error = np.mean(relative_error)
    print(f"Mean Relative Error: {mean_relative_error:.4f}")

    # Add residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted_actions, true_actions - predicted_actions, alpha=0.5)
    plt.xlabel("Predicted Actions")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig('residual_plot.png')
    plt.close()

    # Calculate metrics
    mse = mean_squared_error(true_actions, predicted_actions)
    r2 = r2_score(true_actions, predicted_actions)
    avg_mse_loss = total_mse_loss / len(dataloader)
    avg_physics_loss = total_physics_loss / len(dataloader)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Average MSE Loss: {avg_mse_loss:.4f}")
    print(f"Average Physics Loss: {avg_physics_loss:.4f}")

    # Plot true vs predicted actions
    plt.figure(figsize=(10, 6))
    plt.scatter(true_actions, predicted_actions, alpha=0.5)
    plt.plot([min(true_actions), max(true_actions)], [min(true_actions), max(true_actions)], 'r--')
    plt.xlabel("True Actions")
    plt.ylabel("Predicted Actions")
    plt.title("True vs Predicted Actions")
    plt.savefig('true_vs_predicted_actions.png')
    plt.close()

    # Plot error distribution
    errors = true_actions - predicted_actions
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.savefig('error_distribution.png')
    plt.close()

    # Plot actions vs state variables
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    state_labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']
    for i in range(4):
        axs[i // 2, i % 2].scatter(states[:, i], true_actions, alpha=0.5, label='True')
        axs[i // 2, i % 2].scatter(states[:, i], predicted_actions, alpha=0.5, label='Predicted')
        axs[i // 2, i % 2].set_xlabel(state_labels[i])
        axs[i // 2, i % 2].set_ylabel('Action')
        axs[i // 2, i % 2].legend()
    plt.tight_layout()
    plt.savefig('actions_vs_states.png')
    plt.close()

    return mse, r2, avg_mse_loss, avg_physics_loss