import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def remove_nan_entries(batch):
    print("Removing NaN entries from batch")
    return batch[~torch.isnan(batch).any(dim=1)]

def evaluate_pinn(model, dataloader, params):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    true_actions = []
    predicted_actions = []
    states = []

    with torch.no_grad():
        for batch in dataloader:
            if torch.isnan(batch).any():
                print("NaN found in input batch:")
                print(batch)
                #return
        
        for batch in dataloader:
            # TODO: Handle NaN entries in batch better
            if torch.isnan(batch).any():
                batch = remove_nan_entries(batch)
                
            x, x_dot, theta, theta_dot, action = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]
            x, x_dot, theta, theta_dot = x.to(device), x_dot.to(device), theta.to(device), theta_dot.to(device)

            t = torch.zeros_like(x)
            if model.predict_friction:
                # TODO: Handle friction predictions
                F, _, _ = model(t, x, x_dot, theta, theta_dot)
            else:
                F = model(t, x, x_dot, theta, theta_dot)

            if torch.isnan(F).any():
                print("NaN found in model output:")
                print(F)
                return

            true_actions.extend(action.cpu().numpy())
            predicted_actions.extend(F.cpu().numpy())
            states.extend(torch.stack([x, x_dot, theta, theta_dot], dim=1).cpu().numpy())

    true_actions = np.array(true_actions)
    predicted_actions = np.array(predicted_actions)
    states = np.array(states)

    # Calculate metrics
    mse = mean_squared_error(true_actions, predicted_actions)
    r2 = r2_score(true_actions, predicted_actions)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

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

    return mse, r2