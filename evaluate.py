import torch
import matplotlib.pyplot as plt

def evaluate_pinn(model, dataloader, params):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    true_actions = []
    predicted_actions = []

    with torch.no_grad():
        for batch in dataloader:
            x, x_dot, theta, theta_dot, action = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]
            x, x_dot, theta, theta_dot = x.to(device), x_dot.to(device), theta.to(device), theta_dot.to(device)

            t = torch.zeros_like(x)
            if model.predict_friction:
                F, _, _ = model(t, x, x_dot, theta, theta_dot)
            else:
                F = model(t, x, x_dot, theta, theta_dot)

            true_actions.extend(action.cpu().numpy())
            predicted_actions.extend(F.cpu().numpy())

    plt.figure(figsize=(10, 6))
    plt.scatter(true_actions, predicted_actions, alpha=0.5)
    plt.plot([min(true_actions), max(true_actions)], [min(true_actions), max(true_actions)], 'r--')
    plt.xlabel("True Actions")
    plt.ylabel("Predicted Actions")
    plt.title("True vs Predicted Actions")
    plt.show()

    mse = torch.mean((torch.tensor(true_actions) - torch.tensor(predicted_actions))**2)
    print(f"Mean Squared Error: {mse.item():.4f}")