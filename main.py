from data_loader import get_dataloaders
from pinn_model import CartpolePINN
from loss_functions import pinn_loss
from train_utils import train_pinn, optimize_hyperparameters
from evaluate import evaluate_pinn
from gym_integration import PINNCartPoleEnv

import torch

def main():
    # Define file paths
    file_paths = [
        "demonstration_data2018_1.csv",
        "demonstration_data2018_2.csv",
        "demonstration_data2024.csv"
    ]

    # Load and split data
    train_dataloader, test_dataloader = get_dataloaders(file_paths, batch_size=32, test_size=0.2)

    # Define parameters
    params = {
        "m_c": 0.466,
        "m_p": 0.06,
        "l": 0.201,
        "g": 9.81,
        "mu_c": 0.1,  # Example value, adjust as needed
        "mu_p": 0.01  # Example value, adjust as needed
    }

    # Hyperparameter optimization
    print("Starting hyperparameter optimization...")
    best_config = optimize_hyperparameters(train_dataloader, params)

    # Create and train model with best hyperparameters
    print("Training final model with best hyperparameters...")
    model = CartpolePINN(predict_friction=best_config["predict_friction"])
    optimizer = torch.optim.Adam(model.parameters(), lr=best_config["lr"])
    
    trained_model = train_pinn(model, train_dataloader, optimizer, pinn_loss, params, best_config["num_epochs"])

    # Evaluate model
    print("Evaluating model...")
    evaluate_pinn(trained_model, test_dataloader, params)

    # Create Gym environment with trained PINN
    print("Creating Gym environment with trained PINN...")
    env = PINNCartPoleEnv(trained_model, params)

    # Example usage of the environment
    print("Running example episode in Gym environment...")
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()

    print("Done!")

if __name__ == "__main__":
    main()