from data_loader import get_dataloaders
from pinn_model import CartpolePINN
from loss_functions import pinn_loss
from train_utils import train_pinn, optimize_hyperparameters
from evaluate import evaluate_pinn
from compare_environments import compare_environments

import torch

def main():
    # TODO: Implement option to choose friction prediction or only force prediction or both can compare models performance
    
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
        "mu_p": 0.01,  # Example value, adjust as needed
        "force_mag": 10.0  # Add this parameter for the PINN environment
    }

    models = {}
    for predict_friction in [False]: # Currently only False for simplicity
        print(f"{'With' if predict_friction else 'Without'} friction prediction:")
        
        # Hyperparameter optimization
        print("Starting hyperparameter optimization...")
        best_config = optimize_hyperparameters(train_dataloader, test_dataloader, params, predict_friction)

        # Create and train model with best hyperparameters
        print("Training final model with best hyperparameters...")
        model = CartpolePINN(predict_friction=predict_friction)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_config["lr"])
        
        trained_model, _ = train_pinn(model, train_dataloader, optimizer, pinn_loss, params, best_config["num_epochs"], best_config["physics_weight"])
        models[predict_friction] = trained_model

        # Evaluate model
        print("Evaluating model...")
        mse, r2, avg_mse_loss, avg_physics_loss, mean_relative_error = evaluate_pinn(trained_model, test_dataloader, params)

        # Save the trained model
        torch.save(trained_model.state_dict(), f'trained_pinn_model_{"with" if predict_friction else "without"}_friction.pth')

        print(f"Results for {'with' if predict_friction else 'without'} friction prediction:")
        print(f"MSE: {mse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"Avg MSE Loss: {avg_mse_loss:.4f}")
        print(f"Avg Physics Loss: {avg_physics_loss:.4f}")
        print(f"Mean Relative Error: {mean_relative_error:.4f}")
        print("\n" + "="*50 + "\n")

    # Compare environments
    print("Comparing CartPole environments...")
    rewards_orig, rewards_pinn_with_friction = compare_environments(models[True], params)
    rewards_orig, rewards_pinn_without_friction = compare_environments(models[False], params)

    print("Average reward (Original): {:.2f}".format(sum(rewards_orig) / len(rewards_orig)))
    print("Average reward (PINN with friction): {:.2f}".format(sum(rewards_pinn_with_friction) / len(rewards_pinn_with_friction)))
    print("Average reward (PINN without friction): {:.2f}".format(sum(rewards_pinn_without_friction) / len(rewards_pinn_without_friction)))

    print("Done!")

if __name__ == "__main__":
    main()