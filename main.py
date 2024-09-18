from data.data_loader import get_dataloaders
from model.pinn_model import CartpolePINN
from training.train_utils import train_pinn, optimize_hyperparameters
from evaluation.evaluate import evaluate_pinn
from integration.compare_environments import compare_environments
import datetime
import torch

def main():
    file_paths = ["data/cartpole_data.csv"]
    sequence_length = 5  # Adjust this value as needed

    # Load and preprocess the data
    train_dataloader, test_dataloader, scaler = get_dataloaders(file_paths, batch_size=32, sequence_length=sequence_length, test_size=0.2)

    params = {
        "m_c": 0.466,
        "m_p": 0.06,
        "l": 0.201,
        "g": 9.81,
        "mu_c": 0.1,
        "mu_p": 0.01,
        "force_mag": 10.0
    }

    models = {}
    for predict_friction in [False]:
        print(f"{'With' if predict_friction else 'Without'} friction prediction:")
        
        # Hyperparameter optimization
        print("Starting hyperparameter optimization...")
        best_config = optimize_hyperparameters(train_dataloader, test_dataloader, params, predict_friction, sequence_length)
        print("Best hyperparameters found:", best_config)        

        # Create and train model with best hyperparameters
        print("Training final model with best hyperparameters...")
        model = CartpolePINN(sequence_length, predict_friction=predict_friction)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_config["lr"])

        trained_model, _ = train_pinn(model, train_dataloader, optimizer, params, best_config["num_epochs"], best_config["physics_weight"])
        models[predict_friction] = trained_model
        
        # Save the trained model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(trained_model.state_dict(), f'model_archive/trained_pinn_model_{"with" if predict_friction else "without"}_friction_{timestamp}.pth')
        
        # Uncomment this block to load a saved model and evaluate it directly. Make sure to comment out previous training block and hyperparameter optimization block. At evaluate_pinn use the loaded model instead of trained_model.
        """ # Load the already saved model
        saved_model_path = f'model_archive/trained_pinn_model_without_friction_20240918_022442.pth'
        loaded_model = CartpolePINN(sequence_length, predict_friction=predict_friction)
        loaded_model.load_state_dict(torch.load(saved_model_path)) """
        
        # Evaluate model
        print("Evaluating model...")
        mse, r2, avg_mse_loss, avg_physics_loss, mean_relative_error = evaluate_pinn(trained_model, test_dataloader, params, scaler)

        print(f"Results for {'with' if predict_friction else 'without'} friction prediction:")
        print(f"MSE: {mse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"Avg MSE Loss: {avg_mse_loss:.4f}")
        print(f"Avg Physics Loss: {avg_physics_loss:.4f}")
        print(f"Mean Relative Error: {mean_relative_error:.4f}")
        print("\n" + "="*50 + "\n")

    # Compare environments
    #rewards_orig, rewards_pinn_with_friction = compare_environments(models[True], params, True)
    #rewards_orig, rewards_pinn_without_friction = compare_environments(models[False], params, False)
    
    """ # Load the trained model
    trained_model = CartpolePINN(predict_friction=False, sequence_length=sequence_length)
    trained_model.load_state_dict(torch.load('model_archive/trained_pinn_model_without_friction_20240918_022442.pth')) """

    # Compare environments
    print("Comparing CartPole environments...")
    rewards_orig, rewards_pinn_without_friction = compare_environments(trained_model, params, False)

    print("Average reward (Original): {:.2f}".format(sum(rewards_orig) / len(rewards_orig)))
    #print("Average reward (PINN with friction): {:.2f}".format(sum(rewards_pinn_with_friction) / len(rewards_pinn_with_friction)))
    print("Average reward (PINN without friction): {:.2f}".format(sum(rewards_pinn_without_friction) / len(rewards_pinn_without_friction)))

    print("Done!")

if __name__ == "__main__":
    main()