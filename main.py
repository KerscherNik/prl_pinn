from data.data_loader import get_dataloaders
from integration.gym_integration import PINNCartPoleEnv
from model.pinn_model import CartpolePINN
from training.train_utils import train_pinn, optimize_hyperparameters
from evaluation.evaluate import evaluate_pinn
from integration.compare_environments import compare_environments
import datetime
import torch
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('app.log')
                    ])

logger = logging.getLogger(__name__)

def main():
    
    ############################################################################################################
    #    Currently best trained model: model_archive/trained_pinn_model_without_friction_20240922_183400.pth   #
    ############################################################################################################
    file_paths = ["data/cartpole_data.csv"]
    sequence_length = 5  # Adjust this value as needed

    # Load and preprocess the data
    train_dataloader, test_dataloader, scaler = get_dataloaders(file_paths, batch_size=32, sequence_length=sequence_length, test_size=0.2, verbose=False)

    # Params for simulated cartpole
    params = {
        "m_c": 1.0,
        "m_p": 0.1,
        "l": 1.0,
        "g": 9.8,
        "mu_c": 0.0,
        "mu_p": 0.0,
        "force_mag": 10.0,
        "tau" : 0.02  # Assuming a 50Hz sampling rate (1/50 = 0.02 seconds)
    }
    num_epochs = 100
    
    models = {}
    for predict_friction in [True, False]:
        logger.info(f"{'With' if predict_friction else 'Without'} friction prediction:")
        save_folder_name = "with_friction" if predict_friction else "without_friction"
        # Hyperparameter optimization
        logger.info("Starting hyperparameter optimization...")
        best_config = optimize_hyperparameters(train_dataloader, test_dataloader, params, predict_friction, sequence_length)
        best_config["num_epochs"] = num_epochs
        logger.info(f"Best hyperparameters found: {best_config}")
        
        
        # Create and train model with best hyperparameters
        logger.info("Training final model with best hyperparameters...")
        model = CartpolePINN(sequence_length, predict_friction=predict_friction)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_config["lr"])

        trained_model, _ = train_pinn(model, train_dataloader, optimizer, params, best_config["num_epochs"], best_config["physics_weight"], t_span=1.0, reg_weight=best_config["reg_weight"], predict_friction=predict_friction)
        models[predict_friction] = trained_model
        
        # Save the trained model
        logger.info("Saving model...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(trained_model.state_dict(), f'model_archive/{save_folder_name}/trained_pinn_model_{"with" if predict_friction else "without"}_friction_{timestamp}.pth')
        logger.info(f"Model saved under name: trained_pinn_model_{'with' if predict_friction else 'without'}_friction_{timestamp}.pth")
        
        # Evaluate model
        logger.info("Evaluating model...")
        mse, r2, avg_mse_loss, avg_physics_loss, mean_relative_error = evaluate_pinn(trained_model, test_dataloader, params, scaler, predict_friction, save_folder_name)

        logger.info(f"Results for {'with' if predict_friction else 'without'} friction prediction:")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"R2: {r2:.4f}")
        logger.info(f"Avg MSE Loss: {avg_mse_loss:.4f}")
        logger.info(f"Avg Physics Loss: {avg_physics_loss:.4f}")
        logger.info(f"Mean Relative Error: {mean_relative_error:.4f}")
        logger.info("\n" + "="*50 + "\n")

    # Compare environments
    logger.info("Comparing CartPole environments...")
    logger.info("Start comparison without friction prediction:")
    rewards_orig, rewards_pinn_without_friction = compare_environments(models[False], params, predict_friction=False, num_episodes=100, max_steps=500, visualize=False, save_folder_name=save_folder_name)
    rewards_pinn_with_friction = None
    if True in models:
        logger.info("Start comparison with friction prediction:")
        _, rewards_pinn_with_friction = compare_environments(models[True], params, predict_friction=True, num_episodes=100, max_steps=500, visualize=False, save_folder_name=save_folder_name)

    logger.info("Average reward (Original): {:.2f}".format(sum(rewards_orig) / len(rewards_orig)))
    logger.info("Average reward (PINN without friction): {:.2f}".format(sum(rewards_pinn_without_friction) / len(rewards_pinn_without_friction)))
    if rewards_pinn_with_friction is not None:
        logger.info("Average reward (PINN with friction): {:.2f}".format(sum(rewards_pinn_with_friction) / len(rewards_pinn_with_friction)))

    logger.info("Done!")

if __name__ == "__main__":
    main()