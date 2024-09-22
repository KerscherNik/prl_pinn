from data.data_loader import get_dataloaders
from integration.compare_envs_interactively import visualize_interactive
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
                        logging.FileHandler('main_app.log')
                    ])

logger = logging.getLogger(__name__)

def main():
    file_paths = ["data/cartpole_data.csv"]
    sequence_length = 50  # Adjust this value as needed

    # Load and preprocess the data
    train_dataloader, test_dataloader, scaler = get_dataloaders(file_paths, batch_size=32, sequence_length=sequence_length, test_size=0.2, verbose=False)

    """ Params for real cartpole
    params = {
        "m_c": 0.466,
        "m_p": 0.06,
        "l": 0.201,
        "g": 9.81,
        "mu_c": 0.1,
        "mu_p": 0.01,
        "force_mag": 10.0,
        "tau" : 0.02  # Assuming a 50Hz sampling rate (1/50 = 0.02 seconds)
    }"""

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

    models = {}
    for predict_friction in [False]:
        logger.info(f"{'With' if predict_friction else 'Without'} friction prediction:")

        # Hyperparameter optimization
        logger.info("Starting hyperparameter optimization...")
        best_config = optimize_hyperparameters(train_dataloader, test_dataloader, params, predict_friction, sequence_length)
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
        torch.save(trained_model.state_dict(), f'model_archive/trained_pinn_model_{"with" if predict_friction else "without"}_friction_{timestamp}.pth')
        logger.info(f"Model saved under name: trained_pinn_model_{'with' if predict_friction else 'without'}_friction_{timestamp}.pth")
        
        # Uncomment this block to load a saved model and evaluate it directly. Make sure to comment out previous training block and hyperparameter optimization block. At evaluate_pinn use the loaded model instead of trained_model.
        # Load the already saved model
        """ saved_model_path = f'model_archive/trained_pinn_model_without_friction_20240922_033729.pth'
        loaded_model = CartpolePINN(sequence_length, predict_friction=predict_friction)
        loaded_model.load_state_dict(torch.load(saved_model_path, weights_only=True)) """
       
        # visualize interactively: uncomment only this block and the block above to load the pinn model
        #original_env = Monitor(gym.make('CartPole-v1'))
        #pinn_env = Monitor(PINNCartPoleEnv(loaded_model, params))
        #visualize_interactive(pinn_env, original_env, max_steps=500, visualize=True, slow_motion_factor=3)


        # Evaluate model
        logger.info("Evaluating model...")
        mse, r2, avg_mse_loss, avg_physics_loss, mean_relative_error = evaluate_pinn(trained_model, test_dataloader, params, scaler, predict_friction)
        """ logger.info("Evaluating model...")
        mse, r2, avg_mse_loss, avg_physics_loss, mean_relative_error = evaluate_pinn(loaded_model, test_dataloader, params, scaler, predict_friction) #TODO: trained_model """

        logger.info(f"Results for {'with' if predict_friction else 'without'} friction prediction:")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"R2: {r2:.4f}")
        logger.info(f"Avg MSE Loss: {avg_mse_loss:.4f}")
        logger.info(f"Avg Physics Loss: {avg_physics_loss:.4f}")
        logger.info(f"Mean Relative Error: {mean_relative_error:.4f}")
        logger.info("\n" + "="*50 + "\n")


    # Compare environments
    #rewards_orig, rewards_pinn_with_friction = compare_environments(models[True], params, True)
    #rewards_orig, rewards_pinn_without_friction = compare_environments(models[False], params, False)

    """
    # Load the trained model
    trained_model = CartpolePINN(predict_friction=False, sequence_length=sequence_length)
    trained_model.load_state_dict(torch.load('model_archive/trained_pinn_model_without_friction_20240921_224527.pth'))
    """

    # Compare environments
    logger.info("Comparing CartPole environments...")
    rewards_orig, rewards_pinn_without_friction = compare_environments(trained_model, params, False)
    """ rewards_orig, rewards_pinn_without_friction = compare_environments(loaded_model, params, False) #TODO: trained_model """


    logger.info("Average reward (Original): {:.2f}".format(sum(rewards_orig) / len(rewards_orig)))
    #logger.info("Average reward (PINN with friction): {:.2f}".format(sum(rewards_pinn_with_friction) / len(rewards_pinn_with_friction)))
    logger.info("Average reward (PINN without friction): {:.2f}".format(sum(rewards_pinn_without_friction) / len(rewards_pinn_without_friction)))

    logger.info("Done!")

if __name__ == "__main__":
    main()