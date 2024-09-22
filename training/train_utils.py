import logging
import torch
from tqdm import tqdm
from ray import tune
from ray.tune import CLIReporter
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from model.pinn_model import CartpolePINN
from model.loss_functions import pinn_loss
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('app.log')
                    ])

logger = logging.getLogger(__name__)

def train_pinn(model, train_dataloader, optimizer, params, num_epochs, physics_weight, t_span=1.0, reg_weight=1e-5, predict_friction=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")
    model.to(device)
    logger.info(f"Training for {num_epochs} epochs")
    
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)
    
    for epoch in epoch_pbar:
        model.train()
        total_loss, total_mse_loss, total_physics_loss = 0, 0, 0

        batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", position=1, leave=False)
        
        for batch_idx, (sequences, targets) in enumerate(batch_pbar):
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()

            sim_pbar = tqdm(total=100, desc="Simulation", position=2, leave=False)
            def simulation_callback(t, state):
                sim_pbar.update(int(t.item() / t_span * 100) - sim_pbar.n)

            loss_pbar = tqdm(total=100, desc="Loss Calculation", position=3, leave=False)
            def loss_calculation_callback(progress):
                loss_pbar.update(int(progress * 100) - loss_pbar.n)

            loss, mse_loss, physics_loss = pinn_loss(
                model, sequences, targets, params, physics_weight, t_span=t_span, reg_weight=reg_weight, predict_friction=predict_friction,
                simulation_callback=simulation_callback, loss_calculation_callback=loss_calculation_callback
            )

            sim_pbar.close()
            loss_pbar.close()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_physics_loss += physics_loss.item()

            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{mse_loss.item():.4f}',
                'physics': f'{physics_loss.item():.4f}'
            })

        avg_loss = total_loss / len(train_dataloader)
        avg_mse = total_mse_loss / len(train_dataloader)
        avg_physics = total_physics_loss / len(train_dataloader)

        epoch_pbar.set_postfix({
            'avg_loss': f'{avg_loss:.4f}',
            'avg_mse': f'{avg_mse:.4f}',
            'avg_physics': f'{avg_physics:.4f}'
        })

        logger.info(f"\nEpoch [{epoch+1}/{num_epochs}]")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        logger.info(f"MSE Loss: {avg_mse:.4f}, Physics Loss: {avg_physics:.4f}")

    return model, avg_loss

def objective(config, train_dataloader=None, test_dataloader=None, params=None, predict_friction=False, sequence_length=None):
    # Initialize model
    model = CartpolePINN(sequence_length=sequence_length, predict_friction=predict_friction)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Train the model for one epoch
    model, avg_train_loss = train_pinn(
        model, 
        train_dataloader, 
        optimizer, 
        params, 
        num_epochs=config["num_epochs"],
        physics_weight=config["physics_weight"], 
        t_span=1.0, 
        reg_weight=config["reg_weight"], 
        predict_friction=predict_friction
    )

    # Evaluate the model on the test set
    model.eval()
    
    total_test_loss = 0
    total_mse_loss = 0
    total_physics_loss = 0
    predicted_mu_c = []
    predicted_mu_p = []
    
    with torch.no_grad():
        for sequences, targets in test_dataloader:
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
            total_test_loss += loss.item()
    
    # Average test loss
    avg_test_loss = total_test_loss / len(test_dataloader)

    # Report the losses to Ray Tune
    session.report({"train_loss": avg_train_loss, "test_loss": avg_test_loss})

def optimize_hyperparameters(train_dataloader, test_dataloader, params, predict_friction, sequence_length):
    # Smaller epoch size for faster hyperparameter optimization to find the best configuration
    # num_epochs is not optimized here
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "num_epochs": 10,
        "physics_weight": tune.uniform(0.1, 10.0),
        "reg_weight": tune.loguniform(1e-6, 1e-4)
    }

    scheduler = ASHAScheduler(
        metric="test_loss",
        mode="min",
        max_t=200,
        grace_period=10,
        reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=["lr", "num_epochs", "physics_weight", "reg_weight"],
        metric_columns=["train_loss", "test_loss", "training_iteration"]
    )
    
    result = tune.run(
        tune.with_parameters(
            objective, 
            train_dataloader=train_dataloader, 
            test_dataloader=test_dataloader, 
            params=params,
            predict_friction=predict_friction,
            sequence_length=sequence_length
        ),
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter,
        log_to_file=("ray_stdout.log", "ray_stderr.log")  # Redirect stdout and stderr
    )

    best_trial = result.get_best_trial("test_loss", "min", "last")
    
    if best_trial is None:
        logger.warning("Warning: Could not find best trial. Using default configuration.")
        return {
            "lr": 1e-3,
            "num_epochs": 100,
            "physics_weight": 1.0,
            "reg_weight": 1e-5
        }
    else:
        logger.info(f"Best trial config: {best_trial.config}")
        logger.info(f"Best trial final train loss: {best_trial.last_result['train_loss']}")
        logger.info(f"Best trial final test loss: {best_trial.last_result['test_loss']}")
        return best_trial.config
