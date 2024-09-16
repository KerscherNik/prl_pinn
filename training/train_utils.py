import torch
from tqdm import tqdm
from ray import tune, train
from ray.tune import CLIReporter
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from model.pinn_model import CartpolePINN
import numpy as np
from model.loss_functions import pinn_loss

def train_pinn(model, train_dataloader, optimizer, loss_fn, params, num_epochs, physics_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        model.train()
        total_loss = 0
        mse_loss = 0
        physics_loss = 0

        for batch in train_dataloader:
            x, x_dot, theta, theta_dot, action = [b.to(device) for b in batch]

            optimizer.zero_grad()
            loss, mse, phys = loss_fn(model, x, x_dot, theta, theta_dot, action, params, physics_weight)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            mse_loss += mse.item()
            physics_loss += phys.item()

        avg_loss = total_loss / len(train_dataloader)
        avg_mse = mse_loss / len(train_dataloader)
        avg_physics = physics_loss / len(train_dataloader)

        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'mse': f'{avg_mse:.4f}',
            'physics': f'{avg_physics:.4f}'
        })

        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"MSE Loss: {avg_mse:.4f}, Physics Loss: {avg_physics:.4f}")

    return model, avg_loss

# Defines the objective for Ray Tune's hyperparameter optimization
# training loss and test loss
def objective(config, train_dataloader, test_dataloader, params, predict_friction):
    model = CartpolePINN(predict_friction=predict_friction)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    try:
        for epoch in range(config["num_epochs"]):
            train_loss = train_pinn(model, train_dataloader, optimizer, pinn_loss, params, 1, config["physics_weight"])
            """ if not np.isfinite(train_loss):
                raise ValueError("Training loss is not finite") """
            scheduler.step(train_loss)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_dataloader:
                batch = [b.to(device) for b in batch]
                x, x_dot, theta, theta_dot, action = batch
                loss, _, _ = pinn_loss(model, x, x_dot, theta, theta_dot, action, params, config["physics_weight"])
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_dataloader)
        
        session.report({"train_loss": train_loss, "test_loss": avg_test_loss})
    except Exception as e:
        print(f"Error during training: {str(e)}")
        session.report({"train_loss": float('inf'), "test_loss": float('inf')})
    finally:
        torch.cuda.empty_cache()

def optimize_hyperparameters(train_dataloader, test_dataloader, params, predict_friction):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "num_epochs": tune.choice([50, 100, 200]),
        "physics_weight": tune.uniform(0.1, 10.0)
    }

    scheduler = ASHAScheduler(
        metric="test_loss",
        mode="min",
        max_t=200,
        grace_period=10,
        reduction_factor=2
    )

    # Command line reporter
    reporter = CLIReporter(
        parameter_columns=["lr", "num_epochs", "physics_weight"],
        metric_columns=["train_loss", "test_loss", "training_iteration"]
    )

    # Tuning the hyperparameters
    result = tune.run(
        tune.with_parameters(
            objective, 
            train_dataloader=train_dataloader, 
            test_dataloader=test_dataloader, 
            params=params,
            predict_friction=predict_friction
        ),
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("test_loss", "min", "last")
    
    if best_trial is None:
        print("Warning: Could not find best trial. Using default configuration.")
        return {
            "lr": 1e-3,
            "num_epochs": 100,
            "physics_weight": 1.0
        }
    else:
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final train loss: {best_trial.last_result['train_loss']}")
        print(f"Best trial final test loss: {best_trial.last_result['test_loss']}")
        return best_trial.config