import torch
from tqdm import tqdm
from ray import tune
from ray.tune import CLIReporter
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from model.pinn_model import CartpolePINN
from model.loss_functions import pinn_loss

def train_pinn(model, train_dataloader, optimizer, params, num_epochs, physics_weight, t_span=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)
    print(f"Training for {num_epochs} epochs")
    
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)
    
    for epoch in epoch_pbar:
        model.train()
        total_loss, total_mse_loss, total_physics_loss = 0, 0, 0

        batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", position=1, leave=False)
        
        for batch_idx, (sequences, targets) in enumerate(batch_pbar):
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()

            # Simulation progress bar
            sim_pbar = tqdm(total=100, desc="Simulation", position=2, leave=False)
            
            def simulation_callback(t, state):
                sim_pbar.update(int(t.item() / t_span * 100) - sim_pbar.n)

            # Loss calculation progress bar
            loss_pbar = tqdm(total=100, desc="Loss Calculation", position=3, leave=False)

            def loss_calculation_callback(progress):
                loss_pbar.update(int(progress * 100) - loss_pbar.n)

            loss, mse_loss, physics_loss = pinn_loss(
                model, sequences, targets, params, physics_weight, t_span=t_span,
                simulation_callback=simulation_callback,
                loss_calculation_callback=loss_calculation_callback
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

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"MSE Loss: {avg_mse:.4f}, Physics Loss: {avg_physics:.4f}")

    return model, avg_loss

def objective(config, train_dataloader, test_dataloader, params, predict_friction, sequence_length):
    model = CartpolePINN(sequence_length=sequence_length, predict_friction=predict_friction)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(config["num_epochs"]):
        model, avg_train_loss = train_pinn(
            model, train_dataloader, optimizer, params, 1, config["physics_weight"], t_span=1.0
        )
        scheduler.step(avg_train_loss)
        
        # Evaluate on test set
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for sequences, targets in test_dataloader:
                sequences, targets = sequences.to(device), targets.to(device)
                loss, _, _ = pinn_loss(
                    model, sequences, targets, params, config["physics_weight"], t_span=1.0
                )
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_dataloader)
        
        # Report metrics to Ray Tune
        session.report({"train_loss": avg_train_loss, "test_loss": avg_test_loss})

def optimize_hyperparameters(train_dataloader, test_dataloader, params, predict_friction, sequence_length):
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

    reporter = CLIReporter(
        parameter_columns=["lr", "num_epochs", "physics_weight"],
        metric_columns=["train_loss", "test_loss", "training_iteration"]
    )
    # TODO: Uncomment this block to run hyperparameter optimization
    """ result = tune.run(
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
        progress_reporter=reporter
    ) """

    #best_trial = result.get_best_trial("test_loss", "min", "last")
    best_trial = None
    if best_trial is None:
        print("Warning: Could not find best trial. Using default configuration.")
        return {
            "lr": 1e-3,
            "num_epochs": 10,
            "physics_weight": 1.0
        }
    else:
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final train loss: {best_trial.last_result['train_loss']}")
        print(f"Best trial final test loss: {best_trial.last_result['test_loss']}")
        return best_trial.config