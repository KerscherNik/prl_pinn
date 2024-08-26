import torch
from tqdm import tqdm
from ray import tune
from ray.tune import CLIReporter
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from pinn_model import CartpolePINN
from loss_functions import pinn_loss

def train_pinn(model, dataloader, optimizer, loss_fn, params, num_epochs):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)

    for epoch in tqdm(range(num_epochs), desc="Training"):
        total_loss = 0
        for batch in dataloader:
            x, x_dot, theta, theta_dot, action = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]
            x, x_dot, theta, theta_dot, action = x.to(device), x_dot.to(device), theta.to(device), theta_dot.to(device), action.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model, x, x_dot, theta, theta_dot, action, params)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        torch.cuda.empty_cache()

    return model

def objective(config, train_dataloader, params):
    model = CartpolePINN(predict_friction=config["predict_friction"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    try:
        for epoch in range(config["num_epochs"]):
            model = train_pinn(model, train_dataloader, optimizer, pinn_loss, params, 1)
        
        batch = next(iter(train_dataloader))
        x, x_dot, theta, theta_dot, action = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]
        x, x_dot, theta, theta_dot, action = x.to(device), x_dot.to(device), theta.to(device), theta_dot.to(device), action.to(device)
        
        final_loss = pinn_loss(model, x, x_dot, theta, theta_dot, action, params).item()
        
        # Use session.report instead of tune.report
        session.report({"loss": final_loss})
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("| WARNING: ran out of memory, skipping this trial")
            session.report({"loss": float('inf')})
        else:
            raise e
    finally:
        torch.cuda.empty_cache()

def optimize_hyperparameters(train_dataloader, params):
    # TODO: More friction searches in hyperparameter optimization
    
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "num_epochs": tune.choice([50, 100, 200]),
        "predict_friction": tune.choice([True, False])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=200,
        grace_period=10,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"]
    )

    result = tune.run(
        tune.with_parameters(objective, train_dataloader=train_dataloader, params=params),
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config:", best_trial.config)
    print("Best trial final loss:", best_trial.last_result["loss"])

    return best_trial.config