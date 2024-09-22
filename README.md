# Physics-Informed Neural Network for CartPole Simulation

This project implements a Physics-Informed Neural Network (PINN) to simulate the CartPole system, combining real-world data with known physical equations.

## Overview

The CartPole system consists of a pendulum attached to a cart moving along a frictionless track. Our PINN approach aims to accurately model this system by learning from both demonstration data and the underlying physical laws.

![CartPole System](media/cart_pole.gif)

*GIF Source: [Gymnasium Documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/#cart-pole)*

## Key Features

- Data-driven learning from real CartPole demonstrations
- Physics-informed loss function incorporating known differential equations
- Hyperparameter optimization using Ray Tune
- Integration with OpenAI Gym for reinforcement learning applications
- Visulization of original and integrated gym environment side-by-side using the compare_envs_interactively script (call with python -m integration.compare_envs_interactively)

## Physical Model

The CartPole system is governed by the following differential equations:

![Differential Equations](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Cbegin%7Balign*%7D%20%5Cddot%7B%5Ctheta%7D%20%26%3D%20%5Cfrac%7Bg%20%5Csin%5Ctheta%20&plus;%20%5Ccos%5Ctheta%20%5Cleft%5B%20%5Cfrac%7B-F%20-%20m_p%20l%20%5Cdot%7B%5Ctheta%7D%5E2%20%5Csin%5Ctheta%20&plus;%20%5Cmu_c%20%5Ctext%7Bsgn%7D%28%5Cdot%7Bx%7D%29%7D%7Bm_c%20&plus;%20m_p%7D%20%5Cright%5D%20-%20%5Cfrac%7B%5Cmu_p%20%5Cdot%7B%5Ctheta%7D%7D%7Bm_p%20l%7D%7D%7Bl%20%5Cleft%5B%20%5Cfrac%7B4%7D%7B3%7D%20-%20%5Cfrac%7Bm_p%20%5Ccos%5E2%5Ctheta%7D%7Bm_c%20&plus;%20m_p%7D%20%5Cright%5D%7D%20%5C%5C%20%5Cddot%7Bx%7D%20%26%3D%20%5Cfrac%7BF%20&plus;%20m_p%20l%20%5Cleft%5B%20%5Cdot%7B%5Ctheta%7D%5E2%20%5Csin%5Ctheta%20-%20%5Cddot%7B%5Ctheta%7D%20%5Ccos%5Ctheta%20%5Cright%5D%20-%20%5Cmu_c%20%5Ctext%7Bsgn%7D%28%5Cdot%7Bx%7D%29%7D%7Bm_c%20&plus;%20m_p%7D%20%5Cend%7Balign*%7D)

Where:

- θ: Angular position of the pendulum
- x: Position of the cart
- F: Force applied to the cart
- m_c: Mass of the cart
- m_p: Mass of the pendulum
- l: Length of the pendulum
- g: Gravitational constant
- μ_c: Friction coefficient of the cart
- μ_p: Friction coefficient of the pendulum

## Model Architecture

Our PINN model is implemented as a feedforward neural network with the option to predict friction coefficients:

```python
class CartpolePINN(nn.Module):
    def __init__(self, predict_friction=False):
        super().__init__()
        self.predict_friction = predict_friction
        
        self.lstm = nn.LSTM(input_size=5, hidden_size=128, num_layers=2, batch_first=True)

        self.network = nn.Sequential(
            nn.Linear(128, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128)
        )
```

## Data

The model is trained on real CartPole demonstration data with the following structure:

| datetime | cartPos | cartVel | pendPos | pendVel | action | cartController | exploration |
|----------|---------|---------|---------|---------|--------|----------------|-------------|
| 14.02.2019 11:12 | 339.18 | 0.0 | 0.04 | 0.0 | 0 | Gamepad | 0 |
| 14.02.2019 11:12 | 338.83 | -0.35 | 0.04 | 0.0 | 0 | Gamepad | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... |

## Usage

1. Prepare your CartPole demonstration data in CSV format.
2. Install the required dependencies from the requirements.txt file. We used python3.10 for our implementation. You may also want to install cuda ([Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) | [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)) and use the appropriate installation link for the [pytorch with cuda](https://pytorch.org/get-started/locally/) installation. We recommend using a virtual environment with conda or virtualenv, to manage dependencies.
3. Run the main script:

   ```bash
   python main.py
   ```

4. The script will:
   - Load and preprocess the data
   - Perform hyperparameter optimization
   - Train the PINN model
   - Evaluate the model's performance
   - Create a Gym environment using the trained PINN
   - Evaluate the custom Gym environment against the original cartpole environment
  
5. Custom evaluation
  You can also run a custom evaluation of your own trained model, by executing the compare_environments.py script and changing the location in the main function to you .pth file. This will wrap the model in a gymnasium environment and compare it against the original cartpole environment on a trained PPO model (on original model).

  ```bash
   python compare_environments.py
   ```

## Dependencies

- PyTorch
- NumPy
- Pandas
- Ray Tune
- Gymnasium

## Future Work

- Correctly integrate the option to choose between training with prediction of friction coefficients and force or only force prediction
- Implement more sophisticated PINN architectures
- Explore transfer learning to other control tasks
- Better interface for gym integration of cartpole simulation

## References

1. Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. IEEE transactions on systems, man, and cybernetics, (5), 834-846.
2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.
