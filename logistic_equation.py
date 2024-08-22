from typing import Callable
import argparse

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import torchopt

from pinn import make_forward_fn, LinearNN


R = 1.0  # rate of maximum population growth parameterizing the equation
X_BOUNDARY = 0.0  # boundary condition coordinate
F_BOUNDARY = 0.5  # boundary condition value


if __name__ == "__main__":

    # Make it reproducible
    torch.manual_seed(42)

    # Parse input from user
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num-hidden", type=int, default=5)
    parser.add_argument("-d", "--dim-hidden", type=int, default=5)
    parser.add_argument("-b", "--batch-size", type=int, default=100)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3)
    parser.add_argument("-e", "--num-epochs", type=int, default=1000)

    args = parser.parse_args()

    # Configuration
    num_hidden = args.num_hidden
    dim_hidden = args.dim_hidden
    batch_size = args.batch_size
    num_iter = args.num_epochs
    tolerance = 1e-8
    learning_rate = args.learning_rate
    domain = (-1.0, 1.0)

    # Function versions of model forward, gradient and loss
    model = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=3)
    funcs = make_forward_fn(model, derivative_order=1)

    f = funcs[0]
    dfdx, dfdy, dfdz = funcs[1]
    loss_fn = make_loss_fn(f, dfdx, dfdy, dfdz)

    # Choose optimizer with functional API using functorch
    optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))

    # Initial parameters randomly initialized
    params = tuple(model.parameters())

    # Train the model
    loss_evolution = []
    for i in range(num_iter):

        # Sample points in the domain randomly for each epoch
        x = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1])
        y = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1])
        z = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1])

        # Compute the loss with the current parameters
        loss = loss_fn(params, x, y, z)

        # Update the parameters with functional optimizer
        params = optimizer.step(loss, params)

        print(f"Iteration {i} with loss {float(loss)}")
        loss_evolution.append(float(loss))

    # Plot the final solution (this will be handled by OpenGL)
    x_eval = torch.linspace(domain[0], domain[1], steps=20).reshape(-1, 1)
    y_eval = torch.linspace(domain[0], domain[1], steps=20).reshape(-1, 1)
    z_eval = torch.linspace(domain[0], domain[1], steps=20).reshape(-1, 1)

    # Model evaluation for each point in the 3D space
    f_eval = f(torch.cat([x_eval, y_eval, z_eval], dim=-1), params)
    print(f"Model output at evaluation points: {f_eval}")
