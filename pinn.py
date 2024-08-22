from collections import OrderedDict
from typing import Callable

import torch
from torch import nn
from torch.func import functional_call, grad, vmap



class LinearNN(nn.Module):
    def __init__(
        self,
        num_inputs: int = 3,  # Adjust to 3 for (x, y, z)
        num_layers: int = 3,  # Increase layers for complexity
        num_neurons: int = 10,  # Increase neurons to handle 3D complexity
        act: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.num_layers = num_layers

        layers = []

        # input layer
        layers.append(nn.Linear(self.num_inputs, num_neurons))

        # hidden layers with linear layer and activation
        for _ in range(num_layers):
            layers.extend([nn.Linear(num_neurons, num_neurons), act])

        # output layer
        layers.append(nn.Linear(num_neurons, 1))

        # build the network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)






def make_loss_fn(f: Callable, dfdx: Callable) -> Callable:
    def loss_fn(params: torch.Tensor, inputs: torch.Tensor):
        # Ensure inputs tensor has the correct shape: [batch_size, 3]
        if inputs.shape[1] != 3:
            raise ValueError(f"Expected inputs to have shape [batch_size, 3], but got {inputs.shape}")

        # Compute the derivatives (Laplacian components)
        dfdx_val = dfdx(inputs, params)  # Pass the full input tensor
        print(f"dfdx_val shape: {dfdx_val.shape}")

        # Compute the Laplacian as the sum of the second derivatives
        laplacian = dfdx_val  # Assuming `dfdx_val` includes all partial derivatives

        print(f"laplacian shape: {laplacian.shape}")

        # Boundary condition (example: zero boundary condition)
        boundary = torch.zeros_like(inputs[:, 0:1])  # Ensure this has the same shape as one input column

        # Compute the loss as the sum of the Laplacian loss and boundary loss
        loss = nn.MSELoss()
        laplacian_loss = loss(laplacian, torch.zeros_like(laplacian))  # Interior points loss
        boundary_loss = loss(f(inputs, params), boundary)  # Boundary condition loss

        # Total loss is the sum of interior and boundary losses
        loss_value = laplacian_loss + boundary_loss

        return loss_value

    return loss_fn








def make_forward_fn(model: nn.Module, derivative_order: int = 1) -> list[Callable]:
    """Make a functional forward pass and gradient functions for 3D"""

    def f(x: torch.Tensor, params: dict[str, torch.nn.Parameter] | tuple[torch.nn.Parameter, ...]) -> torch.Tensor:
        if isinstance(params, tuple):
            params_dict = tuple_to_dict_parameters(model, params)
        else:
            params_dict = params
        return functional_call(model, params_dict, (x, ))

    fns = [f]

    dfunc = f
    for _ in range(derivative_order):
        # Derivative with respect to the entire input (x, y, z together)
        dfdx = grad(dfunc, argnums=0)  # Gradient w.r.t. x, y, z together

        # Use vmap to support batching, ensure you map over the first dimension
        fns.append(lambda x, params: vmap(dfdx, in_dims=(0, None))(x, params))

    return fns




def tuple_to_dict_parameters(
        model: nn.Module, params: tuple[torch.nn.Parameter, ...]
) -> OrderedDict[str, torch.nn.Parameter]:
    """Convert a set of parameters stored as a tuple into a dictionary form

    This conversion is required to be able to call the `functional_call` API which requires
    parameters in a dictionary form from the results of a functional optimization step which 
    returns the parameters as a tuple

    Args:
        model (nn.Module): the model to make the functional calls for. It can be any subclass of
            a nn.Module
        params (tuple[Parameter, ...]): the model parameters stored as a tuple
    
    Returns:
        An OrderedDict instance with the parameters stored as an ordered dictionary
    """
    keys = list(dict(model.named_parameters()).keys())
    values = list(params)
    return OrderedDict(({k:v for k,v in zip(keys, values)}))


if __name__ == "__main__":

    # TODO: turn this into a unit test
    
    model = LinearNN(num_layers=2)
    fns = make_forward_fn(model, derivative_order=2)

    batch_size = 10
    x = torch.randn(batch_size)
    # params = dict(model.named_parameters())
    params = dict(model.named_parameters())

    fn_x = fns[0](x, params)
    assert fn_x.shape[0] == batch_size

    dfn_x = fns[1](x, params)
    assert dfn_x.shape[0] == batch_size

    ddfn_x = fns[2](x, params)
    assert ddfn_x.shape[0] == batch_size
