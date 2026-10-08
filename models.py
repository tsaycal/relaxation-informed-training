"""
Neural network model definitions and bound computation.

All models use ReLU activations in hidden layers and a linear output layer,
compatible with big-M MILP formulation.  Interval bound propagation (IBP)
computes [lb, ub] on pre-activation and post-activation values at each neuron.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ========================================================================== #
#  Model                                                                      #
# ========================================================================== #

@dataclass
class ModelConfig:
    """Configuration for a ReLU neural network."""
    input_dim: int
    hidden_dims: List[int]
    output_dim: int = 1

    @property
    def architecture_str(self) -> str:
        """String representation like '2-32-32-1'."""
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        return "-".join(str(d) for d in dims)


class ReLUNet(nn.Module):
    """Feedforward neural network with ReLU activations.

    Structure: input -> [Linear -> ReLU] x L -> Linear -> output

    Attributes:
        layers: nn.ModuleList of linear layers
        config: ModelConfig used to construct the network
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        dims = [config.input_dim] + config.hidden_dims + [config.output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: shape (batch_size, input_dim)
        Returns:
            shape (batch_size, output_dim)
        """
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def get_weights_and_biases(self):
        """Extract weights and biases as numpy arrays.

        Returns:
            List of (W, b) tuples, one per layer.
            W[l] has shape (out_features, in_features),
            b[l] has shape (out_features,).
        """
        params = []
        for layer in self.layers:
            W = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy()
            params.append((W, b))
        return params

    @property
    def n_hidden_layers(self) -> int:
        return len(self.layers) - 1

    @property
    def n_relu_neurons(self) -> int:
        return sum(self.config.hidden_dims)


def create_model(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int = 1,
    seed: Optional[int] = None,
) -> ReLUNet:
    """Create a ReLU network with optional seed for reproducibility.

    Args:
        input_dim: dimension of input
        hidden_dims: list of hidden layer widths
        output_dim: dimension of output
        seed: random seed for weight initialization

    Returns:
        ReLUNet instance
    """
    if seed is not None:
        torch.manual_seed(seed)

    config = ModelConfig(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    model = ReLUNet(config)

    # Kaiming initialization for ReLU networks
    for layer in model.layers[:-1]:
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)
    # Xavier for output layer (linear activation)
    nn.init.xavier_normal_(model.layers[-1].weight)
    nn.init.zeros_(model.layers[-1].bias)

    return model


# ========================================================================== #
#  Interval bound propagation                                                 #
# ========================================================================== #

def interval_bound_propagation_numpy(
    weights_and_biases: list,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Non-differentiable IBP using numpy arrays (for MILP formulation).

    Args:
        weights_and_biases: list of (W, b) numpy arrays from model.get_weights_and_biases()
        input_lb: shape (input_dim,), lower bounds on input
        input_ub: shape (input_dim,), upper bounds on input

    Returns:
        List of (pre_lb, pre_ub, post_lb, post_ub) for each layer.
        For the output layer, post = pre (no ReLU).
    """
    layer_bounds = []
    lb, ub = input_lb.copy(), input_ub.copy()

    for l, (W, b) in enumerate(weights_and_biases):
        is_output = l == len(weights_and_biases) - 1

        W_pos = np.maximum(W, 0)
        W_neg = np.minimum(W, 0)

        pre_lb = W_pos @ lb + W_neg @ ub + b
        pre_ub = W_pos @ ub + W_neg @ lb + b

        if is_output:
            post_lb, post_ub = pre_lb, pre_ub
        else:
            post_lb = np.maximum(pre_lb, 0)
            post_ub = np.maximum(pre_ub, 0)

        layer_bounds.append((pre_lb, pre_ub, post_lb, post_ub))
        lb, ub = post_lb, post_ub

    return layer_bounds


def interval_bound_propagation(
    model: ReLUNet,
    input_lb: torch.Tensor,
    input_ub: torch.Tensor,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Differentiable IBP using PyTorch tensors.

    Computes interval arithmetic bounds through the network.
    All operations use torch ops, so gradients flow back to model weights.

    Args:
        model: ReLUNet instance
        input_lb: shape (input_dim,), lower bounds on input
        input_ub: shape (input_dim,), upper bounds on input

    Returns:
        List of (pre_lb, pre_ub, post_lb, post_ub) for each layer.
        For the output layer, post = pre (no ReLU).
    """
    layer_bounds = []
    lb, ub = input_lb, input_ub

    for l, layer in enumerate(model.layers):
        is_output = l == len(model.layers) - 1
        W = layer.weight
        b = layer.bias

        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)

        pre_lb = W_pos @ lb + W_neg @ ub + b
        pre_ub = W_pos @ ub + W_neg @ lb + b

        if is_output:
            post_lb, post_ub = pre_lb, pre_ub
        else:
            post_lb = torch.clamp(pre_lb, min=0)
            post_ub = torch.clamp(pre_ub, min=0)

        layer_bounds.append((pre_lb, pre_ub, post_lb, post_ub))
        lb, ub = post_lb, post_ub

    return layer_bounds


def count_unstable_neurons(
    layer_bounds: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> int:
    """Count neurons whose pre-activation bounds straddle zero.

    Args:
        layer_bounds: output from interval_bound_propagation

    Returns:
        Number of unstable (ambiguous) ReLU neurons
    """
    count = 0
    for pre_lb, pre_ub, _, _ in layer_bounds[:-1]:
        unstable = (pre_lb < 0) & (pre_ub > 0)
        count += unstable.sum().item()
    return int(count)


def get_bound_widths(
    layer_bounds: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Get pre-activation bound widths for all hidden neurons.

    Args:
        layer_bounds: output from interval_bound_propagation

    Returns:
        1D tensor of bound widths (pre_ub - pre_lb) for all hidden neurons
    """
    widths = []
    for pre_lb, pre_ub, _, _ in layer_bounds[:-1]:
        widths.append(pre_ub - pre_lb)
    return torch.cat(widths)
