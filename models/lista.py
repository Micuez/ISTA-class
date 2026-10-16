"""LISTA in feed-forward form."""

import torch
from torch import nn

from .ops import numpy_to_tensor, sensing_init, shrink_free


class ListaCell(nn.Module):
    def __init__(self, measurement_dim, signal_dim, layer_id, share_W):
        super().__init__()
        self.measurement_dim = measurement_dim
        self.signal_dim = signal_dim
        self.layer_id = layer_id
        self.share_W = share_W

    def forward(self, inputs, basis, tied, step_size, theta):
        output = inputs[:, : self.measurement_dim] @ basis.T
        if self.layer_id != 0:
            previous = inputs[:, -self.signal_dim :]
            if self.share_W:
                previous = previous * step_size
            output = output + previous @ tied.T
        output = shrink_free(output, theta)
        return torch.cat([inputs, output], dim=1)


class Lista(nn.Module):
    """Feed-forward LISTA.

    Layer 0 applies only ``B``. Later layers apply a learned ``W``. With
    ``share_W=False`` (the training default) the per-layer step size is the
    constant 1 and is absorbed into ``W``.
    """

    def __init__(self, A, T, lam, share_W=False, D=None, name="Lista"):
        super().__init__()
        del D  # The original cells never apply the dictionary.
        matrix, _, theta, basis, tied = sensing_init(A, lam)
        self.name = name
        self.share_W = share_W
        self.num_layers = int(T)
        self.M = int(matrix.shape[0])
        self.N = int(matrix.shape[1])
        self.output_interval = self.N

        self.register_buffer("A", numpy_to_tensor(matrix))
        self.B = nn.Parameter(numpy_to_tensor(basis))
        self.theta = nn.ParameterList(
            [nn.Parameter(numpy_to_tensor(theta).reshape(())) for _ in range(self.num_layers)]
        )
        if share_W:
            self.W = nn.Parameter(numpy_to_tensor(tied))
            self.step_size = nn.ParameterList(
                [nn.Parameter(torch.tensor(1.0)) for _ in range(self.num_layers)]
            )
        else:
            self.W = nn.ParameterList(
                [nn.Parameter(numpy_to_tensor(tied)) for _ in range(1, self.num_layers)]
            )
            self.step_size = [1.0] * self.num_layers
        self.cells = nn.ModuleList()

    def create_cell(self, layer_id):
        if layer_id != len(self.cells):
            raise RuntimeError(f"Expected to add layer {len(self.cells)}, got {layer_id}.")
        self.cells.append(ListaCell(self.M, self.N, layer_id, self.share_W))

    def layer_param_names(self, layer_id):
        names = ["B", f"theta.{layer_id}"]
        if self.share_W:
            names.append(f"step_size.{layer_id}")
            if layer_id != 0:
                names.append("W")
        elif layer_id != 0:
            names.append(f"W.{layer_id - 1}")
        return names

    def forward(self, measurements):
        state = measurements
        for cell in self.cells:
            layer_id = cell.layer_id
            tied = self.W if self.share_W else (None if layer_id == 0 else self.W[layer_id - 1])
            state = cell(state, self.B, tied, self.step_size[layer_id], self.theta[layer_id])
        return state
