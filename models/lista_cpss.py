"""LISTA-CPSS: LISTA-CP with support selection."""

import numpy as np
import torch
from torch import nn

from .ops import numpy_to_tensor, sensing_init, shrink_ss


class ListaCpssCell(nn.Module):
    def __init__(self, measurement_dim, signal_dim, layer_id, support_fraction):
        super().__init__()
        self.measurement_dim = measurement_dim
        self.signal_dim = signal_dim
        self.layer_id = layer_id
        self.support_fraction = float(support_fraction)

    def forward(self, inputs, sensing, weight, step_size, theta):
        if self.layer_id == 0:
            residual = inputs[:, : self.measurement_dim]
            code = 0.0
        else:
            measurements = inputs[:, : self.measurement_dim]
            code = inputs[:, -self.signal_dim :]
            residual = measurements - code @ sensing.T
        output = code + step_size * (residual @ weight)
        output = shrink_ss(output, theta, self.support_fraction)
        return torch.cat([inputs, output], dim=1)


class ListaCpss(nn.Module):
    def __init__(
        self,
        A,
        T,
        lam,
        q_per_layer,
        maxq,
        share_W=False,
        D=None,
        name="ListaCpss",
    ):
        super().__init__()
        del D
        matrix, _, theta, _, _ = sensing_init(A, lam)
        self.name = name
        self.share_W = share_W
        self.num_layers = int(T)
        self.M = int(matrix.shape[0])
        self.N = int(matrix.shape[1])
        self.output_interval = self.N
        self.q = np.clip(
            [(t + 1) * q_per_layer for t in range(self.num_layers)], 0.0, maxq
        )
        self.register_buffer("A", numpy_to_tensor(matrix))
        weight = numpy_to_tensor(matrix)
        if share_W:
            self.W = nn.Parameter(weight)
            self.step_size = nn.ParameterList(
                [nn.Parameter(torch.tensor(1.0)) for _ in range(self.num_layers)]
            )
        else:
            self.W = nn.ParameterList(
                [nn.Parameter(weight.clone()) for _ in range(self.num_layers)]
            )
            self.step_size = [1.0] * self.num_layers
        self.theta = nn.ParameterList(
            [nn.Parameter(numpy_to_tensor(theta).reshape(())) for _ in range(self.num_layers)]
        )
        self.cells = nn.ModuleList()

    def create_cell(self, layer_id):
        if layer_id != len(self.cells):
            raise RuntimeError(f"Expected to add layer {len(self.cells)}, got {layer_id}.")
        self.cells.append(ListaCpssCell(self.M, self.N, layer_id, self.q[layer_id]))

    def layer_param_names(self, layer_id):
        names = [f"theta.{layer_id}"]
        if self.share_W:
            names.extend(["W", f"step_size.{layer_id}"])
        else:
            names.append(f"W.{layer_id}")
        return names

    def forward(self, measurements):
        state = measurements
        for cell in self.cells:
            layer_id = cell.layer_id
            weight = self.W if self.share_W else self.W[layer_id]
            state = cell(state, self.A, weight, self.step_size[layer_id], self.theta[layer_id])
        return state
