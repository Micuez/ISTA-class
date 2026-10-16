"""Learned AMP (LAMP)."""

import torch
from torch import nn

from .ops import numpy_to_tensor, sensing_init, shrink_lamp


class LampCell(nn.Module):
    def __init__(self, measurement_dim, signal_dim, layer_id):
        super().__init__()
        self.measurement_dim = measurement_dim
        self.signal_dim = signal_dim
        self.layer_id = layer_id

    def forward(self, inputs, sensing, weight, step_size, lam):
        batch = inputs.shape[0]
        if self.layer_id == 0:
            residual = inputs[:, : self.measurement_dim]
            code = inputs.new_zeros(batch, self.signal_dim)
            onsager = inputs.new_zeros(batch, self.measurement_dim)
            divergence = inputs.new_zeros(batch, 1)
        else:
            measurements = inputs[:, : self.measurement_dim]
            code = inputs[:, -self.signal_dim :]
            # count_nonzero has no gradient in the TensorFlow graph.
            divergence = (code != 0).sum(dim=1, keepdim=True).to(code.dtype).detach()
            divergence = divergence / self.measurement_dim
            residual = measurements - code @ sensing.T
            onsager = inputs[:, -(self.signal_dim + self.measurement_dim) : -self.signal_dim]

        onsager = residual + divergence * onsager
        variance = torch.sum(onsager ** 2, dim=1, keepdim=True) / self.measurement_dim
        residual_code = code + step_size * (onsager @ weight)
        output = shrink_lamp(residual_code, variance, lam)
        return torch.cat([inputs, onsager, output], dim=1)


class Lamp(nn.Module):
    def __init__(self, A, T, lam, share_W=False, D=None, name="Lamp"):
        super().__init__()
        del D
        matrix, scale, _, _, _ = sensing_init(A, lam)
        self.name = name
        self.share_W = share_W
        self.num_layers = int(T)
        self.M = int(matrix.shape[0])
        self.N = int(matrix.shape[1])
        self.output_interval = self.M + self.N
        self.register_buffer("A", numpy_to_tensor(matrix))

        weight = (matrix / scale).astype(matrix.dtype)
        if share_W:
            self.W = nn.Parameter(numpy_to_tensor(weight))
            self.step_size = nn.ParameterList(
                [nn.Parameter(torch.tensor(1.0)) for _ in range(self.num_layers)]
            )
        else:
            self.W = nn.ParameterList(
                [nn.Parameter(numpy_to_tensor(weight)) for _ in range(self.num_layers)]
            )
            self.step_size = [1.0] * self.num_layers
        self.lam = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(lam))) for _ in range(self.num_layers)]
        )
        self.cells = nn.ModuleList()

    def create_cell(self, layer_id):
        if layer_id != len(self.cells):
            raise RuntimeError(f"Expected to add layer {len(self.cells)}, got {layer_id}.")
        self.cells.append(LampCell(self.M, self.N, layer_id))

    def layer_param_names(self, layer_id):
        names = [f"lam.{layer_id}"]
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
            state = cell(state, self.A, weight, self.step_size[layer_id], self.lam[layer_id])
        return state
