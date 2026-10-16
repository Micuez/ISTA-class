"""Step-LISTA: only the ISTA step size is learned."""

import torch
from torch import nn

from .ops import numpy_to_tensor, sensing_init, shrink_free


class StepListaCell(nn.Module):
    def __init__(self, measurement_dim, signal_dim, layer_id):
        super().__init__()
        self.measurement_dim = measurement_dim
        self.signal_dim = signal_dim
        self.layer_id = layer_id

    def forward(self, inputs, sensing, step_size, lam):
        if self.layer_id == 0:
            residual = inputs[:, : self.measurement_dim]
            code = 0.0
        else:
            measurements = inputs[:, : self.measurement_dim]
            code = inputs[:, -self.signal_dim :]
            residual = measurements - code @ sensing.T
        output = code + step_size * (residual @ sensing)
        output = shrink_free(output, lam * step_size)
        return torch.cat([inputs, output], dim=1)


class StepLista(nn.Module):
    def __init__(self, A, T, lam, D=None, name="StepLista"):
        super().__init__()
        del D
        matrix, scale, _, _, _ = sensing_init(A, lam)
        self.name = name
        self.num_layers = int(T)
        self._lam = float(lam)
        self.M = int(matrix.shape[0])
        self.N = int(matrix.shape[1])
        self.output_interval = self.N
        self.register_buffer("A", numpy_to_tensor(matrix))
        init_step = float(1.0 / scale)
        self.step_size = nn.ParameterList(
            [nn.Parameter(torch.tensor(init_step)) for _ in range(self.num_layers)]
        )
        self.cells = nn.ModuleList()

    def create_cell(self, layer_id):
        if layer_id != len(self.cells):
            raise RuntimeError(f"Expected to add layer {len(self.cells)}, got {layer_id}.")
        self.cells.append(StepListaCell(self.M, self.N, layer_id))

    def layer_param_names(self, layer_id):
        return [f"step_size.{layer_id}"]

    def forward(self, measurements):
        state = measurements
        for cell in self.cells:
            state = cell(state, self.A, self.step_size[cell.layer_id], self._lam)
        return state
