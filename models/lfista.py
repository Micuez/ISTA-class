"""LFISTA, the learned fast ISTA unfolding."""

import math

import torch
from torch import nn

from .ops import numpy_to_tensor, sensing_init, shrink_free


class LfistaCell(nn.Module):
    def __init__(self, measurement_dim, signal_dim, layer_id):
        super().__init__()
        self.measurement_dim = measurement_dim
        self.signal_dim = signal_dim
        self.layer_id = layer_id

    def forward(self, inputs, we, wg, wm, theta):
        output = inputs[:, : self.measurement_dim] @ we.T
        if self.layer_id > 0:
            output = output + inputs[:, -self.signal_dim :] @ wg.T
        if self.layer_id > 1:
            previous = inputs[:, -2 * self.signal_dim : -self.signal_dim]
            output = output + previous @ wm.T
        output = shrink_free(output, theta)
        return torch.cat([inputs, output], dim=1)


class Lfista(nn.Module):
    def __init__(self, A, T, lam, share_W=False, D=None, name="Lfista"):
        super().__init__()
        del share_W, D
        matrix, _, theta, basis, tied = sensing_init(A, lam)
        self.name = name
        self.num_layers = int(T)
        self.M = int(matrix.shape[0])
        self.N = int(matrix.shape[1])
        self.output_interval = self.N
        self.register_buffer("A", numpy_to_tensor(matrix))

        # The TensorFlow implementation builds a one-element list and repeats
        # that same Variable ``T`` times, so every layer shares one ``We``.
        self.We = nn.Parameter(numpy_to_tensor(basis))
        momentum = _fista_momentum(self.num_layers)
        self.Wg = nn.ParameterList(
            [
                nn.Parameter(numpy_to_tensor(tied * (1.0 + momentum[i])))
                for i in range(1, self.num_layers)
            ]
        )
        self.Wm = nn.ParameterList(
            [
                nn.Parameter(numpy_to_tensor(-momentum[i] * tied))
                for i in range(1, self.num_layers)
            ]
        )
        self.theta = nn.ParameterList(
            [nn.Parameter(numpy_to_tensor(theta).reshape(())) for _ in range(self.num_layers)]
        )
        self.cells = nn.ModuleList()

    def create_cell(self, layer_id):
        if layer_id != len(self.cells):
            raise RuntimeError(f"Expected to add layer {len(self.cells)}, got {layer_id}.")
        self.cells.append(LfistaCell(self.M, self.N, layer_id))

    def layer_param_names(self, layer_id):
        names = ["We", f"theta.{layer_id}"]
        if layer_id != 0:
            names.extend([f"Wg.{layer_id - 1}", f"Wm.{layer_id - 1}"])
        return names

    def forward(self, measurements):
        state = measurements
        for cell in self.cells:
            layer_id = cell.layer_id
            wg = None if layer_id == 0 else self.Wg[layer_id - 1]
            wm = None if layer_id == 0 else self.Wm[layer_id - 1]
            state = cell(state, self.We, wg, wm, self.theta[layer_id])
        return state


def _fista_momentum(num_layers):
    tk = [1.0, 1.0]
    mk = []
    for _ in range(num_layers):
        tk.append((1.0 + math.sqrt(1.0 + 4.0 * tk[-1] ** 2)) / 2.0)
        mk.append((tk[-2] - 1.0) / tk[-1])
    return mk
