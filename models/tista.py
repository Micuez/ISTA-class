"""TISTA: trainable iterative soft-thresholding with a Gaussian denoiser."""

import math

import numpy as np
import torch
from torch import nn

from .ops import numpy_to_tensor, sensing_init


class TistaCell(nn.Module):
    def __init__(self, measurement_dim, signal_dim, layer_id, taa, tww, sigma2):
        super().__init__()
        self.measurement_dim = measurement_dim
        self.signal_dim = signal_dim
        self.layer_id = layer_id
        self.taa = float(taa)
        self.tww = float(tww)
        self.sigma2 = float(sigma2)

    def _gauss(self, value, variance):
        return torch.exp(-value ** 2 / (2.0 * variance)) / torch.pow(
            2.0 * math.pi * variance, 0.5
        )

    def _shrink(self, estimate, tau2, prior, alpha2):
        signal_var = alpha2 + tau2
        gauss_signal = self._gauss(estimate, signal_var)
        gauss_noise = self._gauss(estimate, tau2)
        return (estimate * alpha2 / signal_var) * prior * gauss_signal / (
            (1.0 - prior) * gauss_noise + prior * gauss_signal
        )

    def _tau2(self, residual, step_size):
        energy = torch.sum(residual ** 2, dim=1, keepdim=True)
        variance = (energy - self.measurement_dim * self.sigma2) / self.taa
        variance = variance.clamp(min=1e-9, max=1e6)
        return (variance / self.signal_dim) * (
            self.signal_dim + (step_size ** 2 - 2.0 * step_size) * self.measurement_dim
        ) + step_size ** 2 * self.tww * self.sigma2 / self.signal_dim

    def forward(self, inputs, sensing, weight, step_size, prior, alpha2):
        measurements = inputs[:, : self.measurement_dim]
        if self.layer_id == 0:
            code = 0.0
            residual = measurements
        else:
            code = inputs[:, -self.signal_dim :]
            residual = measurements - code @ sensing.T
        tau2 = self._tau2(residual, step_size)
        output = code + step_size * (residual @ weight)
        output = self._shrink(output, tau2, prior, alpha2)
        return torch.cat([inputs, output], dim=1)


class Tista(nn.Module):
    def __init__(self, A, T, lam, sigma2, share_W=False, D=None, name="Tista"):
        super().__init__()
        del lam, D
        if sigma2 is None:
            raise ValueError("TISTA requires tista_sigma2.")
        matrix, _, _, _, _ = sensing_init(A, 0.0)
        self.name = name
        self.share_W = share_W
        self.sigma2 = float(sigma2)
        self.num_layers = int(T)
        self.M = int(matrix.shape[0])
        self.N = int(matrix.shape[1])
        self.output_interval = self.N
        gram = matrix.T @ matrix
        pseudo = np.linalg.pinv(matrix)
        self.taa = float(np.trace(gram))
        self.tww = float(np.trace(pseudo.T @ pseudo))
        self.register_buffer("A", numpy_to_tensor(matrix))
        if share_W:
            self.W = nn.Parameter(numpy_to_tensor(matrix))
        else:
            self.W = nn.ParameterList(
                [nn.Parameter(numpy_to_tensor(matrix)) for _ in range(self.num_layers)]
            )
        self.step_size = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0)) for _ in range(self.num_layers)]
        )
        self.p = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(self.num_layers)]
        )
        self.alpha2 = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0)) for _ in range(self.num_layers)]
        )
        self.cells = nn.ModuleList()

    def create_cell(self, layer_id):
        if layer_id != len(self.cells):
            raise RuntimeError(f"Expected to add layer {len(self.cells)}, got {layer_id}.")
        self.cells.append(TistaCell(self.M, self.N, layer_id, self.taa, self.tww, self.sigma2))

    def layer_param_names(self, layer_id):
        names = [f"step_size.{layer_id}", f"p.{layer_id}", f"alpha2.{layer_id}"]
        names.append("W" if self.share_W else f"W.{layer_id}")
        return names

    def forward(self, measurements):
        state = measurements
        for cell in self.cells:
            layer_id = cell.layer_id
            weight = self.W if self.share_W else self.W[layer_id]
            state = cell(
                state,
                self.A,
                weight,
                self.step_size[layer_id],
                self.p[layer_id],
                self.alpha2[layer_id],
            )
        return state
