"""GLISTA: gated LISTA with gain and overshoot."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .ops import numpy_to_tensor, sensing_init, shrink_ss


def _reweight_relu(code, gain, theta, alti):
    gate = F.relu(1.0 - F.relu(gain * code.abs()))
    return 1.0 + alti * theta * gate


def _reweight_inverse(code, gain, theta, alti):
    return 1.0 + alti * theta * 0.2 / (0.001 + (gain * code).abs())


def _reweight_exp(code, gain, theta, alti):
    return 1.0 + alti * theta * torch.exp(-gain * code.abs())


def _reweight_sigmoid(code, gain, theta, alti):
    return 1.0 + alti * theta * torch.sigmoid(-gain * code.abs())


_GAIN_FUNCTIONS = {
    "relu": _reweight_relu,
    "inv": _reweight_inverse,
    "exp": _reweight_exp,
    "sigm": _reweight_sigmoid,
}


class GlistaCell(nn.Module):
    def __init__(self, measurement_dim, signal_dim, layer_id, support_fraction, gain_func):
        super().__init__()
        self.measurement_dim = measurement_dim
        self.signal_dim = signal_dim
        self.layer_id = layer_id
        self.support_fraction = float(support_fraction)
        if gain_func not in _GAIN_FUNCTIONS and gain_func != "none":
            raise ValueError(f"Unknown gain_func {gain_func!r}.")
        self.gain_func = gain_func

    def _gain(self, code, gain, alti):
        if self.gain_func == "none":
            return code.new_tensor(1.0)
        # The original gain call passes theta=1, so the threshold inside the
        # gate is absorbed into ``alti`` rather than the layer threshold.
        return _GAIN_FUNCTIONS[self.gain_func](code, gain, 1.0, alti)

    def forward(self, inputs, sensing, weight, step_size, theta, gain, alti, alti_over):
        measurements = inputs[:, : self.measurement_dim]
        if self.layer_id == 0:
            code = inputs.new_zeros(inputs.shape[0], self.signal_dim)
            part2 = theta
        else:
            code = inputs[:, -self.signal_dim :]
            support = inputs[:, -2 * self.signal_dim : -self.signal_dim]
            part2 = theta * support

        gate = self._gain(code, gain, alti)
        residual = measurements - (gate * code) @ sensing.T
        proposed = gate * code + step_size * (residual @ weight)
        shrunk, complement = shrink_ss(
            proposed, theta, self.support_fraction, return_index=True
        )
        part1 = 1.0 / ((shrunk - code).abs() + 0.1)
        overshoot = 1.0 - alti_over * part1 * part2
        output = overshoot * shrunk + (1.0 - overshoot) * code
        return torch.cat([inputs, complement, output], dim=1)


class Glista(nn.Module):
    def __init__(
        self,
        A,
        T,
        lam,
        q_per_layer,
        maxq,
        share_W=False,
        D=None,
        alti=5.0,
        gain_func="relu",
        name="Glista",
    ):
        super().__init__()
        del D
        matrix, scale, theta, _, _ = sensing_init(A, lam)
        self.name = name
        self.share_W = share_W
        self.gain_func = gain_func
        self.num_layers = int(T)
        self.M = int(matrix.shape[0])
        self.N = int(matrix.shape[1])
        self.output_interval = self.N * 2
        self.q = np.clip(
            [(t + 1) * q_per_layer for t in range(self.num_layers)], 0.0, maxq
        )
        self.register_buffer("A", numpy_to_tensor(matrix))
        weight = (matrix / scale).astype(matrix.dtype)
        gain = np.ones((1, self.N), dtype=np.float32)
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
        self.theta = nn.ParameterList(
            [nn.Parameter(numpy_to_tensor(theta).reshape(())) for _ in range(self.num_layers)]
        )
        self.alti = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(alti))) for _ in range(self.num_layers)]
        )
        self.alti_over = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(alti))) for _ in range(self.num_layers)]
        )
        self.D_gain = nn.ParameterList(
            [nn.Parameter(numpy_to_tensor(gain)) for _ in range(self.num_layers)]
        )
        self.cells = nn.ModuleList()

    def create_cell(self, layer_id):
        if layer_id != len(self.cells):
            raise RuntimeError(f"Expected to add layer {len(self.cells)}, got {layer_id}.")
        self.cells.append(
            GlistaCell(self.M, self.N, layer_id, self.q[layer_id], self.gain_func)
        )

    def layer_param_names(self, layer_id):
        names = [
            f"theta.{layer_id}",
            f"alti.{layer_id}",
            f"alti_over.{layer_id}",
            f"D_gain.{layer_id}",
        ]
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
            state = cell(
                state,
                self.A,
                weight,
                self.step_size[layer_id],
                self.theta[layer_id],
                self.D_gain[layer_id],
                self.alti[layer_id],
                self.alti_over[layer_id],
            )
        return state
