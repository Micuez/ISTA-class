"""Unrolled sparse recovery models. Observation batches are row-major."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


METHODS = ("EBT-LISTA", "HLISTA", "ALISTA", "ELISTA", "NA-ALISTA")


def scalar(value):
    return nn.Parameter(torch.tensor(float(value), dtype=torch.float32))


def shrink(z, theta):
    return z.sign() * F.relu(z.abs() - theta)


def shrink_ss(z, theta, rank):
    # Nearest upper-tail order statistic, then strict comparisons as in utils/tf.py.
    cutoff = z.detach().abs().topk(rank + 1, dim=1).values[:, -1:]
    selected = (z.abs() > cutoff) & (z.abs() > theta)
    return torch.where(selected, z, shrink(z, theta))


class AnalyticLayer(nn.Module):
    def __init__(self, threshold, rank):
        super().__init__()
        self.alpha = scalar(1)
        self.theta = scalar(threshold)
        self.rank = rank

    def forward(self, x, y, A, W):
        direction = (y - x @ A.T) @ W
        return shrink_ss(x + self.alpha * direction, self.theta, self.rank)


def multistage_threshold(z, first, second):
    # Li et al., Eq. (6). Keep the two breakpoints ordered during optimization.
    a, b = first.abs(), second.abs()
    lo, hi = torch.minimum(a, b), torch.maximum(a, b)
    width = hi - lo
    safe_width = torch.where(width > 0, width, torch.ones_like(width))
    middle = z.sign() * hi * ((z.abs() - lo) / safe_width).clamp(0, 1)
    return torch.where(z.abs() >= hi, z, middle)


class ExtragradientLayer(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.alpha1, self.alpha2 = scalar(1), scalar(1)
        self.theta1, self.upper1 = scalar(threshold - 0.1), scalar(threshold)
        self.theta2, self.upper2 = scalar(threshold - 0.1), scalar(threshold)

    def forward(self, x, y, A, W):
        mid = multistage_threshold(x + self.alpha1 * ((y - x @ A.T) @ W),
                                   self.theta1, self.upper1)
        # The second residual is evaluated at mid, but the update starts at x.
        return multistage_threshold(x + self.alpha2 * ((y - mid @ A.T) @ W),
                                    self.theta2, self.upper2)


class ErrorThresholdLayer(nn.Module):
    def __init__(self, weight, threshold, rank):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.theta = scalar(threshold)
        self.bias = scalar(0)
        self.rank = rank

    def forward(self, x, y, A, W):
        direction = (y - x @ A.T) @ self.weight
        amplitude = torch.linalg.vector_norm(direction, dim=1, keepdim=True) / 50.0
        return shrink_ss(x + direction, self.theta * amplitude + self.bias, self.rank)


class HybridLayer(nn.Module):
    def __init__(self, weight, threshold, rank):
        super().__init__()
        self.weight1 = nn.Parameter(weight.clone())
        self.weight2 = nn.Parameter(weight.clone())
        self.theta1, self.theta2 = scalar(threshold), scalar(threshold)
        self.alpha = scalar(0)
        self.rank = rank
        self.cnn = nn.Sequential(nn.Conv1d(1, 16, 9, padding=4, bias=False), nn.ReLU(),
                                 nn.Conv1d(16, 16, 9, padding=4, bias=False), nn.ReLU(),
                                 nn.Conv1d(16, 1, 9, padding=4, bias=False))
        for layer in self.cnn:
            if isinstance(layer, nn.Conv1d):
                kernel = torch.empty(9 * layer.in_channels, layer.out_channels)
                nn.init.orthogonal_(kernel)
                with torch.no_grad():
                    layer.weight.copy_(kernel.reshape(9, layer.in_channels, -1).permute(2, 1, 0))

    def forward(self, x, y, A, W):
        t1, t2 = self.theta1.abs(), self.theta2.abs()
        v = shrink_ss(x + (y - x @ A.T) @ self.weight1, t1, self.rank)
        # Preserve the association between each sample and its convolution output.
        u = v + self.cnn(v.unsqueeze(1)).squeeze(1)
        w = shrink_ss(u + (y - u @ A.T) @ self.weight2, t2, self.rank)
        mix = self.alpha.sigmoid() * t1 / (t1 + t2).clamp_min(torch.finfo(x.dtype).tiny)
        return mix * w + (1 - mix) * v


class UnfoldedISTA(nn.Module):
    def __init__(self, method, A, W, depth):
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("W", W)
        scale = 1.001 * torch.linalg.matrix_norm(A, ord=2).item() ** 2
        self.shared_weight = nn.Parameter(A.clone() / scale) if method == "ELISTA" else None
        self.layers = nn.ModuleList()
        for t in range(1, depth + 1):
            percent = min(t * (0.7 if method == "HLISTA" else 1.2), 13)
            rank = round((A.shape[1] - 1) * percent / 100)
            if method == "ALISTA":
                layer = AnalyticLayer(0.4 / scale, rank)
            elif method == "ELISTA":
                layer = ExtragradientLayer(0.4 / scale)
            elif method == "HLISTA":
                layer = HybridLayer(A / scale, 0.4 / scale, rank)
            else:
                layer = ErrorThresholdLayer(A / scale, 0.4 / scale, rank)
            self.layers.append(layer)

    def stage_module(self, depth):
        return self.layers[depth - 1]

    def forward(self, y, depth, last_step_grad=False, return_path=False):
        x = y.new_zeros((len(y), self.A.shape[1]))
        path = []
        weight = self.W if self.shared_weight is None else self.shared_weight
        for i in range(depth):
            with torch.set_grad_enabled(torch.is_grad_enabled() and
                                        (not last_step_grad or i == depth - 1)):
                x = self.layers[i](x, y, self.A, weight)
            if return_path:
                path.append(x)
        return path if return_path else x


class NormController(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTMCell(2, 128)
        self.lll = nn.Linear(128, 128)
        self.linear = nn.Linear(128, 2)
        self.hidden = nn.Parameter(torch.tensor(np.random.normal(size=128), dtype=torch.float32))
        self.cellstate = nn.Parameter(torch.tensor(np.random.normal(size=128), dtype=torch.float32))

    def forward(self, features, state):
        hidden, cell = self.lstm(features, state)
        controls = F.softplus(self.linear(self.lll(cell).relu()))
        return controls, (hidden, cell)


class NeuralALISTA(nn.Module):
    def __init__(self, A, W, sparsity, depth):
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("W", W)
        self.register_buffer("mean", torch.zeros(2))
        self.register_buffer("std", torch.ones(2))
        self.regressor = NormController()
        final = 1.2 * sparsity
        self.counts = np.linspace(final // depth, final, depth).astype(int).clip(0, A.shape[1]).tolist()

    def stage_module(self, depth):
        return self.regressor

    def features(self, b, c):
        return torch.stack((b.abs().sum(1), c.abs().sum(1)), dim=1)

    @torch.no_grad()
    def calibrate(self, train_y):
        if len(train_y) < 2:
            raise ValueError("NA-ALISTA calibration needs at least two training observations")
        features = self.features(-train_y, -train_y @ self.W)
        self.mean.copy_(features.mean(0))
        self.std.copy_(features.std(0).clamp_min(torch.finfo(train_y.dtype).eps))

    def forward(self, y, depth, last_step_grad=False, return_path=False):
        x = y.new_zeros((len(y), self.A.shape[1]))
        path = []
        state = (self.regressor.hidden.expand(len(y), -1), self.regressor.cellstate.expand(len(y), -1))
        for i in range(depth):
            with torch.set_grad_enabled(torch.is_grad_enabled() and
                                        (not last_step_grad or i == depth - 1)):
                b = x @ self.A.T - y
                c = b @ self.W
                controls, state = self.regressor((self.features(b, c) - self.mean) / self.std, state)
                gamma, theta = controls.split(1, dim=1)
                z = x - gamma * c
                x = shrink(z, theta)
                count = self.counts[i]
                if count:
                    cutoff = z.detach().abs().topk(count, dim=1).values[:, -1:]
                    x = torch.where(z.abs() > cutoff, z, x)
            if return_path:
                path.append(x)
        return path if return_path else x


def build_model(method, data, depth):
    if method not in METHODS or depth < 1:
        raise ValueError("Unknown method or invalid depth")
    A, W = torch.tensor(data.A), torch.tensor(data.W)
    if method == "NA-ALISTA":
        return NeuralALISTA(A, W, data.config["nonzeros"], depth)
    return UnfoldedISTA(method, A, W, depth)
