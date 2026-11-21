"""Unrolled sparse recovery models. Observation batches are row-major."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from generate import analytic_weights
from salista_constraints import RecoveryCertificate, project_threshold_budget, rescale_stalled_threshold, recovery_prior


BASELINE_METHODS = ("EBT-LISTA", "HLISTA", "ALISTA", "ELISTA", "NA-ALISTA")
METHODS = BASELINE_METHODS + ("SALISTA",)


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


class CoordinateController(nn.Module):
    """One recurrent state per sample, one set of head weights across coordinates."""

    def __init__(self, cell, width, with_momentum):
        super().__init__()
        self.cell = (nn.GRUCell if cell == "gru" else nn.LSTMCell)(5, width)
        self.context = nn.Linear(width, width)
        self.local = nn.Linear(6, width, bias=False)
        self.step = nn.Linear(width, 1)
        self.momentum = nn.Linear(width, 1) if with_momentum else None
        self.threshold = nn.Linear(width + 1, 1)

    def forward(self, global_features, local_features, state):
        state = self.cell(global_features.log1p(), state)
        hidden = state[0] if isinstance(state, tuple) else state
        features = (self.context(hidden).unsqueeze(1) + self.local(local_features)).tanh()
        return features, state


class SALISTA(nn.Module):
    """V3 unroll with fixed [m,n] A/W and row-major [batch,m] observations."""

    def __init__(self, A, W, sparsity, depth, *, initial_l1_bound, noise_l2_bound,
                 cell="gru", sharing="shared", hidden_size=32, gamma_min=0.5,
                 gamma_max=1.1, beta_max=0.2, eta=0.05, stall_threshold_ratio=None,
                 free_threshold=False, free_step=False):
        super().__init__()
        if A.ndim != 2 or min(A.shape) < 1 or A.dtype not in (torch.float32, torch.float64):
            raise ValueError("A must be a real float32/float64 [m,n] tensor")
        if W is None:
            W = torch.from_numpy(analytic_weights(A.detach().cpu().numpy())).to(A)
        if A.shape != W.shape or A.device != W.device or A.dtype != W.dtype:
            raise ValueError("Supplied W must match A in shape, dtype and device")
        if type(depth) is not int or depth < 1 or type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("depth and hidden_size must be positive integers")
        if cell not in ("gru", "lstm") or sharing not in ("shared", "untied"):
            raise ValueError("Choose cell gru/lstm and sharing shared/untied")
        if stall_threshold_ratio is not None and (type(stall_threshold_ratio) not in (int, float)
                                                 or not 0 < stall_threshold_ratio < 1):
            raise ValueError("stall_threshold_ratio must be null or in (0, 1)")
        if type(free_threshold) is not bool:
            raise ValueError("free_threshold must be a bool")
        if type(free_step) is not bool:
            raise ValueError("free_step must be a bool")
        controls = (initial_l1_bound, noise_l2_bound, gamma_min, gamma_max, beta_max, eta)
        if (not all(type(v) in (float, int) and math.isfinite(v) for v in controls)
                or min(initial_l1_bound, noise_l2_bound, beta_max, eta) < 0
                or not 0 < gamma_min <= gamma_max):
            raise ValueError("Require finite nonnegative radii/caps and 0 < gamma_min <= gamma_max")
        self.register_buffer("A", A.detach().clone())
        self.register_buffer("W", W.detach().clone())
        self.certificate = RecoveryCertificate(self.A, self.W, sparsity)
        self.initial_l1_bound, self.noise_l2_bound = initial_l1_bound, noise_l2_bound
        self.gamma_min, self.gamma_max = gamma_min, gamma_max
        self.beta_max, self.eta = beta_max, eta
        self.stall_threshold_ratio = stall_threshold_ratio
        self.free_threshold = free_threshold
        self.free_step = free_step
        self.depth, self.sharing = depth, sharing
        blocks = 1 if sharing == "shared" else depth
        self.layers = nn.ModuleList(CoordinateController(cell, hidden_size, sharing == "shared" or i > 0)
                                    for i in range(blocks)).to(self.A)

    def stage_module(self, depth):
        return self.layers[0 if self.sharing == "shared" else depth - 1]

    def forward(self, y, depth, last_step_grad=False, return_path=False, *, return_diagnostics=False):
        if type(depth) is not int or not 1 <= depth <= self.depth:
            raise ValueError("Requested depth is outside the configured SALISTA unroll")
        if y.ndim != 2 or y.shape[1] != self.A.shape[0]:
            raise ValueError("Expected observations shaped [batch,m]")
        cert = self.certificate
        gram_available = bool(cert.margin > 0)
        x = y.new_zeros((len(y), self.A.shape[1]))
        previous_x = x
        budget = y.new_full((len(y), 1), self.initial_l1_bound)
        previous_budget = budget
        support_certified = torch.ones_like(budget, dtype=torch.bool)
        state = None  # The cell initializes zeros for each new observation batch.
        path, diagnostics = [], []
        for i in range(depth):
            with torch.set_grad_enabled(torch.is_grad_enabled() and (not last_step_grad or i == depth - 1)):
                r = y - x @ self.A.T
                c = r @ self.W
                change = x - previous_x
                error, interference = cert.coordinate_bounds(c, budget, self.noise_l2_bound, gram_available)
                if self.stall_threshold_ratio is not None and gram_available:
                    # After a relaxation the error support may exceed s. Use only l1-based bounds.
                    error = torch.where(support_certified, error, budget)
                    interference = torch.where(support_certified, interference, cert.fallback_factor * budget)
                global_features = torch.stack((r.abs().sum(1), c.abs().sum(1), c.abs().amax(1),
                                               change.abs().sum(1), budget[:, 0]), dim=1)
                local_features = torch.stack((x, c, change, error.log1p(), cert.q.expand_as(x),
                                              cert.omega.expand_as(x)), dim=2)
                block = self.stage_module(i + 1)
                features, state = block(global_features, local_features, state)
                step_raw = block.step(features).squeeze(-1)
                if self.free_step:
                    gamma = F.softplus(step_raw)
                else:
                    gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * step_raw.sigmoid()
                beta = self.beta_max * block.momentum(features).squeeze(-1).sigmoid() if i else torch.zeros_like(x)
                lower = gamma * (interference + self.noise_l2_bound * cert.omega)
                threshold_features = torch.cat((features, lower.log1p().unsqueeze(-1)), dim=2)
                proposal = F.softplus(block.threshold(threshold_features).squeeze(-1))
                z = x + gamma * c + beta * change
                if self.free_threshold:
                    # Learned per-coordinate threshold, without the certificate floor,
                    # the l1 budget projection, or the stall rescaling.
                    delta = proposal
                    reference_theta = proposal
                    theta = proposal
                    scale = torch.ones_like(budget)
                else:
                    delta = project_threshold_budget(proposal, self.eta * budget)
                    reference_theta = lower + delta
                    if self.stall_threshold_ratio is None:
                        theta = reference_theta
                        scale = torch.ones_like(budget)
                    else:
                        theta, scale, stalled = rescale_stalled_threshold(
                            z, reference_theta, self.stall_threshold_ratio)
                        support_certified = support_certified & ~stalled
                next_x = shrink(z, theta)
                if i + 1 < depth or return_diagnostics:
                    triangle = self.initial_l1_bound + next_x.abs().sum(1, keepdim=True)
                    next_budget = triangle
                    if self.stall_threshold_ratio is None or support_certified.any():
                        recurrence = cert.next_budget(gamma, budget, previous_budget, self.noise_l2_bound,
                                                      self.eta, self.beta_max)
                        next_budget = torch.where(support_certified, torch.minimum(recurrence, triangle), triangle)
                    if return_diagnostics:
                        diagnostics.append(dict(gamma=gamma, beta=beta, theta=theta, lower=lower, delta=delta,
                                                error_bound=error, budget=budget, next_budget=next_budget,
                                                reference_theta=reference_theta, threshold_scale=scale,
                                                support_certified=support_certified))
                    previous_budget, budget = budget, next_budget
                previous_x, x = x, next_x
            if return_path:
                path.append(x)
        result = path if return_path else x
        return (result, diagnostics) if return_diagnostics else result


def build_model(method, data, depth, config=None):
    if method not in METHODS or depth < 1:
        raise ValueError("Unknown method or invalid depth")
    A, W = torch.tensor(data.A), torch.tensor(data.W)
    if method == "SALISTA":
        config = config or {}
        options = dict(config.get("salista", {}))
        s = data.config["nonzeros"]
        s, b0, epsilon = recovery_prior(options.pop("bounds", None), A, s, config.get("snr_db"))
        return SALISTA(A, W, s, depth, initial_l1_bound=b0, noise_l2_bound=epsilon, **options)
    if method == "NA-ALISTA":
        return NeuralALISTA(A, W, data.config["nonzeros"], depth)
    return UnfoldedISTA(method, A, W, depth)
