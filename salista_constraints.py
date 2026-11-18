"""Inference-time bounds for SALISTA. All budgets have one value per sample."""

import math

import torch
from torch import nn


def gaussian_tail_parameters(spec, snr_db, events):
    if set(spec) != {"mode", "failure_probability"}:
        raise ValueError("Gaussian bounds require failure_probability")
    p = spec["failure_probability"]
    if type(p) not in (float, int) or not 0 < p < 1:
        raise ValueError("failure_probability must be in (0, 1) per sample")
    if snr_db is None or math.isnan(float(snr_db)) or float(snr_db) == -math.inf:
        raise ValueError("Gaussian bounds require the single training SNR")
    return math.log(events) - math.log(p), 10 ** (-float(snr_db) / 20)


def chi_radius(d, t):
    return math.sqrt(d + 2 * math.sqrt(d * t) + 2 * t)


def resolve_bounds(spec, m, s, snr_db):
    """Use known radii or the exact-s Gaussian law of generate.save_split."""
    if not isinstance(spec, dict):
        raise ValueError("Set salista.bounds to an explicit or gaussian specification")
    if spec.get("mode") == "explicit" and set(spec) == {"mode", "initial_l1_bound", "noise_l2_bound"}:
        return spec["initial_l1_bound"], spec["noise_l2_bound"]
    if spec.get("mode") != "gaussian":
        raise ValueError("Use explicit bounds or gaussian bounds with failure_probability")
    t, amplitude = gaussian_tail_parameters(spec, snr_db, 2)
    return math.sqrt(s) * chi_radius(s, t), math.sqrt(s / m) * amplitude * chi_radius(m, t)


def recovery_prior(spec, A, s, snr_db):
    """The online prior covers either author-online or stored observations of this SNR."""
    m, n = A.shape
    if not isinstance(spec, dict) or spec.get("mode") != "online_gaussian":
        return s, *resolve_bounds(spec, m, s, snr_db)
    if m < 2 or not 1 <= s <= n:
        raise ValueError("Online bounds need at least two measurements and 1 <= mean sparsity <= n")
    t, amplitude = gaussian_tail_parameters(spec, snr_db, 3)
    variance = s * (1 - s / n)
    cap = min(n, math.ceil(s + math.sqrt(2 * variance * t) + 2 * t / 3))
    x_radius = chi_radius(cap, t)
    # std(Ax, correction=1) <= ||A||_2 ||x||_2 / sqrt(m-1).
    online_std = torch.linalg.matrix_norm(A.double(), ord=2).item() * x_radius / math.sqrt(m - 1)
    noise_bound = max(math.sqrt(s / m), online_std) * amplitude * chi_radius(m, t)
    return cap, math.sqrt(cap) * x_radius, noise_bound


def project_threshold_budget(proposal, radius):
    """Euclidean projection onto {delta >= 0, sum(delta) <= radius}, by row."""
    sorted_values = proposal.sort(dim=1, descending=True).values
    counts = torch.arange(1, proposal.shape[1] + 1, dtype=proposal.dtype, device=proposal.device)
    # Fixed-order inclusive scan. Floating CUDA cumsum is incompatible with
    # the training entry point's deterministic-algorithms setting in PyTorch 2.8.
    sums = sorted_values
    stride = 1
    while stride < proposal.shape[1]:
        sums = torch.cat((sums[:, :stride], sums[:, stride:] + sums[:, :-stride]), dim=1)
        stride *= 2
    shift = ((sums - radius) / counts).amax(1, keepdim=True).clamp_min(0)
    return (proposal - shift).clamp_min(0)


def rescale_stalled_threshold(z, theta, ratio):
    """Shrink thresholds together only when a nonzero proposal would vanish."""
    amplitude = z.abs()
    stalled = (amplitude <= theta).all(1, keepdim=True) & (amplitude > 0).any(1, keepdim=True)
    denominator = torch.where(theta > 0, theta, torch.ones_like(theta))
    # On stalled rows this is |z|/theta, with 0/0 mapped to 0. Bound the unused branch.
    best_ratio = (torch.minimum(amplitude, theta) / denominator).amax(1, keepdim=True)
    scale = torch.where(stalled, ratio * best_ratio, torch.ones_like(best_ratio))
    return theta * scale, scale, stalled


def weighted_top_sum(matrix, weights, count):
    """For each row, sum its largest count entries after sample-specific weighting."""
    if count == 0:
        return torch.zeros_like(weights)
    # Bound the size of each temporary instead of materializing [batch, n, n].
    sums = []
    for rows in matrix.split(32, dim=0):
        products = rows.unsqueeze(0) * weights.unsqueeze(1)
        sums.append(products.topk(count, dim=2, sorted=False).values.sum(2))
    return torch.cat(sums, dim=1)


class RecoveryCertificate(nn.Module):
    def __init__(self, A, W, sparsity):
        super().__init__()
        n = A.shape[1]
        if type(sparsity) is not int or not 1 <= sparsity <= n:
            raise ValueError("sparsity must be an integer in [1, n]")
        self.sparsity = sparsity
        gram = W.double().T @ A.double()
        diagonal = gram.diagonal().clone()
        if not torch.isfinite(gram).all() or (diagonal - 1).abs().max() > 1e-5:
            raise ValueError("SALISTA requires finite analytic weights with diag(W.T @ A) near one")
        off = gram.abs()
        off.fill_diagonal_(0)
        q = off.topk(sparsity - 1, dim=1, sorted=False).values.sum(1)
        row_sum = off.topk(min(sparsity, n - 1), dim=1, sorted=False).values.sum(1)
        self.register_buffer("off", off.to(A))
        self.register_buffer("diagonal", diagonal.to(A))
        self.register_buffer("omega", torch.linalg.vector_norm(W.double(), dim=0).to(A))
        self.register_buffer("q", q.to(A))
        self.register_buffer("margin", (diagonal.abs() - q).min().to(A))
        self.register_buffer("mu", off.max().to(A))
        self.register_buffer("fallback_factor", torch.minimum(row_sum, off.max()).to(A))

    def coordinate_bounds(self, c, budget, noise_radius, gram_available):
        if not gram_available:
            # When E_j = B for every j, the row top-s sum factors exactly.
            return budget.expand_as(c), self.fallback_factor * budget
        observed = c.abs() + noise_radius * self.omega
        uniform = (c.abs().amax(1, keepdim=True) + noise_radius * self.omega.max()) / self.margin
        error = torch.minimum(budget, (observed + self.q * uniform) / self.diagonal.abs())
        row_sum = weighted_top_sum(self.off, error, self.sparsity)
        return error, torch.minimum(row_sum, self.mu * budget)

    def next_budget(self, gamma, budget, previous_budget, noise_radius, eta, beta_max):
        # Columns enter the induced l1 norm. The threshold certificate above uses rows.
        col_sum = weighted_top_sum(self.off.T, gamma, self.sparsity - 1)
        a0 = ((1 - gamma * self.diagonal).abs() + col_sum).amax(1, keepdim=True)
        max_step = gamma.amax(1, keepdim=True)
        a = a0 + self.sparsity * max_step * self.mu + eta
        noise = 2 * self.sparsity * max_step * self.omega.max() * noise_radius
        return (a + beta_max) * budget + beta_max * previous_budget + noise
