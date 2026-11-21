"""Ada-LISTA, following Adaptive_ISTA in aaberdam/AdaLISTA (984411e4)."""

import math

import torch
from torch import nn


class AdaLayer(nn.Module):
    def __init__(self, reference, initial_step):
        super().__init__()
        self.gamma = nn.Parameter(reference.new_tensor(initial_step))
        self.eta = nn.Parameter(reference.new_tensor(initial_step))


class AdaLISTA(nn.Module):
    """Encode a batch [B,m] using one [m,n] or B separate [m,n] dictionaries."""

    supports_variable_dictionary = True

    def __init__(self, A, depth, *, lambd=1.0, initial_step=None):
        super().__init__()
        if A.ndim != 2 or min(A.shape) < 1 or A.dtype not in (torch.float32, torch.float64):
            raise ValueError("Expected a float32/float64 reference dictionary [m,n]")
        if not torch.isfinite(A).all() or type(depth) is not int or depth < 1:
            raise ValueError("Use a finite reference dictionary and a positive integer depth")
        if not math.isfinite(lambd) or lambd < 0:
            raise ValueError("lambd must be finite and nonnegative")
        self.register_buffer("A", A.detach().clone())
        if initial_step is None:
            norm = torch.linalg.matrix_norm(self.A.double(), ord=2).item()
            initial_step = 1 / norm**2 if norm else 1.0
        if not math.isfinite(initial_step) or initial_step <= 0:
            raise ValueError("initial_step must be finite and positive")
        self.lambd = lambd
        eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        self.shared_matrices = nn.ParameterList([nn.Parameter(eye.clone()), nn.Parameter(eye.clone())])
        self.layers = nn.ModuleList(AdaLayer(A, initial_step) for _ in range(depth))

    def stage_module(self, depth):
        return self.layers[depth - 1]

    def forward(self, y, depth, last_step_grad=False, return_path=False, *, dictionary=None):
        if type(depth) is not int or not 1 <= depth <= len(self.layers):
            raise ValueError("Requested depth is outside the configured AdaLISTA unroll")
        D = self.A if dictionary is None else dictionary
        if y.ndim != 2 or y.shape[1] != self.A.shape[0]:
            raise ValueError("Expected observations [batch,m]")
        if D.ndim not in (2, 3) or D.shape[-2:] != self.A.shape or (D.ndim == 3 and D.shape[0] != len(y)):
            raise ValueError("Expected a dictionary [m,n] or [batch,m,n] matched to y")
        if D.device != y.device or D.dtype != y.dtype or y.device != self.A.device or y.dtype != self.A.dtype:
            raise ValueError("Observations, dictionary, and model must have matching dtype and device")

        analysis, feedback = self.shared_matrices
        transpose = D.transpose(-1, -2)
        drive = transpose @ (analysis.T @ y.unsqueeze(-1))
        x = torch.zeros_like(drive)
        path = []
        for i, layer in enumerate(self.layers[:depth]):
            with torch.set_grad_enabled(torch.is_grad_enabled() and (not last_step_grad or i + 1 == depth)):
                # Apply (W2 D)^T (W2 D) to x without building Gram or transformed dictionaries.
                weighted_signal = feedback @ (D @ x)
                correction = transpose @ (feedback.T @ weighted_signal)
                z = x + layer.gamma * (drive - correction)
                x = z.sign() * torch.relu(z.abs() - self.lambd * layer.eta.abs())
            if return_path:
                path.append(x.squeeze(-1))
        return path if return_path else x.squeeze(-1)
