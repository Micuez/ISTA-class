"""Elementwise shrinkage used by the unfolded sparse-recovery models."""

import torch
import torch.nn.functional as F


def shrink_free(data, theta):
    """Soft-threshold. Matches ``tf.sign(data) * relu(|data| - theta)``."""
    return torch.sign(data) * F.relu(data.abs() - theta)


def shrink_lamp(r_, rvar_, lam_):
    """Thresholding neuron from Learned AMP."""
    theta_ = torch.maximum(torch.sqrt(rvar_) * lam_, torch.zeros_like(rvar_))
    return torch.sign(r_) * torch.maximum(r_.abs() - theta_, torch.zeros_like(r_))


def shrink_ss(inputs_, theta_, q, return_index=False):
    """Soft-threshold everywhere except the largest-magnitude support.

    ``q`` is a percentage. Entries that are both above ``theta`` and inside the
    top ``q`` percent are copied through. The support mask is treated as a
    constant, matching ``tf.stop_gradient`` in the TensorFlow code.
    """
    abs_ = inputs_.abs()
    quantile = min(max((100.0 - float(q)) / 100.0, 0.0), 1.0)
    thres_ = torch.quantile(abs_.detach(), quantile, dim=1, keepdim=True, interpolation="linear")
    index_ = ((abs_ > theta_) & (abs_ > thres_)).to(dtype=inputs_.dtype).detach()
    cindex_ = 1.0 - index_
    output = index_ * inputs_ + shrink_free(cindex_ * inputs_, theta_)
    if return_index:
        return output, cindex_
    return output


def sensing_init(A, lam):
    """LISTA closed-form initialization.

    Returns ``A`` as float32, the Lipschitz scale, the scalar threshold, and
    ``B = A.T / scale``. Division is done in float64 and then cast, matching
    the original NumPy initialization.
    """
    import numpy as np

    matrix = np.asarray(A, dtype=np.float32)
    scale = 1.001 * np.linalg.norm(matrix, ord=2) ** 2
    theta = np.asarray(lam / scale, dtype=np.float32)
    basis = (matrix.T / scale).astype(np.float32)
    tied = np.eye(matrix.shape[1], dtype=np.float32) - np.matmul(basis, matrix)
    return matrix, scale, theta, basis, tied


def numpy_to_tensor(array):
    import numpy as np

    return torch.tensor(np.asarray(array, dtype=np.float32), dtype=torch.float32)
