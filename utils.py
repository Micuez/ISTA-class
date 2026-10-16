"""Losses, metrics, checkpoints, and the layer-wise Adam schedule."""

import os

import numpy as np
import torch


def mse_loss(y_true, y_pred, signal_dim, extra_dim=None):
    """Match ``tf.nn.l2_loss``, which is ``sum(diff**2) / 2`` over the batch.

    Keras applies ``SUM_OVER_BATCH_SIZE`` to this already-scalar value, and the
    element count of a scalar is 1, so the optimizer sees the sum itself.
    """
    diff = y_pred[:, -signal_dim:] - y_true[:, -signal_dim:]
    loss = 0.5 * torch.sum(diff * diff)
    if extra_dim is not None:
        coefficients = y_pred[:, -extra_dim - signal_dim : -signal_dim]
        loss = loss + 0.001 * torch.sum(coefficients.abs())
    return loss


def lasso_loss(y_true, y_pred, sensing, lam, signal_dim):
    """Training objective used by the original ``LassoLoss``.

    ``0.5 * tf.nn.l2_loss(Ax - y) + lam * ||x||_1``. This is not the same
    reduction as the reported Lasso metric below.
    """
    sparse = y_pred[:, -signal_dim:]
    residual = sparse @ sensing.T - y_true[:, :-signal_dim]
    l2 = 0.5 * torch.sum(residual * residual)
    l1 = torch.sum(sparse.abs())
    return 0.5 * l2 + lam * l1


def nmse_db(y_true, y_pred, signal_dim, layer_id=-1, measurement_dim=None, interval=None):
    """Per-sample NMSE in dB. The caller averages over the evaluation set."""
    if layer_id < 0:
        prediction = y_pred[:, -signal_dim:]
    else:
        end = measurement_dim + (layer_id + 1) * interval
        prediction = y_pred[:, end - signal_dim : end]
    target = y_true[:, -signal_dim:]
    mse = torch.mean((target - prediction) ** 2, dim=-1) + 1e-10
    power = torch.mean(target ** 2, dim=-1) + 1e-10
    return 10.0 * torch.log10(mse / power)


def lasso_objective(y_true, y_pred, sensing, lam, measurement_dim, signal_dim, layer_id=-1):
    """Per-sample ``0.5 ||Ax - y||^2 + lam ||x||_1``."""
    if layer_id < 0:
        sparse = y_pred[:, -signal_dim:]
    else:
        start = measurement_dim + layer_id * signal_dim
        sparse = y_pred[:, start : start + signal_dim]
    residual = sparse @ sensing.T - y_true[:, :measurement_dim]
    l2 = torch.sum(residual * residual, dim=-1)
    l1 = torch.sum(sparse.abs(), dim=-1)
    return 0.5 * l2 + lam * l1


def psnr(y_true, y_pred):
    mse = torch.mean((y_true - y_pred.clamp(0, 255)) ** 2)
    return 20.0 * torch.log10(torch.tensor(255.0, device=y_true.device) / torch.sqrt(mse))


def im2cols(image):
    height, width = image.shape
    columns = []
    for i in range(0, height, 16):
        for j in range(0, width, 16):
            columns.append(image[i : i + 16, j : j + 16])
    return np.asarray(columns).reshape(-1, 256)


def col2im(columns):
    patches = columns.shape[0]
    height = width = int(np.sqrt(patches)) * 16
    image = np.zeros((height, width))
    index = 0
    for i in range(0, height, 16):
        for j in range(0, width, 16):
            image[i : i + 16, j : j + 16] += columns[index].reshape(16, 16)
            index += 1
    return image


def save_partial(checkpoint_dir, layer):
    model_dir = os.path.join(checkpoint_dir, f"layer_{layer + 1}")
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, "model.pt")


def check_and_load_partial(checkpoint_dir, num_layers):
    """Return ``(checkpoint_path, num_completed_layers)``."""
    previous = None
    for layer_id in range(num_layers):
        path = os.path.join(checkpoint_dir, f"layer_{layer_id + 1}", "model.pt")
        if not os.path.isfile(path):
            return previous, layer_id
        previous = path
    return previous, num_layers


def update_var_ages(ages, names):
    """Age every parameter that already exists, then introduce new ones at age 0."""
    for name in list(ages):
        ages[name] += 1
    for name in names:
        if name not in ages:
            ages[name] = 0
    return ages


class AgeAdam:
    """Adam with the original per-layer gradient schedule.

    Keras Adam here uses ``beta=(0.9, 0.999)`` and ``epsilon=1e-7``.
    During the frozen phase only parameters introduced by the current layer
    (age 0) are updated. Later phases multiply gradients by ``0.3 ** age``
    before Adam sees them, and each phase builds a fresh optimizer.
    """

    def __init__(self, model, ages, learning_rate, freeze_current_only):
        named = dict(model.named_parameters())
        missing = [name for name in ages if name not in named]
        if missing:
            raise KeyError(f"Parameters missing from the model: {missing}")
        params = []
        self.age_of = {}
        for name, age in ages.items():
            param = named[name]
            params.append(param)
            self.age_of[id(param)] = age
        self.freeze_current_only = freeze_current_only
        self.opt = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-7)

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)

    def step(self):
        for group in self.opt.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                age = self.age_of[id(param)]
                if self.freeze_current_only:
                    if age != 0:
                        param.grad = None
                else:
                    param.grad.mul_(0.3 ** age)
        self.opt.step()
