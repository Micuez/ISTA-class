"""Use stored (y, x) pairs with the author's real-valued NA-ALISTA solver."""

import math
from pathlib import Path

from experiment_inputs import torch_batches


def epoch_metrics(model, data, split, batch_size, snr_db=None, epoch=0, optimizer=None):
    import torch
    from utils.conf import device

    training = optimizer is not None
    model.train(training)
    error_energy, target_energy, elements = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for y, x in torch_batches(data, split, batch_size, snr_db, epoch):
            y, x = y.to(device), x.to(device)
            if training:
                optimizer.zero_grad()
            prediction, _, _ = model(y, None)
            squared_error = (prediction - x) ** 2
            if training:
                squared_error.mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            error_energy += squared_error.detach().double().sum().item()
            target_energy += (x.double() ** 2).sum().item()
            elements += x.numel()
    nmse_db = 10 * math.log10(error_energy / target_energy) if error_energy else float("-inf")
    return error_energy / elements, nmse_db


def save_model(model, filename):
    import torch

    # These train-batch statistics are plain Python attributes in the author model.
    reg = model.regressor
    stats = {key: getattr(reg, key) for key in ("cl1_mean", "cl1_std", "bl1_mean", "bl1_std")}
    target = Path(filename)
    staging = target.with_suffix(".tmp")
    torch.save({"weights": model.state_dict(), "normalizers": stats}, str(staging))
    staging.replace(target)


def load_model(model, filename):
    import torch
    from utils.conf import device

    checkpoint = torch.load(str(filename), map_location=device)
    model.load_state_dict(checkpoint["weights"])
    for key, value in checkpoint["normalizers"].items():
        setattr(model.regressor, key, value)
    model.regressor.initialized_normalizers = True
