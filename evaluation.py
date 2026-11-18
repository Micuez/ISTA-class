"""Shared observation batches and sparse recovery evaluation."""

import math

import torch

from dataset import level_key


def tensors(data, split, batch_size, device, level=None, epoch=0):
    for y, x in data.batches(split, batch_size, level, epoch):
        yield torch.from_numpy(y).to(device), torch.from_numpy(x).to(device)


@torch.no_grad()
def evaluate(model, data, split, depth, batch_size, device):
    model.eval()
    metrics = {}
    ratios = []
    k = data.config["nonzeros"]
    for level in data.levels:
        err, energy, hits, n_signals = 0.0, 0.0, 0, 0
        for y, x in tensors(data, split, batch_size, device, level):
            pred = model(y, depth)
            err += (pred.double() - x.double()).square().sum().item()
            energy += x.double().square().sum().item()
            # At equal magnitudes, give the lower coordinate index precedence.
            indices = pred.abs().argsort(dim=1, descending=True, stable=True)[:, :k]
            hits += x.gather(1, indices).count_nonzero().item()
            n_signals += len(x)
        ratio = err / energy
        if not math.isfinite(ratio):
            raise FloatingPointError("Non-finite {} NMSE at {} dB".format(split, level))
        ratios.append(ratio)
        # The string represents an exact zero error without writing nonstandard JSON floats.
        metrics[level_key(level)] = {
            "nmse": ratio, "nmse_db": 10 * math.log10(ratio) if ratio else "-inf",
            "top_k_support_f1": hits / (k * n_signals), "signals": n_signals,
        }
    return metrics, sum(ratios) / len(ratios)
