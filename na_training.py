"""Synthetic real-valued training batches from the NA-ALISTA author's protocol."""

import torch
from torch.utils.data import DataLoader


def sparse_signals(size, n, k):
    mask = torch.bernoulli(torch.full((size, n), k / n))
    return mask * torch.normal(torch.zeros_like(mask), torch.ones_like(mask))


def author_batches(x, matrix, snr, batch, shuffle):
    """Use complete batches and the sample standard deviation of each clean observation."""
    for target in DataLoader(x, batch_size=batch, shuffle=shuffle, drop_last=True):
        target = target.to(matrix.device)
        clean = target @ matrix.T
        if float(snr) != float("inf"):
            std = clean.std(dim=1, keepdim=True) * 10 ** (-float(snr) / 20)
            clean = clean + torch.normal(torch.zeros_like(clean), std.expand_as(clean))
        yield clean, target


@torch.no_grad()
def author_holdout_score(model, holdout, snr, batch, depth):
    model.eval()
    error, energy = 0.0, 0.0
    for y, x in author_batches(holdout, model.A, snr, batch, shuffle=False):
        prediction = model(y, depth)
        error += (prediction - x).square().mean().item()
        energy += x.square().mean().item()
    return error / energy
