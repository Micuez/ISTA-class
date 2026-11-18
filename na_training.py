"""Synthetic real-valued training batches from the NA-ALISTA author's protocol."""

import torch
from torch.utils.data import DataLoader


def sparse_signals(size, n, k, generator=None):
    mask = torch.bernoulli(torch.full((size, n), k / n), generator=generator)
    return mask * torch.normal(torch.zeros_like(mask), torch.ones_like(mask), generator=generator)


def author_batches(x, matrix, snr, batch, shuffle, generator=None, noise_generator=None):
    """Use complete batches and the sample standard deviation of each clean observation."""
    for target in DataLoader(x, batch_size=batch, shuffle=shuffle, drop_last=True, generator=generator):
        target = target.to(matrix.device)
        clean = target @ matrix.T
        if float(snr) != float("inf"):
            std = clean.std(dim=1, keepdim=True) * 10 ** (-float(snr) / 20)
            clean = clean + torch.normal(torch.zeros_like(clean), std.expand_as(clean), generator=noise_generator)
        yield clean, target


def paired_author_batches(size, k, matrix, snr, batch, seed):
    """Separate epoch RNGs from model initialization so both recurrent models see the same draws."""
    cpu_rng = torch.Generator().manual_seed(seed)
    noise_rng = torch.Generator(device=matrix.device).manual_seed(seed + 2**32)
    signals = sparse_signals(size, matrix.shape[1], k, generator=cpu_rng)
    return author_batches(signals, matrix, snr, batch, True, cpu_rng, noise_rng)


@torch.no_grad()
def author_holdout_score(model, holdout, snr, batch, depth):
    model.eval()
    error, energy = 0.0, 0.0
    for y, x in author_batches(holdout, model.A, snr, batch, shuffle=False):
        prediction = model(y, depth)
        error += (prediction - x).square().mean().item()
        energy += x.square().mean().item()
    return error / energy
