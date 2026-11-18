"""Generate the fixed-dictionary sparse recovery experiment without training any model."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dataset import FORMAT_VERSION, SPLITS, level_key


def validate(config):
    for key in ("measurements", "signal_dimension", "nonzeros", "seed"):
        if type(config[key]) is not int:
            raise ValueError(key + " must be an integer")
    m, n, k = (config[key] for key in ("measurements", "signal_dimension", "nonzeros"))
    if not (0 < k <= m < n and config["seed"] >= 0):
        raise ValueError("Require 0 < nonzeros <= measurements < signal_dimension and seed >= 0")
    if set(config["clean_signals"]) != set(SPLITS) or any(
        type(size) is not int or size < 1 for size in config["clean_signals"].values()
    ):
        raise ValueError("train, val and test must have positive integer sizes")
    levels = config["snr_db"]
    if not levels or any(type(v) not in (float, int) or not math.isfinite(v) for v in levels):
        raise ValueError("snr_db must be a nonempty list of finite numbers")
    if len({level_key(v) for v in levels}) != len(levels):
        raise ValueError("SNR levels must be distinct")


def make_dictionary(m, n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    A /= np.linalg.norm(A, axis=0)
    A = A.astype(np.float32)
    # Solve the constrained Frobenius problem for the actual saved dictionary.
    a64 = A.astype(np.float64)
    v = np.linalg.solve(a64 @ a64.T, a64)
    W = v / np.sum(a64 * v, axis=0)
    return A, W.astype(np.float32)


def save_split(root, size, A, k, levels, seed):
    support_rng, value_rng, noise_rng = map(np.random.default_rng, seed.spawn(3))
    m, n = A.shape
    x = np.zeros((size, n), dtype=np.float32)
    values = value_rng.standard_normal((size, k)).astype(np.float32)
    zeros = values == 0
    while zeros.any():
        values[zeros] = value_rng.standard_normal(int(zeros.sum())).astype(np.float32)
        zeros = values == 0
    for row in range(size):
        x[row, support_rng.choice(n, size=k, replace=False)] = values[row]
    clean = x @ A.T
    g = noise_rng.standard_normal((size, m)).astype(np.float32)
    root.mkdir()
    np.save(root / "x.npy", x)
    signal_energy = np.sum(np.square(clean, dtype=np.float64))
    actual = {}
    for db in levels:
        sigma = math.sqrt(k / m) * 10 ** (-db / 20)
        if not (0 < np.float32(sigma) < np.inf):
            raise ValueError("Noise scale is outside the float32 range")
        y = clean + np.float32(sigma) * g
        np.save(root / ("y_" + level_key(db) + "dB.npy"), y)
        error = np.subtract(y, clean, dtype=np.float64)
        actual[level_key(db)] = float(10 * np.log10(signal_energy / np.sum(error**2)))
    return actual


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def generate(config, dest):
    validate(config)
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError("Dataset already exists. Choose a new --out for a new version: " + str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as staging:
        root = Path(staging) / "dataset"
        root.mkdir()
        seeds = np.random.SeedSequence(config["seed"]).spawn(1 + len(SPLITS))
        A, W = make_dictionary(config["measurements"], config["signal_dimension"], seeds[0])
        np.save(root / "A.npy", A)
        np.save(root / "W.npy", W)
        stats = {}
        for split, seed in zip(SPLITS, seeds[1:]):
            stats[split] = save_split(root / split, config["clean_signals"][split],
                                      A, config["nonzeros"], config["snr_db"], seed)
        manifest = {
            "format_version": FORMAT_VERSION, "numpy_version": np.__version__,
            "config": config, "measured_snr_db": stats,
            "sha256": {p.relative_to(root).as_posix(): sha256(p)
                       for p in sorted(root.rglob("*.npy"))},
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        root.rename(dest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment.json"))
    parser.add_argument("--out", type=Path, default=Path("data/main"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        manifest = generate(json.load(stream), args.out)
    print(args.out.resolve())
    print(json.dumps(manifest["measured_snr_db"], indent=2))
