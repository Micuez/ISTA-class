"""Check the published arrays and recompute observation SNR from y - Ax."""

import argparse
import json
from pathlib import Path

import numpy as np

from dataset import RecoveryData, SPLITS, level_key
from generate import sha256


def check_paired_signals(finite, noiseless):
    files = ["A.npy", "W.npy", *(f"{split}/x.npy" for split in SPLITS)]
    for name in files:
        if finite.manifest["sha256"][name] != noiseless.manifest["sha256"][name]:
            raise ValueError(f"The finite-SNR and noiseless datasets differ in {name}")


def verify(root):
    data = RecoveryData(root)
    for name, expected in data.manifest["sha256"].items():
        if sha256(data.root / name) != expected:
            raise ValueError("File checksum differs: " + name)
    a, w, cfg = data.A, data.W, data.config
    if a.shape != (cfg["measurements"], cfg["signal_dimension"]) or w.shape != a.shape:
        raise ValueError("Invalid dictionary or analytic weight shape")
    np.testing.assert_allclose(np.linalg.norm(a.astype("float64"), axis=0), 1, atol=1e-6)
    np.testing.assert_allclose(np.einsum("ij,ij->j", a, w, dtype="float64"), 1, atol=1e-6)
    actual = {}
    # Recompute the clean signal in double precision independently of the generator.
    a64 = a.astype(np.float64)
    for split in SPLITS:
        actual[split] = {}
        noise_reference = None
        _, x = data.arrays(split, data.levels[0])
        if x.shape != (cfg["clean_signals"][split], a.shape[1]):
            raise ValueError("Invalid label shape in " + split)
        if not np.all(np.count_nonzero(x, axis=1) == cfg["nonzeros"]):
            raise ValueError("Unexpected support size in " + split)
        clean = x @ a64.T
        for db in data.levels:
            y, _ = data.arrays(split, db)
            if y.shape != (len(x), a.shape[0]):
                raise ValueError("Invalid observation shape in " + split)
            if float(db) == float("inf"):
                # float32 BLAS may differ by rounding across the server and CPU.
                np.testing.assert_allclose(y, clean, rtol=3e-6, atol=3e-6)
                actual[split]["inf"] = "inf"
                continue
            error = y - clean
            db_measured = 10 * np.log10(np.sum(clean**2) / np.sum(error**2))
            np.testing.assert_allclose(db_measured, data.manifest["measured_snr_db"][split][level_key(db)],
                                       atol=1e-3, rtol=0)
            # Shared noise vectors across SNR; allowance covers the float32 forward product.
            reference_scale = 10 ** ((db - data.levels[0]) / 20)
            if noise_reference is None:
                noise_reference = error
            else:
                np.testing.assert_allclose(error * reference_scale, noise_reference, rtol=2e-4, atol=4e-6)
            actual[split][level_key(db)] = float(db_measured)
    return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=Path("data/main"))
    print(json.dumps(verify(parser.parse_args().data), indent=2))
