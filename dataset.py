"""Read the same float32 observations in modern NumPy and the older model environments."""

import json
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")
FORMAT_VERSION = 1


def level_key(db):
    return format(float(db), ".15g")


class RecoveryData:
    """Each public batch is a pair of row-major arrays (y, x)."""

    def __init__(self, root):
        self.root = Path(root)
        with (self.root / "manifest.json").open(encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        if self.manifest["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported dataset format")
        self.config = self.manifest["config"]
        self.levels = tuple(self.config["snr_db"])
        self.A = self._load("A.npy")
        self.W = self._load("W.npy")

    def _load(self, relative_path):
        return np.load(str(self.root / relative_path), mmap_mode="r", allow_pickle=False)

    def arrays(self, split, snr_db):
        if split not in SPLITS or snr_db not in self.levels:
            raise ValueError("Unknown split or noise level: {}, {}".format(split, snr_db))
        y = self._load(Path(split) / ("y_" + level_key(snr_db) + "dB.npy"))
        x = self._load(Path(split) / "x.npy")
        return y, x

    def batches(self, split, batch_size, snr_db=None, epoch=0):
        """Exactly one visit per (clean signal, selected level) in each epoch."""
        if batch_size < 1 or epoch < 0:
            raise ValueError("batch_size must be positive and epoch nonnegative")
        levels = self.levels if snr_db is None else (snr_db,)
        measurements = []
        for level in levels:
            y, x = self.arrays(split, level)
            measurements.append(y)
        count = len(x)
        # Adjacent indices cycle over levels, so ordered validation is balanced.
        order = np.arange(count * len(levels))
        if split == "train":
            seed = (self.config["seed"] + epoch) % (2**32)
            np.random.RandomState(seed).shuffle(order)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            rows, groups = np.divmod(indices, len(levels))
            y_batch = np.empty((len(rows), self.A.shape[0]), dtype=np.float32)
            for group, matrix in enumerate(measurements):
                mask = groups == group
                y_batch[mask] = matrix[rows[mask]]
            yield y_batch, x[rows]
