"""NumPy ``.npy`` datasets used by the model-based L2O experiments."""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class RecoveryDataset(Dataset):
    """Each row is ``[measurement (M), ... target ...]``.

    The model receives the first ``M`` entries. The loss receives the full row,
    matching ``dataset_parser`` in the TensorFlow loader.
    """

    def __init__(self, array, measurement_dim):
        data = np.asarray(array, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2-D array, got shape {data.shape}.")
        if data.shape[1] < measurement_dim:
            raise ValueError(
                f"Rows have length {data.shape[1]}, shorter than M={measurement_dim}."
            )
        self.data = torch.from_numpy(data)
        self.measurement_dim = int(measurement_dim)

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, index):
        row = self.data[index]
        return row[: self.measurement_dim], row


def input_fn(
    is_training,
    data_dir,
    batch_size,
    task="sc",
    drop_remainder=False,
    A=None,
    filename=None,
    seed=0,
):
    """Build a ``DataLoader``.

    Training shuffles and the training loop cycles batches for
    ``steps_per_epoch``. Evaluation keeps file order.
    """
    del task  # Retained so the call site matches the original arguments.
    if A is None:
        raise ValueError("A is required to determine the measurement dimension.")
    if filename is None:
        filename = "train_data.npy" if is_training else "val_data.npy"
    path = filename if os.path.isabs(filename) else os.path.join(data_dir, filename)
    array = np.load(path, allow_pickle=True)
    dataset = RecoveryDataset(array, A.shape[0])
    generator = None
    if is_training:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        drop_last=drop_remainder,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
