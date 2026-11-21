"""Sparse coding of observed signals by minimizing 0.5 ||Dz-y||² + l1 ||z||₁.

Each row of y is a signal. Dictionary atoms are columns of a shared [m,n]
or per-signal [N,m,n] dictionary. The task requires no coefficient labels.
The CLI trains end to end and selects by mean validation objective.
NPZ splits are train_y/val_y/test_y, with shared D or corresponding split_D.
"""

import argparse
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch


def _check_inputs(signals, dictionary):
    if signals.ndim != 2 or min(signals.shape) < 1:
        raise ValueError("signals must be a nonempty [N,m] tensor")
    if dictionary.ndim not in (2, 3) or dictionary.shape[-2] != signals.shape[1] or dictionary.shape[-1] < 1:
        raise ValueError("dictionary must have shape [m,n] or [N,m,n] with n >= 1")
    if dictionary.ndim == 3 and dictionary.shape[0] != len(signals):
        raise ValueError("A batched dictionary must contain one dictionary per signal")
    if signals.dtype not in (torch.float32, torch.float64) or dictionary.dtype != signals.dtype:
        raise ValueError("Use matching float32 or float64 signals and dictionary")
    if dictionary.device != signals.device:
        raise ValueError("Store signals and dictionary on the same device")


def sparse_encode(model, signals, dictionary, depth):
    """Return [N,n] codes, retaining gradients for training. Inputs match model.A."""
    _check_inputs(signals, dictionary)
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be a positive integer")
    if signals.device != model.A.device or signals.dtype != model.A.dtype:
        raise ValueError("Move the batch or model so their dtype and device match")
    if getattr(model, "supports_variable_dictionary", False):
        return model(signals, depth, dictionary=dictionary)
    if dictionary.ndim != 2 or not torch.equal(dictionary, model.A):
        raise ValueError("This fixed-dictionary model requires its original [m,n] dictionary")
    return model(signals, depth)


def sparse_coding_epoch(model, signals, dictionary, depth, l1, *, batch_size=64,
                        optimizer=None, generator=None):
    """Train one epoch with optimizer, or score without gradients. Include the last batch.

    Returns per-signal means. Reconstruction is half the squared residual norm.
    CPU data may be streamed to model.A.device. The optional CPU generator
    controls sample/dictionary shuffling. Calibrate a fresh NA-ALISTA on
    training observations before using this API, as in the recovery trainer.
    """
    _check_inputs(signals, dictionary)
    if type(batch_size) is not int or batch_size < 1 or not math.isfinite(l1) or l1 < 0:
        raise ValueError("Use a positive batch size and finite, nonnegative l1")
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(signals), generator=generator) if training else torch.arange(len(signals))
    order = order.to(signals.device)
    device = model.A.device
    shared = dictionary.to(device) if dictionary.ndim == 2 else None
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    with torch.set_grad_enabled(training):
        for indices in order.split(batch_size):
            y = signals[indices].to(device)
            d = shared if shared is not None else dictionary[indices].to(device)
            z = sparse_encode(model, y, d, depth)
            predicted = z @ d.T if d.ndim == 2 else torch.bmm(d, z.unsqueeze(2)).squeeze(2)
            reconstruction = 0.5 * (predicted - y).square().sum(1)
            code_l1 = z.abs().sum(1)
            objective = (reconstruction + l1 * code_l1).mean()
            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite sparse-coding objective")
            if training:
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()
            totals += torch.stack((reconstruction.detach().double().sum(), code_l1.detach().double().sum()))
    rec, norm = (totals / len(signals)).tolist()
    return dict(signals=len(signals), objective=rec + l1 * norm, reconstruction=rec, code_l1=norm)


def _load_split(archive, split):
    specific = split + "_D"
    if ("D" in archive) == (specific in archive):
        raise ValueError(f"Provide exactly one dictionary for {split}: D or {specific}")
    key = "D" if "D" in archive else specific
    signals = torch.from_numpy(archive[split + "_y"])
    dictionary = torch.from_numpy(archive[key])
    _check_inputs(signals, dictionary)
    if key == "D" and dictionary.ndim != 2:
        raise ValueError("Shared D must have shape [m,n]. Use split_D for per-signal dictionaries")
    if not torch.isfinite(dictionary).all() or not torch.isfinite(signals).all():
        raise ValueError(f"Non-finite data in {split} split")
    return signals, dictionary


def _make_model(method, dictionary, depth, sparsity, config, l1, weights=None):
    from generate import analytic_weights
    from models import build_model

    if method == "AdaLISTA":
        from adalista_model import AdaLISTA

        options = dict(config.get("adalista", {}))
        if "lambd" in options and options.pop("lambd") != l1:
            raise ValueError("AdaLISTA lambd must match the sparse-coding task's --l1")
        return AdaLISTA(dictionary, depth, lambd=l1, **options).to(dictionary)
    if method in ("NA-ALISTA", "SALISTA") and (type(sparsity) is not int or not 1 <= sparsity <= dictionary.shape[1]):
        raise ValueError(f"{method} needs --sparsity between 1 and the number of atoms")
    if method == "SALISTA" and config.get("salista", {}).get("bounds", {}).get("mode") != "explicit":
        raise ValueError("Supply SALISTA's explicit coefficient and noise bounds under salista.bounds in --model-config")
    a = dictionary.detach().cpu().numpy()
    if weights is not None:
        w = weights.detach().cpu().numpy()
    elif method in ("ALISTA", "NA-ALISTA", "SALISTA"):
        w = analytic_weights(a)
    else:
        # EBT/HLISTA learn layer matrices and ELISTA learns one shared matrix.
        # Their W buffer is unused. Their initialization scales by ||A||².
        if not np.any(a):
            raise ValueError(f"{method} needs a nonzero dictionary for spectral initialization")
        w = a
    data = SimpleNamespace(A=a, W=w, config={"nonzeros": sparsity})
    return build_model(method, data, depth, config).to(dictionary)


def main(argv=None):
    from models import METHODS

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    fit = sub.add_parser("train")
    test = sub.add_parser("test")
    for command in (fit, test):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
        command.add_argument("--batch-size", type=int, default=64)
    fit.add_argument("--l1", type=float, required=True)
    fit.add_argument("--model", choices=METHODS, default="AdaLISTA")
    fit.add_argument("--sparsity", type=int, help="Assumed coefficient sparsity for NA-ALISTA and SALISTA")
    fit.add_argument("--model-config", type=Path, help="JSON model options, including explicit bounds for SALISTA")
    fit.add_argument("--depth", type=int, default=16)
    fit.add_argument("--epochs", type=int, default=100)
    fit.add_argument("--patience", type=int, default=25)
    fit.add_argument("--lr", type=float, default=1e-3)
    fit.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if args.action == "test":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        with np.load(args.data, allow_pickle=False) as archive:
            samples = _load_split(archive, "test")
        state = checkpoint["weights"]
        model = _make_model(checkpoint["method"], state["A"], checkpoint["depth"], checkpoint["sparsity"],
                            checkpoint["model_config"], checkpoint["l1"], state.get("W")).to(args.device)
        model.load_state_dict(state)
        metrics = sparse_coding_epoch(model, *samples, checkpoint["depth"], checkpoint["l1"],
                                      batch_size=args.batch_size)
        print(json.dumps(dict(split="test", **metrics), allow_nan=False))
        return metrics

    if min(args.epochs, args.patience, args.depth) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, patience, depth and learning rate must be positive")
    if not math.isfinite(args.l1) or args.l1 < 0:
        parser.error("l1 must be finite and nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if args.checkpoint.exists():
        parser.error("Choose a new checkpoint path for training")
    with np.load(args.data, allow_pickle=False) as archive:
        training = _load_split(archive, "train")
        validation = _load_split(archive, "val")
    dictionary = training[1]
    if args.model != "AdaLISTA" and (dictionary.ndim != 2 or not torch.equal(dictionary, validation[1])):
        parser.error("Fixed-dictionary models require the same [m,n] dictionary across splits")
    config = json.loads(args.model_config.read_text(encoding="utf-8")) if args.model_config else {}
    if not isinstance(config, dict):
        parser.error("model-config must be a JSON object")
    initial = dictionary if dictionary.ndim == 2 else dictionary[0]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    model = _make_model(args.model, initial, args.depth, args.sparsity, config, args.l1).to(args.device)
    if args.model == "NA-ALISTA":
        model.calibrate(training[0][:args.batch_size].to(args.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, wait = math.inf, 0
    with TemporaryDirectory(dir=args.checkpoint.parent) as staging:
        for epoch in range(1, args.epochs + 1):
            train_score = sparse_coding_epoch(model, *training, args.depth, args.l1, batch_size=args.batch_size,
                                              optimizer=optimizer, generator=generator)
            val = sparse_coding_epoch(model, *validation, args.depth, args.l1, batch_size=args.batch_size)
            print(json.dumps(dict(epoch=epoch, train_objective=train_score["objective"],
                                  val_objective=val["objective"]), allow_nan=False), flush=True)
            if val["objective"] < best:
                best, wait = val["objective"], 0
                tmp = Path(staging) / "best.pt"
                torch.save(dict(method=args.model, depth=args.depth, l1=args.l1, epoch=epoch,
                                seed=args.seed, val_objective=best, batch_size=args.batch_size,
                                lr=args.lr, patience=args.patience, sparsity=args.sparsity,
                                model_config=config, weights=model.state_dict()), tmp)
                tmp.replace(args.checkpoint)
            else:
                wait += 1
                if wait >= args.patience:
                    break


if __name__ == "__main__":
    main()
