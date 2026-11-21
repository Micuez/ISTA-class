"""Sparse coding of observed signals by minimizing 0.5 ||Dz-y||² + l1 ||z||₁.

Each row of y is a signal. Dictionary atoms are columns of a shared [m,n]
or per-signal [N,m,n] dictionary. The task requires no coefficient labels.
Train end to end or at increasing depth, selecting by mean validation objective.
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


def sparse_encode(model, signals, dictionary, depth, *, last_step_grad=False):
    """Return [N,n] codes, retaining gradients for training. Inputs match model.A."""
    _check_inputs(signals, dictionary)
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be a positive integer")
    if signals.device != model.A.device or signals.dtype != model.A.dtype:
        raise ValueError("Move the batch or model so their dtype and device match")
    kwargs = {"last_step_grad": True} if last_step_grad else {}
    if getattr(model, "supports_variable_dictionary", False):
        return model(signals, depth, dictionary=dictionary, **kwargs)
    if dictionary.ndim != 2 or not torch.equal(dictionary, model.A):
        raise ValueError("This fixed-dictionary model requires its original [m,n] dictionary")
    return model(signals, depth, **kwargs)


def sparse_coding_epoch(model, signals, dictionary, depth, l1, *, batch_size=64,
                        optimizer=None, generator=None, last_step_grad=False, gradient_multipliers=()):
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
            z = sparse_encode(model, y, d, depth, last_step_grad=last_step_grad)
            predicted = z @ d.T if d.ndim == 2 else torch.bmm(d, z.unsqueeze(2)).squeeze(2)
            reconstruction = 0.5 * (predicted - y).square().sum(1)
            code_l1 = z.abs().sum(1)
            objective = (reconstruction + l1 * code_l1).mean()
            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite sparse-coding objective")
            if training:
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                for parameter, factor in gradient_multipliers:
                    if parameter.grad is not None:
                        parameter.grad.mul_(factor)
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
    from models import METHODS, SALISTA
    from train import MODES, gradient_factors, stages, supported_modes, trainable_parameters

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
    fit.add_argument("--mode", choices=MODES, default="end_to_end")
    fit.add_argument("--sparsity", type=int, help="Assumed coefficient sparsity for NA-ALISTA and SALISTA")
    fit.add_argument("--model-config", type=Path, help="JSON model options, including explicit bounds for SALISTA")
    fit.add_argument("--depth", type=int, default=16)
    fit.add_argument("--epochs", type=int, default=100, help="end-to-end epoch cap")
    fit.add_argument("--epochs-per-layer", type=int, default=100,
                     help="layerwise epoch cap per active depth, split across its training phases")
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
    if args.epochs_per_layer < 3:
        parser.error("epochs-per-layer must be at least 3 to allocate the three training phases")
    if args.mode not in supported_modes(args.model):
        parser.error(f"{args.model} supports only {', '.join(supported_modes(args.model))} training")
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
    schedule = dict(depth=args.depth, max_epochs_end_to_end=args.epochs,
                    max_epochs_per_layer=args.epochs_per_layer, early_stopping_patience=args.patience,
                    learning_rate=args.lr, layerwise_learning_rate=args.lr, recurrent_learning_rate=args.lr,
                    salista=config.get("salista", {}))
    plan = stages(args.mode, schedule, args.model)
    shared_controller = isinstance(model, SALISTA) and model.sharing == "shared"
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    records = []
    epoch = updates = best_epoch = 0
    updates_per_epoch = math.ceil(len(training[0]) / args.batch_size)
    with TemporaryDirectory(dir=args.checkpoint.parent) as staging:
        stage_best = Path(staging) / "stage.pt"
        pending = Path(staging) / "publish.pt"

        def publish_full_depth(score, complete):
            torch.save(dict(method=args.model, mode=args.mode, depth=args.depth, l1=args.l1, epoch=best_epoch,
                            epochs=epoch, updates=updates, seed=args.seed, val_objective=score,
                            batch_size=args.batch_size, lr=args.lr, patience=args.patience,
                            max_epochs_end_to_end=args.epochs, max_epochs_per_layer=args.epochs_per_layer,
                            sparsity=args.sparsity, stages=records, training_complete=complete,
                            model_config=config, weights=model.state_dict()), pending)
            pending.replace(args.checkpoint)

        for phase in plan:
            params = trainable_parameters(model, phase.name, phase.depth)
            optimizer = torch.optim.Adam(params.values(), lr=phase.lr)
            # Shared SALISTA updates the same controller through every step in its active prefix.
            multipliers = () if shared_controller else gradient_factors(params, phase, 0.3)
            # A lower-rate fine-tune must compete with the incoming best weights at the same depth.
            best = records[-1]["best_val_objective"] if phase.name.startswith("joint_") else math.inf
            wait, stop_reason = 0, "epoch_limit"
            for stage_epoch in range(1, phase.epochs + 1):
                train_score = sparse_coding_epoch(model, *training, phase.depth, args.l1,
                                                  batch_size=args.batch_size, optimizer=optimizer,
                                                  generator=generator, last_step_grad=phase.name == "new_layer",
                                                  gradient_multipliers=multipliers)
                val = sparse_coding_epoch(model, *validation, phase.depth, args.l1, batch_size=args.batch_size)
                epoch += 1
                updates += updates_per_epoch
                if val["objective"] < best:
                    best, best_epoch, wait = val["objective"], epoch, 0
                    torch.save(model.state_dict(), stage_best)
                    if phase.depth == args.depth:
                        publish_full_depth(best, complete=False)
                else:
                    wait += 1
                print(json.dumps(dict(epoch=epoch, updates=updates, active_depth=phase.depth, phase=phase.name,
                                      stage_epoch=stage_epoch, train_objective=train_score["objective"],
                                      val_objective=val["objective"]), allow_nan=False), flush=True)
                if wait >= args.patience:
                    stop_reason = "validation_patience"
                    break
            model.load_state_dict(torch.load(stage_best, map_location=args.device, weights_only=True))
            records.append(dict(depth=phase.depth, phase=phase.name, learning_rate=phase.lr,
                                epoch_cap=phase.epochs, epochs_run=stage_epoch, updates=stage_epoch * updates_per_epoch,
                                best_epoch=best_epoch, best_val_objective=best, stop_reason=stop_reason))
        publish_full_depth(best, complete=True)


if __name__ == "__main__":
    main()
