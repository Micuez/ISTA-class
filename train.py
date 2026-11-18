"""Train and evaluate one noise condition per checkpoint."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import NamedTuple

import numpy as np
import torch

from dataset import RecoveryData, level_key
from evaluation import evaluate, tensors
from models import BASELINE_METHODS, METHODS, NeuralALISTA, SALISTA, build_model
from na_training import author_batches, author_holdout_score, paired_author_batches, sparse_signals


BASE = Path(__file__).resolve().parent
MODES = ("end_to_end", "layerwise")
DEFAULT_RUNS = Path("runs/single-snr-finetuned")
EVALUATION_FILES = ("dataset.py", "models.py", "evaluation.py", "salista_constraints.py")
# The initial release recorded only a full trainer hash. Its prediction and
# scoring routines are identical to the evaluation revision recorded here.
LEGACY_EVALUATORS = {
    "8cdb4a5282cfa1f86a3b7020d704819c9da5e82923387a08cbe8d27a8507911c":
        "e33783d3a46b7c21b13489d2ddc28847965e31e4967e524fa95eb716a3780fa7"
}
# Baseline predictions and metrics from the two pre-SALISTA evaluation revisions.
BASELINE_EVALUATORS = {"e33783d3a46b7c21b13489d2ddc28847965e31e4967e524fa95eb716a3780fa7",
                       "9ed6583231e2398f4b36745b1827acef80ced153bf068049f52a04ed60d3a8b5"}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def implementation_sha256(files=EVALUATION_FILES + ("train.py", "na_training.py")):
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode("ascii"))
        digest.update((BASE / name).read_text(encoding="utf-8").encode("utf-8"))
    return digest.hexdigest()


class Phase(NamedTuple):
    depth: int
    name: str
    epochs: int
    lr: float


def supported_modes(method, na_shared_prefix=False):
    return ("end_to_end",) if method == "NA-ALISTA" and not na_shared_prefix else MODES


def stages(mode, config, method="ALISTA"):
    allowed = supported_modes(method, config.get("na_shared_prefix", False))
    if mode not in allowed:
        raise ValueError(f"{method} supports {', '.join(allowed)} for this configuration")
    depth = config["depth"]
    patience = config["early_stopping_patience"]
    if type(patience) is not int or patience < 1:
        raise ValueError("early_stopping_patience must be a positive integer")
    end_cap, layer_cap = config["max_epochs_end_to_end"], config["max_epochs_per_layer"]
    if type(end_cap) is not int or end_cap < 1 or type(layer_cap) is not int or layer_cap < 3:
        raise ValueError("Use a positive end-to-end budget and at least three epochs per layer")
    recurrent = method in ("NA-ALISTA", "SALISTA")
    recurrent_lr = config.get("recurrent_learning_rate") if recurrent else None
    if recurrent_lr is not None and (not math.isfinite(recurrent_lr) or recurrent_lr <= 0):
        raise ValueError("recurrent_learning_rate must be finite and positive")
    if method == "NA-ALISTA":
        schedule = config.get("na_schedule", "experiment")
        if schedule not in ("experiment", "author"):
            raise ValueError("Unknown NA-ALISTA schedule: " + str(schedule))
        if schedule == "author" and (mode == "layerwise" or config.get("paired_author_online")):
            raise ValueError("The author schedule uses its original end-to-end protocol")
        if mode == "end_to_end":
            cap = 100 + 20 * depth if schedule == "author" else end_cap
            return [Phase(depth, "end_to_end", cap, 0.0002 if schedule == "author" or recurrent_lr is None else recurrent_lr)]
    if mode == "end_to_end":
        return [Phase(depth, mode, end_cap, config["learning_rate"] if recurrent_lr is None else recurrent_lr)]
    factors = config.get("finetune_lr_factors", [0.2, 0.02])
    if len(factors) != 2 or not (0 < factors[1] < factors[0] < 1):
        raise ValueError("Expect two decreasing fine-tuning LR factors between 0 and 1")
    base = config.get("layerwise_learning_rate", 0.0005) if recurrent_lr is None else recurrent_lr
    if not math.isfinite(base) or base <= 0:
        raise ValueError("layerwise_learning_rate must be finite and positive")
    quotient, remainder = divmod(layer_cap, 3)
    shared_prefix = method == "NA-ALISTA" or (method == "SALISTA" and
                    config.get("salista", {}).get("sharing", "shared") == "shared")
    plan = []
    for active in range(1, depth + 1):
        if shared_prefix and active < depth:
            plan.append(Phase(active, "prefix", layer_cap, base))
            continue
        first = "prefix" if shared_prefix else "new_layer"
        for i, (name, factor) in enumerate(zip((first, "joint_1", "joint_2"), (1, *factors))):
            plan.append(Phase(active, name, quotient + int(i < remainder), base * factor))
    return plan


def matched_level(data, snr_db):
    for level in data.levels:
        if float(level) == float(snr_db):
            return level
    raise ValueError("Unknown noise level: " + str(snr_db))


def single_level(data, config):
    """Return the previous level tuple after restricting training to config['snr_db']."""
    if "snr_db" not in config:
        return None
    previous = data.levels
    data.levels = (matched_level(data, config["snr_db"]),)
    return previous


def restore_levels(data, previous):
    if previous is not None:
        data.levels = previous


def trainable_parameters(model, phase, depth):
    model.requires_grad_(phase == "end_to_end")
    if phase != "end_to_end":
        if isinstance(model, NeuralALISTA):
            if phase not in ("prefix", "joint_1", "joint_2"):
                raise ValueError("Shared NA-ALISTA uses full prefix backpropagation")
            model.regressor.requires_grad_(True)
        elif phase == "prefix" and isinstance(model, SALISTA) and model.sharing == "shared":
            model.stage_module(depth).requires_grad_(True)
        elif phase == "new_layer":
            if isinstance(model, SALISTA) and model.sharing == "shared":
                raise ValueError("Shared SALISTA uses full prefix backpropagation")
            model.stage_module(depth).requires_grad_(True)
        elif phase in ("joint_1", "joint_2"):
            for layer in model.layers[:depth]:
                layer.requires_grad_(True)
        else:
            raise ValueError("Unknown training phase: " + phase)
        if getattr(model, "shared_weight", None) is not None:
            model.shared_weight.requires_grad_(phase != "new_layer" or depth == 1)
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def gradient_factors(params, phase, decay):
    # Match get_train_op in the TF references: multiply gradients before Adam.
    if not phase.name.startswith("joint_"):
        return []
    factors = []
    for name, parameter in params.items():
        first_depth = 1 if name == "shared_weight" else int(name.split(".")[1]) + 1
        factors.append((parameter, decay ** (phase.depth - first_depth)))
    return factors


def fit(data, method, mode, config, device, dest, fingerprint):
    if "snr_db" not in config:
        raise ValueError("A single snr_db is required for training")
    stages(mode, config, method)
    previous_levels = single_level(data, config)
    try:
        _fit(data, method, mode, config, device, dest, fingerprint)
    finally:
        restore_levels(data, previous_levels)


def _fit(data, method, mode, config, device, dest, fingerprint):
    identity = {"method": method, "mode": mode, "training": config,
                "data_manifest_sha256": fingerprint, "implementation_sha256": implementation_sha256(),
                "evaluation_sha256": implementation_sha256(EVALUATION_FILES)}
    if dest.exists():
        if (dest / "run.json").is_file() and (dest / "best.pt").is_file():
            old = json.loads((dest / "run.json").read_text(encoding="utf-8"))
            if old["experiment"] == identity:
                print("Already trained:", dest)
                return
        raise FileExistsError("Use a new --out for changed settings: " + str(dest))
    depth, batch = config["depth"], config["batch_size"]
    level, = data.levels
    plan = stages(mode, config, method)
    is_na = method == "NA-ALISTA"
    paired_online = config.get("paired_author_online", False) and method in ("NA-ALISTA", "SALISTA")
    bounds = config.get("salista", {}).get("bounds", {})
    if paired_online and method == "SALISTA" and bounds.get("mode") != "online_gaussian":
        raise ValueError("Paired online SALISTA requires online_gaussian bounds")
    author_schedule = is_na and config.get("na_schedule") == "author"
    patience = None if author_schedule else config["early_stopping_patience"]
    decay = config.get("older_layer_gradient_decay", 0.3)
    if not (0 < decay <= 1):
        raise ValueError("older_layer_gradient_decay must be in (0, 1]")
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    signals = holdout = None
    if is_na or paired_online:
        batch = 512 if author_schedule else config.get("na_batch_size", 512)
        train_size = 50000 if author_schedule else config.get("na_samples_per_epoch", 50000)
        if not 2 <= batch <= train_size:
            raise ValueError("Online training needs at least one complete training batch of size >= 2")
        if not paired_online:
            signals = sparse_signals(train_size, data.A.shape[1], data.config["nonzeros"])
        if author_schedule:
            holdout = sparse_signals(10000, data.A.shape[1], data.config["nonzeros"])
    model = build_model(method, data, depth, config).to(device)
    shared_controller = is_na or (isinstance(model, SALISTA) and model.sharing == "shared")
    tf_layerwise_loss = mode == "layerwise" and method not in ("NA-ALISTA", "SALISTA")
    dest.parent.mkdir(parents=True, exist_ok=True)
    global_epoch, total_updates = 0, 0
    stage_records = []
    started = perf_counter()
    with TemporaryDirectory(dir=dest.parent) as temp:
        work = Path(temp) / "run"
        work.mkdir()
        with (work / "history.jsonl").open("w", encoding="utf-8") as history:
            for phase in plan:
                params = trainable_parameters(model, phase.name, phase.depth)
                optimizer = torch.optim.Adam(params.values(), lr=phase.lr)
                multipliers = [] if shared_controller else gradient_factors(params, phase, decay)
                # Keep the incoming same-depth checkpoint eligible across fine-tuning phases.
                best_score = stage_records[-1]["best_val_nmse"] if phase.name.startswith("joint_") else math.inf
                best_epoch = stage_records[-1]["best_epoch"] if phase.name.startswith("joint_") else global_epoch
                start_epoch, start_updates = global_epoch, total_updates
                wait, stop_reason = 0, "epoch_limit"
                for stage_epoch in range(1, phase.epochs + 1):
                    model.train()
                    if paired_online:
                        batches = paired_author_batches(train_size, data.config["nonzeros"], model.A, level,
                                                        batch, config["seed"] + global_epoch)
                    elif is_na:
                        if global_epoch:
                            signals = sparse_signals(train_size, data.A.shape[1], data.config["nonzeros"])
                        batches = author_batches(signals, model.A, level, batch, shuffle=True)
                    else:
                        batches = tensors(data, "train", batch, device, level=level, epoch=global_epoch)
                    for y, x in batches:
                        if is_na and total_updates == 0:
                            model.calibrate(y)
                        optimizer.zero_grad(set_to_none=True)
                        pred = model(y, phase.depth, last_step_grad=(phase.name == "new_layer"))
                        squared_error = (pred - x).square()
                        # The TF layerwise references minimize tf.nn.l2_loss.
                        loss = 0.5 * squared_error.sum() if tf_layerwise_loss else squared_error.mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite training loss")
                        loss.backward()
                        for parameter, factor in multipliers:
                            if parameter.grad is not None:
                                parameter.grad.mul_(factor)
                        max_norm = math.inf if tf_layerwise_loss else 1.0
                        torch.nn.utils.clip_grad_norm_(params.values(), max_norm, error_if_nonfinite=True)
                        optimizer.step()
                        total_updates += 1
                    global_epoch += 1
                    if holdout is not None:
                        score = author_holdout_score(model, holdout, level, batch, depth)
                        validation = {"author_holdout": {"nmse": score}}
                    else:
                        validation, score = evaluate(model, data, "val", phase.depth, batch, device)
                    if not math.isfinite(score):
                        raise FloatingPointError("Non-finite validation score")
                    if score < best_score or (author_schedule and score == best_score):
                        torch.save(model.state_dict(), work / "best.pt")
                        best_score, best_epoch, wait = score, global_epoch, 0
                    else:
                        wait += 1
                    record = {"epoch": global_epoch, "stage_epoch": stage_epoch, "phase": phase.name,
                              "epochs_without_improvement": wait, "active_depth": phase.depth,
                              "updates": total_updates, "val": validation, "mean_val_nmse": score}
                    history.write(json.dumps(record, allow_nan=False) + "\n")
                    history.flush()
                    if patience is not None and wait >= patience:
                        stop_reason = "validation_patience"
                        break
                model.load_state_dict(torch.load(work / "best.pt", map_location=device, weights_only=True))
                stage_records.append({"depth": phase.depth, "phase": phase.name, "learning_rate": phase.lr,
                                      "epochs": [start_epoch + 1, global_epoch], "epoch_cap": phase.epochs,
                                      "epochs_run": stage_epoch, "updates": total_updates - start_updates,
                                      "stop_reason": stop_reason, "best_epoch": best_epoch,
                                      "best_val_nmse": best_score, "trained_parameters": list(params)})
                print(f"Depth {phase.depth} {phase.name}, epochs {stage_epoch}, {stop_reason}, "
                      f"best NMSE {best_score:.6g}", flush=True)
        details = {}
        if isinstance(model, SALISTA):
            cert = model.certificate
            details["salista_certificate"] = {
                "initial_l1_bound": model.initial_l1_bound, "noise_l2_bound": model.noise_l2_bound,
                "q_row": cert.q.max().item(), "gram_margin": cert.margin.item(), "mu": cert.mu.item(),
                "sparsity_bound": cert.sparsity,
                "parameters": sum(p.numel() for p in model.parameters())}
        write_json(work / "run.json", {"experiment": identity, "device": str(device), **details,
                   "torch": torch.__version__, "epochs": global_epoch, "updates": total_updates,
                   "training_data": "author_online" if is_na or paired_online else "stored_single_snr",
                   "training_validation_seconds": perf_counter() - started,
                   "selection": "author_holdout" if author_schedule else "fixed_validation",
                   "batch_size": batch, "patience": patience, "stages": stage_records})
        work.rename(dest)
    print("Trained:", dest, "updates:", total_updates)


def load_checkpoint(data, path, device, fingerprint):
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))["experiment"]
    if run["data_manifest_sha256"] != fingerprint:
        raise ValueError("The dataset differs from the one used to train this checkpoint")
    signature = run.get("evaluation_sha256", LEGACY_EVALUATORS.get(run["implementation_sha256"]))
    # Earlier checkpoints use the same trained-weight forward equations.
    known_baseline = run["method"] in BASELINE_METHODS and signature in BASELINE_EVALUATORS
    if signature != implementation_sha256(EVALUATION_FILES) and not known_baseline:
        raise ValueError("Use the matching model and evaluation revision for this checkpoint")
    cfg = run["training"]
    model = build_model(run["method"], data, cfg["depth"], cfg).to(device)
    model.load_state_dict(torch.load(path / "best.pt", map_location=device, weights_only=True))
    return run, model.eval()


def test_run(data, path, device, fingerprint):
    run, model = load_checkpoint(data, path, device, fingerprint)
    cfg = run["training"]
    previous_levels = single_level(data, cfg)
    try:
        scores, _ = evaluate(model, data, "test", cfg["depth"], cfg["batch_size"], device)
    finally:
        restore_levels(data, previous_levels)
    with TemporaryDirectory(dir=path) as tmp:
        result = Path(tmp) / "test.json"
        write_json(result, {"method": run["method"], "mode": run["mode"], "snr_db": scores})
        result.replace(path / "test.json")
    print("Evaluated:", path / "test.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "test"))
    parser.add_argument("--model", choices=("all",) + METHODS, default="all",
                        help="'all' runs the published baseline grid; select SALISTA explicitly")
    parser.add_argument("--mode", choices=("both",) + MODES, default="both")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, help="override the model initialization seed")
    parser.add_argument("--snr", type=float, required=True, choices=(40, 20, 10, 5, math.inf))
    parser.add_argument("--na-schedule", choices=("experiment", "author"),
                        help="author uses 100 + 20 * depth epochs and a separate synthetic holdout")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config_path = args.config or BASE / ("training.salista.json" if args.model == "SALISTA" else "training.json")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.na_schedule is not None:
        cfg["na_schedule"] = args.na_schedule
    default_methods = METHODS if cfg.get("paired_author_online") and "salista" in cfg else BASELINE_METHODS
    models = default_methods if args.model == "all" else (args.model,)
    modes = MODES if args.mode == "both" else (args.mode,)
    if not (type(cfg["depth"]) is int and cfg["depth"] >= 1 and cfg["batch_size"] >= 1
            and math.isfinite(cfg["learning_rate"]) and cfg["learning_rate"] > 0
            and 0 <= cfg["seed"] < 2**32):
        parser.error("Invalid training configuration")
    jobs = [(m, mode) for m in models for mode in modes if mode in supported_modes(m, cfg.get("na_shared_prefix", False))]
    if args.action == "test" and args.model == "NA-ALISTA" and args.mode == "layerwise":
        jobs = [(args.model, args.mode)]  # Evaluation of a historical checkpoint.
    if not jobs:
        parser.error("NA-ALISTA is trained end-to-end only")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    args.data = args.data or Path("data/noiseless" if math.isinf(args.snr) else "data/main")
    data = RecoveryData(args.data)
    cfg["snr_db"] = matched_level(data, args.snr)
    fingerprint = hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest()
    for method, mode in jobs:
        dest = args.out / method / mode / ("snr-" + level_key(cfg["snr_db"])) / ("seed-" + str(cfg["seed"]))
        if args.action == "train":
            fit(data, method, mode, cfg, device, dest, fingerprint)
        else:
            test_run(data, dest, device, fingerprint)


if __name__ == "__main__":
    main()
