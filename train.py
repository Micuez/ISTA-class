"""Train or evaluate the four sparse recovery models with the same stored observations."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from dataset import RecoveryData
from evaluation import evaluate, tensors
from models import METHODS, NeuralALISTA, build_model


BASE = Path(__file__).resolve().parent
MODES = ("end_to_end", "layerwise")
EVALUATION_FILES = ("dataset.py", "models.py", "evaluation.py")
# The initial release recorded only a full trainer hash. Its prediction and
# scoring routines are identical to the evaluation revision recorded here.
LEGACY_EVALUATORS = {
    "8cdb4a5282cfa1f86a3b7020d704819c9da5e82923387a08cbe8d27a8507911c":
        "e33783d3a46b7c21b13489d2ddc28847965e31e4967e524fa95eb716a3780fa7"
}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def implementation_sha256(files=EVALUATION_FILES + ("train.py",)):
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode("ascii"))
        digest.update((BASE / name).read_text(encoding="utf-8").encode("utf-8"))
    return digest.hexdigest()


def stages(mode, config):
    depth = config["depth"]
    if ("total_epochs" in config) == ("max_epochs_per_stage" in config):
        raise ValueError("Specify exactly one of total_epochs or max_epochs_per_stage")
    patience = config.get("early_stopping_patience")
    if patience is not None and (type(patience) is not int or patience < 1):
        raise ValueError("early_stopping_patience must be a positive number of epochs")
    stage_cap = config.get("max_epochs_per_stage")
    epochs = config.get("total_epochs", stage_cap)
    if type(epochs) is not int or epochs < 1:
        raise ValueError("The epoch budget must be a positive integer")
    if mode == "end_to_end":
        return [(depth, epochs)]
    if mode != "layerwise":
        raise ValueError("Unknown training mode: " + str(mode))
    if stage_cap is not None:
        return [(i, stage_cap) for i in range(1, depth + 1)]
    if epochs < depth:
        raise ValueError("layerwise needs at least one epoch for each layer")
    base, extra = divmod(epochs, depth)
    return [(i, base + int(i > depth - extra)) for i in range(1, depth + 1)]


def trainable_parameters(model, mode, depth):
    model.requires_grad_(mode == "end_to_end")
    if mode == "layerwise":
        model.stage_module(depth).requires_grad_(True)
        if isinstance(model, NeuralALISTA) and depth > 1:
            # The detached prefix consumed the two learned initial states.
            model.regressor.hidden.requires_grad_(False)
            model.regressor.cellstate.requires_grad_(False)
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def fit(data, method, mode, config, device, dest, fingerprint):
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
    patience = config.get("early_stopping_patience")
    plan = stages(mode, config)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model = build_model(method, data, depth).to(device)
    if isinstance(model, NeuralALISTA):
        y, _ = next(tensors(data, "train", batch, device))
        model.calibrate(y)
    dest.parent.mkdir(parents=True, exist_ok=True)
    global_epoch, total_updates = 0, 0
    stage_records = []
    with TemporaryDirectory(dir=dest.parent) as temp:
        work = Path(temp) / "run"
        work.mkdir()
        with (work / "history.jsonl").open("w", encoding="utf-8") as history:
            for active_depth, epoch_cap in plan:
                params = trainable_parameters(model, mode, active_depth)
                optimizer = torch.optim.Adam(params.values(), lr=config["learning_rate"])
                best_score = math.inf
                start_epoch, start_updates = global_epoch, total_updates
                stop_reason = "epoch_limit"
                for stage_epoch in range(1, epoch_cap + 1):
                    model.train()
                    for y, x in tensors(data, "train", batch, device, epoch=global_epoch):
                        optimizer.zero_grad(set_to_none=True)
                        pred = model(y, active_depth, last_step_grad=(mode == "layerwise"))
                        loss = (pred - x).square().mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite training loss")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params.values(), 1.0, error_if_nonfinite=True)
                        optimizer.step()
                        total_updates += 1
                    global_epoch += 1
                    validation, score = evaluate(model, data, "val", active_depth, batch, device)
                    if score < best_score:
                        torch.save(model.state_dict(), work / "best.pt")
                        best_score, best_epoch = score, global_epoch
                    wait = global_epoch - best_epoch
                    record = {"epoch": global_epoch, "stage_epoch": stage_epoch,
                              "epochs_without_improvement": wait, "active_depth": active_depth,
                              "updates": total_updates, "val": validation, "mean_val_nmse": score}
                    history.write(json.dumps(record, allow_nan=False) + "\n")
                    history.flush()
                    if patience is not None and wait >= patience:
                        stop_reason = "validation_patience"
                        break
                model.load_state_dict(torch.load(work / "best.pt", map_location=device, weights_only=True))
                stage_records.append({"depth": active_depth, "epochs": [start_epoch + 1, global_epoch],
                                      "epoch_cap": epoch_cap, "epochs_run": stage_epoch,
                                      "updates": total_updates - start_updates, "stop_reason": stop_reason,
                                      "best_epoch": best_epoch, "best_val_nmse": best_score,
                                      "trained_parameters": list(params)})
        write_json(work / "run.json", {"experiment": identity, "device": str(device),
                   "torch": str(torch.__version__), "epochs": global_epoch,
                   "updates": total_updates, "stages": stage_records})
        work.rename(dest)
    print("Trained:", dest, "updates:", total_updates)


def test_run(data, path, device, fingerprint):
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))["experiment"]
    if run["data_manifest_sha256"] != fingerprint:
        raise ValueError("The dataset differs from the one used to train this checkpoint")
    signature = run.get("evaluation_sha256", LEGACY_EVALUATORS.get(run["implementation_sha256"]))
    if signature != implementation_sha256(EVALUATION_FILES):
        raise ValueError("Use the matching model and evaluation revision for this checkpoint")
    cfg = run["training"]
    model = build_model(run["method"], data, cfg["depth"]).to(device)
    model.load_state_dict(torch.load(path / "best.pt", map_location=device, weights_only=True))
    scores, _ = evaluate(model, data, "test", cfg["depth"], cfg["batch_size"], device)
    with TemporaryDirectory(dir=path) as tmp:
        result = Path(tmp) / "test.json"
        write_json(result, {"method": run["method"], "mode": run["mode"], "snr_db": scores})
        result.replace(path / "test.json")
    print("Evaluated:", path / "test.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "test"))
    parser.add_argument("--model", choices=("all",) + METHODS, default="all")
    parser.add_argument("--mode", choices=("both",) + MODES, default="both")
    parser.add_argument("--data", type=Path, default=Path("data/main"))
    parser.add_argument("--out", type=Path, default=Path("runs/v1"))
    parser.add_argument("--config", type=Path, default=BASE / "training.json")
    parser.add_argument("--seed", type=int, help="override the model initialization seed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    models = METHODS if args.model == "all" else (args.model,)
    modes = MODES if args.mode == "both" else (args.mode,)
    if not (type(cfg["depth"]) is int and cfg["depth"] >= 1 and cfg["batch_size"] >= 1
            and math.isfinite(cfg["learning_rate"]) and cfg["learning_rate"] > 0
            and 0 <= cfg["seed"] < 2**32):
        parser.error("Invalid training configuration")
    if args.action == "train":
        try:
            for mode in modes:
                stages(mode, cfg)
        except ValueError as error:
            parser.error(str(error))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    data = RecoveryData(args.data)
    fingerprint = hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest()
    for method in models:
        for mode in modes:
            dest = args.out / method / mode / ("seed-" + str(cfg["seed"]))
            if args.action == "train":
                fit(data, method, mode, cfg, device, dest, fingerprint)
            else:
                test_run(data, dest, device, fingerprint)


if __name__ == "__main__":
    main()
