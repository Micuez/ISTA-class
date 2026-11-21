"""Run the sparse-coding methods and training schedules at one observation SNR."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys

import numpy as np
import torch

from dataset import RecoveryData, SPLITS
from generate import sha256
from salista_constraints import resolve_bounds


ROOT = Path(__file__).resolve().parent
VARIANTS = ("EBT-LISTA", "HLISTA", "ALISTA", "ELISTA", "NA-ALISTA", "AdaLISTA",
            "SALISTA-bounded", "SALISTA-free")
JOBS = [(variant, mode) for variant in VARIANTS for mode in ("end_to_end", "layerwise")
        if variant != "NA-ALISTA" or mode == "end_to_end"]


def write_json(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def evaluate_variant(args, variant, mode, gpu, sparsity, test_size):
    folder = args.out / variant / mode
    folder.mkdir(parents=True)
    salista = variant.startswith("SALISTA-")
    method = "SALISTA" if salista else variant
    common = ["--data", str(args.out / "coding.npz"), "--checkpoint", str(folder / "best.pt"),
              "--batch-size", str(args.batch_size), "--device", "cuda"]
    base = [sys.executable, "-u", str(ROOT / "sparse_coding.py")]
    fit = base + ["train", *common, "--model", method, "--mode", mode, "--depth", str(args.depth),
                  "--l1", str(args.l1), "--epochs", str(args.epochs), "--patience", str(args.patience),
                  "--lr", str(args.lr), "--seed", str(args.seed)]
    if mode == "layerwise":
        fit.extend(["--epochs-per-layer", str(args.epochs_per_layer)])
    if method in ("NA-ALISTA", "SALISTA"):
        fit.extend(["--sparsity", str(sparsity)])
    if salista:
        fit.extend(["--model-config", str(args.out / (variant + ".json"))])
    test = base + ["test", *common]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    state = dict(variant=variant, mode=mode, gpu=gpu, started=timestamp(), status="training",
                 train_command=fit, test_command=test)
    status_file = folder / "status.json"
    write_json(status_file, state)
    print(f"Starting {variant}/{mode} on GPU {gpu}", flush=True)
    try:
        with (folder / "train.log").open("w", encoding="utf-8") as log:
            subprocess.run(fit, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        if (not checkpoint["training_complete"] or checkpoint["method"] != method or checkpoint["mode"] != mode
                or checkpoint["seed"] != args.seed or checkpoint["depth"] != args.depth):
            raise ValueError("Final checkpoint does not match the requested run")
        state.update(epochs=checkpoint["epochs"], best_epoch=checkpoint["epoch"], updates=checkpoint["updates"])
        del checkpoint
        state["status"] = "testing"
        write_json(status_file, state)
        with (folder / "test.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(test, stdout=subprocess.PIPE, stderr=log, text=True,
                                    encoding="utf-8", env=env, check=True)
            log.write(result.stdout)
        metrics = json.loads(result.stdout)
        objective = metrics["reconstruction"] + args.l1 * metrics["code_l1"]
        if (metrics["split"] != "test" or metrics["signals"] != test_size
                or not all(math.isfinite(metrics[k]) for k in ("objective", "reconstruction", "code_l1"))
                or not math.isclose(metrics["objective"], objective, rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError("Test metrics failed consistency checks")
        write_json(folder / "test.json", metrics)
        state.update(status="completed", metrics=metrics)
    except Exception as exc:
        state.update(status="failed", failed_phase=state["status"], error=str(exc))
    state["ended"] = timestamp()
    write_json(status_file, state)
    print(f"{variant}/{mode}: {state['status']}", flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-main", type=Path, default=Path("data/main"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snr", type=int, choices=(40, 20, 10, 5), default=20)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--l1", type=float, default=.1)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epochs-per-layer", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--simultaneous", action="store_true",
                        help="Launch every job immediately, reserve one GPU per SALISTA job and share remaining GPUs in pairs")
    args = parser.parse_args()
    if min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU IDs must be distinct and nonnegative")
    if (min(args.depth, args.epochs, args.patience, args.batch_size) < 1 or args.epochs_per_layer < 3
            or not 0 <= args.seed < 2**32
            or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.l1) or args.l1 < 0):
        parser.error("Invalid training settings")
    allocation = {}
    if args.simultaneous:
        salista = [job for job in JOBS if job[0].startswith("SALISTA-")]
        other = [job for job in JOBS if job not in salista]
        shared = args.gpus[len(salista):]
        if 2 * len(shared) < len(other):
            parser.error("Simultaneous launch requires 10 GPUs for four dedicated SALISTA and at most two other jobs per GPU")
        allocation.update(zip(salista, args.gpus))
        allocation.update((job, shared[i % len(shared)]) for i, job in enumerate(other))
    args.out = args.out.resolve()
    print(f"Single SNR {args.snr} dB, seed {args.seed}; {len(JOBS)} tasks", flush=True)
    for variant, mode in JOBS:
        gpu = allocation.get((variant, mode), "next free")
        print(f"  {variant}/{mode} -> GPU {gpu}", flush=True)
    if args.dry_run:
        return
    data = RecoveryData(args.data_main)
    observations = {part: data.arrays(part, args.snr)[0] for part in SPLITS}
    split_sizes = {part: len(y) for part, y in observations.items()}
    sparsity = data.config["nonzeros"]
    radius, noise = resolve_bounds({"mode": "gaussian", "failure_probability": .01},
                                   data.A.shape[0], sparsity, args.snr)
    args.out.mkdir(parents=True, exist_ok=False)
    archive = args.out / "coding.npz"
    np.savez(archive, D=data.A, **{part + "_y": y for part, y in observations.items()})
    settings = json.loads((ROOT / "training.comparison.json").read_text(encoding="utf-8"))["salista"]
    bounds = dict(mode="explicit", initial_l1_bound=radius, noise_l2_bound=noise)
    for variant in ("SALISTA-bounded", "SALISTA-free"):
        write_json(args.out / (variant + ".json"),
                   {"salista": dict(settings, bounds=bounds, free_momentum=variant == "SALISTA-free")})
    revision = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    settings = {name: value for name, value in vars(args).items() if name not in ("out", "data_main", "dry_run")}
    jobs = [dict(variant=v, mode=m, gpu=allocation.get((v, m))) for v, m in JOBS]
    manifest = dict(**settings, jobs=jobs, commit=revision, source=str(data.root.resolve()),
                    manifest_sha256=sha256(data.root / "manifest.json"), npz_sha256=sha256(archive),
                    split_sizes=split_sizes, sparsity=sparsity, bounds=bounds, failure_probability=.01)
    write_json(args.out / "manifest.json", manifest)
    slots = Queue()
    for gpu in args.gpus:
        slots.put(gpu)

    def run(job):
        if allocation:
            return evaluate_variant(args, *job, allocation[job], sparsity, split_sizes["test"])
        gpu = slots.get()
        try:
            return evaluate_variant(args, *job, gpu, sparsity, split_sizes["test"])
        finally:
            slots.put(gpu)

    workers = len(JOBS) if allocation else min(len(args.gpus), len(JOBS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, JOBS))
    write_json(args.out / "summary.json", results)
    completed = {(row["variant"], row["mode"]): row for row in results if row["status"] == "completed"}
    comparison = []
    for mode in ("end_to_end", "layerwise"):
        bounded = completed.get(("SALISTA-bounded", mode))
        free = completed.get(("SALISTA-free", mode))
        if bounded and free:
            comparison.append(dict(mode=mode, bounded_objective=bounded["metrics"]["objective"],
                                   free_objective=free["metrics"]["objective"],
                                   free_minus_bounded=free["metrics"]["objective"] - bounded["metrics"]["objective"],
                                   bounded_updates=bounded["updates"], free_updates=free["updates"]))
    write_json(args.out / "salista_momentum_comparison.json", comparison)
    if any(result["status"] != "completed" for result in results):
        raise SystemExit("One or more tasks failed. See each variant's status.json and logs.")


if __name__ == "__main__":
    main()
