"""Train and evaluate the baseline grid or the 48-run six-method comparison."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys

from dataset import RecoveryData, level_key
from generate import generate_noiseless
from models import BASELINE_METHODS, METHODS
from train import DEFAULT_RUNS, matched_level, stages, supported_modes
from verify_data import check_paired_signals


ROOT = Path(__file__).resolve().parent
SNRS = (40, 20, 10, 5, float("inf"))


def grid_jobs(preset="baseline"):
    comparison = preset == "six-methods"
    levels = SNRS[:-1] if comparison else SNRS
    methods = METHODS if comparison else BASELINE_METHODS
    return [(method, mode, snr) for snr in levels for method in methods
            for mode in supported_modes(method, na_shared_prefix=comparison)]


def write_test_summary(out, jobs, seed):
    rows = []
    for method, mode, snr in jobs:
        key = level_key(snr)
        dest = out / method / mode / ("snr-" + key) / f"seed-{seed}"
        run = json.loads((dest / "run.json").read_text(encoding="utf-8"))
        score = json.loads((dest / "test.json").read_text(encoding="utf-8"))["snr_db"][key]
        rows.append(dict(method=method, mode=mode, snr_db=key, seed=seed, depth=run["experiment"]["training"]["depth"],
                         training_data=run["training_data"], batch_size=run["batch_size"], epochs=run["epochs"],
                         updates=run["updates"], training_validation_seconds=run.get("training_validation_seconds"),
                         **score))
    target = out / f"test_summary_seed_{seed}.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Test comparison:", target, flush=True)


def devices(device):
    if device == "cpu":
        return ["cpu"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        visible = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).replace("\n", ",")
    ids = [value.strip() for value in visible.split(",") if value.strip() and value.strip() != "-1"]
    if not ids:
        raise SystemExit("No GPU is visible. Use --device cpu to run on CPU.")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("baseline", "six-methods"), default="baseline")
    parser.add_argument("--dry-run", action="store_true", help="list the resolved jobs without training or opening any data")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-main", type=Path, default=Path("data/main"))
    parser.add_argument("--data-noiseless", type=Path, default=Path("data/noiseless"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--na-schedule", choices=("experiment", "author"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    comparison = args.preset == "six-methods"
    args.config = args.config or ROOT / ("training.comparison.json" if comparison else "training.json")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.na_schedule:
        cfg["na_schedule"] = args.na_schedule
    if comparison and not (cfg.get("paired_author_online") and cfg.get("na_shared_prefix") and
                           cfg.get("salista", {}).get("bounds", {}).get("mode") == "online_gaussian"):
        parser.error("six-methods needs the paired online, shared prefix, and online_gaussian comparison settings")
    args.seed = cfg["seed"] if args.seed is None else args.seed
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    args.out = args.out or (Path(f"runs/six-methods-seed-{args.seed}") if comparison else DEFAULT_RUNS)
    jobs = grid_jobs(args.preset)
    for method, mode in dict.fromkeys((m, mode) for m, mode, _ in jobs):
        stages(mode, cfg, method)
    print(f"{args.preset}: {len(jobs)} runs, depth {cfg['depth']}, seed {args.seed}, output {args.out}", flush=True)
    if args.dry_run:
        for method, mode, snr in jobs:
            print(method, mode, level_key(snr), "dB")
        return
    main_data, clean_data, out = args.data_main.resolve(), args.data_noiseless.resolve(), args.out.resolve()
    data = RecoveryData(main_data)
    levels = list(dict.fromkeys(snr for _, _, snr in jobs))
    for snr in levels:
        if not math.isinf(snr):
            matched_level(data, snr)
    if math.inf in levels:
        if not clean_data.exists():
            generate_noiseless(main_data, clean_data)
        check_paired_signals(data, RecoveryData(clean_data))
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ids = devices(args.device)
    slots = Queue()
    for gpu in ids:
        slots.put(gpu)

    def run(job):
        method, mode, snr = job
        gpu = slots.get()
        try:
            key = level_key(snr)
            name = f"{method}-{mode}-snr{key}-seed{args.seed}"
            common = ["--model", method, "--mode", mode, "--snr", key, "--seed", str(args.seed),
                      "--device", args.device, "--data", str(clean_data if key == "inf" else main_data),
                      "--config", str(args.config.resolve()), "--out", str(out)]
            if args.na_schedule and method == "NA-ALISTA":
                common += ["--na-schedule", args.na_schedule]
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            if args.device == "cuda":
                env.update(CUDA_VISIBLE_DEVICES=gpu)
            env.setdefault("OMP_NUM_THREADS", "2")
            env.setdefault("MKL_NUM_THREADS", "2")
            print(f"Starting {name} on {gpu}", flush=True)
            with (log_dir / (name + ".log")).open("a", encoding="utf-8") as log:
                for action in ("train", "test"):
                    cmd = [sys.executable, str(ROOT / "train.py"), action, *common]
                    print(cmd, file=log, flush=True)
                    result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        return name
            return None
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=len(ids)) as pool:
        failures = [name for name in pool.map(run, jobs) if name is not None]
    if failures:
        raise SystemExit("Failed runs (see logs): " + ", ".join(failures))
    write_test_summary(out, jobs, args.seed)
    plot = [sys.executable, str(ROOT / "plot_single_snr.py"), "--runs", str(out), "--data-main", str(main_data),
            "--data-noiseless", str(clean_data), "--device", args.device, "--seed", str(args.seed),
            "--snrs", *map(level_key, levels), "--methods", *dict.fromkeys(m for m, _, _ in jobs)]
    if comparison:
        plot.append("--include-na-prefix")
    subprocess.run(plot, check=True)
    print(f"Completed {len(jobs)} runs; figures in {out / 'figures'}")


if __name__ == "__main__":
    main()
