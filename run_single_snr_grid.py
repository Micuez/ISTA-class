"""Train all 45 single-SNR models, test them and draw fixed-checkpoint depth curves."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys

from dataset import RecoveryData, level_key
from generate import generate_noiseless
from models import METHODS
from train import DEFAULT_RUNS, supported_modes
from verify_data import check_paired_signals


ROOT = Path(__file__).resolve().parent
SNRS = (40, 20, 10, 5, float("inf"))


def grid_jobs():
    return [(method, mode, snr) for snr in SNRS for method in METHODS
            for mode in supported_modes(method)]


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
    parser.add_argument("--out", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--config", type=Path, default=ROOT / "training.json")
    parser.add_argument("--data-main", type=Path, default=Path("data/main"))
    parser.add_argument("--data-noiseless", type=Path, default=Path("data/noiseless"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--na-schedule", choices=("experiment", "author"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    main_data, clean_data, out = args.data_main.resolve(), args.data_noiseless.resolve(), args.out.resolve()
    if not clean_data.exists():
        generate_noiseless(main_data, clean_data)
    check_paired_signals(RecoveryData(main_data), RecoveryData(clean_data))
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
        failures = [name for name in pool.map(run, grid_jobs()) if name is not None]
    if failures:
        raise SystemExit("Failed runs (see logs): " + ", ".join(failures))
    subprocess.run([sys.executable, str(ROOT / "plot_single_snr.py"), "--runs", str(out),
                    "--data-main", str(main_data), "--data-noiseless", str(clean_data),
                    "--device", args.device, "--seed", str(args.seed)], check=True)
    print(f"Completed {len(grid_jobs())} runs; figures in {out / 'figures'}")


if __name__ == "__main__":
    main()
