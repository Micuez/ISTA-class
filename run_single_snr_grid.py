"""Train four models at four SNRs with both schedules, one job per GPU."""

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = ("EBT-LISTA", "HLISTA", "ALISTA", "NA-ALISTA")
MODES = ("layerwise", "end_to_end")
SNRS = (40, 20, 10, 5)
OUT = ROOT / "runs" / "single-snr"


def gpu_ids():
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
    return [line.strip() for line in text.splitlines() if line.strip()]


def main():
    gpus = gpu_ids()
    if not gpus:
        raise SystemExit("No GPU is visible")
    jobs = [(model, mode, snr) for model in MODELS for snr in SNRS for mode in MODES]
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    slots = queue.Queue()
    for gpu in gpus:
        slots.put(gpu)
    failures = []
    lock = threading.Lock()

    def run(job):
        model, mode, snr = job
        gpu = slots.get()
        name = "{}-{}-snr{}".format(model, mode, snr)
        log_path = log_dir / (name + ".log")
        common = [
            "--model", model, "--mode", mode, "--snr", str(snr),
            "--seed", "0", "--device", "cuda",
            "--data", "data/main", "--config", "training.json",
            "--out", "runs/single-snr",
        ]
        commands = [
            [sys.executable, "train.py", "train", *common],
            [sys.executable, "train.py", "test", *common],
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["PYTHONUNBUFFERED"] = "1"
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        print("start gpu={} {}".format(gpu, name), flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("gpu={}\n".format(gpu))
            completed = subprocess.CompletedProcess(commands[0], 0)
            for command in commands:
                log.write("command={}\n".format(" ".join(command)))
                log.flush()
                completed = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode != 0:
                    break
        if completed.returncode != 0:
            with lock:
                failures.append((name, gpu, completed.returncode))
            print("fail gpu={} {} code={}".format(gpu, name, completed.returncode), flush=True)
        else:
            print("done gpu={} {}".format(gpu, name), flush=True)
        slots.put(gpu)

    threads = [threading.Thread(target=run, args=(job,)) for job in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        print("failed jobs:", failures, flush=True)
        raise SystemExit(1)
    print("all {} jobs finished on GPUs {}".format(len(jobs), ",".join(gpus)), flush=True)


if __name__ == "__main__":
    main()
