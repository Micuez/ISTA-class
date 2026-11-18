"""Evaluate all depths from one frozen final checkpoint per model, mode and SNR."""

import argparse
import csv
import hashlib
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import RecoveryData, level_key
from evaluation import tensors
from generate import sha256
from models import BASELINE_METHODS, METHODS
from train import DEFAULT_RUNS, MODES, load_checkpoint, matched_level, supported_modes
from verify_data import check_paired_signals


SNRS = (40, 20, 10, 5, float("inf"))
COLORS = {"EBT-LISTA": "#0072B2", "HLISTA": "#D98600", "ALISTA": "#008364",
          "ELISTA": "#7656A4", "NA-ALISTA": "#C45112", "SALISTA": "#394D67"}


@torch.inference_mode()
def fixed_depth_metrics(model, data, snr, depth, batch, device, split):
    model.eval()
    errors = torch.zeros(depth, dtype=torch.float64, device=device)
    hits = torch.zeros(depth, dtype=torch.int64, device=device)
    energy = torch.zeros((), dtype=torch.float64, device=device)
    k = data.config["nonzeros"]
    count, samples = 0, []
    for y, x in tensors(data, split, batch, device, level=snr):
        target = x.double()
        signal_energy = target.square().sum(dim=1)
        energy += signal_energy.sum()
        count += len(x)
        # All depths come from this one unroll, including the same recurrent state.
        for idx, estimate in enumerate(model(y, depth, return_path=True)):
            per_signal = (estimate.double() - target).square().sum(dim=1)
            errors[idx] += per_signal.sum()
            indices = estimate.abs().argsort(dim=1, descending=True, stable=True)[:, :k]
            hits[idx] += x.gather(1, indices).count_nonzero()
        samples.append((10 * torch.log10(per_signal / signal_energy)).cpu().numpy())
    ratios = (errors / energy).cpu().numpy()
    if not np.isfinite(ratios).all():
        raise ValueError("Non-finite fixed-checkpoint NMSE")
    with np.errstate(divide="ignore"):
        db = 10 * np.log10(ratios)
    rows = [{"layer": idx + 1, "signals": count, "nmse": float(ratio), "nmse_db": float(db[idx]),
             "top_k_support_f1": hits[idx].item() / (count * k)} for idx, ratio in enumerate(ratios)]
    samples = np.concatenate(samples)
    if np.isnan(samples).any() or np.isposinf(samples).any():
        raise ValueError("Undefined per-signal NMSE; check target energy and predictions")
    return rows, samples


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def title(snr):
    return "Noiseless" if math.isinf(snr) else f"{level_key(snr)} dB"


def draw_curves(rows, levels, modes, methods, split, directory):
    fig, axes = plt.subplots(len(modes), len(levels), squeeze=False, sharex=True,
                             figsize=(max(5, 3.35 * len(levels)), 3.4 * len(modes) + 0.6))
    handles = {}
    depth = max(item["layer"] for item in rows)
    ticks = sorted({1, depth, *range(4, depth + 1, 4)})
    for row_idx, mode in enumerate(modes):
        for col, snr in enumerate(levels):
            axis = axes[row_idx, col]
            for method in methods:
                series = [v for v in rows if (v["model"], v["mode"], v["snr_db"]) ==
                          (method, mode, level_key(snr))]
                if series:
                    layers = [v["layer"] for v in series]
                    db = np.array([v["nmse_db"] for v in series])
                    line, = axis.plot(layers, np.ma.masked_invalid(db), ".-", color=COLORS[method],
                                      linewidth=1.6, label=method)
                    handles[method] = line
                    if np.isneginf(db).any():
                        for layer in np.asarray(layers)[np.isneginf(db)]:
                            axis.text(layer, 0.015, "−∞", transform=axis.get_xaxis_transform(),
                                      ha="center", color=COLORS[method], fontsize=8)
            axis.set_title(title(snr))
            axis.set_xticks(ticks)
            if col == 0:
                label = "Layerwise" if mode == "layerwise" else "End-to-end"
                axis.set_ylabel(f"{label}\n{split.capitalize()} NMSE (dB)")
            if row_idx == len(modes) - 1:
                axis.set_xlabel("Unfolded depth")
            axis.grid(alpha=0.22)
    fig.legend([handles[m] for m in methods if m in handles], [m for m in methods if m in handles],
               loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"fixed_depth_{split}_nmse.{ext}", dpi=160)
    plt.close(fig)


def draw_boxes(values, levels, methods, split, directory):
    fig, axes = plt.subplots(1, len(levels), squeeze=False, figsize=(max(5, len(levels) * 3.3), 4))
    for axis, snr in zip(axes[0], levels):
        groups = [values[(m, level_key(snr))] for m in methods]
        finite = [group[np.isfinite(group)] for group in groups]
        boxes = axis.boxplot(finite, patch_artist=True, widths=0.55, medianprops={"color": "black"})
        for patch, method in zip(boxes["boxes"], methods):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.78)
        labels = [method if len(group) == len(shown) else f"{method}\n{len(group) - len(shown)} exact"
                  for method, group, shown in zip(methods, groups, finite)]
        axis.set_xticks(range(1, len(methods) + 1), labels=labels, rotation=28, ha="right")
        axis.set_title(title(snr))
        axis.grid(axis="y", alpha=0.22)
    axes[0, 0].set_ylabel("End-to-end sample NMSE (dB)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"end_to_end_{split}_nmse_boxplot.{ext}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--data-main", type=Path, default=Path("data/main"))
    parser.add_argument("--data-noiseless", type=Path, default=Path("data/noiseless"))
    parser.add_argument("--snrs", type=float, nargs="+", default=SNRS, choices=SNRS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=BASELINE_METHODS)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=("layerwise", "end_to_end"))
    parser.add_argument("--include-na-prefix", "--include-legacy-na", dest="include_na_layerwise", action="store_true",
                        help="also read the NA-ALISTA layerwise checkpoints from this run directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    rows, samples, quartiles, boxes = [], [], [], {}
    datasets = {}
    for snr in args.snrs:
        root = args.data_noiseless if math.isinf(snr) else args.data_main
        if root not in datasets:
            datasets[root] = RecoveryData(root), hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    if len(datasets) == 2:
        check_paired_signals(datasets[args.data_main][0], datasets[args.data_noiseless][0])
    for snr in args.snrs:
        key = level_key(snr)
        root = args.data_noiseless if math.isinf(snr) else args.data_main
        data, fingerprint = datasets[root]
        level = matched_level(data, snr)
        for method in args.methods:
            for mode in args.modes:
                if mode not in supported_modes(method, args.include_na_layerwise):
                    continue
                path = args.runs / method / mode / ("snr-" + key) / f"seed-{args.seed}"
                run, model = load_checkpoint(data, path, device, fingerprint)
                cfg = run["training"]
                if (run["method"], run["mode"], cfg["seed"], level_key(cfg["snr_db"])) != (
                        method, mode, args.seed, key):
                    raise ValueError(f"Checkpoint identity differs from {path}")
                metrics, scores = fixed_depth_metrics(model, data, level, cfg["depth"],
                                                       cfg["batch_size"], device, args.split)
                metadata = {"model": method, "mode": mode, "snr_db": key, "seed": args.seed,
                            "split": args.split, "checkpoint_sha256": sha256(path / "best.pt")}
                rows.extend({**metadata, **item} for item in metrics)
                if mode == "end_to_end":
                    boxes[method, key] = scores
                    samples.extend({"model": method, "snr_db": key, "sample": i, "nmse_db": float(v)}
                                   for i, v in enumerate(scores))
                    finite = scores[np.isfinite(scores)]
                    q1, med, q3 = np.percentile(finite, [25, 50, 75]) if len(finite) else (None, None, None)
                    quartiles.append({"model": method, "snr_db": key, "signals": len(scores),
                                      "finite_db": len(finite), "exact_recoveries": len(scores) - len(finite),
                                      "q1_nmse_db": q1, "median_nmse_db": med, "q3_nmse_db": q3})
    if not rows:
        parser.error("The selection has no trained model/mode combinations")
    figures = args.runs / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(args.runs / f"fixed_depth_{args.split}_nmse.csv", rows)
    drawn_modes = [mode for mode in args.modes if any(row["mode"] == mode for row in rows)]
    draw_curves(rows, args.snrs, drawn_modes, args.methods, args.split, figures)
    if boxes:
        write_csv(args.runs / f"end_to_end_{args.split}_samples.csv", samples)
        write_csv(args.runs / f"end_to_end_{args.split}_boxplot.csv", quartiles)
        draw_boxes(boxes, args.snrs, args.methods, args.split, figures)
    print(f"{len(rows)} fixed-checkpoint depth scores, figures in {figures}")


if __name__ == "__main__":
    main()
