"""Plot layer-wise validation NMSE and end-to-end per-sample NMSE."""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import RecoveryData, level_key
from evaluation import tensors
from models import METHODS, build_model


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "single-snr"
FIGURES = OUT / "figures"
SNRS = (40, 20, 10, 5, float("inf"))
COLORS = {
    "EBT-LISTA": "#0072B2",
    "HLISTA": "#E69F00",
    "ALISTA": "#009E73",
    "NA-ALISTA": "#D55E00",
}


def use_cjk_font():
    from matplotlib import font_manager
    for path in (
        ROOT / "fonts" / "wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ):
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def run_dir(method, mode, snr):
    return OUT / method / mode / ("snr-" + level_key(snr)) / "seed-0"


def snr_title(snr):
    if math.isinf(float(snr)):
        return "SNR = 无穷"
    return "SNR = {} dB".format(snr)


def same_snr(left, right):
    left, right = float(left), float(right)
    return math.isinf(left) and math.isinf(right) or left == right


def panel_axes(count):
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    flat = list(axes.ravel())
    for axis in flat[count:]:
        axis.axis("off")
    return figure, flat[:count]


def dataset_for(snr):
    root = ROOT / "data" / ("noiseless" if math.isinf(float(snr)) else "main")
    data = RecoveryData(root)
    for level in data.levels:
        if same_snr(level, snr):
            return data, level
    raise ValueError("SNR is not in {}: {}".format(root, snr))


def metric_db(value):
    if value == "-inf":
        return -np.inf
    return float(value)


def layerwise_table():
    rows = []
    for method in METHODS:
        for snr in SNRS:
            path = run_dir(method, "layerwise", snr)
            run = json.loads((path / "run.json").read_text(encoding="utf-8"))
            history = {
                record["epoch"]: record
                for record in map(json.loads, (path / "history.jsonl").read_text(encoding="utf-8").splitlines())
            }
            key = level_key(snr)
            for stage in run["stages"]:
                record = history[stage["best_epoch"]]
                metrics = record["val"][key]
                rows.append({
                    "model": method,
                    "snr_db": snr,
                    "layer": stage["depth"],
                    "best_epoch": stage["best_epoch"],
                    "epochs_run": stage["epochs_run"],
                    "updates": stage["updates"],
                    "stop_reason": stage["stop_reason"],
                    "val_nmse": metrics["nmse"],
                    "val_nmse_db": metric_db(metrics["nmse_db"]),
                    "val_top_k_support_f1": metrics["top_k_support_f1"],
                })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_layerwise(rows):
    figure, axes = panel_axes(len(SNRS))
    for axis, snr in zip(axes, SNRS):
        for method in METHODS:
            selected = [row for row in rows if row["model"] == method and same_snr(row["snr_db"], snr)]
            selected.sort(key=lambda row: row["layer"])
            axis.plot(
                [row["layer"] for row in selected],
                [row["val_nmse_db"] for row in selected],
                marker="o", markersize=3.5, linewidth=1.4,
                color=COLORS[method], label=method,
            )
        axis.set_title(snr_title(snr))
        axis.set_xlabel("层数")
        axis.set_ylabel("恢复误差 NMSE (dB)")
        axis.set_xticks(range(1, 17))
        axis.grid(True, linewidth=0.4, alpha=0.5)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / "layerwise_val_nmse.png", dpi=160)
    figure.savefig(FIGURES / "layerwise_val_nmse.pdf")
    plt.close(figure)
    for snr in SNRS:
        single, axis = plt.subplots(figsize=(7.2, 4.4))
        for method in METHODS:
            selected = [row for row in rows if row["model"] == method and same_snr(row["snr_db"], snr)]
            selected.sort(key=lambda row: row["layer"])
            axis.plot(
                [row["layer"] for row in selected],
                [row["val_nmse_db"] for row in selected],
                marker="o", markersize=3.5, linewidth=1.4,
                color=COLORS[method], label=method,
            )
        axis.set_title(snr_title(snr))
        axis.set_xlabel("层数")
        axis.set_ylabel("恢复误差 NMSE (dB)")
        axis.set_xticks(range(1, 17))
        axis.grid(True, linewidth=0.4, alpha=0.5)
        axis.legend(frameon=False, fontsize=8)
        single.tight_layout()
        stem = "layerwise_snr_{}".format(level_key(snr))
        single.savefig(FIGURES / (stem + ".png"), dpi=160)
        single.savefig(FIGURES / (stem + ".pdf"))
        plt.close(single)


@torch.no_grad()
def per_sample_nmse_db(model, data, snr, depth, batch_size, device):
    values = []
    for observations, signals in tensors(data, "val", batch_size, device, snr):
        prediction = model(observations, depth)
        error = (prediction.double() - signals.double()).square().sum(dim=1)
        energy = signals.double().square().sum(dim=1).clamp_min(torch.finfo(torch.float64).tiny)
        values.append((10 * torch.log10(error / energy)).cpu().numpy())
    return np.concatenate(values)


def plot_end_to_end(device):
    batch_size = json.loads((ROOT / "training.json").read_text(encoding="utf-8"))["batch_size"]
    samples = []
    summaries = []
    for method in METHODS:
        for snr in SNRS:
            data, level = dataset_for(snr)
            path = run_dir(method, "end_to_end", snr)
            state = torch.load(path / "best.pt", map_location=device, weights_only=True)
            model = build_model(method, data, 16).to(device)
            model.load_state_dict(state)
            model.eval()
            scores = per_sample_nmse_db(model, data, level, 16, batch_size, device)
            for value in scores:
                samples.append({"model": method, "snr_db": snr, "val_nmse_db": float(value)})
            summaries.append({
                "model": method,
                "snr_db": snr,
                "signals": int(scores.size),
                "median_nmse_db": float(np.median(scores)),
                "q1_nmse_db": float(np.percentile(scores, 25)),
                "q3_nmse_db": float(np.percentile(scores, 75)),
            })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    figure, axes = panel_axes(len(SNRS))
    for axis, snr in zip(axes, SNRS):
        groups = [
            [row["val_nmse_db"] for row in samples if row["model"] == method and same_snr(row["snr_db"], snr)]
            for method in METHODS
        ]
        drawn = axis.boxplot(
            groups, patch_artist=True, widths=0.6,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch, method in zip(drawn["boxes"], METHODS):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.85)
        axis.set_title(snr_title(snr))
        axis.set_xticks(range(1, len(METHODS) + 1))
        axis.set_xticklabels(METHODS, rotation=15)
        axis.set_ylabel("恢复误差 NMSE (dB)")
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    figure.tight_layout()
    figure.savefig(FIGURES / "end_to_end_val_nmse_boxplot.png", dpi=160)
    figure.savefig(FIGURES / "end_to_end_val_nmse_boxplot.pdf")
    plt.close(figure)
    return samples, summaries


def main():
    use_cjk_font()
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = layerwise_table()
    write_csv(OUT / "layerwise_val_nmse.csv", rows)
    plot_layerwise(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, summaries = plot_end_to_end(device)
    write_csv(OUT / "end_to_end_val_samples.csv", samples)
    write_csv(OUT / "end_to_end_val_boxplot.csv", summaries)
    print("wrote", FIGURES)


if __name__ == "__main__":
    main()
