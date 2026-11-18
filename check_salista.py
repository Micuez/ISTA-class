"""Compare the analytic eligibility condition and the actual first SALISTA output."""

import argparse
import json
from pathlib import Path

import torch

from dataset import RecoveryData
from models import build_model


@torch.no_grad()
def check_first_layer(data, config, level):
    model = build_model("SALISTA", data, config["depth"], dict(config, snr_db=level))
    cert = model.certificate
    gram_available = bool(cert.margin > 0)
    observations, _ = data.arrays("train", level)
    batch = config["batch_size"]
    eligible, active, max_direction, min_floor = 0, 0, 0., float("inf")
    for start in range(0, len(observations), batch):
        y = torch.tensor(observations[start:start + batch])
        c = y @ model.W
        budget = c.new_full((len(c), 1), model.initial_l1_bound)
        _, psi = cert.coordinate_bounds(c, budget, model.noise_l2_bound, gram_available)
        floor_per_step = psi + model.noise_l2_bound * cert.omega
        eligible += (c.abs() > floor_per_step).any(1).sum().item()
        active += model(y, 1).ne(0).any(1).sum().item()
        max_direction = max(max_direction, c.abs().max().item())
        min_floor = min(min_floor, floor_per_step.min().item())
    return dict(snr_db=level, observations=len(observations), strict_eligible_signals=eligible,
                actual_nonzero_outputs=active, stall_threshold_ratio=model.stall_threshold_ratio,
                max_abs_direction=max_direction, min_floor_per_step=min_floor,
                gram_available=gram_available, q_row=cert.q.max().item(), mu=cert.mu.item(),
                initial_l1_bound=model.initial_l1_bound, noise_l2_bound=model.noise_l2_bound)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/main"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("training.salista.json"))
    args = parser.parse_args()
    data = RecoveryData(args.data)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for level in data.levels:
        torch.manual_seed(config["seed"])
        print(json.dumps(check_first_layer(data, config, level), allow_nan=False))
