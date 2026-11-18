import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset import RecoveryData
from experiment_inputs import column_batches, torch_batches
from generate import generate
from models import METHODS, NeuralALISTA, build_model
from train import evaluate, fit, stages, test_run, trainable_parameters
from verify_data import verify


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.config = dict(seed=28, measurements=9, signal_dimension=18, nonzeros=3,
                          snr_db=[40, 20, 10, 5], clean_signals=dict(train=9, val=4, test=5))
        cls.manifest = generate(cls.config, cls.root / "data")
        cls.data = RecoveryData(cls.root / "data")

    @classmethod
    def tearDownClass(cls):
        cls.data = None
        cls.tmp.cleanup()

    def test_dataset_reproducibility_and_split_isolation(self):
        verify(self.data.root)
        repeat = generate(self.config, self.root / "repeat")
        self.assertEqual(self.manifest["sha256"], repeat["sha256"])
        cfg = dict(self.config, clean_signals=dict(train=10, val=4, test=5))
        changed = generate(cfg, self.root / "changed")
        for name, checksum in repeat["sha256"].items():
            if not name.startswith("train/"):
                self.assertEqual(checksum, changed["sha256"][name])

    def test_all_measurements_seen_once_per_epoch(self):
        expected = set()
        for db in self.data.levels:
            y, x = self.data.arrays("train", db)
            expected.update((a.tobytes(), b.tobytes()) for a, b in zip(y, x))
        orders = []
        for epoch in (0, 1):
            seen = [(a.tobytes(), b.tobytes())
                    for y, x in self.data.batches("train", 7, epoch=epoch) for a, b in zip(y, x)]
            self.assertEqual(len(seen), len(expected))
            self.assertEqual(set(seen), expected)
            orders.append(seen)
        self.assertNotEqual(*orders)

    def test_framework_input_layouts_contain_identical_observations(self):
        for split in ("train", "val", "test"):
            level = 5 if split == "test" else None
            columns = column_batches(self.data, split, 7, level)
            rows = torch_batches(self.data, split, 7, level)
            for (y_col, x_col), (y_row, x_row) in zip(columns, rows, strict=True):
                np.testing.assert_array_equal(y_col.T, y_row.numpy())
                np.testing.assert_array_equal(x_col.T, x_row.numpy())

    def test_threshold_updates_against_numpy(self):
        # Compare row-wise PyTorch against independent column-wise arithmetic.
        y, _ = next(self.data.batches("train", 6))
        y = y.T
        for name in ("ALISTA", "EBT-LISTA"):
            net = build_model(name, self.data, 2)
            x = np.zeros((self.data.A.shape[1], y.shape[1]), dtype=np.float32)
            for layer in net.layers:
                layer.rank = 2
                residual = y - self.data.A @ x
                if name == "ALISTA":
                    v = self.data.W.T @ residual
                    z = x + layer.alpha.item() * v
                    theta = layer.theta.item()
                else:
                    v = layer.weight.detach().numpy().T @ residual
                    z = x + v
                    theta = layer.theta.item() * np.linalg.norm(v, axis=0) / 50 + layer.bias.item()
                cutoff = np.sort(np.abs(z), axis=0)[-3]
                keep = (np.abs(z) > cutoff) & (np.abs(z) > theta)
                x = np.where(keep, z, np.sign(z) * np.maximum(np.abs(z) - theta, 0))
            with self.subTest(method=name):
                actual = net(torch.from_numpy(y.T), 2).detach().numpy()
                np.testing.assert_allclose(actual, x.T, rtol=5e-5, atol=1e-6)

    def test_batch_independence_and_layer_freezing(self):
        y, x = next(self.data.batches("train", 8))
        y, x = torch.from_numpy(y), torch.from_numpy(x)
        for method in METHODS:
            with self.subTest(method=method):
                model = build_model(method, self.data, 2)
                if isinstance(model, NeuralALISTA):
                    model.calibrate(y)
                torch.testing.assert_close(model(y, 2), torch.cat([model(v[None], 2) for v in y]),
                                           rtol=4e-5, atol=3e-6)
                params = trainable_parameters(model, "layerwise", 2)
                before = {name: p.detach().clone() for name, p in model.named_parameters()}
                optimizer = torch.optim.Adam(params.values(), lr=0.002)
                (model(y, 2, last_step_grad=True) - x).square().mean().backward()
                optimizer.step()
                for name, param in model.named_parameters():
                    if name not in params:
                        self.assertIsNone(param.grad)
                        self.assertTrue(torch.equal(param, before[name]))
                self.assertTrue(any(not torch.equal(p, before[n]) for n, p in params.items()))

    def test_both_training_modes_and_checkpoint_evaluation(self):
        cfg = dict(depth=2, total_epochs=2, batch_size=8, learning_rate=0.001, seed=3)
        for method in METHODS:
            for mode in ("end_to_end", "layerwise"):
                with self.subTest(method=method, mode=mode):
                    dest = self.root / method / mode
                    fit(self.data, method, mode, cfg, "cpu", dest, "test-id")
                    self.assertFalse((dest / "test.json").exists())
                    log = json.loads((dest / "run.json").read_text())
                    self.assertEqual(log["updates"], 10)
                    self.assertEqual(len(log["stages"]), 1 if mode == "end_to_end" else 2)
                    test_run(self.data, dest, "cpu", "test-id")
                    metrics = json.loads((dest / "test.json").read_text())["snr_db"]
                    self.assertEqual(set(metrics), {"40", "20", "10", "5"})
                    self.assertTrue(all(np.isfinite(m["nmse"]) and m["signals"] == 5
                                        for m in metrics.values()))
                    with self.assertRaisesRegex(ValueError, "dataset differs"):
                        test_run(self.data, dest, "cpu", "different-dataset")

    def test_score_is_global_energy_ratio(self):
        class Zero(torch.nn.Module):
            def forward(self, y, depth):
                return y.new_zeros((len(y), 18))
        result, mean = evaluate(Zero(), self.data, "test", 1, 3, "cpu")
        self.assertEqual(mean, 1.0)
        self.assertTrue(all(row["nmse_db"] == 0.0 for row in result.values()))

    def test_formal_budget_is_per_stage(self):
        config = json.loads(Path(__file__).with_name("training.json").read_text())
        self.assertEqual(config["early_stopping_patience"], 5)
        self.assertEqual(stages("end_to_end", config), [(16, 100000)])
        self.assertEqual(stages("layerwise", config), [(i, 100000) for i in range(1, 17)])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            stages("end_to_end", dict(config, total_epochs=1))

    def test_patience_counts_consecutive_epochs_and_restores_best(self):
        cfg = dict(depth=2, max_epochs_per_stage=12, early_stopping_patience=5,
                   batch_size=8, learning_rate=0.001, seed=3)
        # A tie counts as a miss. Epoch 3 resets the wait for end-to-end training.
        scores = {"end_to_end": {2: [1, 1, 0.9, 0.9, 1.1, 1, 0.9, 1]},
                  "layerwise": {1: [1] * 6, 2: [2, 1.9, 2, 1.9, 2, 2, 1.9]}}
        for mode, traces in scores.items():
            dest = self.root / "patience" / mode
            seen = {depth: [] for depth in traces}

            def validation(model, data, split, depth, batch_size, device):
                self.assertEqual(split, "val")
                index = len(seen[depth])
                seen[depth].append({name: tensor.detach().clone()
                                    for name, tensor in model.state_dict().items()})
                return {}, traces[depth][index]

            with self.subTest(mode=mode), patch("train.evaluate", new=validation):
                fit(self.data, "ALISTA", mode, cfg, "cpu", dest, "early-stop-test")
                run = json.loads((dest / "run.json").read_text())
                self.assertEqual(run["epochs"], sum(map(len, traces.values())))
                self.assertEqual(run["updates"], 5 * run["epochs"])
                for record in run["stages"]:
                    curve = traces[record["depth"]]
                    first_best = curve.index(min(curve))
                    self.assertEqual(record["stop_reason"], "validation_patience")
                    self.assertEqual(record["epochs_run"], len(curve))
                    self.assertEqual(record["best_epoch"], record["epochs"][0] + first_best)
                best_idx = traces[2].index(min(traces[2]))
                saved = torch.load(dest / "best.pt", weights_only=True)
                for name, value in saved.items():
                    torch.testing.assert_close(value, seen[2][best_idx][name], rtol=0, atol=0)
                history = [json.loads(s) for s in (dest / "history.jsonl").read_text().splitlines()]
                self.assertTrue(all(r["epochs_without_improvement"] == 0 for r in history
                                    if r["stage_epoch"] == 1))
                self.assertEqual(history[-1]["epochs_without_improvement"], 5)

    def test_improving_validation_reaches_epoch_cap(self):
        cfg = dict(depth=1, max_epochs_per_stage=3, early_stopping_patience=5,
                   batch_size=8, learning_rate=0.001, seed=3)
        dest = self.root / "cap"
        scores = iter([1.0, 0.9, 0.8])
        with patch("train.evaluate", new=lambda *args: ({}, next(scores))):
            fit(self.data, "ALISTA", "end_to_end", cfg, "cpu", dest, "cap-test")
        record = json.loads((dest / "run.json").read_text())["stages"][0]
        self.assertEqual((record["epochs_run"], record["best_epoch"], record["stop_reason"]),
                         (3, 3, "epoch_limit"))

    def test_na_forward_matches_author_for_identical_weights(self):
        source = Path(__file__).parent / "upstream" / "NA-ALISTA"
        if not (source / "utils" / "algorithms.py").exists():
            self.skipTest("Optional author checkout is not prepared")
        sys.path.insert(0, str(source))
        try:
            from utils.algorithms import NA_ALISTA, device
        finally:
            sys.path.pop(0)
        ours = build_model("NA-ALISTA", self.data, 3).to(device)
        m, n = self.data.A.shape
        ref = NA_ALISTA(m, n, 3, self.data.A.copy(), self.data.W.copy(), 3, ours.counts).to(device)
        ours.regressor.load_state_dict(ref.regressor.state_dict())
        y, _ = next(self.data.batches("train", 8))
        y = torch.from_numpy(y).to(device)
        ours.calibrate(y)
        with torch.no_grad():
            expected, _, _ = ref(y, None)
            torch.testing.assert_close(ours(y, 3), expected, rtol=4e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
