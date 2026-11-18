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
from generate import generate, generate_noiseless
from models import BASELINE_METHODS, NeuralALISTA, build_model, multistage_threshold
from na_training import author_batches
from plot_single_snr import fixed_depth_metrics
from run_single_snr_grid import grid_jobs
from train import evaluate, fit, load_checkpoint, stages, supported_modes, tensors, test_run, trainable_parameters
from verify_data import check_paired_signals, verify


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
        for method in BASELINE_METHODS:
            with self.subTest(method=method):
                model = build_model(method, self.data, 2)
                if isinstance(model, NeuralALISTA):
                    model.calibrate(y)
                torch.testing.assert_close(model(y, 2), torch.cat([model(v[None], 2) for v in y]),
                                           rtol=4e-5, atol=3e-6)
                new_only = method != "NA-ALISTA"
                params = trainable_parameters(model, "new_layer" if new_only else "end_to_end", 2)
                before = {name: p.detach().clone() for name, p in model.named_parameters()}
                optimizer = torch.optim.Adam(params.values(), lr=0.002)
                (model(y, 2, last_step_grad=new_only) - x).square().mean().backward()
                optimizer.step()
                for name, param in model.named_parameters():
                    if name not in params:
                        self.assertIsNone(param.grad)
                        self.assertTrue(torch.equal(param, before[name]))
                self.assertTrue(any(not torch.equal(p, before[n]) for n, p in params.items()))

    def test_both_training_modes_and_checkpoint_evaluation(self):
        cfg = dict(depth=2, max_epochs_per_layer=3, max_epochs_end_to_end=2,
                   early_stopping_patience=50, batch_size=8, learning_rate=0.001, seed=3,
                   snr_db=5, na_batch_size=4, na_samples_per_epoch=12)
        for method in BASELINE_METHODS:
            for mode in supported_modes(method):
                with self.subTest(method=method, mode=mode):
                    dest = self.root / method / mode
                    fit(self.data, method, mode, cfg, "cpu", dest, "test-id")
                    log = json.loads((dest / "run.json").read_text())
                    expected = 12 if mode == "layerwise" else (6 if method == "NA-ALISTA" else 4)
                    self.assertEqual(log["updates"], expected)
                    test_run(self.data, dest, "cpu", "test-id")
                    metrics = json.loads((dest / "test.json").read_text())["snr_db"]
                    self.assertEqual(set(metrics), {"5"})
                    run, model = load_checkpoint(self.data, dest, "cpu", "test-id")
                    rows, _ = fixed_depth_metrics(model, self.data, 5, 2, 8, "cpu", "test")
                    self.assertAlmostEqual(rows[-1]["nmse"], metrics["5"]["nmse"], places=11)
                    with self.assertRaisesRegex(ValueError, "dataset differs"):
                        test_run(self.data, dest, "cpu", "different-dataset")
        self.assertEqual(tuple(self.data.levels), (40, 20, 10, 5))

    def test_score_is_global_energy_ratio(self):
        class Zero(torch.nn.Module):
            def forward(self, y, depth):
                return y.new_zeros((len(y), 18))
        result, mean = evaluate(Zero(), self.data, "test", 1, 3, "cpu")
        self.assertEqual(mean, 1.0)
        self.assertTrue(all(row["nmse_db"] == 0.0 for row in result.values()))

    def test_formal_budget_and_grid(self):
        config = json.loads(Path(__file__).with_name("training.json").read_text())
        self.assertEqual(config["early_stopping_patience"], 50)
        self.assertEqual(stages("end_to_end", config)[0].epochs, 100000)
        phases = stages("layerwise", config)
        self.assertEqual(len(phases), 48)
        for d in range(1, 17):
            group = [p for p in phases if p.depth == d]
            self.assertEqual([p.name for p in group], ["new_layer", "joint_1", "joint_2"])
            self.assertEqual([p.epochs for p in group], [334, 333, 333])
            np.testing.assert_allclose([p.lr for p in group], [0.0005, 0.0001, 0.00001])
        self.assertEqual(len(set(grid_jobs())), 45)
        self.assertTrue(all(mode == "end_to_end" for m, mode, _ in grid_jobs() if m == "NA-ALISTA"))
        self.assertEqual(stages("end_to_end", dict(config, na_schedule="author"), "NA-ALISTA")[0].epochs, 420)
        with self.assertRaisesRegex(ValueError, "supports"):
            stages("layerwise", config, "NA-ALISTA")

    def test_fifty_misses_and_best_weights(self):
        cfg = dict(depth=1, max_epochs_per_layer=3, max_epochs_end_to_end=60,
                   early_stopping_patience=50, batch_size=8, learning_rate=0.001, seed=3, snr_db=5)
        trace = iter([1.0, 1.0, 0.9] + [0.9, 1.0] * 25)
        seen = []
        def validation(model, data, split, depth, batch_size, device):
            self.assertEqual(split, "val")
            seen.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            return {}, next(trace)
        out = self.root / "fifty"
        with patch("train.evaluate", new=validation):
            fit(self.data, "ALISTA", "end_to_end", cfg, "cpu", out, "id")
        stage, = json.loads((out / "run.json").read_text())["stages"]
        self.assertEqual((stage["epochs_run"], stage["best_epoch"], stage["stop_reason"]),
                         (53, 3, "validation_patience"))
        for name, value in torch.load(out / "best.pt", weights_only=True).items():
            torch.testing.assert_close(value, seen[2][name], rtol=0, atol=0)

    def test_joint_phases_keep_incoming_best_and_can_update_prefix(self):
        cfg = dict(depth=1, max_epochs_per_layer=9, max_epochs_end_to_end=3, early_stopping_patience=2,
                   batch_size=8, learning_rate=0.001, seed=3, snr_db=5)
        scores = iter([0.9, 0.8, 0.7, 0.9, 0.9, 0.6, 0.7, 0.8])
        out = self.root / "joint"
        with patch("train.evaluate", new=lambda *args: ({}, next(scores))):
            fit(self.data, "ALISTA", "layerwise", cfg, "cpu", out, "id")
        records = json.loads((out / "run.json").read_text())["stages"]
        self.assertEqual([s["epochs_run"] for s in records], [3, 2, 3])
        self.assertEqual([s["best_epoch"] for s in records], [3, 3, 6])
        self.assertEqual(records[0]["stop_reason"], "epoch_limit")
        y, x = next(tensors(self.data, "train", 8, "cpu", level=5))
        for method in ("ELISTA", "ALISTA", "EBT-LISTA", "HLISTA"):
            net = build_model(method, self.data, 3)
            params = trainable_parameters(net, "joint_1", 2)
            self.assertTrue(any(n.startswith("layers.0.") for n in params))
            self.assertFalse(any(n.startswith("layers.2.") for n in params))
            self.assertEqual("shared_weight" in params, method == "ELISTA")
            prefix_before = [p.detach().clone() for p in net.layers[0].parameters()]
            opt = torch.optim.Adam(params.values(), lr=1e-4)
            (net(y, 2) - x).square().mean().backward()
            opt.step()
            self.assertTrue(any(not torch.equal(p, v) for p, v in zip(net.layers[0].parameters(), prefix_before)))

    def test_elista_equation8_with_tied_matrix_and_mt(self):
        net = build_model("ELISTA", self.data, 3)
        y, _ = next(self.data.batches("train", 8, 5))
        w = net.shared_weight.detach().numpy()
        self.assertEqual(sum(p.numel() for p in net.parameters()), w.size + 6 * 3)
        x = np.zeros((len(y), w.shape[1]), dtype=np.float32)
        def mt(z, a, b):
            lo, hi = sorted([abs(a.item()), abs(b.item())])
            return np.sign(z) * np.where(abs(z) <= lo, 0, np.where(
                abs(z) >= hi, abs(z), hi * (abs(z) - lo) / (hi - lo)))
        paths = net(torch.from_numpy(y), 3, return_path=True)
        for i, block in enumerate(net.layers):
            mid = mt(x + block.alpha1.item() * ((y - x @ self.data.A.T) @ w), block.theta1, block.upper1)
            x = mt(x + block.alpha2.item() * ((y - mid @ self.data.A.T) @ w), block.theta2, block.upper2)
            np.testing.assert_allclose(paths[i].detach().numpy(), x, atol=2e-6)
        z = torch.tensor([-2., -1., -0.5, 0., .5, 1., 2.], requires_grad=True)
        a, b = torch.tensor(1., requires_grad=True), torch.tensor(1., requires_grad=True)
        value = multistage_threshold(z, a, b)
        value.sum().backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in (a, b, z)))

    def test_noiseless_sibling_and_na_noise(self):
        dest = self.root / "clean"
        generate_noiseless(self.data.root, dest)
        self.assertEqual(verify(dest)["val"]["inf"], "inf")
        sibling = RecoveryData(dest)
        check_paired_signals(self.data, sibling)
        for name in ("A.npy", "W.npy", "train/x.npy", "val/x.npy", "test/x.npy"):
            self.assertEqual(sibling.manifest["sha256"][name], self.manifest["sha256"][name])
        sibling.manifest["sha256"]["test/x.npy"] = "other-signals"
        with self.assertRaisesRegex(ValueError, "test/x.npy"):
            check_paired_signals(self.data, sibling)
        matrix = torch.tensor(self.data.A)
        x = torch.tensor(np.array(self.data.arrays("train", 5)[1]))
        y0, x0 = next(author_batches(x, matrix, "inf", 4, False))
        torch.testing.assert_close(y0, x0 @ matrix.T)
        torch.manual_seed(44)
        actual, _ = next(author_batches(x, matrix, 5, 4, False))
        torch.manual_seed(44)
        original_batch = next(iter(torch.utils.data.DataLoader(x, batch_size=4, drop_last=True)))
        original_y = original_batch @ matrix.T
        std = original_y.std(dim=1, keepdim=True) * 10 ** (-5 / 20)
        torch.testing.assert_close(actual, original_y + torch.normal(torch.zeros_like(original_y), std.expand_as(original_y)))
        self.assertEqual(len(list(author_batches(x, matrix, "inf", 4, False))), 2)
        self.assertEqual(list(author_batches(x[:2], matrix, 5, 4, False)), [])

    def test_na_forward_and_adam_update_match_author(self):
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
        y, x = (torch.from_numpy(v).to(device) for v in next(self.data.batches("train", 8)))
        ours.calibrate(y)
        expected, _, _ = ref(y, None)
        actual = ours(y, 3)
        torch.testing.assert_close(actual, expected, rtol=4e-5, atol=3e-6)
        for net, prediction in ((ours, actual), (ref, expected)):
            optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)
            (prediction - x).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
        for (our_name, ours_p), (ref_name, ref_p) in zip(
                ours.regressor.named_parameters(), ref.regressor.named_parameters(), strict=True):
            self.assertEqual(our_name, ref_name)
            torch.testing.assert_close(ours_p.grad, ref_p.grad, rtol=5e-5, atol=1e-6)
        with torch.no_grad():
            expected, _, _ = ref(y, None)
            torch.testing.assert_close(ours(y, 3), expected, rtol=4e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
