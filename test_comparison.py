"""Regression checks for the online 48-run comparison and deterministic projection."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from dataset import RecoveryData
from generate import generate
from models import METHODS
from na_training import paired_author_batches
from run_single_snr_grid import grid_jobs
from salista_constraints import project_threshold_budget, recovery_prior
from train import MODES, fit, load_checkpoint, stages


class ComparisonTests(unittest.TestCase):
    def test_grid_and_shared_prefix_protocol(self):
        cfg = json.loads(Path(__file__).with_name("training.comparison.json").read_text())
        expected = {(m, mode, r) for m in METHODS for mode in MODES for r in (40, 20, 10, 5)}
        self.assertEqual(len(grid_jobs("six-methods")), 48)
        self.assertEqual(set(grid_jobs("six-methods")), expected)
        self.assertEqual((cfg["depth"], cfg["seed"], cfg["na_batch_size"]), (16, 20261118, 512))
        self.assertEqual(cfg["na_samples_per_epoch"] // cfg["na_batch_size"], 97)
        self.assertTrue(cfg["paired_author_online"])
        for mode in MODES:
            self.assertEqual(stages(mode, cfg, "NA-ALISTA"), stages(mode, cfg, "SALISTA"))
        plan = stages("layerwise", cfg, "NA-ALISTA")
        self.assertEqual([p.name for p in plan], ["prefix"] * 16 + ["joint_1", "joint_2"])
        self.assertEqual([p.depth for p in plan if p.name == "prefix"], list(range(1, 17)))
        with self.assertRaisesRegex(ValueError, "author schedule"):
            stages("layerwise", dict(cfg, na_schedule="author"), "NA-ALISTA")

    def test_online_pairs_do_not_depend_on_model_rng(self):
        matrix = torch.eye(8)
        for r in (20, float("inf")):
            reference = list(paired_author_batches(11, 3, matrix, r, 4, 20261118))
            torch.randn(314)
            same = list(paired_author_batches(11, 3, matrix, r, 4, 20261118))
            next_epoch = list(paired_author_batches(11, 3, matrix, r, 4, 20261119))
            self.assertEqual(len(reference), 2)
            for (y, x), (yy, xx) in zip(reference, same, strict=True):
                torch.testing.assert_close(x, xx, rtol=0, atol=0)
                torch.testing.assert_close(y, yy, rtol=0, atol=0)
                if math.isinf(r):
                    torch.testing.assert_close(y, x)
            self.assertFalse(torch.equal(reference[0][1], next_epoch[0][1]))

    def test_online_prior_includes_signal_count_and_conditional_noise(self):
        A = 2.5 * torch.eye(6, 12, dtype=torch.float64)
        spec = {"mode": "online_gaussian", "failure_probability": .03}
        s, b0, eps = recovery_prior(spec, A, 3, 20)
        t = math.log(100)
        self.assertEqual(s, min(12, math.ceil(3 + math.sqrt(2 * 2.25 * t) + 2 * t / 3)))
        r = math.sqrt(s + 2 * math.sqrt(s * t) + 2 * t)
        self.assertAlmostEqual(b0, math.sqrt(s) * r)
        self.assertAlmostEqual(eps, 0.1 * 2.5 * r / math.sqrt(5) * math.sqrt(6 + 2 * math.sqrt(6 * t) + 2 * t))
        self.assertEqual(recovery_prior(spec, A, 3, "inf")[2], 0)

    def test_both_recurrent_trainers_consume_paired_prefix_batches(self):
        cfg = json.loads(Path(__file__).with_name("training.comparison.json").read_text())
        cfg.update(depth=2, max_epochs_per_layer=3, max_epochs_end_to_end=1,
                   batch_size=4, na_samples_per_epoch=8, na_batch_size=4, snr_db=20)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate(dict(seed=9, measurements=6, signal_dimension=10, nonzeros=2, snr_db=[20],
                          clean_signals=dict(train=8, val=4, test=4)), root / "data")
            data = RecoveryData(root / "data")
            self.addCleanup(torch.set_num_threads, torch.get_num_threads())
            torch.set_num_threads(1)
            reference = None
            for method in ("NA-ALISTA", "SALISTA"):
                draws = []
                def capture(*args):
                    for y, x in paired_author_batches(*args):
                        draws.append((y.clone(), x.clone()))
                        yield y, x
                with patch("train.paired_author_batches", new=capture):
                    fit(data, method, "layerwise", cfg, "cpu", root / method, "fixture")
                record = json.loads((root / method / "run.json").read_text())
                self.assertEqual((record["training_data"], record["updates"]), ("author_online", 12))
                self.assertEqual([row["phase"] for row in record["stages"]], ["prefix", "prefix", "joint_1", "joint_2"])
                _, model = load_checkpoint(data, root / method, "cpu", "fixture")
                self.assertTrue(torch.isfinite(model(draws[0][0], 2)).all())
                if reference is None:
                    reference = draws
                else:
                    for actual, expected in zip(draws, reference, strict=True):
                        for v, ref in zip(actual, expected, strict=True):
                            torch.testing.assert_close(v, ref, rtol=0, atol=0)
            del data

    def test_projection_value_and_gradient_at_non_power_of_two_width(self):
        # The CPU reference can use native cumsum. Test the implementation used by CUDA.
        rng = torch.Generator().manual_seed(17)
        x = torch.rand((3, 500), generator=rng, dtype=torch.float64).requires_grad_()
        r = torch.tensor([[2.], [55.], [300.]], dtype=x.dtype, requires_grad=True)
        scale = torch.randn(x.shape, generator=rng, dtype=x.dtype)
        columns = torch.arange(1, 501)
        offset = ((x.sort(descending=True).values.cumsum(1) - r) / columns).amax(1, keepdim=True).clamp_min(0)
        expected = (x - offset).clamp_min(0)
        expected_grad = torch.autograd.grad((expected * scale).sum(), (x, r))
        previous = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, previous)
        torch.use_deterministic_algorithms(True)
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            a, b = (p.detach().to(device).requires_grad_() for p in (x, r))
            actual = project_threshold_budget(a, b)
            gradient = torch.autograd.grad((actual * scale.to(device)).sum(), (a, b))
            torch.testing.assert_close(actual.cpu(), expected)
            for g, ref in zip(gradient, expected_grad):
                torch.testing.assert_close(g.cpu(), ref)


if __name__ == "__main__":
    unittest.main()
