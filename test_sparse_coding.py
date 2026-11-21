import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models import UnfoldedISTA
from sparse_coding import _make_model, main, sparse_coding_epoch, sparse_encode


class OrthonormalEncoder(torch.nn.Module):
    """Exact LASSO solution for the orthonormal dictionaries in the test."""

    supports_variable_dictionary = True

    def __init__(self, dictionary, l1):
        super().__init__()
        self.register_buffer("A", dictionary)
        self.l1 = l1

    def forward(self, y, depth, dictionary):
        transformed = torch.bmm(y.unsqueeze(1), dictionary).squeeze(1)
        return transformed.sign() * torch.relu(transformed.abs() - self.l1)


class SparseCodingTests(unittest.TestCase):
    def test_lasso_objective_and_partial_batch(self):
        y = torch.tensor([[2., -0.1], [0.6, 0.2], [-1., 2.]], dtype=torch.float64)
        a = torch.eye(2, dtype=y.dtype)
        dictionaries = torch.stack((a, -a, a.flip(0)))
        l1 = 0.5
        encoder = OrthonormalEncoder(a, l1)
        result = sparse_coding_epoch(encoder, y, dictionaries, 1, l1, batch_size=2)
        # At an orthonormal optimum each transformed scalar contributes its Huber value.
        expected = torch.where(y.abs() <= l1, y.square() / 2, l1 * y.abs() - l1**2 / 2)
        self.assertEqual(result["signals"], 3)
        self.assertAlmostEqual(result["objective"], expected.sum(1).mean().item())
        self.assertAlmostEqual(result["objective"], result["reconstruction"] + l1 * result["code_l1"])
        with self.assertRaisesRegex(ValueError, "per signal"):
            sparse_encode(encoder, y, dictionaries[:2], 1)

    def test_existing_model_requires_its_dictionary(self):
        a = torch.eye(3)
        model = UnfoldedISTA("ALISTA", a, a, 2)
        y = torch.ones(4, 3)
        torch.testing.assert_close(sparse_encode(model, y, a.clone(), 2), model(y, 2))
        with self.assertRaisesRegex(ValueError, "original"):
            sparse_encode(model, y, -a, 2)
        with self.assertRaisesRegex(ValueError, "original"):
            sparse_encode(model, y, a.expand(4, -1, -1), 2)

    def test_learned_matrix_models_accept_rank_deficient_dictionary(self):
        # Second row is dependent and column 3 is zero; analytic_weights rejects it.
        d = torch.tensor([[1., 2., 0.], [2., 4., 0.]])
        y = torch.tensor([[1., 2.]])
        for method in ("EBT-LISTA", "HLISTA", "ELISTA"):
            with self.subTest(method=method):
                model = _make_model(method, d, 2, None, {}, 0.1)
                self.assertTrue(torch.isfinite(sparse_encode(model, y, d, 2)).all())

    def test_cli_best_checkpoint_for_variable_and_fixed_dictionaries(self):
        rng = np.random.default_rng(14)
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            shared = np.eye(3, dtype=np.float32)
            signals = {f"{split}_y": rng.normal(size=(n, 3)).astype(np.float32)
                       for split, n in (("train", 5), ("val", 3), ("test", 2))}
            dictionaries = {key.replace("_y", "_D"): np.stack([shared * (-1)**i for i in range(len(rows))])
                            for key, rows in signals.items()}
            configs = root / "model.json"
            configs.write_text(json.dumps({"salista": {"bounds": {"mode": "explicit", "initial_l1_bound": 4.0,
                                                                    "noise_l2_bound": 0.0},
                                                        "free_threshold": True, "free_step": True}}))
            salista_options = ["--sparsity", "1", "--model-config", str(configs)]
            for method, mode, arrays, options in (
                ("AdaLISTA", "end_to_end", dictionaries, []),
                ("SALISTA", "end_to_end", {"D": shared}, salista_options),
                ("AdaLISTA", "layerwise", {"D": shared}, []),
                ("SALISTA", "layerwise", {"D": shared}, salista_options),
            ):
                with self.subTest(method=method, mode=mode):
                    data, best = root / f"{method}.npz", root / f"{method}-{mode}.pt"
                    np.savez(data, **signals, **arrays)
                    common = ["--data", str(data), "--checkpoint", str(best), "--batch-size", "2"]
                    main(["train", *common, "--model", method, "--l1", "0.1", "--depth", "2",
                          "--mode", mode, "--epochs", "2", "--epochs-per-layer", "3", *options])
                    saved = torch.load(best, weights_only=True)
                    self.assertEqual(saved["method"], method)
                    self.assertEqual(saved["depth"], 2)
                    self.assertEqual(saved["mode"], mode)
                    if mode == "end_to_end":
                        self.assertIn(saved["epoch"], (1, 2))
                    else:
                        depth_phases = [(s["depth"], s["phase"]) for s in saved["stages"]]
                        if method == "SALISTA":
                            expected = [(1, "prefix"), (2, "prefix"), (2, "joint_1"), (2, "joint_2")]
                        else:
                            expected = [(depth, phase) for depth in (1, 2)
                                        for phase in ("new_layer", "joint_1", "joint_2")]
                        self.assertEqual(depth_phases, expected)
                        self.assertEqual(saved["epochs"], 6)
                        self.assertEqual(saved["updates"], 18)  # Five rows in batches of two.
                    # Verify the saved weights reproduce the selected validation score.
                    val = {k.replace("val_", "test_"): v for k, v in {**signals, **arrays}.items()
                           if k == "D" or k.startswith("val_")}
                    validation = root / "validation-as-test.npz"
                    np.savez(validation, **val)
                    selected = main(["test", "--data", str(validation), "--checkpoint", str(best), "--batch-size", "2"])
                    self.assertAlmostEqual(selected["objective"], saved["val_objective"])
                    score = main(["test", *common])
                    self.assertEqual(score["signals"], len(signals["test_y"]))
                    self.assertTrue(np.isfinite(score["objective"]))

    def test_na_layerwise_is_rejected_before_loading_the_dataset(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as failure:
            main(["train", "--model", "NA-ALISTA", "--mode", "layerwise", "--data", "missing.npz",
                  "--checkpoint", "unused.pt", "--l1", "0.1"])
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("end_to_end", errors.getvalue())

    def test_layerwise_finetuning_retains_incoming_best_at_each_depth(self):
        # Each mock update increments a weight. Only the first update at each
        # depth is a validation improvement, so later increments must be undone.
        observations = {1: [], 2: []}

        def epoch(model, signals, dictionary, depth, l1, *, optimizer=None, **kwargs):
            parameter = model.layers[0].gamma
            if optimizer is not None:
                observations[depth].append(parameter.item())
                with torch.no_grad():
                    parameter.add_(1.)
            return {"objective": float(len(observations[depth]))}

        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            data, best = Path(directory) / "codes.npz", Path(directory) / "best.pt"
            signals = np.eye(2, dtype=np.float32)
            np.savez(data, train_y=signals, val_y=signals, D=signals)
            with patch("sparse_coding.sparse_coding_epoch", side_effect=epoch):
                main(["train", "--model", "AdaLISTA", "--mode", "layerwise", "--data", str(data),
                      "--checkpoint", str(best), "--l1", "0.1", "--depth", "2", "--epochs-per-layer",
                      "6", "--patience", "1"])
            checkpoint = torch.load(best, weights_only=True)
            # The first substage trains twice; each subsequent joint stage
            # stops immediately and restores the optimal weight from epoch 1.
            self.assertEqual(observations[1], [1., 2., 2., 2.])
            self.assertEqual(observations[2], [2., 3., 3., 3.])
            self.assertEqual(checkpoint["weights"]["layers.0.gamma"].item(), 3.)
            self.assertEqual(checkpoint["epoch"], 5)
            self.assertEqual(checkpoint["epochs"], 8)
            self.assertEqual(checkpoint["updates"], 8)
            self.assertEqual(checkpoint["val_objective"], 1.)
            self.assertTrue(all(stage["stop_reason"] == "validation_patience"
                                for stage in checkpoint["stages"]))


if __name__ == "__main__":
    unittest.main()
