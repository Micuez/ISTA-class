"""Small CPU tests for the V3 certificate, gradients, and training entry points."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from dataset import RecoveryData
from generate import generate, make_dictionary
from models import SALISTA, build_model
from salista_constraints import project_threshold_budget, rescale_stalled_threshold, resolve_bounds
from train import fit, load_checkpoint, stages, test_run, trainable_parameters


class SALISTATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(12)

    def test_euclidean_budget_projection(self):
        values = torch.tensor([[3., 2., 1.], [3., 2., 1.], [.1, .2, .3], [1., 1., 1.]])
        radius = torch.tensor([[2.], [0.], [1.], [1.5]])
        result = project_threshold_budget(values, radius)
        expected = torch.tensor([[1.5, .5, 0.], [0., 0., 0.], [.1, .2, .3], [.5, .5, .5]])
        torch.testing.assert_close(result, expected)

    def test_offline_weights_for_an_external_dictionary(self):
        a, w = make_dictionary(6, 8, 31)
        model = SALISTA(torch.tensor(a), None, 2, 2, initial_l1_bound=2., noise_l2_bound=.1)
        torch.testing.assert_close(model.W, torch.tensor(w), rtol=0, atol=0)
        torch.testing.assert_close((model.W.T @ model.A).diag(), torch.ones(8))
        with self.assertRaisesRegex(ValueError, "full-row-rank"):
            SALISTA(torch.ones(2, 4), None, 1, 1, initial_l1_bound=1., noise_l2_bound=0.)

    def test_prior_bounds_and_invalid_inputs(self):
        spec = dict(mode="gaussian", failure_probability=0.01)
        b, eps = resolve_bounds(spec, 250, 50, 20)
        self.assertAlmostEqual(b, 68.24555199698756)
        self.assertAlmostEqual(eps, 0.8165612975576763)
        self.assertEqual(resolve_bounds(spec, 250, 50, "inf"), (b, 0.0))
        self.assertGreater(resolve_bounds(dict(spec, failure_probability=1e-5), 250, 50, 20)[1], eps)
        for invalid in (0., -0.1, 1., math.nan, True):
            with self.assertRaises(ValueError):
                resolve_bounds(dict(spec, failure_probability=invalid), 3, 1, 20)
        with self.assertRaisesRegex(ValueError, "SNR"):
            resolve_bounds(spec, 3, 1, None)
        with self.assertRaisesRegex(ValueError, "diag"):
            SALISTA(torch.eye(4), torch.eye(4) / 2, 2, 2, initial_l1_bound=1., noise_l2_bound=0.)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            SALISTA(torch.eye(4), torch.eye(4), 2, 2, initial_l1_bound=1., noise_l2_bound=-1.)

    def test_asymmetric_gram_recurrence_and_support_against_numpy(self):
        A, W = (v.astype(float) for v in make_dictionary(6, 8, 31))
        C = W.T @ A
        off = abs(C)
        np.fill_diagonal(off, 0.)
        self.assertGreater(abs(C - C.T).max(), .1)
        omega = np.linalg.norm(W, axis=0)
        target = np.array([[1.8, -.2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, .1, 0, -1.4]])
        eps, b0 = .02, 2.
        noise = np.zeros((2, 6))
        noise[:, 0] = [eps, -eps]
        y = target @ A.T + noise
        for s in (2, 3):  # same nonsymmetric W, Gram branch and fallback
            net = SALISTA(torch.from_numpy(A), torch.from_numpy(W), s, 3,
                          initial_l1_bound=b0, noise_l2_bound=eps, hidden_size=6, eta=.08)
            with torch.no_grad():
                path, trace = net(torch.from_numpy(y), 3, return_path=True, return_diagnostics=True)
            q = np.sort(off, axis=1)[:, -(s - 1):].sum(1)
            gap = (abs(C.diagonal()) - q).min()
            self.assertEqual(gap > 0, s == 2)
            x = previous_x = np.zeros_like(target)
            budget = previous_budget = np.full((2, 1), b0)
            for i, (estimate, controls) in enumerate(zip(path, trace)):
                v = {key: tensor.numpy() for key, tensor in controls.items()}
                gamma, beta = v["gamma"], v["beta"]
                self.assertTrue(((net.gamma_min <= gamma) & (gamma <= net.gamma_max)).all())
                self.assertTrue(((0 <= beta) & (beta <= net.beta_max)).all())
                if i == 0:
                    self.assertFalse(beta.any())
                c = (y - x @ A.T) @ W
                error = np.broadcast_to(budget, target.shape)
                if gap > 0:
                    sup_error = (abs(c).max(1, keepdims=True) + eps * omega.max()) / gap
                    error = np.minimum(error, (abs(c) + eps * omega + q * sup_error) / abs(C.diagonal()))
                row_sum = np.stack([np.sort(off * row, axis=1)[:, -s:].sum(1) for row in error])
                lower = gamma * (np.minimum(row_sum, off.max() * budget) + eps * omega)
                np.testing.assert_allclose(v["budget"], budget, atol=1e-12)
                np.testing.assert_allclose(v["error_bound"], error, atol=1e-12)
                np.testing.assert_allclose(v["lower"], lower, atol=1e-12)
                self.assertTrue((abs(x - target) <= error + 1e-10).all())
                self.assertTrue((v["delta"] >= 0).all())
                self.assertTrue((v["delta"].sum(1, keepdims=True) <= net.eta * budget + 1e-10).all())
                np.testing.assert_allclose(v["theta"], lower + v["delta"], atol=1e-12)
                z = x + gamma * c + beta * (x - previous_x)
                previous_x, x = x, np.sign(z) * np.maximum(abs(z) - v["theta"], 0)
                np.testing.assert_allclose(estimate.numpy(), x, atol=1e-12)
                self.assertFalse(x[target == 0].any())
                col_sum = np.stack([np.sort(off * row[:, None], axis=0)[-(s - 1):].sum(0) for row in gamma])
                a0 = (abs(1 - gamma * C.diagonal()) + col_sum).max(1, keepdims=True)
                high = gamma.max(1, keepdims=True)
                coefficient = a0 + s * high * off.max() + net.eta + net.beta_max
                recurrence = coefficient * budget + net.beta_max * previous_budget + 2 * s * high * omega.max() * eps
                # Check the raw column-wise recurrence even if the triangle bound is tighter.
                raw = net.certificate.next_budget(torch.from_numpy(gamma), torch.from_numpy(budget),
                                                  torch.from_numpy(previous_budget), eps, net.eta, net.beta_max)
                np.testing.assert_allclose(raw.numpy(), recurrence, atol=1e-12)
                previous_budget, budget = budget, np.minimum(recurrence, b0 + abs(x).sum(1, keepdims=True))
                np.testing.assert_allclose(v["next_budget"], budget, atol=1e-12)
                self.assertTrue((abs(x - target).sum(1, keepdims=True) <= budget + 1e-10).all())

    def test_pointwise_heads_gradients_and_sharing(self):
        y = torch.tensor([[2., -.5, 0., 0.], [0., 0., -1.8, .7]])
        source = torch.eye(4, requires_grad=True)
        for cell in ("gru", "lstm"):
            for sharing in ("shared", "untied"):
                with self.subTest(cell=cell, sharing=sharing):
                    model = SALISTA(source, source, 2, 3, initial_l1_bound=3., noise_l2_bound=.01,
                                    hidden_size=8, eta=.5, cell=cell, sharing=sharing)
                    self.assertEqual(model.stage_module(1) is model.stage_module(3), sharing == "shared")
                    path, trace = model(y, 3, return_path=True, return_diagnostics=True)
                    torch.testing.assert_close(path[1], model(y, 2))
                    torch.testing.assert_close(path[-1], torch.cat([model(row[None], 3) for row in y]))
                    order = [3, 0, 2, 1]  # identity dictionary with permuted measurement/coefficient axes
                    torch.testing.assert_close(path[-1][:, order], model(y[:, order], 3))
                    for key in ("gamma", "theta", "beta"):
                        self.assertEqual(trace[1][key].shape, y.shape)
                        self.assertGreater(trace[1][key][0].max().item() - trace[1][key][0].min().item(), 0)
                    (path[-1] - y).square().mean().backward()
                    active = model.stage_module(3)
                    for layer in (active.cell, active.step, active.threshold, active.momentum):
                        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                                            and p.grad.abs().sum() > 0 for p in layer.parameters()))
                    self.assertIsNone(source.grad)
                    model.zero_grad(set_to_none=True)
                    if sharing == "untied":
                        params = trainable_parameters(model, "new_layer", 2)
                        (model(y, 2, last_step_grad=True) - y).square().mean().backward()
                        self.assertTrue(all(n.startswith("layers.1.") for n in params))
                        self.assertTrue(all(p.grad is None for p in model.layers[0].parameters()))
                        self.assertTrue(all(p.grad is None for p in model.layers[2].parameters()))
                    else:
                        trainable_parameters(model, "prefix", 2)
                        prefix = model(y, 2, return_path=True)
                        self.assertGreater(torch.autograd.grad(prefix[1].sum(), prefix[0])[0].abs().sum(), 0.)

    def test_zero_margin_zero_budget_and_single_coordinate(self):
        for A, s in ((torch.ones(1, 3), 2), (torch.ones(1, 1), 1)):
            model = SALISTA(A, A, s, 3, initial_l1_bound=0., noise_l2_bound=0., eta=0.)
            if s == 2:
                self.assertEqual(model.certificate.margin.item(), 0.)
            output, trace = model(torch.zeros(2, 1), 3, return_diagnostics=True)
            self.assertTrue(torch.equal(output, torch.zeros_like(output)))
            self.assertTrue(all(torch.isfinite(t).all() for step in trace for t in step.values()))
            self.assertTrue(all(t["next_budget"].count_nonzero() == 0 for t in trace))
            with self.assertRaisesRegex(ValueError, "depth"):
                model(torch.zeros(1, 1), 4)

    def test_exact_identity_step_and_absorbing_zero(self):
        x = torch.tensor([[-2., 0., 0.]])
        identity = torch.eye(3)
        net = SALISTA(identity, identity, 1, 2, initial_l1_bound=2., noise_l2_bound=0.,
                      gamma_min=1., gamma_max=1., eta=0., beta_max=0.)
        path, trace = net(x, 2, return_path=True, return_diagnostics=True)
        for value, controls in zip(path, trace):
            torch.testing.assert_close(value, x)
            self.assertEqual(controls["next_budget"].item(), 0.)
        ambiguous = torch.ones(1, 3)
        net = SALISTA(ambiguous, ambiguous, 2, 3, initial_l1_bound=3., noise_l2_bound=0.)
        y = x @ ambiguous.T
        path, trace = net(y, 3, return_path=True, return_diagnostics=True)
        for value, controls in zip(path, trace):
            self.assertEqual(value.count_nonzero(), 0)
            self.assertEqual(controls["next_budget"].item(), 3.)
        (path[-1] - x).square().mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.count_nonzero() == 0 for p in net.parameters()))

    def test_stall_rescaling_only_changes_blocked_nonzero_samples(self):
        z = torch.tensor([[2., 1., 0.], [2., .5, 0.], [0., 0., 0.]])
        threshold = torch.tensor([[4., 3., 0.], [1., 2., 0.], [1., 0., 1.]])
        effective, scale, stalled = rescale_stalled_threshold(z, threshold, .8)
        torch.testing.assert_close(scale[:, 0], torch.tensor([.4, 1., 1.]))
        self.assertEqual(stalled[:, 0].tolist(), [True, False, False])
        torch.testing.assert_close(effective, scale * threshold)
        A = torch.eye(2)
        settings = dict(initial_l1_bound=4., noise_l2_bound=1., gamma_min=1., gamma_max=1.,
                        eta=0., beta_max=0.)
        strict = SALISTA(A, A, 1, 3, **settings)
        relaxed = SALISTA(A, A, 1, 3, stall_threshold_ratio=.8, **settings)
        relaxed.load_state_dict(strict.state_dict())
        y = torch.tensor([[3., 0.], [.4, -.1], [0., 0.]])
        path, trace = relaxed(y, 3, return_path=True, return_diagnostics=True)
        reference = strict(y, 3, return_path=True)
        for i, (value, diag) in enumerate(zip(path, trace)):
            torch.testing.assert_close(value[[0, 2]], reference[i][[0, 2]])
            self.assertTrue(value[1].ne(0).any())
            self.assertEqual(diag["support_certified"].flatten().tolist(), [True, False, True])
            torch.testing.assert_close(diag["next_budget"][1], 4. + value[1].abs().sum().reshape(1))
            if i:
                torch.testing.assert_close(diag["error_bound"][1], diag["budget"][1].expand(2))
        with self.assertRaisesRegex(ValueError, "stall_threshold_ratio"):
            SALISTA(A, A, 1, 3, stall_threshold_ratio=1., **settings)

    def test_free_threshold_is_not_bound_to_the_certificate(self):
        A = torch.eye(4)
        settings = dict(initial_l1_bound=3., noise_l2_bound=0.5, stall_threshold_ratio=.8, eta=0.05)
        bound = SALISTA(A, A, 2, 2, **settings)
        free = SALISTA(A, A, 2, 2, free_threshold=True, **settings)
        free.load_state_dict(bound.state_dict())
        free.layers[0].threshold.bias.data.fill_(-2.)
        y = torch.tensor([[1., 0., 0., 0.], [.2, -.1, .4, 0.]])
        _, bound_trace = bound(y, 2, return_diagnostics=True)
        prediction, free_trace = free(y, 2, return_diagnostics=True)
        self.assertFalse(torch.allclose(bound_trace[0]["theta"], free_trace[0]["theta"]))
        self.assertGreater(free_trace[0]["theta"].sum().item(), (free.eta * free_trace[0]["budget"]).sum().item())
        self.assertTrue(torch.equal(free_trace[0]["threshold_scale"], torch.ones(len(y), 1)))
        self.assertFalse(free_trace[0]["support_certified"].any())
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.square().mean().backward()
        self.assertGreater(free.layers[0].threshold.weight.grad.abs().sum().item(), 0.)
        with self.assertRaisesRegex(ValueError, "free_threshold"):
            SALISTA(A, A, 1, 1, initial_l1_bound=1., noise_l2_bound=0., free_threshold=1)

    def test_free_step_is_not_boxed_into_the_configured_interval(self):
        A = torch.eye(3)
        settings = dict(initial_l1_bound=2., noise_l2_bound=0., gamma_min=0.5, gamma_max=1.1, beta_max=0.)
        boxed = SALISTA(A, A, 1, 1, **settings)
        free = SALISTA(A, A, 1, 1, free_step=True, **settings)
        free.load_state_dict(boxed.state_dict())
        free.layers[0].step.bias.data.fill_(4.)
        y = torch.tensor([[1., 0., 0.]])
        _, boxed_trace = boxed(y, 1, return_diagnostics=True)
        _, free_trace = free(y, 1, return_diagnostics=True)
        self.assertTrue((boxed_trace[0]["gamma"] >= 0.5).all())
        self.assertTrue((boxed_trace[0]["gamma"] <= 1.1).all())
        self.assertGreater(free_trace[0]["gamma"].min().item(), 1.1)
        free.layers[0].step.bias.data.fill_(-8.)
        _, tiny = free(y, 1, return_diagnostics=True)
        self.assertLess(tiny[0]["gamma"].max().item(), 0.5)
        with self.assertRaisesRegex(ValueError, "free_step"):
            SALISTA(A, A, 1, 1, initial_l1_bound=1., noise_l2_bound=0., free_step=1)

    def test_free_signed_momentum_and_budget(self):
        A = torch.eye(2)
        settings = dict(initial_l1_bound=1., noise_l2_bound=0., gamma_min=.5,
                        gamma_max=.5, eta=0., beta_max=.2)
        bounded = SALISTA(A, A, 1, 3, **settings)
        free = SALISTA(A, A, 1, 3, free_momentum=True, **settings)
        free.load_state_dict(bounded.state_dict())
        y = torch.tensor([[1., 0.], [0., -.5]])
        for value in (-1.2, 1.25):
            with torch.no_grad():
                free.layers[0].momentum.weight.zero_()
                free.layers[0].momentum.bias.fill_(value)
            bounded.load_state_dict(free.state_dict())
            path, trace = free(y, 3, return_path=True, return_diagnostics=True)
            _, capped = bounded(y, 3, return_diagnostics=True)
            self.assertEqual(trace[0]["beta"].count_nonzero().item(), 0)
            self.assertTrue(((capped[1]["beta"] >= 0) & (capped[1]["beta"] <= .2)).all())
            torch.testing.assert_close(trace[1]["beta"], torch.full_like(y, value))
            expected = path[0] + .5 * (y - path[0]) + value * path[0]
            torch.testing.assert_close(path[1], expected)
            for estimate, step in zip(path, trace):
                self.assertTrue(((estimate - y).abs().sum(1, keepdim=True) <= step["next_budget"] + 1e-6).all())
            rec = free.certificate.next_budget(trace[1]["gamma"], trace[1]["budget"], trace[0]["budget"],
                                               0., 0., abs(value))
            torch.testing.assert_close(trace[1]["next_budget"], torch.minimum(rec, 1 + path[1].abs().sum(1, keepdim=True)))
            free.zero_grad()
            path[1].sum().backward()
            self.assertGreater(free.layers[0].momentum.bias.grad.abs().item(), 0.)
        with self.assertRaisesRegex(ValueError, "free_momentum"):
            SALISTA(A, A, 1, 2, free_momentum=1, **settings)

    def test_production_trainer_and_checkpoint(self):
        cfg = json.loads(Path(__file__).with_name("training.salista.json").read_text())
        cfg.update(depth=2, batch_size=4, max_epochs_end_to_end=1, snr_db=20)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate(dict(seed=30, measurements=5, signal_dimension=6, nonzeros=1,
                          clean_signals=dict(train=8, val=4, test=3), snr_db=[20, 40]), root / "data")
            data = RecoveryData(root / "data")
            y = torch.tensor(data.arrays("val", 20)[0])
            explicit = dict(cfg, salista={"bounds": {"mode": "explicit", "initial_l1_bound": 5.,
                                                     "noise_l2_bound": .25}})
            self.assertEqual(build_model("SALISTA", data, 2, explicit).noise_l2_bound, .25)
            for mode, sharing, cell in (("end_to_end", "shared", "gru"), ("layerwise", "shared", "gru"),
                                        ("layerwise", "untied", "lstm")):
                cfg["salista"].update(sharing=sharing, cell=cell, hidden_size=8)
                dest = root / (mode + sharing)
                fit(data, "SALISTA", mode, cfg, "cpu", dest, "tiny")
                run, model = load_checkpoint(data, dest, "cpu", "tiny")
                self.assertEqual(run["training"], cfg)
                self.assertEqual(tuple(data.levels), (20, 40))
                record = json.loads((dest / "run.json").read_text())
                self.assertEqual(record["salista_certificate"]["noise_l2_bound"], model.noise_l2_bound)
                plan = stages(mode, cfg, "SALISTA")
                if mode == "layerwise":
                    expected = (["prefix", "prefix", "joint_1", "joint_2"] if sharing == "shared" else
                                ["new_layer", "joint_1", "joint_2"] * 2)
                    self.assertEqual([stage.name for stage in plan], expected)
                self.assertEqual([v["phase"] for v in record["stages"]], [p.name for p in plan])
                clone = build_model("SALISTA", data, cfg["depth"], cfg)
                clone.load_state_dict(torch.load(dest / "best.pt", weights_only=True))
                torch.testing.assert_close(clone(y, 2), model(y, 2), atol=0, rtol=0)
                test_run(data, dest, "cpu", "tiny")
                self.assertEqual(set(json.loads((dest / "test.json").read_text())["snr_db"]), {"20"})
            del data  # Release the read-only A/W mappings before Windows removes the temp directory.


if __name__ == "__main__":
    unittest.main()
