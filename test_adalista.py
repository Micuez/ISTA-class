"""Tests for the project depth convention and dictionary-dependent AdaLISTA update."""

import ast
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from adalista_model import AdaLISTA


class AdaLISTATests(unittest.TestCase):
    def test_identity_dictionary_reduces_to_lasso_soft_threshold(self):
        dictionary = torch.eye(3, dtype=torch.float64)
        model = AdaLISTA(dictionary, 3, lambd=0.25)
        y = torch.tensor([[1.0, -0.65, 0.05]], dtype=dictionary.dtype)
        expected = F.softshrink(y, lambd=0.25)
        path = model(y, 3, return_path=True)
        self.assertEqual(len(path), 3)
        for estimate in path:
            torch.testing.assert_close(estimate, expected)

    def test_one_dictionary_per_sample_and_column_permutation(self):
        rng = torch.Generator().manual_seed(392)
        dictionary = torch.randn(3, 5, dtype=torch.float64, generator=rng)
        y = torch.randn(3, 3, dtype=dictionary.dtype, generator=rng)
        model = AdaLISTA(dictionary, 3, lambd=0.02)
        perm = [1, 3, 0, 4, 2]
        dictionaries = torch.stack([dictionary, dictionary[:, perm], 0.7 * dictionary])
        result = model(y, 3, dictionary=dictionaries)
        for index, current in enumerate(dictionaries):
            expected = model(y[index:index+1], 3, dictionary=current)
            torch.testing.assert_close(result[index], expected[0])
        torch.testing.assert_close(result[1], model(y[1:2], 3)[0, perm])
        with self.assertRaises(ValueError):
            model(y, 3, dictionary=dictionaries[:1])

    def test_only_final_layer_scalars_receive_gradients_in_greedy_phase(self):
        dictionary = torch.eye(2)
        model = AdaLISTA(dictionary, 3, lambd=0.03, initial_step=0.4)
        pred = model(torch.tensor([[1.0, -0.7]]), 3, last_step_grad=True)
        pred.square().sum().backward()
        for layer in model.layers[:2]:
            self.assertIsNone(layer.gamma.grad)
            self.assertIsNone(layer.eta.grad)
        for parameter in model.stage_module(3).parameters():
            self.assertGreater(parameter.grad.abs().item(), 0)

    def test_forward_and_backward_match_authors_adaptive_ista(self):
        # The pinned source imports SciPy for an unrelated solver. Extract only
        # the author class so this optional oracle needs only the normal PyTorch dependency.
        source = Path(__file__).parent / "upstream" / "AdaLISTA" / "models.py"
        if not source.exists():
            self.skipTest("Prepare the pinned AdaLISTA source to run the author parity test")
        syntax = ast.parse(source.read_text(encoding="utf-8"))
        author_class = next(node for node in syntax.body
                            if isinstance(node, ast.ClassDef) and node.name == "Adaptive_ISTA")
        namespace = {"torch": torch, "nn": nn, "F": F}
        exec(compile(ast.Module(body=[author_class], type_ignores=[]), str(source), "exec"), namespace)
        reference_type = namespace["Adaptive_ISTA"]
        rng = torch.Generator().manual_seed(726)
        D = torch.randn(2, 3, 4, generator=rng)
        y = torch.randn(2, 3, generator=rng)
        reference = reference_type(3, 4, T=2, lambd=0.04)
        model = AdaLISTA(D[0], 3, lambd=0.04, initial_step=1.0)
        # Non-symmetric weights and independent layer scalars exercise both
        # transpose orientations. A negative eta checks the even threshold scaling.
        with torch.no_grad():
            for p, ref in zip(model.shared_matrices, [reference.W1.weight, reference.W2.weight]):
                p.add_(0.1 * torch.randn(p.shape, generator=rng))
                ref.copy_(p)
            for index, layer in enumerate(model.layers):
                layer.gamma.fill_(0.2 + 0.03 * index)
                layer.eta.fill_(0.5 * (-1)**index)
                reference.gammas[index].copy_(layer.gamma)
                reference.etas[index].copy_(layer.eta)
        inputs = [tensor.clone().requires_grad_() for tensor in (y, D)]
        ref_inputs = [tensor.clone().requires_grad_() for tensor in (y, D)]
        actual = model(inputs[0], 3, dictionary=inputs[1])
        expected = reference(*ref_inputs)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for value, ref in zip([*inputs, *model.shared_matrices],
                              [*ref_inputs, reference.W1.weight, reference.W2.weight]):
            torch.testing.assert_close(value.grad, ref.grad)
        for index, layer in enumerate(model.layers):
            torch.testing.assert_close(layer.gamma.grad, reference.gammas.grad[index].squeeze())
            torch.testing.assert_close(layer.eta.grad, reference.etas.grad[index].squeeze())


if __name__ == "__main__":
    unittest.main()
