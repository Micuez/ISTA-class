"""Check the PyTorch port against the closed-form initializations and a short train."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import models
import models.ops
import utils


def soft(values, theta):
    return np.sign(values) * np.maximum(np.abs(values) - theta, 0.0)


def make_problem(measurement_dim=8, signal_dim=16, samples=4, seed=0):
    rng = np.random.default_rng(seed)
    sensing = rng.normal(size=(measurement_dim, signal_dim)).astype(np.float32)
    sensing /= np.linalg.norm(sensing, ord=2)
    code = rng.normal(size=(samples, signal_dim)).astype(np.float32)
    code[np.abs(code) < 1.0] = 0.0
    measurements = code @ sensing.T
    return sensing, measurements, code


def assert_close(actual, expected, name):
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    print(f"  match {name}")


def forward_numpy_lista(sensing, measurements, lam, layers):
    matrix, _, theta, basis, tied = models.ops.sensing_init(sensing, lam)
    state = measurements.astype(np.float32)
    outputs = []
    for layer_id in range(layers):
        update = state[:, : matrix.shape[0]] @ basis.T
        if layer_id != 0:
            update = update + state[:, -matrix.shape[1] :] @ tied.T
        output = soft(update, float(theta))
        state = np.concatenate([state, output], axis=1)
        outputs.append(output)
    return outputs


def check_lista():
    sensing, measurements, _ = make_problem()
    lam = 0.1
    layers = 3
    model = models.Lista(sensing, layers, lam, share_W=False)
    for layer_id in range(layers):
        model.create_cell(layer_id)
    with torch.no_grad():
        prediction = model(torch.from_numpy(measurements)).numpy()
    expected = forward_numpy_lista(sensing, measurements, lam, layers)
    measurement_dim, signal_dim = sensing.shape
    for layer_id, output in enumerate(expected):
        start = measurement_dim + layer_id * signal_dim
        assert_close(prediction[:, start : start + signal_dim], output, f"lista layer {layer_id}")


def check_step_and_cp_and_lfista():
    sensing, measurements, _ = make_problem()
    lam = 0.05
    matrix, scale, theta, basis, tied = models.ops.sensing_init(sensing, lam)
    step = np.float32(1.0 / scale)

    step_model = models.StepLista(sensing, 2, lam)
    cp_model = models.ListaCp(sensing, 2, lam, share_W=False)
    fista_model = models.Lfista(sensing, 2, lam)
    for layer_id in range(2):
        step_model.create_cell(layer_id)
        cp_model.create_cell(layer_id)
        fista_model.create_cell(layer_id)

    measurement = torch.from_numpy(measurements)
    with torch.no_grad():
        step_out = step_model(measurement).numpy()
        cp_out = cp_model(measurement).numpy()
        fista_out = fista_model(measurement).numpy()

    measurement_dim, signal_dim = matrix.shape
    first = soft(step * (measurements @ matrix), lam * step)
    residual = measurements - first @ matrix.T
    second = soft(first + step * (residual @ matrix), lam * step)
    assert_close(step_out[:, measurement_dim : measurement_dim + signal_dim], first, "step lista layer 0")
    assert_close(step_out[:, -signal_dim:], second, "step lista layer 1")

    cp_first = soft(measurements @ matrix, float(theta))
    cp_residual = measurements - cp_first @ matrix.T
    cp_second = soft(cp_first + cp_residual @ matrix, float(theta))
    assert_close(cp_out[:, measurement_dim : measurement_dim + signal_dim], cp_first, "lista-cp layer 0")
    assert_close(cp_out[:, -signal_dim:], cp_second, "lista-cp layer 1")

    momentum = []
    tk = [1.0, 1.0]
    for _ in range(2):
        tk.append((1.0 + np.sqrt(1.0 + 4.0 * tk[-1] ** 2)) / 2.0)
        momentum.append((tk[-2] - 1.0) / tk[-1])
    tied_gain = tied * (1.0 + momentum[1])
    fista_first = soft(measurements @ basis.T, float(theta))
    fista_second = soft(measurements @ basis.T + fista_first @ tied_gain.T, float(theta))
    assert_close(fista_out[:, measurement_dim : measurement_dim + signal_dim], fista_first, "lfista layer 0")
    assert_close(fista_out[:, -signal_dim:], fista_second, "lfista layer 1")


def check_lamp():
    sensing, measurements, _ = make_problem()
    lam = 0.2
    matrix, scale, _, _, _ = models.ops.sensing_init(sensing, lam)
    model = models.Lamp(sensing, 1, lam, share_W=False)
    model.create_cell(0)
    with torch.no_grad():
        prediction = model(torch.from_numpy(measurements)).numpy()
    weight = (matrix / scale).astype(np.float32)
    variance = np.sum(measurements ** 2, axis=1, keepdims=True) / matrix.shape[0]
    residual = measurements @ weight
    threshold = np.maximum(np.sqrt(variance) * lam, 0.0)
    expected = np.sign(residual) * np.maximum(np.abs(residual) - threshold, 0.0)
    assert_close(prediction[:, -matrix.shape[1] :], expected, "lamp layer 0")


def check_all_models_train_one_step():
    sensing, measurements, code = make_problem(samples=8)
    rows = np.concatenate([measurements, code], axis=1).astype(np.float32)
    batch = torch.from_numpy(rows[:, : sensing.shape[0]])
    target = torch.from_numpy(rows)
    weight = sensing.astype(np.float32)
    specs = [
        ("lista", models.Lista(sensing, 2, 0.1)),
        ("lfista", models.Lfista(sensing, 2, 0.1)),
        ("lamp", models.Lamp(sensing, 2, 0.1)),
        ("step_lista", models.StepLista(sensing, 2, 0.1)),
        ("lista_cp", models.ListaCp(sensing, 2, 0.1)),
        ("lista_cpss", models.ListaCpss(sensing, 2, 0.1, 1.2, 13.0)),
        ("alista", models.Alista(sensing, weight, 2, 0.1, 1.2, 13.0)),
        ("glista", models.Glista(sensing, 2, 0.1, 1.2, 13.0, gain_func="relu")),
        ("tista", models.Tista(sensing, 2, 0.1, sigma2=1e-4)),
    ]
    for name, model in specs:
        for layer_id in range(2):
            model.create_cell(layer_id)
            missing = [item for item in model.layer_param_names(layer_id) if item not in dict(model.named_parameters())]
            if missing:
                raise AssertionError(f"{name} missing parameter names {missing}")
        prediction = model(batch)
        end = sensing.shape[0] + 2 * model.output_interval
        if prediction.shape != (batch.shape[0], end):
            raise AssertionError(f"{name} output shape {tuple(prediction.shape)} != {(batch.shape[0], end)}")
        loss = utils.mse_loss(target, prediction, sensing.shape[1])
        loss.backward()
        grads = [param.grad for param in model.parameters() if param.grad is not None]
        if not grads or not all(torch.isfinite(grad).all() for grad in grads):
            raise AssertionError(f"{name} did not produce finite gradients")
        if not torch.isfinite(prediction).all():
            raise AssertionError(f"{name} forward is non-finite")
        print(f"  train step {name} loss {float(loss.detach()):.4f}")


def check_freeze_keeps_old_layers():
    sensing, measurements, code = make_problem(samples=16, seed=1)
    rows = np.concatenate([measurements, code], axis=1).astype(np.float32)
    model = models.Lista(sensing, 2, 0.1)
    ages = {}
    for layer_id in range(2):
        model.create_cell(layer_id)
        utils.update_var_ages(ages, model.layer_param_names(layer_id))
    if ages["B"] != 1 or ages["theta.0"] != 1 or ages["W.0"] != 0 or ages["theta.1"] != 0:
        raise AssertionError(f"unexpected ages {ages}")
    before_b = model.B.detach().clone()
    before_w = model.W[0].detach().clone()
    optimizer = utils.AgeAdam(model, ages, learning_rate=1e-2, freeze_current_only=True)
    batch = torch.from_numpy(rows[:, : sensing.shape[0]])
    target = torch.from_numpy(rows)
    optimizer.zero_grad()
    loss = utils.mse_loss(target, model(batch), sensing.shape[1])
    loss.backward()
    optimizer.step()
    if not torch.equal(before_b, model.B.detach()):
        raise AssertionError("frozen B changed during the current-layer phase")
    if torch.equal(before_w, model.W[0].detach()):
        raise AssertionError("new tied weight did not move")
    print("  freeze schedule keeps previous LISTA weights fixed")


def write_split(directory, name, sensing, samples, seed):
    rng = np.random.default_rng(seed)
    code = rng.normal(size=(samples, sensing.shape[1])).astype(np.float32)
    mask = rng.random(code.shape) > 0.9
    code *= mask
    measurements = code @ sensing.T
    np.save(directory / name, np.concatenate([measurements, code], axis=1).astype(np.float32))


def check_cli_train_and_resume():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="openl2o_smoke_") as tmp:
        tmp = Path(tmp)
        data = tmp / "data"
        data.mkdir()
        rng = np.random.default_rng(2)
        sensing = rng.normal(size=(8, 12)).astype(np.float32)
        sensing /= np.linalg.norm(sensing, ord=2)
        np.save(data / "A.npy", sensing)
        write_split(data, "train_data.npy", sensing, 64, 3)
        write_split(data, "val_data.npy", sensing, 32, 4)
        write_split(data, "test_data.npy", sensing, 32, 5)
        common = [
            sys.executable,
            str(root / "train.py"),
            "--model_name",
            "lista",
            "--task",
            "sc",
            "--num_layers",
            "1",
            "--epochs",
            "2",
            "--patience",
            "5",
            "--train_batch_size",
            "32",
            "--val_batch_size",
            "32",
            "--num_train_images",
            "64",
            "--num_val_images",
            "32",
            "--base_dir",
            str(tmp / "exp"),
            "--data_dir",
            str(data),
            "--exp_name",
            "smoke_lista",
            "--gpu",
            "0",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "5"
        full = common.copy()
        full[full.index("--num_layers") + 1] = "2"
        subprocess.check_call(full, cwd=root, env=env)
        layer2 = tmp / "exp" / "models" / "smoke_lista" / "replicate_1" / "layer_2" / "model.pt"
        if not layer2.is_file():
            raise AssertionError(f"missing checkpoint {layer2}")
        subprocess.check_call(full + ["--test", "--test_files", "test_data.npy"], cwd=root, env=env)
        layer2.unlink()
        log = subprocess.check_output(full, cwd=root, env=env, stderr=subprocess.STDOUT, text=True)
        if "Checkpoint restored" not in log or "Building layer 2 / 2." not in log:
            raise AssertionError("resume did not restore layer 1 and train layer 2:\n" + log)
        print("  cli train, test, and resume passed")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    print("closed-form forwards")
    check_lista()
    check_step_and_cp_and_lfista()
    check_lamp()
    print("one optimization step")
    check_all_models_train_one_step()
    check_freeze_keeps_old_layers()
    print("gpu training")
    check_cli_train_and_resume()
    print("smoke test passed")


if __name__ == "__main__":
    main()
