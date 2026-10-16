"""Layer-wise training and evaluation for the unfolded models."""

import json
import logging
import os

import numpy as np
import torch

import data_preprocessing
import models
import utils


def resolve_device(gpu):
    if not torch.cuda.is_available():
        logging.warning("CUDA is unavailable. Training will run on CPU.")
        return torch.device("cpu")
    count = torch.cuda.device_count()
    if gpu is None:
        gpu = 5 if count > 5 else 0
    if gpu < 0 or gpu >= count:
        raise ValueError(f"GPU index {gpu} is outside the visible range 0..{count - 1}.")
    return torch.device(f"cuda:{gpu}")


def _iterate_forever(loader):
    while True:
        for batch in loader:
            yield batch


def _move(batch, device):
    measurements, target = batch
    return measurements.to(device, non_blocking=True), target.to(device, non_blocking=True)


def _reduce_metric(values, total, count):
    return total + float(values.sum().detach()), count + int(values.numel())


@torch.no_grad()
def _evaluate(model, loader, device, loss_fn, metric_fn, max_batches=None):
    model.eval()
    loss_total = 0.0
    metric_total = 0.0
    metric_count = 0
    batches = 0
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        measurements, target = _move(batch, device)
        prediction = model(measurements)
        loss_total += float(loss_fn(target, prediction).detach())
        metric_total, metric_count = _reduce_metric(
            metric_fn(target, prediction), metric_total, metric_count
        )
        batches += 1
    if batches == 0 or metric_count == 0:
        raise ValueError("The evaluation loader produced no batches.")
    if max_batches is not None and batches < max_batches:
        raise ValueError(
            f"Evaluation data produced {batches} batches, but {max_batches} were requested. "
            "Check num_val_images, the batch size, and the .npy file length."
        )
    return loss_total / batches, metric_total / metric_count


def _fit_phase(
    model,
    train_loader,
    val_loader,
    device,
    loss_fn,
    metric_fn,
    optimizer,
    epochs,
    train_steps,
    val_steps,
    patience,
    monitor_name,
    layer_number,
    phase_name,
):
    stream = _iterate_forever(train_loader)
    best = float("inf")
    wait = 0
    last = None
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for _ in range(train_steps):
            measurements, target = _move(next(stream), device)
            optimizer.zero_grad()
            prediction = model(measurements)
            loss = loss_fn(target, prediction)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at layer {layer_number} epoch {epoch}."
                )
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach())
        train_loss = loss_total / train_steps
        val_loss, val_metric = _evaluate(
            model, val_loader, device, loss_fn, metric_fn, max_batches=val_steps
        )
        logging.info(
            "layer %d %s epoch %d  loss %.6f  val_loss %.6f  %s %.6f",
            layer_number,
            phase_name,
            epoch,
            train_loss,
            val_loss,
            monitor_name,
            val_metric,
        )
        last = {
            "loss": train_loss,
            "val_loss": val_loss,
            monitor_name: val_metric,
        }
        if val_metric < best:
            best = val_metric
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logging.info(
                    "Early stopping at layer %d %s epoch %d (best %s %.6f).",
                    layer_number,
                    phase_name,
                    epoch,
                    monitor_name,
                    best,
                )
                break
    return last


def _load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    logging.info("Checkpoint restored from %s.", path)


def _save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    logging.info("Checkpoint saved at %s.", path)


def _log_parameters(model):
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        logging.info("param %-24s %s", name, tuple(param.shape))
    logging.info("Trainable parameters: %d", total)


@torch.no_grad()
def _test_files(
    model,
    files,
    data_dir,
    batch_size,
    task,
    device,
    sensing,
    lam,
    measurement_dim,
    signal_dim,
    output_interval,
    model_dir,
    seed,
):
    results = {}
    for filename in files:
        loader = data_preprocessing.input_fn(
            False,
            data_dir,
            batch_size,
            task=task,
            drop_remainder=False,
            A=sensing.detach().cpu().numpy(),
            filename=filename,
            seed=seed,
        )
        metric_sums = [0.0 for _ in range(len(model.cells))]
        metric_count = 0
        predictions = []
        model.eval()
        for measurements, target in loader:
            measurements, target = _move((measurements, target), device)
            prediction = model(measurements)
            predictions.append(prediction[:, -signal_dim:].cpu())
            for layer_id in range(len(model.cells)):
                if task == "lasso":
                    values = utils.lasso_objective(
                        target,
                        prediction,
                        sensing,
                        lam,
                        measurement_dim,
                        signal_dim,
                        layer_id=layer_id,
                    )
                else:
                    values = utils.nmse_db(
                        target,
                        prediction,
                        signal_dim,
                        layer_id=layer_id,
                        measurement_dim=measurement_dim,
                        interval=output_interval,
                    )
                metric_sums[layer_id] += float(values.sum().detach())
            metric_count += int(target.shape[0])
        if metric_count == 0:
            raise ValueError(f"Test file {filename} produced no samples.")
        metrics = [value / metric_count for value in metric_sums]
        results[filename] = metrics
        logging.info("%s : %s", filename, metrics)
        if task == "lasso":
            final = torch.cat(predictions, dim=0).numpy()
            stem = os.path.basename(filename)
            if stem.endswith(".npy"):
                stem = stem[: -len(".npy")]
            np.save(os.path.join(model_dir, stem + "_final_output.npy"), final)
    return results


def run(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device(args.gpu)
    if device.type == "cuda":
        logging.info("Using %s (%s).", device, torch.cuda.get_device_name(device))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_dir = os.path.abspath(args.base_dir)
    data_dir = os.path.abspath(args.data_dir)
    model_dir = os.path.join(
        base_dir, "models", args.exp_name, "replicate_" + str(args.replicate)
    )
    log_dir = os.path.join(
        base_dir, "logs", args.exp_name, "replicate_" + str(args.replicate)
    )
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logging.info("Saving checkpoints at %s", model_dir)
    logging.info("Saving logs at %s", log_dir)

    if args.task in ("sc", "lasso"):
        sensing_np = np.load(os.path.join(data_dir, "A.npy"), allow_pickle=True).astype(np.float32)
        dictionary = None
        loss_signal_dim = int(sensing_np.shape[1])
        loss_extra_dim = None
        train_steps = int(args.num_train_images // args.train_batch_size)
        val_steps = int(args.num_val_images // args.val_batch_size)
    elif args.task == "cs":
        sensing_np = np.load(
            os.path.join(data_dir, "A_128_512.npy"), allow_pickle=True
        ).astype(np.float32)
        dictionary = np.load(
            os.path.join(data_dir, "D_256_512.npy"), allow_pickle=True
        ).astype(np.float32)
        loss_signal_dim = int(dictionary.shape[0])
        loss_extra_dim = int(dictionary.shape[1])
        train_steps = 3125
        val_steps = 10
    else:
        raise ValueError(f"Invalid task {args.task!r}.")

    if train_steps < 1 or val_steps < 1:
        raise ValueError(
            f"Need at least one train step and one val step, got {train_steps} and {val_steps}."
        )

    alista_w = None
    if args.model_name.startswith("alista"):
        alista_w = np.load(
            os.path.join(data_dir, args.alista_W_file), allow_pickle=True
        ).astype(np.float32)

    measurement_dim = int(sensing_np.shape[0])
    signal_dim = int(sensing_np.shape[1])
    mode = "test" if args.test else "train"
    logging.info("mode=%s task=%s model=%s layers=%d", mode, args.task, args.model_name, args.num_layers)
    logging.info("train batch %d (%d steps), eval batch %d (%d steps)",
                 args.train_batch_size, train_steps, args.val_batch_size, val_steps)

    model = models.build_model(
        args.model_name,
        sensing_np,
        args.num_layers,
        args.model_lam,
        share_W=args.share_W,
        D=dictionary,
        alista_W=alista_w,
        ss_q_per_layer=args.ss_q_per_layer,
        ss_maxq=args.ss_maxq,
        glista_alti=args.glista_alti,
        gain_func=args.gain_func,
        tista_sigma2=args.tista_sigma2,
        lasso_lam=args.lasso_lam,
    )
    if hasattr(model, "q"):
        logging.info("Support schedule: %s", np.array2string(model.q, precision=4))
    model.to(device)
    sensing = model.A

    if args.task == "lasso":
        monitor = "val_lasso"

        def loss_fn(y_true, y_pred):
            return utils.lasso_loss(y_true, y_pred, sensing, args.lasso_lam, signal_dim)

        def metric_fn(y_true, y_pred):
            return utils.lasso_objective(
                y_true, y_pred, sensing, args.lasso_lam, measurement_dim, signal_dim, layer_id=-1
            )
    else:
        monitor = "val_nmse"

        def loss_fn(y_true, y_pred):
            return utils.mse_loss(y_true, y_pred, loss_signal_dim, loss_extra_dim)

        def metric_fn(y_true, y_pred):
            return utils.nmse_db(y_true, y_pred, loss_signal_dim, layer_id=-1)

    previous_ckpt, completed_layers = utils.check_and_load_partial(model_dir, args.num_layers)

    if mode == "test":
        if completed_layers != args.num_layers:
            raise ValueError(
                f"Testing needs checkpoints for all {args.num_layers} layers, "
                f"found {completed_layers} in {model_dir}."
            )
        if not args.test_files:
            raise ValueError("Pass --test_files for evaluation.")
        for layer_id in range(args.num_layers):
            model.create_cell(layer_id)
        _load_checkpoint(model, previous_ckpt, device)
        return _test_files(
            model,
            args.test_files,
            data_dir,
            args.val_batch_size,
            args.task,
            device,
            sensing,
            args.lasso_lam,
            measurement_dim,
            signal_dim,
            model.output_interval,
            model_dir,
            args.seed,
        )

    train_loader = data_preprocessing.input_fn(
        True,
        data_dir,
        args.train_batch_size,
        task=args.task,
        drop_remainder=False,
        A=sensing_np,
        seed=args.seed,
    )
    val_loader = data_preprocessing.input_fn(
        False,
        data_dir,
        args.val_batch_size,
        task=args.task,
        drop_remainder=False,
        A=sensing_np,
        seed=args.seed,
    )

    ages = {}
    val_metric = None
    history_path = os.path.join(log_dir, "metrics.jsonl")
    phases = (
        ("freeze", args.base_lr, True),
        ("finetune", args.base_lr * 0.2, False),
        ("finetune_1e-1", args.base_lr * 0.2 * 0.1, False),
    )

    for layer_id in range(args.num_layers):
        logging.info("Building layer %d / %d.", layer_id + 1, args.num_layers)
        model.create_cell(layer_id)
        utils.update_var_ages(ages, model.layer_param_names(layer_id))

        if layer_id == completed_layers - 1 and previous_ckpt:
            _load_checkpoint(model, previous_ckpt, device)
            _, val_metric = _evaluate(
                model, val_loader, device, loss_fn, metric_fn, max_batches=val_steps
            )
            logging.info("Restored layer %d %s %.6f", layer_id + 1, monitor, val_metric)
            continue
        if layer_id < completed_layers:
            logging.info("Skip layer %d.", layer_id + 1)
            continue

        last = None
        for phase_name, learning_rate, freeze in phases:
            logging.info(
                "Layer %d phase %s, lr %.3e, freeze_previous=%s",
                layer_id + 1,
                phase_name,
                learning_rate,
                freeze,
            )
            optimizer = utils.AgeAdam(model, ages, learning_rate, freeze)
            last = _fit_phase(
                model,
                train_loader,
                val_loader,
                device,
                loss_fn,
                metric_fn,
                optimizer,
                args.epochs,
                train_steps,
                val_steps,
                args.patience,
                monitor,
                layer_id + 1,
                phase_name,
            )
        _log_parameters(model)
        val_metric = last[monitor]
        record = {
            "layer": layer_id + 1,
            "train_loss": last["loss"],
            "val_loss": last["val_loss"],
            monitor: val_metric,
        }
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        _save_checkpoint(model, utils.save_partial(model_dir, layer_id))

    if args.task == "cs":
        raise NotImplementedError(
            "Compressive-sensing image testing was left unimplemented in the "
            "original training script. Layer training itself has finished."
        )
    return val_metric
