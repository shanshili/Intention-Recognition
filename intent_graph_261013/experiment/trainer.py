"""Shared CPU/CUDA trainer with complete masked validation and best checkpoints."""

import copy
import math
import time

import numpy as np

from .losses import compute_loss, validate_loss_config


def validation_mae(model, dataset, batch_size):
    error_sum, loss_sum, count = 0.0, 0.0, 0
    loss_cfg = getattr(model, "validation_loss_config", {"prediction": "mae"})
    for batch in dataset.iter_batches(batch_size):
        prediction = model.predict(batch)["prediction"]
        mask = np.asarray(batch["target_mask"], dtype=bool)
        if not np.isfinite(prediction[mask]).all():
            raise FloatingPointError("Validation prediction contains nonfinite observed outputs.")
        error = np.abs(prediction - batch["target"])[mask].astype(np.float64)
        error_sum += float(error.sum(dtype=np.float64))
        if loss_cfg["prediction"] == "huber":
            delta = loss_cfg["huber_delta"]
            values = np.where(error <= delta, 0.5 * error ** 2, delta * (error - 0.5 * delta))
        else:
            values = error
        loss_sum += float(values.sum(dtype=np.float64))
        count += int(mask.sum())
    if count == 0:
        raise ValueError("Validation dataset contains no observed targets.")
    model.last_validation_prediction_loss = loss_sum / count
    return error_sum / count


def fit_neural(model, train_dataset, val_dataset, config):
    import torch

    started = time.perf_counter()
    allowed = {"epochs", "patience", "batch_size", "learning_rate", "weight_decay", "min_delta",
               "gradient_clip", "cpu_threads", "loss", "seeds", "device", "deterministic"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown training parameters: {sorted(unknown)}")
    epochs = int(config.get("epochs", 50))
    patience = int(config.get("patience", 5))
    batch_size = int(config.get("batch_size", 32))
    cpu_threads = int(config.get("cpu_threads", 2))
    learning_rate = float(config.get("learning_rate", 0.001))
    weight_decay = float(config.get("weight_decay", 0.0))
    min_delta = float(config.get("min_delta", 0.0))
    gradient_clip = float(config.get("gradient_clip", 5.0))
    if min(epochs, patience, batch_size, cpu_threads) <= 0:
        raise ValueError("epochs, patience, batch_size, and cpu_threads must be positive.")
    if not all(math.isfinite(value) for value in (learning_rate, weight_decay, min_delta, gradient_clip)):
        raise ValueError("Optimizer settings must be finite.")
    if learning_rate <= 0 or weight_decay < 0 or min_delta < 0 or gradient_clip < 0:
        raise ValueError("learning_rate must be positive; weight_decay/min_delta/gradient_clip nonnegative.")
    loss_config = validate_loss_config(config.get("loss"), model)
    model.validation_loss_config = loss_config
    model.configure_device(config.get("device", "auto"))
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    torch.manual_seed(model.seed)
    if model.device.type == "cuda":
        torch.cuda.manual_seed_all(model.seed)
    deterministic = config.get("deterministic", False)
    if not isinstance(deterministic, bool):
        raise ValueError("deterministic must be true or false.")
    # Strict mode may reject CUDA index_add backward on runtimes without a
    # deterministic implementation. Never hide that restriction or claim
    # CPU/GPU outputs are bitwise identical.
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    torch.set_num_threads(cpu_threads)
    optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_mae = validation_mae(model, val_dataset, batch_size)
    best_state = copy.deepcopy(model.module.state_dict())
    best_epoch, no_improve = 0, 0
    history = [{"epoch": 0, "val_mae": best_mae, "train_loss": None,
                "val_prediction_loss": getattr(model, "last_validation_prediction_loss", None),
                "train_observed_targets": 0, "batches": 0}]
    print(f"[epoch] model={model.name} seed={model.seed} epoch=0/{epochs} "
          f"train_loss=N/A val_MAE_std={best_mae:.8g} initialization=persistence", flush=True)
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.module.train()
        sums = {name: 0.0 for name in ("prediction", "sparse", "direction", "temporal", "total")}
        observed_total, batch_count = 0, 0
        for batch in train_dataset.iter_batches(batch_size, shuffle_seed=model.seed + epoch,
                                                with_previous=loss_config["temporal"] > 0):
            # Entirely missing batches provide neither valid prediction gradients
            # nor a reason to optimize graph priors in isolation.
            if not np.asarray(batch["target_mask"], dtype=bool).any():
                continue
            optimizer.zero_grad(set_to_none=True)
            output = model.forward(batch)
            loss, pieces, observed = compute_loss(model, output, batch, loss_config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {batch_count}.")
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            for name, value in pieces.items():
                sums[name] += value * observed
            observed_total += observed
            batch_count += 1
        if observed_total == 0:
            raise ValueError("Training dataset contains no observed targets.")
        val_mae = validation_mae(model, val_dataset, batch_size)
        record = {"epoch": epoch, "val_mae": val_mae,
                  "val_prediction_loss": getattr(model, "last_validation_prediction_loss", None),
                  "train_loss": {name: value / observed_total for name, value in sums.items()},
                  "train_observed_targets": observed_total, "batches": batch_count}
        history.append(record)
        if val_mae < best_mae - min_delta:
            best_mae, best_epoch = val_mae, epoch
            best_state = copy.deepcopy(model.module.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        record["best_epoch"] = best_epoch
        record["no_improvement_epochs"] = no_improve
        record["epoch_seconds"] = time.perf_counter() - epoch_started
        pieces_text = " ".join(f"{key}={value:.8g}" for key, value in record["train_loss"].items())
        val_loss = record["val_prediction_loss"]
        val_loss_text = "N/A" if val_loss is None else f"{val_loss:.8g}"
        print(f"[epoch] model={model.name} seed={model.seed} epoch={epoch}/{epochs} "
              f"{pieces_text} val_prediction_loss={val_loss_text} val_MAE_std={val_mae:.8g} "
              f"best_epoch={best_epoch} no_improve={no_improve}/{patience} "
              f"seconds={record['epoch_seconds']:.3f}", flush=True)
        if no_improve >= patience:
            print(f"[early_stop] model={model.name} seed={model.seed} epoch={epoch}; "
                  f"restore_best_epoch={best_epoch}", flush=True)
            break
    model.module.load_state_dict(best_state)
    model.module.eval()
    return {"kind": model.kind, "epochs": len(history) - 1,
            "requested_epochs": epochs, "actual_epochs": len(history) - 1,
            "stopped_early": len(history) - 1 < epochs, "best_epoch": best_epoch,
            "selection_metric": "val_mae_standardized", "patience": patience, "min_delta": min_delta,
            "val_mae": best_mae, "history": history, "loss": loss_config,
            "fit_seconds": time.perf_counter() - started, "parameter_count": model.parameter_count,
            "restored_best_checkpoint": True, "initialization": "zero_residual_persistence",
            "deterministic_algorithms": deterministic,
            "peak_gpu_allocated_mb": (torch.cuda.max_memory_allocated(model.device) / (1024 ** 2)
                                      if model.device.type == "cuda" else None),
            **model.device_metadata()}
