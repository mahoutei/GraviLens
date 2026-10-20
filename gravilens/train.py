"""Train one run. This module deliberately never loads the test split.

One loop for GraviLens variants and the parameter-matched baselines. The test
split is touched only by ``evaluate.py``, after checkpoint selection.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig, PhysicsConfig, RunConfig, runs_root
from .dataset import SplitData, fingerprint, load_split
from .models import BASELINE_SPECS, VARIANTS, build_model, count_trainable_params


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL).strip())
    except Exception:
        return None


def library_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    names = ("torch", "numpy", "PyYAML", "scipy", "scikit-image",
             "scikit-learn", "lenstronomy", "pyHalo", "astropy")
    found = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            pass
    return found


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_name(model_name: str, variant: str, seed: int) -> str:
    if model_name == "gravilens":
        return f"gravilens_{variant}_seed{seed}"
    return f"{model_name}_seed{seed}"


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    os.replace(tmp, path)


def _atomic_torch_save(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cpu_state(model: torch.nn.Module) -> dict:
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Data + augmentation
# ---------------------------------------------------------------------------

class LensDataset(Dataset):
    def __init__(self, data: SplitData, augment: bool = False,
                 max_rotation_deg: float = 15.0):
        self.images = torch.from_numpy(data.images).float().unsqueeze(1)
        self.labels = torch.from_numpy(data.labels).long()
        self.theta = torch.from_numpy(data.theta_e).float()
        self.augment = augment
        self.max_rotation_deg = max_rotation_deg

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.images[index]
        if self.augment:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=[-1])
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=[-2])
            angle = (torch.rand(()) * 2 - 1) * self.max_rotation_deg
            image = _rotate(image, float(angle))
        return image, self.labels[index], self.theta[index]


def _rotate(image: torch.Tensor, angle_deg: float) -> torch.Tensor:
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    affine = torch.tensor([[[c, -s, 0.0], [s, c, 0.0]]],
                          dtype=image.dtype, device=image.device)
    grid = F.affine_grid(affine, (1, 1) + image.shape[-2:], align_corners=False)
    return F.grid_sample(image.unsqueeze(0), grid, align_corners=False,
                         padding_mode="zeros").squeeze(0)


def make_loader(data: SplitData, batch_size: int, shuffle: bool,
                augment: bool = False, max_rotation_deg: float = 15.0,
                pin_memory: bool = False) -> DataLoader:
    dataset = LensDataset(data, augment=augment, max_rotation_deg=max_rotation_deg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=pin_memory)


# ---------------------------------------------------------------------------
# Metrics + EMA
# ---------------------------------------------------------------------------

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(f1.mean()), f1.tolist(), cm.tolist()


def theta_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    std = true.std()
    pearson = float(np.corrcoef(pred, true)[0, 1]) if std > 1e-9 else float("nan")
    return {
        "theta_mae": float(np.abs(err).mean()),
        "theta_rmse": float(np.sqrt((err ** 2).mean())),
        "theta_bias": float(err.mean()),
        "theta_pearson": pearson,
        "theta_within_10pct": float(np.mean(np.abs(err) <= 0.1 * np.abs(true))),
    }


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {key: value.detach().clone()
                       for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].mul_(self.decay).add_(value.detach(),
                                                      alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


# ---------------------------------------------------------------------------
# Validation + checkpoint loading
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                    model_name: str, label_smoothing: float = 0.0) -> dict:
    model.eval()
    logits_all, labels_all, theta_all, true_all = [], [], [], []
    loss_sum, count = 0.0, 0
    for images, labels, theta_true in loader:
        images = images.to(device)
        labels = labels.to(device)
        if model_name == "gravilens":
            logits, theta, _ = model(images, return_aux=True)
            theta_all.append(theta.detach().cpu())
            true_all.append(theta_true.detach().cpu())
        else:
            logits = model(images)
        loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        loss_sum += float(loss) * len(labels)
        count += len(labels)
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())

    y_true = torch.cat(labels_all).numpy()
    y_pred = torch.cat(logits_all).argmax(dim=1).numpy()
    f1, per_class, confusion = macro_f1(y_true, y_pred)
    result = {
        "loss": loss_sum / max(1, count),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": f1,
        "per_class_f1": per_class,
        "confusion": confusion,
        "n": int(count),
    }
    if theta_all:
        result.update(theta_metrics(torch.cat(theta_all).numpy(),
                                     torch.cat(true_all).numpy()))
    return result


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**checkpoint["model_config"])
    physics = PhysicsConfig(**checkpoint["physics"])
    spec = checkpoint.get("baseline_spec")
    kwargs = dict(spec) if spec else {}
    if checkpoint["model_name"] == "gravilens":
        kwargs.update(theta_min=physics.theta_min, theta_max=physics.theta_max,
                      pixel_scale=physics.pixel_scale, core_radius=physics.core_radius)
    model = build_model(checkpoint["model_name"], cfg, checkpoint["variant"], **kwargs)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(cfg: RunConfig, model_name: str, variant: str, seed: int,
                 data_dir: str | Path | None = None,
                 out_dir: str | Path | None = None,
                 device: str | None = None,
                 epochs: int | None = None,
                 batch_size: int | None = None,
                 patience: int | None = None,
                 lr: float | None = None) -> dict:
    if model_name == "gravilens" and variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if model_name != "gravilens" and model_name not in BASELINE_SPECS:
        raise ValueError(f"unknown model {model_name!r}")

    train_cfg = replace(
        cfg.train, seed=seed,
        epochs=cfg.train.epochs if epochs is None else epochs,
        batch_size=cfg.train.batch_size if batch_size is None else batch_size,
        patience=cfg.train.patience if patience is None else patience,
        lr=cfg.train.lr if lr is None else lr,
    )
    data_dir = Path(data_dir or cfg.data.root)
    name = run_name(model_name, variant, seed)
    out_dir = Path(out_dir or runs_root() / name)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    torch_device = pick_device(device or cfg.device)
    pin_memory = torch_device.type == "cuda"

    train_data = load_split(data_dir, "train")
    val_data = load_split(data_dir, "val")
    train_loader = make_loader(train_data, train_cfg.batch_size, shuffle=True,
                               max_rotation_deg=train_cfg.max_rotation_deg,
                               pin_memory=pin_memory)
    val_loader = make_loader(val_data, train_cfg.batch_size * 2, shuffle=False,
                             pin_memory=pin_memory)

    kwargs = dict(BASELINE_SPECS.get(model_name, {}))
    if model_name == "gravilens":
        kwargs.update(theta_min=cfg.physics.theta_min, theta_max=cfg.physics.theta_max,
                      pixel_scale=cfg.physics.pixel_scale,
                      core_radius=cfg.physics.core_radius)
    model = build_model(model_name, cfg.model, variant, **kwargs).to(torch_device)
    params = count_trainable_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                                  weight_decay=train_cfg.weight_decay)
    use_ema = train_cfg.use_ema
    ema = EMA(model, train_cfg.ema_decay) if use_ema else None
    eval_model = copy.deepcopy(model) if use_ema else model

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, train_cfg.epochs * steps_per_epoch)
    warmup_steps = max(0, train_cfg.warmup_epochs * steps_per_epoch)

    manifest = {
        "run": name,
        "status": "running",
        "model": model_name,
        "variant": variant,
        "seed": seed,
        "params": params,
        "baseline_spec": BASELINE_SPECS.get(model_name),
        "model_config": asdict(cfg.model),
        "physics": asdict(cfg.physics),
        "train_config": asdict(train_cfg),
        "data": {
            "root": str(data_dir),
            "fingerprint": fingerprint(data_dir),
            "train_size": len(train_data),
            "val_size": len(val_data),
            "train_classes": train_data.class_counts(),
            "val_classes": val_data.class_counts(),
            "train_theta_mean": float(train_data.theta_e.mean()),
        },
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "library_versions": library_versions(),
        "torch": torch.__version__,
        "device": str(torch_device),
        "started": _now(),
    }
    _write_json(out_dir / "manifest.json", manifest)

    history = []
    best_score, best_epoch, best_val, best_state = -1.0, -1, None, None
    step = 0

    for epoch in range(train_cfg.epochs):
        model.train()
        train_loader.dataset.augment = epoch >= train_cfg.aug_start
        loss_sum, seen = 0.0, 0
        for images, labels, theta_true in train_loader:
            images = images.to(torch_device)
            labels = labels.to(torch_device)
            theta_true = theta_true.to(torch_device)

            lr_now = _lr_at(step, total_steps, warmup_steps,
                            train_cfg.lr, train_cfg.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)
            if model_name == "gravilens":
                logits, theta, _ = model(images, return_aux=True)
            else:
                logits, theta = model(images), None
            loss = F.cross_entropy(logits, labels,
                                   label_smoothing=train_cfg.label_smoothing)
            if (theta is not None and variant in ("full", "dist")
                    and train_cfg.theta_loss_weight > 0):
                loss = loss + train_cfg.theta_loss_weight * F.smooth_l1_loss(
                    theta, theta_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            loss_sum += loss.detach().item() * len(labels)
            seen += len(labels)
            step += 1

        if ema is not None:
            ema.copy_to(eval_model)
        val = evaluate_loader(eval_model, val_loader, torch_device, model_name,
                              train_cfg.label_smoothing)
        record = {"epoch": epoch, "lr": lr_now,
                  "train_loss": loss_sum / max(1, seen), **val}
        history.append(record)

        score = val["macro_f1"]
        if score > best_score + 1e-6:
            best_score, best_epoch, best_val = score, epoch, val
            best_state = _cpu_state(eval_model)
            _atomic_torch_save(out_dir / "best.pt", {
                "model": best_state,
                "model_name": model_name,
                "variant": variant,
                "model_config": asdict(cfg.model),
                "physics": asdict(cfg.physics),
                "baseline_spec": BASELINE_SPECS.get(model_name),
                "epoch": epoch,
                "seed": seed,
                "val": val,
                "params": params,
                "train_theta_mean": float(train_data.theta_e.mean()),
                "ema": use_ema,
                "data_fingerprint": manifest["data"]["fingerprint"],
                "git_sha": manifest["git_sha"],
            })
            _write_json(out_dir / "val_metrics.json", dict(best_val, epoch=epoch))
            manifest["best_epoch"] = best_epoch
            manifest["best_val"] = best_val
            _write_json(out_dir / "manifest.json", manifest)

        _atomic_torch_save(out_dir / "last.pt", {
            "model": _cpu_state(model),
            "optimizer": optimizer.state_dict(),
            "ema": {key: value.detach().cpu() for key, value in ema.shadow.items()}
            if ema is not None else None,
            "epoch": epoch,
            "history": history,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "best_val": best_val,
        })

        print(f"{name}: epoch {epoch:02d} train {record['train_loss']:.4f} "
              f"val {val['loss']:.4f} acc {val['accuracy']:.3f} "
              f"f1 {val['macro_f1']:.3f}", flush=True)

        if best_epoch >= 0 and epoch - best_epoch >= train_cfg.patience:
            print(f"{name}: early stop at {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")

    _write_json(out_dir / "history.json", history)
    manifest.update({
        "status": "finished",
        "finished": _now(),
        "best_epoch": best_epoch,
        "best_val": best_val,
    })
    _write_json(out_dir / "manifest.json", manifest)
    return {
        "run": name,
        "run_dir": str(out_dir),
        "params": params,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "history": history,
    }


def _lr_at(step: int, total: int, warmup: int, base: float, min_lr: float) -> float:
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * progress))
