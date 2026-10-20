"""Evaluate a trained checkpoint on the held-out split.

This is the only module that loads ``test``. It writes ``metrics.json`` once
per run; ``ablate`` skips runs that already have one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import fingerprint, load_split, stratified_indices
from .physics import sis_invert_bilinear, sis_invert_scatter
from .train import (
    LensDataset,
    git_sha,
    load_checkpoint,
    macro_f1,
    pick_device,
    theta_metrics,
)


def _ssim_mean(pred: np.ndarray, target: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(np.mean([
        structural_similarity(p, t, data_range=1.0)
        for p, t in zip(pred, target)
    ]))


def _macro_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except ValueError:
        return float("nan")


@torch.no_grad()
def evaluate_run(run_dir: str | Path, data_dir: str | Path | None = None,
                 split: str = "test", device: str | None = None,
                 batch_size: int = 128, physics_per_class: int = 167) -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    data_dir = Path(data_dir or manifest["data"]["root"])
    torch_device = pick_device(device or manifest.get("device", "auto"))
    model, checkpoint = load_checkpoint(run_dir / "best.pt", torch_device)
    model_name = checkpoint["model_name"]
    variant = checkpoint["variant"]

    data = load_split(data_dir, split)
    dataset = LensDataset(data, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=torch_device.type == "cuda")

    logits_all, labels_all, theta_all, source_all = [], [], [], []
    for images, labels, _ in loader:
        images = images.to(torch_device)
        labels = labels.to(torch_device)
        if model_name == "gravilens":
            logits, theta, source = model(images, return_aux=True)
            theta_all.append(theta.detach().cpu())
            source_all.append(source.detach().cpu())
        else:
            logits = model(images)
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())

    logits = torch.cat(logits_all)
    y_true = torch.cat(labels_all).numpy()
    y_pred = logits.argmax(dim=1).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    f1, per_class, confusion = macro_f1(y_true, y_pred)
    metrics = {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": f1,
        "macro_auc": _macro_auc(y_true, probs),
        "per_class_f1": per_class,
        "confusion": confusion,
    }

    per_class = min(int(physics_per_class),
                    int(np.bincount(data.labels, minlength=3).min()))
    physics_index = stratified_indices(data.labels, per_class, seed=42)
    physics_meta = {
        "n": int(len(physics_index)),
        "per_class": per_class,
        "seed": 42,
        "sha256": hashlib.sha256(
            physics_index.astype(np.int64).tobytes()).hexdigest()[:16],
    }

    theta_payload = None
    source_payload = None
    if model_name == "gravilens":
        theta_pred = torch.cat(theta_all).numpy()[physics_index]
        theta_true = data.theta_e[physics_index]
        theta_payload = theta_metrics(theta_pred, theta_true)
        theta_payload["n"] = int(len(physics_index))
        theta_payload["train_mean_baseline_mae"] = float(np.mean(np.abs(
            theta_true - checkpoint["train_theta_mean"])))
        theta_payload["supervised"] = variant in ("full", "dist")

        model_source = torch.cat(source_all).numpy()[physics_index, 0]
        images = torch.from_numpy(data.images[physics_index]).float()
        true_theta = torch.from_numpy(theta_true).float()
        oracle_bilinear = sis_invert_bilinear(images, true_theta).numpy()[:, 0]
        oracle_scatter = sis_invert_scatter(images, true_theta).numpy()[:, 0]
        sources = data.sources[physics_index]
        source_payload = {
            "ssim_model": _ssim_mean(model_source, sources),
            "ssim_oracle_bilinear": _ssim_mean(oracle_bilinear, sources),
            "ssim_oracle_scatter": _ssim_mean(oracle_scatter, sources),
            "ssim_lensed_image": _ssim_mean(data.images[physics_index], sources),
        }

    payload = {
        "run": manifest["run"],
        "model": model_name,
        "variant": variant,
        "seed": checkpoint.get("seed", manifest.get("seed")),
        "params": int(manifest.get("params", 0)),
        "checkpoint": str(run_dir / "best.pt"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": split,
        "data_fingerprint": fingerprint(data_dir),
        "git_sha": git_sha(),
        "test": metrics,
        "physics_slice": physics_meta,
        "theta": theta_payload,
        "source_ssim": source_payload,
    }
    out = run_dir / "metrics.json"
    tmp = out.with_name("metrics.json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    tmp.replace(out)

    print(f"{manifest['run']}: {split} acc {metrics['accuracy']:.4f} "
          f"macro-F1 {metrics['macro_f1']:.4f}", flush=True)
    if theta_payload:
        print(f"  theta MAE {theta_payload['theta_mae']:.4f} "
              f"(constant {theta_payload['train_mean_baseline_mae']:.4f})", flush=True)
    return payload
