"""Artifact audit gates: all 21 runs present, figures from JSON only."""

from __future__ import annotations

import json
from pathlib import Path

from gravilens.figures import audit_runs, build_figures, expected_runs
from gravilens.models import GraviLensP, ModelConfig, count_trainable_params
from gravilens.train import run_name

TARGET = count_trainable_params(GraviLensP(ModelConfig(), "full"))
BASELINE = 4_035_459


def _fake_run(root: Path, model: str, variant: str, seed: int) -> None:
    name = run_name(model, variant, seed)
    run_dir = root / name
    run_dir.mkdir(parents=True)
    params = TARGET if model == "gravilens" else BASELINE
    fingerprint = "0123456789abcdef"
    manifest = {
        "run": name, "status": "finished", "model": model, "variant": variant,
        "seed": seed, "params": params, "library_versions": {"torch": "test"},
        "data": {"root": "data/test", "fingerprint": fingerprint},
    }
    test = {
        "n": 18, "accuracy": 0.90, "macro_f1": 0.90, "macro_auc": 0.95,
        "per_class_f1": [0.9, 0.9, 0.9],
        "confusion": [[6, 0, 0], [0, 6, 0], [0, 0, 6]],
    }
    theta = None
    source = None
    if model == "gravilens":
        theta = {
            "theta_mae": 0.10, "theta_rmse": 0.12, "theta_bias": 0.0,
            "theta_pearson": 0.8, "theta_within_10pct": 0.7,
            "train_mean_baseline_mae": 0.30, "supervised": variant in ("full", "dist"),
        }
        source = {
            "ssim_model": 0.8, "ssim_oracle_bilinear": 0.9,
            "ssim_oracle_scatter": 0.85, "ssim_lensed_image": 0.3,
        }
    metrics = {
        "run": name, "model": model, "variant": variant, "seed": seed,
        "params": params, "split": "test", "data_fingerprint": fingerprint,
        "physics_slice": {"n": 6}, "test": test,
        "theta": theta, "source_ssim": source,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "best.pt").write_bytes(b"fake checkpoint")


def test_audit_passes_on_complete_grid_and_figures_read_only_artifacts(tmp_path):
    root = tmp_path / "runs"
    for model, variant, seed in expected_runs():
        _fake_run(root, model, variant, seed)

    assert audit_runs(root, verbose=False) == []
    assert build_figures(root) == 0
    for name in ("summary.json", "main_table.md", "macro_f1.png", "physics.png"):
        assert (root / "figures" / name).exists(), name


def test_audit_fails_when_a_run_is_missing(tmp_path):
    root = tmp_path / "runs"
    for model, variant, seed in expected_runs():
        _fake_run(root, model, variant, seed)
    (root / run_name("gravilens", "full", 42) / "metrics.json").unlink()
    assert audit_runs(root, verbose=False)
