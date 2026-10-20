"""Phase 3 gates: tiny-overfit, train/val-only isolation, evaluation, CLI grid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gravilens.config import DataConfig, ModelConfig, PhysicsConfig, RunConfig, TrainConfig
from gravilens.evaluate import evaluate_run
from gravilens.train import run_training

TINY_MODEL = ModelConfig(
    image_size=16, patch_size=4, embed_dim=32, num_heads=4,
    num_blocks=1, window_size=2, dropout=0.0, hidden_dim=16, hidden_layers=1,
)
TRAIN_CONFIG = TrainConfig(
    epochs=40, batch_size=6, lr=3e-3, min_lr=1e-5, weight_decay=1e-3,
    warmup_epochs=1, label_smoothing=0.0, theta_loss_weight=0.1,
    patience=50, use_ema=False, ema_decay=0.999, aug_start=100,
    max_rotation_deg=15.0, seed=0,
)


def _write_split(root: Path, split: str, n_systems: int, seed: int,
                 size: int = 16) -> None:
    """Write one class-separable matched shard without invoking the simulator."""
    rng = np.random.default_rng(seed)
    n = 3 * n_systems
    images = np.zeros((n, size, size), dtype=np.float32)
    sources = np.zeros_like(images)
    labels = np.tile(np.arange(3, dtype=np.int8), n_systems)
    for system in range(n_systems):
        source = rng.random((size, size), dtype=np.float32)
        source[size // 2 - 2:size // 2 + 2, size // 2 - 2:size // 2 + 2] += 2.0
        source /= source.max()
        for cls in range(3):
            index = 3 * system + cls
            image = rng.random((size, size), dtype=np.float32) * 0.05
            row = 2 + 4 * cls
            image[row:row + 2, 3:size - 3] += 1.0
            images[index] = image
            sources[index] = source
    images /= images.max(axis=(1, 2), keepdims=True)

    out = root / split
    out.mkdir(parents=True)
    np.savez(
        out / "shard_000000.npz",
        images=images, sources=sources, labels=labels,
        theta_e=np.ones(n, dtype=np.float32),
        system_id=np.repeat(np.arange(n_systems), 3),
        shear1=np.zeros(n, dtype=np.float32),
        shear2=np.zeros(n, dtype=np.float32),
        lens_angle=np.zeros(n, dtype=np.float32),
        galaxy_index=np.zeros(n, dtype=np.int32),
        source_mag=np.full(n, 16.0, dtype=np.float32),
        meta=np.asarray(json.dumps({"split": split, "seed": seed})),
    )


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("tiny-data")
    _write_split(root, "train", 12, 0)
    _write_split(root, "val", 6, 1)
    _write_split(root, "test", 6, 2)
    return root


@pytest.fixture(scope="module")
def tiny_run(tiny_data: Path, tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny-runs") / "gravilens_full_seed0"
    cfg = RunConfig(
        exp_name="tiny", device="cpu", physics=PhysicsConfig(),
        model=TINY_MODEL,
        data=DataConfig(root=str(tiny_data), sizes=(36, 18, 18)),
        train=TRAIN_CONFIG,
    )
    result = run_training(cfg, "gravilens", "full", seed=0,
                          data_dir=tiny_data, out_dir=out, device="cpu",
                          epochs=40, batch_size=6, patience=50, lr=3e-3)
    return tiny_data, out, result


def test_tiny_run_overfits_and_writes_artifacts(tiny_run):
    _, run_dir, result = tiny_run
    assert result["best_val"]["accuracy"] >= 0.9
    assert result["best_val"]["macro_f1"] >= 0.9
    for name in ("best.pt", "last.pt", "manifest.json", "history.json", "val_metrics.json"):
        assert (run_dir / name).exists(), name
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "finished"
    assert manifest["data"]["train_size"] == 36
    assert manifest["data"]["val_size"] == 18
    assert manifest["params"] > 0
    assert "git_sha" in manifest
    assert manifest["library_versions"]
    assert len(json.loads((run_dir / "history.json").read_text())) == 40


def test_training_loads_only_train_and_val(tiny_data, tmp_path_factory, monkeypatch):
    import gravilens.train as train_module

    calls = []
    real_load_split = train_module.load_split

    def spy(root, split, *args, **kwargs):
        calls.append(split)
        return real_load_split(root, split, *args, **kwargs)

    monkeypatch.setattr(train_module, "load_split", spy)
    cfg = RunConfig(
        exp_name="spy", device="cpu", physics=PhysicsConfig(),
        model=TINY_MODEL,
        data=DataConfig(root=str(tiny_data), sizes=(36, 18, 18)),
        train=TRAIN_CONFIG,
    )
    out = tmp_path_factory.mktemp("spy-run") / "run"
    run_training(cfg, "gravilens", "full", seed=0, data_dir=tiny_data,
                 out_dir=out, device="cpu", epochs=1, batch_size=12)
    assert calls == ["train", "val"]


def test_evaluate_writes_metrics_on_test_only(tiny_run):
    tiny_data, run_dir, _ = tiny_run
    payload = evaluate_run(run_dir, data_dir=tiny_data, device="cpu",
                           physics_per_class=2)
    assert (run_dir / "metrics.json").exists()
    assert payload["split"] == "test"
    assert payload["test"]["n"] == 18
    assert payload["test"]["accuracy"] >= 0.9
    assert "macro_auc" in payload["test"]
    assert payload["physics_slice"]["n"] == 6
    assert payload["theta"] is not None
    assert payload["theta"]["n"] == 6
    assert payload["theta"]["supervised"] is True
    assert "train_mean_baseline_mae" in payload["theta"]
    assert set(payload["source_ssim"]) == {
        "ssim_model", "ssim_oracle_bilinear",
        "ssim_oracle_scatter", "ssim_lensed_image",
    }
    assert all(np.isfinite(value) for value in payload["source_ssim"].values())


def test_baseline_run_skips_physics_metrics(tiny_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("baseline-run") / "vit_seed0"
    cfg = RunConfig(
        exp_name="baseline", device="cpu", physics=PhysicsConfig(),
        model=TINY_MODEL,
        data=DataConfig(root=str(tiny_data), sizes=(36, 18, 18)),
        train=TRAIN_CONFIG,
    )
    result = run_training(cfg, "vit", "full", seed=0, data_dir=tiny_data,
                          out_dir=out, device="cpu", epochs=1, batch_size=12)
    assert result["params"] > 1_000_000
    payload = evaluate_run(out, data_dir=tiny_data, device="cpu")
    assert payload["theta"] is None
    assert payload["source_ssim"] is None
    assert payload["test"]["n"] == 18


def test_generate_command_delegates_with_its_options(monkeypatch):
    import gravilens.simulate as simulate
    from gravilens.cli import main

    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(simulate, "main", fake_main)
    assert main(["generate", "--config", "configs/data.yaml",
                 "--splits", "test", "--limit", "3", "--shard-size", "9"]) == 0
    assert captured["argv"][:2] == ["--config", "configs/data.yaml"]
    assert "--splits" in captured["argv"] and "test" in captured["argv"]
    assert "--limit" in captured["argv"] and "3" in captured["argv"]


def test_ablation_dry_run_is_21_runs(capsys):
    from gravilens.cli import main

    assert main(["ablate", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "21 runs" in output
    assert "gravilens_full_seed42" in output
    assert "vitsd_seed777" in output
