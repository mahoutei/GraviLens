import json
from pathlib import Path

from gravilens.cli import _run_is_current
from gravilens.config import load_config

CONFIG = Path(__file__).parents[1] / "configs" / "ablation.yaml"
DATA_CONFIG = Path(__file__).parents[1] / "configs" / "data.yaml"


def test_nested_yaml_builds_typed_dataclasses():
    cfg = load_config(CONFIG)
    assert cfg.model.embed_dim == 192
    assert cfg.data.classes == ("no_sub", "cdm", "axion")
    assert cfg.data.sizes == (6999, 1500, 1500)
    assert cfg.train.epochs == 50
    assert cfg.physics.theta_min == 0.5


def test_data_config_loads_with_training_defaults():
    cfg = load_config(DATA_CONFIG)
    assert cfg.data.root == "data/gravilens-v2"
    assert cfg.data.sizes == (6999, 1500, 1500)
    assert cfg.train.epochs == 50


def test_existing_run_counts_as_current_only_for_the_same_recipe_and_data(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({
        "train_config": {"epochs": 50, "batch_size": 64},
        "data": {"fingerprint": "abc"},
    }))
    assert _run_is_current(out, {"epochs": 50, "batch_size": 64}, "abc")
    assert not _run_is_current(out, {"epochs": 2, "batch_size": 64}, "abc")
    assert not _run_is_current(out, {"epochs": 50, "batch_size": 64}, "def")


def test_ablate_chunks_the_grid_without_repeating_a_group(capsys):
    from gravilens.cli import main

    assert main(["ablate", "--config", str(CONFIG), "--dry-run"]) == 0
    assert capsys.readouterr().out.strip().endswith("21 runs")

    assert main(["ablate", "--config", str(CONFIG), "--dry-run",
                 "--variants", "full", "no_sis", "no_sup"]) == 0
    out = capsys.readouterr().out
    assert out.strip().endswith("9 runs")
    assert "vit_seed42" not in out          # baselines are a separate chunk

    assert main(["ablate", "--config", str(CONFIG), "--dry-run",
                 "--baselines", "vit", "cait", "vitsd"]) == 0
    assert capsys.readouterr().out.strip().endswith("9 runs")
