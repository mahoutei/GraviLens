from pathlib import Path

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
