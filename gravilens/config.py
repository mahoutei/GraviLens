"""Typed configuration. YAML in, dataclasses out, no machine-specific paths.

Environment overrides:
  GRAVILENS_DATA  dataset root   (default: data/gravilens-v2)
  GRAVILENS_RUNS  artifact root  (default: runs)
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class PhysicsConfig:
    theta_min: float = 0.5          # arcsec
    theta_max: float = 2.0          # arcsec
    pixel_scale: float = 0.101      # arcsec / pixel (Euclid VIS)
    core_radius: float = 0.01       # arcsec, regularises the SIS singularity


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 64
    patch_size: int = 8
    embed_dim: int = 192
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_blocks: int = 4             # per stack (physics + classifier)
    window_size: int = 4
    dropout: float = 0.1
    in_channels: int = 1
    num_classes: int = 3
    hidden_dim: int = 64
    hidden_layers: int = 3


@dataclass(frozen=True)
class DataConfig:
    root: str = "data/gravilens-v2"
    classes: tuple[str, ...] = ("no_sub", "cdm", "axion")
    sizes: tuple[int, int, int] = (6999, 1500, 1500)   # train, val, test
    shard_size: int = 500
    seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-3
    warmup_epochs: int = 5
    label_smoothing: float = 0.05
    theta_loss_weight: float = 0.1
    patience: int = 25
    use_ema: bool = True
    ema_decay: float = 0.999
    aug_start: int = 10
    max_rotation_deg: float = 15.0
    seed: int = 42


@dataclass(frozen=True)
class RunConfig:
    exp_name: str = "v2"
    device: str = "auto"
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(cls: type, raw: Any) -> Any:
    """Recursively build nested frozen dataclasses from a plain dict."""
    if not dataclasses.is_dataclass(cls):
        return raw
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name in raw:
            value = raw[f.name]
            target = hints[f.name]
            if dataclasses.is_dataclass(target):
                kwargs[f.name] = _build(target, value)
            elif get_origin(target) is tuple and isinstance(value, list):
                kwargs[f.name] = tuple(value)
            else:
                kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _build(RunConfig, raw)


def data_root(cfg: RunConfig) -> Path:
    return Path(os.environ.get("GRAVILENS_DATA", cfg.data.root)).expanduser()


def runs_root() -> Path:
    return Path(os.environ.get("GRAVILENS_RUNS", "runs")).expanduser()
