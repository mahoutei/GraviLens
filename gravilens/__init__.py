"""GraviLens: physics-informed lensing for dark-matter substructure classification."""

from .config import (
    DataConfig,
    ModelConfig,
    PhysicsConfig,
    RunConfig,
    TrainConfig,
    load_config,
)
from .models import (
    VARIANTS,
    CaiTModel,
    GraviLensP,
    ViTModel,
    ViTSDModel,
    build_model,
    count_trainable_params,
    match_budget,
)

__all__ = [
    "DataConfig",
    "ModelConfig",
    "PhysicsConfig",
    "RunConfig",
    "TrainConfig",
    "load_config",
    "VARIANTS",
    "CaiTModel",
    "GraviLensP",
    "ViTModel",
    "ViTSDModel",
    "build_model",
    "count_trainable_params",
    "match_budget",
]
