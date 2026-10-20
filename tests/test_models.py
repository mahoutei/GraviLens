"""Model gates: shapes, bounds, dead parameters, source sensitivity, budget match."""

import pytest
import torch
import torch.nn.functional as F

from gravilens.config import ModelConfig
from gravilens.models import (
    BASELINES,
    BASELINE_SPECS,
    VARIANTS,
    GraviLensP,
    build_model,
    count_trainable_params,
    match_budget,
)

TINY = ModelConfig(
    image_size=32, patch_size=8, embed_dim=32, num_heads=4,
    num_blocks=1, window_size=4, dropout=0.0, num_classes=3,
)


def _batch(n: int = 2, size: int = 32):
    torch.manual_seed(0)
    return torch.rand(n, size, size)


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_shapes_and_bounds(variant):
    model = GraviLensP(TINY, variant)
    logits, theta, source = model(_batch(), return_aux=True)
    assert logits.shape == (2, 3)
    assert theta.shape == (2,)
    assert source.shape == (2, 1, 32, 32)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(theta).all()
    assert torch.isfinite(source).all()
    assert (theta >= 0.5 - 1e-6).all() and (theta <= 2.0 + 1e-6).all()


def test_full_model_has_no_dead_parameters():
    model = GraviLensP(TINY, "full")
    logits = model(_batch())
    F.cross_entropy(logits, torch.tensor([0, 1])).backward()
    dead = [name for name, p in model.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"parameters receive no gradient: {dead}"


def test_dist_variant_has_no_dead_parameters():
    model = GraviLensP(TINY, "dist")
    logits = model(_batch())
    F.cross_entropy(logits, torch.tensor([0, 1])).backward()
    dead = [name for name, p in model.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"parameters receive no gradient: {dead}"


def test_no_sis_only_leaves_the_physics_stack_unused():
    """no_sis is a control: the physics stack is intentionally disconnected."""
    model = GraviLensP(TINY, "no_sis")
    logits = model(_batch())
    F.cross_entropy(logits, torch.tensor([0, 1])).backward()
    dead = [name for name, p in model.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    assert dead, "no_sis should leave the physics stack unused"
    assert all(name.startswith(("physics_stack", "theta_head")) for name in dead), dead


def test_full_and_no_sis_have_the_same_parameter_count():
    full = count_trainable_params(GraviLensP(TINY, "full"))
    no_sis = count_trainable_params(GraviLensP(TINY, "no_sis"))
    assert full == no_sis


def test_source_intervention_changes_logits():
    model = GraviLensP(TINY, "full").eval()
    image = _batch()

    with torch.no_grad():
        base = model(image)

    handle = model.source_tokenizer.register_forward_hook(
        lambda module, inputs, output: torch.zeros_like(output))
    try:
        with torch.no_grad():
            intervened = model(image)
    finally:
        handle.remove()

    assert (base - intervened).abs().max() > 1e-4


def test_forward_rejects_wrong_input_shape():
    model = GraviLensP(TINY, "full")
    with pytest.raises(ValueError):
        model(torch.rand(2, 3, 32, 32))


def test_baselines_match_gravilens_parameter_budget():
    cfg = ModelConfig()
    target = count_trainable_params(GraviLensP(cfg, "full"))
    for name in BASELINES:
        match = match_budget(cfg, name, target)
        assert abs(match["params"] - target) / target <= 0.10


def test_shipped_baseline_specs_are_matched():
    cfg = ModelConfig()
    target = count_trainable_params(GraviLensP(cfg, "full"))
    for name, spec in BASELINE_SPECS.items():
        params = count_trainable_params(build_model(name, cfg, **spec))
        assert abs(params - target) / target <= 0.10
