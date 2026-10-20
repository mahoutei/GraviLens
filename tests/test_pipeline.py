"""Pipeline gates: sharding, resume, matched triplets, and one real render."""

from pathlib import Path

import numpy as np
import pytest

from gravilens.dataset import load_split, stratified_indices, validate_matched
from gravilens.simulate import (
    SimConfig,
    generate_split,
    render_triplet,
    sample_system,
    seed_for,
)


def _fake_render(system, seed):
    rng = np.random.default_rng(seed)
    images = rng.random((3, 8, 8), dtype=np.float32)
    source = system.galaxy[:8, :8].astype(np.float32)
    source = np.clip(source, 0, None)
    source = np.repeat(source[None], 3, axis=0)
    source = source / max(float(source.max()), 1e-8)
    return images, source, system.theta_e


def _tiny_config():
    return SimConfig(image_size=8, galaxy_smoothing_sigma=0.5)


def test_shards_resume_and_validate(tmp_path):
    pool = np.random.default_rng(0).random((3, 64, 64), dtype=np.float32)
    stats = generate_split(
        "test", n_images=6, galaxy_pool=pool, cfg=_tiny_config(),
        out_root=tmp_path, seed=42, shard_size=3, render_fn=_fake_render,
        verbose=False)
    assert stats["written_shards"] == 2
    assert stats["generated_triplets"] == 2

    data = load_split(tmp_path, "test")
    assert len(data) == 6
    assert data.class_counts() == [2, 2, 2]
    assert set(np.unique(data.system_id)) == {0, 1}
    assert len(np.unique(data.theta_e)) == 2

    resumed = generate_split(
        "test", n_images=6, galaxy_pool=pool, cfg=_tiny_config(),
        out_root=tmp_path, seed=42, shard_size=3, render_fn=_fake_render,
        verbose=False)
    assert resumed["written_shards"] == 0
    assert resumed["skipped_shards"] == 2


def test_corrupt_shard_is_regenerated(tmp_path):
    pool = np.random.default_rng(1).random((2, 64, 64), dtype=np.float32)
    kwargs = dict(n_images=3, galaxy_pool=pool, cfg=_tiny_config(),
                  out_root=tmp_path, seed=7, shard_size=3,
                  render_fn=_fake_render, verbose=False)
    generate_split("val", **kwargs)
    shard = Path(tmp_path) / "val" / "shard_000000.npz"
    shard.write_bytes(b"not an npz")

    stats = generate_split("val", **kwargs)
    assert stats["written_shards"] == 1
    assert len(load_split(tmp_path, "val")) == 3


def test_matched_gate_rejects_a_tampered_triplet(tmp_path):
    pool = np.random.default_rng(2).random((2, 64, 64), dtype=np.float32)
    generate_split(
        "test", n_images=3, galaxy_pool=pool, cfg=_tiny_config(),
        out_root=tmp_path, seed=3, shard_size=3, render_fn=_fake_render,
        verbose=False)
    data = load_split(tmp_path, "test")
    validate_matched(data)

    data.theta_e[1] += 0.1
    with pytest.raises(ValueError, match="theta_e differs"):
        validate_matched(data)


def test_generation_is_deterministic_with_a_fixed_renderer(tmp_path):
    pool = np.random.default_rng(9).random((2, 64, 64), dtype=np.float32)
    for name in ("first", "second"):
        generate_split(
            "test", n_images=3, galaxy_pool=pool, cfg=_tiny_config(),
            out_root=Path(tmp_path) / name, seed=11, shard_size=3,
            render_fn=_fake_render, verbose=False)
    first = load_split(Path(tmp_path) / "first", "test")
    second = load_split(Path(tmp_path) / "second", "test")
    for field in ("images", "sources", "labels", "theta_e", "system_id",
                  "shear1", "shear2", "lens_angle", "galaxy_index", "source_mag"):
        assert np.array_equal(getattr(first, field), getattr(second, field)), field


def test_stratified_slice_is_deterministic():
    labels = np.tile(np.arange(3), 20)
    first = stratified_indices(labels, 4, seed=11)
    second = stratified_indices(labels, 4, seed=11)
    assert np.array_equal(first, second)
    assert np.bincount(labels[first], minlength=3).tolist() == [4, 4, 4]


def test_seed_for_is_stable_and_split_specific():
    assert seed_for("train", 5, 42) == seed_for("train", 5, 42)
    assert seed_for("train", 5, 42) != seed_for("test", 5, 42)


def test_one_real_triplet_has_a_shared_macro_lens():
    pytest.importorskip("scipy")
    pytest.importorskip("lenstronomy")
    pytest.importorskip("pyHalo")

    pool = np.random.default_rng(0).random((2, 256, 256), dtype=np.float32)
    cfg = SimConfig()
    system = sample_system(pool, 123, cfg)
    images, sources, theta = render_triplet(system, cfg, 123)

    assert images.shape == (3, 64, 64)
    assert sources.shape == (3, 64, 64)
    assert np.isfinite(images).all() and np.isfinite(sources).all()
    assert images.min() >= 0
    assert np.allclose(images.max(axis=(1, 2)), 1.0, atol=1e-5)
    assert theta == pytest.approx(system.theta_e)
    assert np.allclose(sources, sources[0])
    diffs = [np.abs(images[i] - images[j]).max() for i in range(3) for j in range(i)]
    assert max(diffs) > 1e-4
