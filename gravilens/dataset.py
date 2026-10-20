"""Load and validate the sharded v2 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

CLASSES = ("no_sub", "cdm", "axion")
ARRAY_KEYS = (
    "images", "sources", "labels", "theta_e", "system_id",
    "shear1", "shear2", "lens_angle", "galaxy_index", "source_mag",
)


@dataclass
class SplitData:
    images: np.ndarray          # [N, S, S] float32, max-normalised
    sources: np.ndarray         # [N, S, S] float32, unlensed ground truth
    labels: np.ndarray          # [N] int8, 0=no_sub 1=cdm 2=axion
    theta_e: np.ndarray         # [N] float32, arcsec
    system_id: np.ndarray       # [N] int64, same for the three classes
    shear1: np.ndarray
    shear2: np.ndarray
    lens_angle: np.ndarray
    galaxy_index: np.ndarray
    source_mag: np.ndarray

    def __len__(self) -> int:
        return int(len(self.labels))

    def class_counts(self) -> list[int]:
        return np.bincount(self.labels, minlength=len(CLASSES)).tolist()

    def select(self, index: np.ndarray) -> "SplitData":
        return SplitData(**{f.name: getattr(self, f.name)[index] for f in fields(self)})


def load_split(root: str | Path, split: str, validate: bool = True) -> SplitData:
    split_dir = Path(root) / split
    files = sorted(split_dir.glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(f"no shards in {split_dir}")

    chunks = {key: [] for key in ARRAY_KEYS}
    for path in files:
        with np.load(path) as shard:
            missing = [key for key in ARRAY_KEYS if key not in shard]
            if missing:
                raise ValueError(f"{path} is missing {missing}")
            for key in ARRAY_KEYS:
                chunks[key].append(shard[key])

    concatenated = {key: np.concatenate(value) for key, value in chunks.items()}
    order = np.argsort(concatenated["system_id"], kind="stable")
    data = SplitData(**{key: value[order] for key, value in concatenated.items()})
    _validate_arrays(data)
    if validate:
        validate_matched(data)
    return data


def load_all(root: str | Path, splits=("train", "val", "test"),
             validate: bool = True) -> dict[str, SplitData]:
    return {split: load_split(root, split, validate) for split in splits}


def _validate_arrays(data: SplitData) -> None:
    n, size, _ = data.images.shape
    if data.sources.shape != data.images.shape:
        raise ValueError("images and sources have different shapes")
    if len(data.labels) != n or len(data.theta_e) != n:
        raise ValueError("array lengths disagree")
    if not np.isfinite(data.images).all() or not np.isfinite(data.sources).all():
        raise ValueError("non-finite pixels")
    if data.images.min() < 0 or data.sources.min() < 0:
        raise ValueError("negative pixels; expected clipped images")
    if data.labels.min() < 0 or data.labels.max() >= len(CLASSES):
        raise ValueError("labels out of range")
    if data.theta_e.min() < 0.5 - 1e-4 or data.theta_e.max() > 2.0 + 1e-4:
        raise ValueError("theta_E outside [0.5, 2.0]")


def validate_matched(data: SplitData, atol: float = 1e-6) -> None:
    """Every system must contain all three classes with identical macro lens."""
    ids, counts = np.unique(data.system_id, return_counts=True)
    if not np.all(counts == 3):
        bad = ids[counts != 3][:5]
        raise ValueError(f"systems without exactly 3 images: {bad.tolist()}")
    shared = ("theta_e", "shear1", "shear2", "lens_angle", "galaxy_index", "source_mag")
    for system_id in ids:
        index = np.flatnonzero(data.system_id == system_id)
        if sorted(data.labels[index].tolist()) != [0, 1, 2]:
            raise ValueError(f"system {system_id} does not contain all classes")
        for key in shared:
            if np.ptp(getattr(data, key)[index]) > 1e-5:
                raise ValueError(f"system {system_id}: {key} differs across classes")
        sources = data.sources[index]
        if np.abs(sources - sources[0]).max() > atol:
            raise ValueError(f"system {system_id}: source differs across classes")


def stratified_indices(labels: np.ndarray, n_per_class: int,
                       seed: int = 42, num_classes: int = len(CLASSES)) -> np.ndarray:
    """Fixed, reproducible stratified slice (the physics evaluation set)."""
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in range(num_classes):
        index = np.flatnonzero(labels == cls)
        if len(index) < n_per_class:
            raise ValueError(f"class {cls} has only {len(index)} images")
        chosen.append(rng.choice(index, size=n_per_class, replace=False))
    return np.sort(np.concatenate(chosen))


def fingerprint(root: str | Path) -> str:
    """SHA-256 over the dataset's arrays and data-defining manifest fields.

    The arrays are hashed, not the ``.npz`` bytes: numpy's zip container
    embeds a timestamp, so two independent generations of identical data
    would otherwise hash differently. Provenance (git SHA, per-shard meta)
    lives in the manifest and is excluded here on purpose -- this hash
    answers "are these the same data?", not "did the same code run?".
    """
    digest = hashlib.sha256()
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for key in ("sim_version", "classes", "image_size", "seed", "config"):
            digest.update(key.encode())
            digest.update(json.dumps(manifest.get(key), sort_keys=True,
                                     default=str).encode())
    for split in ("train", "val", "test"):
        digest.update(split.encode())
        for path in sorted((root / split).glob("shard_*.npz")):
            digest.update(path.name.encode())
            with np.load(path) as shard:
                for key in sorted(shard.files):
                    if key == "meta":
                        continue
                    array = shard[key]
                    digest.update(f"{key}:{array.dtype}:{array.shape}".encode())
                    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a sharded v2 dataset")
    parser.add_argument("--data-root", default="data/gravilens-v2")
    parser.add_argument("--split", default="test",
                        help="train, val, test, or all")
    args = parser.parse_args(argv)
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        data = load_split(args.data_root, split)
        print(f"{split}: {len(data):,} images | classes {data.class_counts()}")
        print(f"theta_E: {data.theta_e.min():.3f} .. {data.theta_e.max():.3f} "
              f"(mean {data.theta_e.mean():.3f})")
        print("matched triplets: OK")
    if args.split == "all":
        test = load_split(args.data_root, "test")
        per_class = min(167, int(np.bincount(test.labels, minlength=3).min()))
        index = stratified_indices(test.labels, per_class, seed=42)
        print(f"physics slice: {len(index)} images "
              f"({per_class}/class, seed 42) from the test split")
    print(f"fingerprint: {fingerprint(args.data_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
