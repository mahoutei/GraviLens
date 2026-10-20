"""Matched-triplet simulator for the v2 dataset.

Each triplet is one macro lens observed three times:

    no_sub  SIE + shear, no substructure
    cdm     same macro lens + CDM subhaloes (pyHalo)
    axion   same macro lens + ULDM haloes (pyHalo)

Everything that is not substructure is shared *exactly*: theta_E, shear,
lens-light ellipticity and orientation, source galaxy, source position and
angle, source magnitude, and the noise realisation. A classifier therefore
cannot win on the macro lens.

The old dataset changed theta_E, shear, source position, source angle, and
noise independently per class; that is the confound this file removes.

Shards are atomic ``.npz`` files named ``shard_<first_triplet>.npz``. A run can
be interrupted at any point and resumed; existing valid shards are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

SIM_VERSION = "v2.0"
CLASSES = ("no_sub", "cdm", "axion")
GALAXY_URL = "https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5"

# One cosmology for lenstronomy and pyHalo (plan section 5).
# Ob0 must be non-zero: Colossus refuses the Eisenstein & Hu transfer function at Ob0=0.
H0, OM0, OB0 = 70.0, 0.3, 0.049
KWARGS_COSMO = {
    "H0": H0, "Om0": OM0, "Ob0": OB0, "Odm0": OM0 - OB0,
    "sigma8": 0.81, "flat": "flat", "ns": 0.965, "power_law": False,
}


@dataclass(frozen=True)
class SimConfig:
    image_size: int = 64
    z_lens: float = 0.5
    z_source: float = 2.5
    theta_min: float = 0.5
    theta_max: float = 2.0
    theta_mean: float = 1.4
    theta_std: float = 0.35
    shear_magnitude: float = 0.05
    source_scale: float = 0.015
    source_pos_max: float = 0.4
    axion_log10_mass: float = -21.5
    # 6.5 arcsec covers the 6.36 arcsec image; log_mlow=7 is the resolution floor.
    cone_opening_angle_arcsec: float = 6.5
    log_mlow: float = 7.0
    log_mhigh: float = 10.0
    galaxy_smoothing_sigma: float = 3.0
    galaxy_redshift_max: float = 0.02
    cdm_sigma_sub: float = 0.025


@dataclass(eq=False)
class System:
    """The shared macro lens for one triplet."""

    galaxy_index: int
    galaxy: np.ndarray
    theta_e: float
    source_x: float
    source_y: float
    source_angle: float
    shear_gamma1: float
    shear_gamma2: float
    lens_angle: float
    lens_light_e1: tuple[float, float]
    lens_light_e2: tuple[float, float]
    source_magnitude: float


def seed_for(split: str, triplet: int, base_seed: int) -> int:
    split_id = {"train": 0, "val": 1, "test": 2}.get(split, 3)
    state = np.random.SeedSequence([int(base_seed), split_id, int(triplet)]).generate_state(1)
    return int(state[0])


def _rotate_spin2(e1: float, e2: float, angle: float) -> tuple[float, float]:
    c, s = np.cos(2 * angle), np.sin(2 * angle)
    return float(e1 * c - e2 * s), float(e1 * s + e2 * c)


def sample_system(galaxy_pool: np.ndarray, seed: int, cfg: SimConfig) -> System:
    """Draw one macro lens. Uses a local RNG so it does not disturb pyHalo."""
    from scipy.ndimage import gaussian_filter
    from scipy.stats import truncnorm

    rng = np.random.default_rng(seed)
    index = int(rng.integers(len(galaxy_pool)))
    galaxy = np.asarray(galaxy_pool[index], dtype=np.float64)
    if galaxy.ndim == 3:
        galaxy = galaxy[..., 0]
    galaxy = gaussian_filter(galaxy, cfg.galaxy_smoothing_sigma)
    h = min(50, galaxy.shape[0])
    w = min(50, galaxy.shape[1])
    galaxy = galaxy - np.median(galaxy[:h, :w])

    a = (cfg.theta_min - cfg.theta_mean) / cfg.theta_std
    b = (cfg.theta_max - cfg.theta_mean) / cfg.theta_std
    theta_e = float(truncnorm.rvs(
        a, b, loc=cfg.theta_mean, scale=cfg.theta_std, random_state=rng))

    source_x, source_y = rng.uniform(-cfg.source_pos_max, cfg.source_pos_max, 2)
    source_angle = float(rng.uniform(-np.pi, np.pi))
    shear_angle = float(rng.uniform(0, 2 * np.pi))
    shear_gamma1 = float(cfg.shear_magnitude * np.cos(2 * shear_angle))
    shear_gamma2 = float(cfg.shear_magnitude * np.sin(2 * shear_angle))
    lens_angle = float(rng.uniform(0, 2 * np.pi))
    e1a, e2a = _rotate_spin2(0.0, 0.05, lens_angle)
    e1b, e2b = _rotate_spin2(0.0, 0.30, lens_angle)

    return System(
        galaxy_index=index,
        galaxy=galaxy,
        theta_e=theta_e,
        source_x=float(source_x),
        source_y=float(source_y),
        source_angle=source_angle,
        shear_gamma1=shear_gamma1,
        shear_gamma2=shear_gamma2,
        lens_angle=lens_angle,
        lens_light_e1=(e1a, e1b),
        lens_light_e2=(e2a, e2b),
        source_magnitude=float(rng.normal(16.0, 0.1)),
    )


# ---------------------------------------------------------------------------
# lenstronomy rendering
# ---------------------------------------------------------------------------

def _euclid_band() -> dict:
    from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
    return Euclid(band="VIS", psf_type="GAUSSIAN", coadd_years=6).kwargs_single_band()


def _astropy_cosmo():
    from astropy.cosmology import FlatLambdaCDM
    return FlatLambdaCDM(H0=H0, Om0=OM0, Ob0=OB0)


def _render_class(system: System, cls: str, cfg: SimConfig):
    from lenstronomy.SimulationAPI.sim_api import SimAPI

    macro_models = ["SIE", "SHEAR"]
    macro_kwargs = [
        {"theta_E": system.theta_e, "e1": 0.0, "e2": 0.0,
         "center_x": 0.0, "center_y": 0.0},
        {"gamma1": system.shear_gamma1, "gamma2": system.shear_gamma2},
    ]
    macro_redshifts = [cfg.z_lens, cfg.z_lens]
    cosmo = _astropy_cosmo()

    if cls == "no_sub":
        lens_models, lens_redshifts, lens_kwargs = macro_models, macro_redshifts, macro_kwargs
    elif cls == "cdm":
        from pyHalo.PresetModels.cdm import CDM
        realization = CDM(
            cfg.z_lens, cfg.z_source,
            cone_opening_angle_arcsec=cfg.cone_opening_angle_arcsec,
            log_mlow=cfg.log_mlow, log_mhigh=cfg.log_mhigh,
            sigma_sub=cfg.cdm_sigma_sub, kwargs_cosmo=KWARGS_COSMO)
        models, redshifts, halos, _ = realization.lensing_quantities()
        lens_models = macro_models + list(models)
        lens_redshifts = macro_redshifts + list(redshifts)
        lens_kwargs = macro_kwargs + list(halos)
        cosmo = realization.astropy_instance
    elif cls == "axion":
        from pyHalo.PresetModels.uldm import ULDM
        realization = ULDM(
            cfg.z_lens, cfg.z_source,
            log10_m_uldm=cfg.axion_log10_mass,
            uldm_plaw=1 / 3,
            flucs=False,
            sigma_sub=cfg.cdm_sigma_sub,
            log10_sigma_sub=None,       # the preset default (0.1) would be 4x CDM
            log_mlow=cfg.log_mlow, log_mhigh=cfg.log_mhigh,
            cone_opening_angle_arcsec=cfg.cone_opening_angle_arcsec,
            kwargs_cosmo=KWARGS_COSMO)
        models, redshifts, halos, _ = realization.lensing_quantities()
        lens_models = macro_models + list(models)
        lens_redshifts = macro_redshifts + list(redshifts)
        lens_kwargs = macro_kwargs + list(halos)
        cosmo = realization.astropy_instance
    else:
        raise ValueError(f"unknown class {cls!r}")

    kwargs_model = {
        "lens_model_list": lens_models,
        "lens_redshift_list": lens_redshifts,
        "lens_light_model_list": ["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"],
        "source_light_model_list": ["INTERPOL"],
        "source_redshift_list": [cfg.z_source],
        "cosmo": cosmo,
        "z_source_convention": cfg.z_source,
        "z_source": cfg.z_source,
    }
    sim = SimAPI(cfg.image_size, kwargs_single_band=_euclid_band(),
                 kwargs_model=kwargs_model)
    image_model = sim.image_model_class({"point_source_supersampling_factor": 1})

    lens_light_mag = [
        {"magnitude": 40, "R_sersic": 0.4, "n_sersic": 2.3,
         "e1": system.lens_light_e1[0], "e2": system.lens_light_e2[0],
         "center_x": 0.0, "center_y": 0.0},
        {"magnitude": 40, "R_sersic": 1.5, "n_sersic": 1.2,
         "e1": system.lens_light_e1[1], "e2": system.lens_light_e2[1],
         "center_x": 0.0, "center_y": 0.0},
    ]
    source_mag = [{
        "magnitude": system.source_magnitude,
        "image": system.galaxy,
        "scale": cfg.source_scale,
        "phi_G": system.source_angle,
        "center_x": system.source_x,
        "center_y": system.source_y,
    }]
    kwargs_lens_light, kwargs_source, _ = sim.magnitude2amplitude(
        lens_light_mag, source_mag)
    image = image_model.image(lens_kwargs, kwargs_source, kwargs_lens_light)
    return np.asarray(image, dtype=np.float32), sim, kwargs_source


def _render_unlensed_source(sim, kwargs_source) -> np.ndarray:
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel

    nolens = ImageModel(
        data_class=sim.data_class, psf_class=sim.psf_class,
        lens_model_class=LensModel([]),
        source_model_class=LightModel(["INTERPOL"]),
        kwargs_numerics={"point_source_supersampling_factor": 1})
    source = nolens.image(
        kwargs_lens=[], kwargs_source=kwargs_source, source_add=True,
        lens_light_add=False, point_source_add=False)
    source = np.clip(np.asarray(source, dtype=np.float32), 0, None)
    return source / max(float(source.max()), 1e-12)


def render_triplet(system: System, cfg: SimConfig, seed: int):
    """Render all three classes with one shared noise realisation and source."""
    np.random.seed(int(seed))
    rendered = {cls: _render_class(system, cls, cfg) for cls in CLASSES}

    no_sub_image, no_sub_sim, kwargs_source = rendered["no_sub"]
    noise = np.asarray(no_sub_sim.noise_for_model(model=no_sub_image), dtype=np.float32)
    source = _render_unlensed_source(no_sub_sim, kwargs_source)

    images = []
    sources = []
    for cls in CLASSES:
        image = np.clip(rendered[cls][0] + noise, 0, None)
        image = image / max(float(image.max()), 1e-12)
        images.append(image.astype(np.float32))
        sources.append(source)
    return np.stack(images), np.stack(sources), float(system.theta_e)


# ---------------------------------------------------------------------------
# Sharded, resumable generation
# ---------------------------------------------------------------------------

def _valid_shard(path: Path, expected: int, image_size: int) -> bool:
    try:
        with np.load(path) as z:
            required = ("images", "sources", "labels", "theta_e", "system_id",
                        "shear1", "shear2", "lens_angle", "galaxy_index", "source_mag")
            if any(k not in z for k in required):
                return False
            return (z["images"].shape == (expected, image_size, image_size)
                    and z["sources"].shape == (expected, image_size, image_size)
                    and z["labels"].shape == (expected,)
                    and z["theta_e"].shape == (expected,))
    except Exception:
        return False


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def _write_manifest(root: Path, split: str, target_images: int, shard_images: int,
                    seed: int, cfg: SimConfig, completed: list[int]) -> None:
    try:
        import fcntl
    except ImportError:  # non-Unix; shards remain authoritative
        fcntl = None
    path = root / "manifest.json"
    lock = open(root / "manifest.lock", "w")
    try:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = {}
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest.get("config") not in (None, asdict(cfg)):
                raise ValueError(
                    "existing manifest was generated with a different SimConfig; "
                    "use a new data root or delete the old manifest")
        manifest.update({
            "version": 1,
            "sim_version": SIM_VERSION,
            "git_sha": _git_sha(),
            "classes": list(CLASSES),
            "image_size": cfg.image_size,
            "seed": seed,
            "shard_size": shard_images,
            "config": asdict(cfg),
        })
        manifest.setdefault("splits", {})[split] = {
            "target_images": target_images,
            "n_triplets": target_images // 3,
            "completed_shards": sorted(completed),
        }
        tmp = path.with_name("manifest.tmp.json")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(tmp, path)
    finally:
        lock.close()


def generate_split(split: str, n_images: int, galaxy_pool: np.ndarray,
                   cfg: SimConfig, out_root: str | Path, seed: int = 42,
                   shard_size: int = 500, limit: int | None = None,
                   render_fn: Callable | None = None,
                   verbose: bool = True) -> dict:
    """Generate one split. ``shard_size`` is in images and must be a multiple of 3."""
    n_triplets = n_images // 3
    if limit is not None:
        n_triplets = min(n_triplets, int(limit))
    if n_triplets <= 0:
        raise ValueError("n_images must provide at least one triplet")

    shard_images = max(3, (int(shard_size) // 3) * 3)
    shard_triplets = shard_images // 3
    root = Path(out_root)
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    render_fn = render_fn or (lambda system, s: render_triplet(system, cfg, s))

    completed: list[int] = []
    written = skipped = generated = 0
    started = time.time()

    for start in range(0, n_triplets, shard_triplets):
        stop = min(start + shard_triplets, n_triplets)
        expected = (stop - start) * 3
        path = split_dir / f"shard_{start:06d}.npz"
        if _valid_shard(path, expected, cfg.image_size):
            completed.append(start)
            skipped += 1
            continue

        images, sources, labels, theta_e, system_id = [], [], [], [], []
        shear1, shear2, lens_angle, galaxy_index, source_mag = [], [], [], [], []
        for triplet in range(start, stop):
            last_error = None
            for attempt in range(3):
                sample_seed = seed_for(split, triplet, seed) + attempt * 1_000_003
                try:
                    system = sample_system(galaxy_pool, sample_seed, cfg)
                    imgs, srcs, theta = render_fn(system, sample_seed)
                    break
                except Exception as exc:  # a single pyHalo draw can fail
                    last_error = exc
            else:
                raise RuntimeError(
                    f"{split} triplet {triplet} failed after 3 attempts") from last_error

            images.append(np.asarray(imgs, dtype=np.float32))
            sources.append(np.asarray(srcs, dtype=np.float32))
            labels.extend(range(3))
            theta_e.extend([theta] * 3)
            system_id.extend([triplet] * 3)
            shear1.extend([system.shear_gamma1] * 3)
            shear2.extend([system.shear_gamma2] * 3)
            lens_angle.extend([system.lens_angle] * 3)
            galaxy_index.extend([system.galaxy_index] * 3)
            source_mag.extend([system.source_magnitude] * 3)
            generated += 1

        meta = {
            "version": 1,
            "sim_version": SIM_VERSION,
            "git_sha": _git_sha(),
            "split": split,
            "start": start,
            "stop": stop,
            "seed": seed,
            "config": asdict(cfg),
        }
        _atomic_savez(
            path,
            images=np.concatenate(images, axis=0),
            sources=np.concatenate(sources, axis=0),
            labels=np.asarray(labels, dtype=np.int8),
            theta_e=np.asarray(theta_e, dtype=np.float32),
            system_id=np.asarray(system_id, dtype=np.int64),
            shear1=np.asarray(shear1, dtype=np.float32),
            shear2=np.asarray(shear2, dtype=np.float32),
            lens_angle=np.asarray(lens_angle, dtype=np.float32),
            galaxy_index=np.asarray(galaxy_index, dtype=np.int32),
            source_mag=np.asarray(source_mag, dtype=np.float32),
            meta=np.asarray(json.dumps(meta)),
        )
        completed.append(start)
        written += 1
        _write_manifest(root, split, n_triplets * 3, shard_images, seed, cfg, completed)

        if verbose:
            elapsed = time.time() - started
            rate = elapsed / max(1, generated)
            remaining = (n_triplets - stop) * rate
            print(f"{split}: {stop}/{n_triplets} triplets "
                  f"({rate:.2f}s/triplet, ETA {remaining / 60:.1f} min)", flush=True)

    stats = {
        "split": split,
        "triplets": n_triplets,
        "images": n_triplets * 3,
        "written_shards": written,
        "skipped_shards": skipped,
        "generated_triplets": generated,
        "seconds": time.time() - started,
    }
    if verbose:
        print(f"{split}: done {stats}", flush=True)
    return stats


# ---------------------------------------------------------------------------
# Galaxy pool
# ---------------------------------------------------------------------------

def _valid_galaxy_file(path: Path) -> bool:
    try:
        import h5py
        with h5py.File(path, "r") as handle:
            return all(k in handle for k in ("images", "ans", "redshift"))
    except Exception:
        return False


def ensure_galaxy_file(path: str | Path | None = None) -> Path:
    """Download Galaxy10 DECals once. Resumes a partial ``.part`` file."""
    path = Path(path or Path("data/raw/Galaxy10_DECals.h5"))
    if path.exists() and _valid_galaxy_file(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(GALAXY_URL, headers=headers)
    print(f"downloading Galaxy10 DECals -> {path} (resume at {offset / 1e6:.0f} MB)")
    with urllib.request.urlopen(request) as response:
        append = offset and response.status == 206
        mode = "ab" if append else "wb"
        with open(part, mode) as handle:
            downloaded = offset if append else 0
            next_report = 256e6
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"  {downloaded / 1e9:.2f} GB", flush=True)
                    next_report += 256e6
    os.replace(part, path)
    return path


def load_galaxy_pool(path: str | Path, cfg: SimConfig) -> np.ndarray:
    path = Path(path)
    if path.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as handle:
            ans = np.asarray(handle["ans"])
            redshift = np.asarray(handle["redshift"])
            index = np.flatnonzero((ans == 6) & (redshift < cfg.galaxy_redshift_max))
            pool = np.asarray(handle["images"][index], dtype=np.float32)
    elif path.suffix == ".npy":
        pool = np.load(path)
    else:
        raise ValueError(f"galaxy pool must be .h5 or .npy, got {path}")
    if len(pool) == 0:
        raise ValueError(f"no galaxies selected from {path}")
    return pool


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    from .config import load_config

    parser = argparse.ArgumentParser(description="Generate the matched v2 dataset")
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--galaxy-file", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="triplets per split (pilot); use with --splits")
    args = parser.parse_args(argv)

    run_cfg = load_config(args.config)
    cfg = SimConfig()
    sizes = dict(zip(("train", "val", "test"), run_cfg.data.sizes))
    seed = run_cfg.data.seed if args.seed is None else args.seed
    shard_size = run_cfg.data.shard_size if args.shard_size is None else args.shard_size
    out_root = Path(args.data_root or run_cfg.data.root)

    galaxy_file = Path(args.galaxy_file) if args.galaxy_file else Path("data/raw/Galaxy10_DECals.h5")
    if not galaxy_file.exists():
        galaxy_file = ensure_galaxy_file(galaxy_file)
    pool = load_galaxy_pool(galaxy_file, cfg)
    print(f"galaxy pool: {len(pool):,} images from {galaxy_file}")
    print(f"output: {out_root} | seed={seed} | shard={shard_size} images")

    for split in args.splits:
        if split not in sizes:
            raise SystemExit(f"unknown split {split!r}; choose from {list(sizes)}")
        generate_split(split, sizes[split], pool, cfg, out_root, seed,
                       shard_size, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
