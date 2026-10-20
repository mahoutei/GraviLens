"""Artifact-only summary, figures, and claim audit.

Every number and every panel is read from ``runs/<run>/metrics.json`` and
``manifest.json``. ``audit`` exits non-zero if a run is missing or lacks the
fields the paper/poster is allowed to quote.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import (
    BASELINES,
    VARIANTS,
    GraviLensP,
    ModelConfig,
    count_trainable_params,
)
from .train import run_name

SEEDS = (42, 123, 777)


def expected_runs() -> list[tuple[str, str, int]]:
    return ([("gravilens", variant, seed) for variant in VARIANTS for seed in SEEDS]
            + [(name, "full", seed) for name in BASELINES for seed in SEEDS])


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def audit_runs(runs_root: str | Path, verbose: bool = True) -> list[str]:
    """Return human-readable problems; empty means the grid is audit-clean."""
    runs_root = Path(runs_root)
    target = count_trainable_params(GraviLensP(ModelConfig(), "full"))
    problems: list[str] = []

    for model, variant, seed in expected_runs():
        name = run_name(model, variant, seed)
        run_dir = runs_root / name
        missing = [file for file in ("manifest.json", "metrics.json", "best.pt")
                   if not (run_dir / file).exists()]
        if missing:
            problems.append(f"{name}: missing {', '.join(missing)}")
            continue
        try:
            manifest = _read(run_dir / "manifest.json")
            metrics = _read(run_dir / "metrics.json")
        except json.JSONDecodeError as exc:
            problems.append(f"{name}: invalid JSON ({exc})")
            continue

        if manifest.get("run") != name:
            problems.append(f"{name}: manifest run is {manifest.get('run')!r}")
        if metrics.get("run") != name:
            problems.append(f"{name}: metrics run is {metrics.get('run')!r}")
        if manifest.get("status") != "finished":
            problems.append(f"{name}: status is {manifest.get('status')!r}")
        if manifest.get("seed") != seed:
            problems.append(f"{name}: manifest seed is {manifest.get('seed')!r}")
        if metrics.get("split") != "test":
            problems.append(f"{name}: metrics split is {metrics.get('split')!r}")

        params = manifest.get("params")
        if not isinstance(params, int) or params <= 0:
            problems.append(f"{name}: invalid param count {params!r}")
        elif metrics.get("params") != params:
            problems.append(f"{name}: metrics/manifest param mismatch")
        elif model == "gravilens" and params != target:
            problems.append(f"{name}: {params:,} != target {target:,}")
        elif model != "gravilens" and abs(params - target) / target > 0.10:
            problems.append(f"{name}: baseline outside ±10% of target")

        data = manifest.get("data", {})
        fingerprint = data.get("fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) < 8:
            problems.append(f"{name}: missing data fingerprint")
        elif metrics.get("data_fingerprint") != fingerprint:
            problems.append(f"{name}: data fingerprint changed")
        if not isinstance(manifest.get("library_versions"), dict):
            problems.append(f"{name}: missing library versions")

        test = metrics.get("test", {})
        for key in ("accuracy", "macro_f1", "macro_auc", "confusion",
                    "per_class_f1"):
            if key not in test:
                problems.append(f"{name}: test metrics missing {key}")
        if "confusion" in test and np.shape(test["confusion"]) != (3, 3):
            problems.append(f"{name}: confusion matrix is not 3x3")
        if not isinstance(metrics.get("physics_slice"), dict):
            problems.append(f"{name}: missing physics slice")

        if model == "gravilens":
            theta = metrics.get("theta")
            source = metrics.get("source_ssim")
            if not isinstance(theta, dict):
                problems.append(f"{name}: physics model has no theta metrics")
            else:
                for key in ("theta_mae", "theta_rmse", "theta_bias",
                            "theta_pearson", "theta_within_10pct",
                            "train_mean_baseline_mae"):
                    if key not in theta:
                        problems.append(f"{name}: theta metrics missing {key}")
                if theta.get("supervised") != (variant in ("full", "dist")):
                    problems.append(f"{name}: theta supervision flag is wrong")
            if not isinstance(source, dict) or len(source) != 4:
                problems.append(f"{name}: source SSIM references incomplete")
        else:
            if metrics.get("theta") is not None:
                problems.append(f"{name}: classifier baseline has theta metrics")
            if metrics.get("source_ssim") is not None:
                problems.append(f"{name}: classifier baseline has SSIM metrics")

    if verbose:
        if problems:
            print(f"audit: {len(problems)} problem(s) across {len(expected_runs())} runs")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"audit: {len(expected_runs())}/{len(expected_runs())} runs complete")
    return problems


def load_records(runs_root: str | Path) -> list[dict]:
    records = []
    for model, variant, seed in expected_runs():
        run_dir = Path(runs_root) / run_name(model, variant, seed)
        records.append({
            "model": model,
            "variant": variant,
            "seed": seed,
            "manifest": _read(run_dir / "manifest.json"),
            "metrics": _read(run_dir / "metrics.json"),
        })
    return records


def _stat(values: list[float]) -> dict:
    array = np.asarray([value for value in values
                        if value is not None and np.isfinite(value)], dtype=float)
    if len(array) == 0:
        return {"n": 0, "mean": None, "sd": None}
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return {"n": int(len(array)), "mean": float(array.mean()), "sd": sd}


def summarize(records: list[dict]) -> dict:
    groups = {}
    for model in ("gravilens",) + BASELINES:
        for variant in (VARIANTS if model == "gravilens" else ("full",)):
            selected = [record for record in records
                        if record["model"] == model and record["variant"] == variant]
            if not selected:
                continue
            metrics = selected[0]["metrics"]
            test = metrics["test"]
            group = {
                "label": f"{model}/{variant}" if model == "gravilens" else model,
                "model": model,
                "variant": variant,
                "seeds": [record["seed"] for record in selected],
            }
            for key in ("accuracy", "macro_f1", "macro_auc"):
                group[key] = _stat([record["metrics"]["test"][key]
                                    for record in selected])
            if model == "gravilens":
                for key in ("theta_mae", "theta_rmse", "theta_bias",
                            "theta_pearson", "theta_within_10pct",
                            "train_mean_baseline_mae"):
                    group[key] = _stat([record["metrics"]["theta"][key]
                                        for record in selected])
                for key in ("ssim_model", "ssim_oracle_bilinear",
                            "ssim_oracle_scatter", "ssim_lensed_image"):
                    group[key] = _stat([record["metrics"]["source_ssim"][key]
                                        for record in selected])
            groups[group["label"]] = group
    return {"runs": len(records), "groups": groups}


def _fmt(stat: dict, digits: int = 3) -> str:
    if stat["mean"] is None:
        return "—"
    return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}"


def write_table(summary: dict, path: Path) -> None:
    lines = [
        "| model | seeds | macro-F1 | AUC (OvR) | accuracy | θE MAE | constant θE MAE | SSIM model | SSIM lensed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary["groups"].values():
        theta = _fmt(group["theta_mae"]) if "theta_mae" in group else "—"
        constant = (_fmt(group["train_mean_baseline_mae"])
                    if "train_mean_baseline_mae" in group else "—")
        ssim = _fmt(group["ssim_model"]) if "ssim_model" in group else "—"
        lensed = _fmt(group["ssim_lensed_image"]) if "ssim_lensed_image" in group else "—"
        lines.append(
            f"| {group['label']} | {group['macro_f1']['n']} | "
            f"{_fmt(group['macro_f1'])} | {_fmt(group['macro_auc'])} | "
            f"{_fmt(group['accuracy'])} | {theta} | {constant} | {ssim} | {lensed} |")
    path.write_text("\n".join(lines) + "\n")


def _plot_macro_f1(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(summary["groups"])
    means = [summary["groups"][label]["macro_f1"]["mean"] or 0 for label in labels]
    sds = [summary["groups"][label]["macro_f1"]["sd"] or 0 for label in labels]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(range(len(labels)), means, yerr=sds, capsize=3,
             color=["#4C72B0" if "gravilens" in label else "#999999"
                    for label in labels])
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_ylabel("test macro-F1")
    axis.set_ylim(0, 1)
    axis.set_title("Seed-level test macro-F1 (error bars: SD over 3 seeds)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_physics(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [group for label, group in summary["groups"].items()
              if "theta_mae" in group]
    labels = [group["label"].split("/")[-1] for group in groups]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))

    left.bar(labels, [group["theta_mae"]["mean"] for group in groups],
             yerr=[group["theta_mae"]["sd"] for group in groups], capsize=3,
             color="#4C72B0", label="model")
    left.plot(labels, [group["train_mean_baseline_mae"]["mean"] for group in groups],
              "o", color="#C44E52", label="train-mean constant")
    left.set_ylabel("θE MAE (arcsec)")
    left.set_title("θE error vs constant guess")
    left.legend()

    width = 0.25
    positions = np.arange(len(groups))
    for offset, key, label in (
            (-width, "ssim_model", "model"),
            (0.0, "ssim_oracle_bilinear", "oracle bilinear"),
            (width, "ssim_lensed_image", "lensed image")):
        right.bar(positions + offset, [group[key]["mean"] for group in groups],
                  width=width, label=label)
    right.set_xticks(positions, labels)
    right.set_ylabel("SSIM vs source")
    right.set_ylim(0, 1)
    right.set_title("Source reconstruction")
    right.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def build_figures(runs_root: str | Path, out_dir: str | Path | None = None) -> int:
    runs_root = Path(runs_root)
    problems = audit_runs(runs_root)
    if problems:
        return 1
    records = load_records(runs_root)
    summary = summarize(records)
    out = Path(out_dir or runs_root / "figures")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_table(summary, out / "main_table.md")
    _plot_macro_f1(summary, out / "macro_f1.png")
    _plot_physics(summary, out / "physics.png")
    print(f"wrote {out / 'summary.json'}, {out / 'main_table.md'}, "
          f"{out / 'macro_f1.png'}, {out / 'physics.png'}")
    return 0
