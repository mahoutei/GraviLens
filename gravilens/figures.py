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

# The pre-registered recipe (docs/method.md). A run that used anything else
# -- a 2-epoch smoke, a different batch size -- must not pass the audit.
EXPECTED_RECIPE = {
    "epochs": 50,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 1e-3,
    "warmup_epochs": 5,
    "label_smoothing": 0.05,
    "theta_loss_weight": 0.1,
    "patience": 25,
    "use_ema": True,
    "ema_decay": 0.999,
    "aug_start": 10,
    "max_rotation_deg": 15.0,
}


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
        train_cfg = manifest.get("train_config")
        if not isinstance(train_cfg, dict):
            problems.append(f"{name}: missing training config")
        else:
            for key, expected in EXPECTED_RECIPE.items():
                if train_cfg.get(key) != expected:
                    problems.append(
                        f"{name}: {key} is {train_cfg.get(key)!r}, expected {expected!r}")
            if train_cfg.get("seed") != seed:
                problems.append(f"{name}: train seed is {train_cfg.get('seed')!r}")
        for key, expected in (("train_size", 6999), ("val_size", 1500)):
            if data.get(key) != expected:
                problems.append(f"{name}: {key} is {data.get(key)!r}, expected {expected}")
        if data.get("train_classes") != [2333, 2333, 2333]:
            problems.append(f"{name}: unexpected train class counts")
        if data.get("val_classes") != [500, 500, 500]:
            problems.append(f"{name}: unexpected val class counts")

        test = metrics.get("test", {})
        for key in ("accuracy", "macro_f1", "macro_auc", "confusion",
                    "per_class_f1"):
            if key not in test:
                problems.append(f"{name}: test metrics missing {key}")
        if test.get("n") != 1500:
            problems.append(f"{name}: test n is {test.get('n')!r}, expected 1500")
        if "confusion" in test and np.shape(test["confusion"]) != (3, 3):
            problems.append(f"{name}: confusion matrix is not 3x3")
        if "macro_auc" in test and (test["macro_auc"] is None
                                    or not np.isfinite(test["macro_auc"])):
            problems.append(f"{name}: macro AUC is not finite")

        physics = metrics.get("physics_slice")
        if not isinstance(physics, dict):
            problems.append(f"{name}: missing physics slice")
        else:
            if (physics.get("n"), physics.get("per_class"), physics.get("seed")) != (501, 167, 42):
                problems.append(f"{name}: physics slice is not 167/class seed 42")

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
            expected_ssim = {"ssim_model", "ssim_oracle_bilinear",
                             "ssim_oracle_scatter", "ssim_lensed_image"}
            if not isinstance(source, dict) or set(source) != expected_ssim:
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
                "theta_supervised": (variant in ("full", "dist")
                                     if model == "gravilens" else None),
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

    summary = {"runs": len(records), "groups": groups}
    full = groups.get("gravilens/full")
    no_sis = groups.get("gravilens/no_sis")
    if (full and no_sis and full["macro_f1"]["mean"] is not None
            and no_sis["macro_f1"]["mean"] is not None):
        delta = full["macro_f1"]["mean"] - no_sis["macro_f1"]["mean"]
        summary["decision"] = {
            "comparison": "full - no_sis",
            "delta_macro_f1": float(delta),
            "null_result": bool(delta <= 0),
            "rule": "three-seed mean; null if <= 0",
        }
    return summary


def _fmt(stat: dict, digits: int = 3) -> str:
    if stat["mean"] is None:
        return "—"
    return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}"


def write_table(summary: dict, path: Path) -> None:
    lines = [
        "| model | seeds | θE sup. | macro-F1 | AUC (OvR) | accuracy | θE MAE | constant θE MAE | SSIM model | SSIM lensed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary["groups"].values():
        theta = _fmt(group["theta_mae"]) if "theta_mae" in group else "—"
        constant = (_fmt(group["train_mean_baseline_mae"])
                    if "train_mean_baseline_mae" in group else "—")
        ssim = _fmt(group["ssim_model"]) if "ssim_model" in group else "—"
        lensed = _fmt(group["ssim_lensed_image"]) if "ssim_lensed_image" in group else "—"
        flag = group.get("theta_supervised")
        supervised = "—" if flag is None else ("yes" if flag else "no")
        lines.append(
            f"| {group['label']} | {group['macro_f1']['n']} | {supervised} | "
            f"{_fmt(group['macro_f1'])} | {_fmt(group['macro_auc'])} | "
            f"{_fmt(group['accuracy'])} | {theta} | {constant} | {ssim} | {lensed} |")
    if "decision" in summary:
        decision = summary["decision"]
        lines += [
            "",
            f"Pre-registered decision: `full - no_sis` = "
            f"{decision['delta_macro_f1']:+.4f} macro-F1; "
            f"{'null' if decision['null_result'] else 'non-null'} "
            f"(null if <= 0, three-seed mean).",
        ]
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
    left.set_title("θE error vs constant guess\n(no_sup/no_sis are unsupervised)")
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
    if "decision" in summary:
        decision = summary["decision"]
        print(f"decision: full - no_sis macro-F1 = "
              f"{decision['delta_macro_f1']:+.4f} "
              f"({'null' if decision['null_result'] else 'non-null'})")
    print(f"wrote {out / 'summary.json'}, {out / 'main_table.md'}, "
          f"{out / 'macro_f1.png'}, {out / 'physics.png'}")
    return 0
