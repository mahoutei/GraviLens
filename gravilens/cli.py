"""Command line: ``python -m gravilens {generate,train,evaluate,ablate,figures,audit}``."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .config import load_config, runs_root
from .dataset import fingerprint
from .evaluate import evaluate_run
from .models import BASELINES, VARIANTS
from .train import run_name, run_training


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--data-root", default=None,
                        help="override the config's data.root")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, mps")
    parser.add_argument("--runs-root", default=None,
                        help="artifact root (default: $GRAVILENS_RUNS or runs)")


def _artifacts(args) -> Path:
    return Path(args.runs_root) if args.runs_root else runs_root()


def cmd_generate(args) -> int:
    from .simulate import main as simulate_main
    argv = ["--config", args.config]
    if args.data_root:
        argv += ["--data-root", args.data_root]
    if args.galaxy_file:
        argv += ["--galaxy-file", args.galaxy_file]
    argv += ["--splits", *args.splits]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.shard_size is not None:
        argv += ["--shard-size", str(args.shard_size)]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    return simulate_main(argv)


def cmd_train(args) -> int:
    cfg = load_config(args.config)
    out = Path(args.out) if args.out else _artifacts(args) / run_name(
        args.model, args.variant, args.seed)
    result = run_training(
        cfg, args.model, args.variant, args.seed,
        data_dir=args.data_root, out_dir=out, device=args.device,
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)
    summary = {key: value for key, value in result.items() if key != "history"}
    print(json.dumps(summary, indent=2, default=float))
    return 0


def cmd_evaluate(args) -> int:
    evaluate_run(args.run, data_dir=args.data_root, split=args.split,
                 device=args.device, physics_per_class=args.physics_per_class)
    return 0


def cmd_figures(args) -> int:
    from .figures import build_figures
    return build_figures(_artifacts(args), args.out)


def cmd_audit(args) -> int:
    from .figures import audit_runs
    return 0 if not audit_runs(_artifacts(args)) else 1


def cmd_ablate(args) -> int:
    cfg = load_config(args.config)
    seeds = args.seeds or [42, 123, 777]
    variants = args.variants or list(VARIANTS)
    baselines = args.baselines or list(BASELINES)
    jobs = ([("gravilens", variant, seed) for variant in variants for seed in seeds]
            + [(name, "full", seed) for name in baselines for seed in seeds])

    if args.dry_run:
        for model, variant, seed in jobs:
            print(run_name(model, variant, seed))
        print(f"{len(jobs)} runs")
        return 0

    data_dir = Path(args.data_root or cfg.data.root)
    data_fingerprint = fingerprint(data_dir)

    for index, (model, variant, seed) in enumerate(jobs, start=1):
        name = run_name(model, variant, seed)
        out = _artifacts(args) / name
        expected = asdict(replace(
            cfg.train, seed=seed,
            epochs=cfg.train.epochs if args.epochs is None else args.epochs,
            batch_size=(cfg.train.batch_size if args.batch_size is None
                        else args.batch_size),
            patience=(cfg.train.patience if args.patience is None
                      else args.patience),
            lr=cfg.train.lr))
        if (out / "metrics.json").exists() and not args.force:
            if _run_is_current(out, expected, data_fingerprint):
                print(f"[{index}/{len(jobs)}] skip {name} (metrics.json exists)")
                continue
            print(f"[{index}/{len(jobs)}] {name}: recipe or dataset changed, "
                  f"rerunning", flush=True)
        else:
            print(f"[{index}/{len(jobs)}] {name}", flush=True)
        if not args.skip_train:
            run_training(
                cfg, model, variant, seed, data_dir=args.data_root, out_dir=out,
                device=args.device, epochs=args.epochs,
                batch_size=args.batch_size, patience=args.patience)
        elif (out / "manifest.json").exists() and not _run_is_current(
                out, expected, data_fingerprint):
            print(f"  warning: {name} was not trained with this recipe/dataset; "
                  f"its stale checkpoint is being scored", flush=True)
        evaluate_run(out, data_dir=args.data_root, device=args.device,
                     physics_per_class=args.physics_per_class)
    return 0


def _run_is_current(out: Path, expected_train_config: dict,
                    data_fingerprint: str) -> bool:
    """True when an existing run used this recipe on this dataset."""
    try:
        manifest = json.loads((out / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("train_config") != expected_train_config:
        return False
    return manifest.get("data", {}).get("fingerprint") == data_fingerprint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gravilens")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate the matched dataset")
    generate.add_argument("--config", default="configs/data.yaml")
    generate.add_argument("--data-root", default=None)
    generate.add_argument("--galaxy-file", default=None)
    generate.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--shard-size", type=int, default=None)
    generate.add_argument("--limit", type=int, default=None)
    generate.set_defaults(func=cmd_generate)

    train = sub.add_parser("train", help="train one run")
    _common(train)
    train.add_argument("--model", choices=("gravilens",) + BASELINES,
                       default="gravilens")
    train.add_argument("--variant", choices=VARIANTS, default="full")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--patience", type=int, default=None)
    train.add_argument("--out", default=None)
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="evaluate a checkpoint on test")
    _common(evaluate)
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--physics-per-class", type=int, default=167)
    evaluate.set_defaults(func=cmd_evaluate)

    ablate = sub.add_parser("ablate", help="train + evaluate the full grid")
    _common(ablate)
    ablate.add_argument("--seeds", nargs="+", type=int, default=None)
    ablate.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    ablate.add_argument("--baselines", nargs="+", choices=BASELINES, default=None)
    ablate.add_argument("--epochs", type=int, default=None)
    ablate.add_argument("--batch-size", type=int, default=None)
    ablate.add_argument("--patience", type=int, default=None)
    ablate.add_argument("--physics-per-class", type=int, default=167)
    ablate.add_argument("--skip-train", action="store_true",
                        help="evaluate existing runs only")
    ablate.add_argument("--force", action="store_true",
                        help="redo runs that already have metrics.json")
    ablate.add_argument("--dry-run", action="store_true")
    ablate.set_defaults(func=cmd_ablate)

    figures = sub.add_parser("figures", help="write the main table and figures")
    _common(figures)
    figures.add_argument("--out", default=None)
    figures.set_defaults(func=cmd_figures)

    audit = sub.add_parser("audit", help="check that all 21 run artifacts exist")
    _common(audit)
    audit.set_defaults(func=cmd_audit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
