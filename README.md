# GraviLens

Physics-informed lens modelling for dark-matter substructure classification.

**Status:** architecture v2, Phases 0–3. The API and the data format are not
frozen until Phase 4. The full 21-run ablation has not been executed yet.

## Install

```bash
git clone https://github.com/<handle>/gravilens
cd gravilens
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Data generation needs the simulation extra:

```bash
pip install -e ".[sim,dev]"
```

## Test

```bash
pytest
```

The suite runs on CPU in under two minutes. It covers the SIS coordinate
convention and differentiability, the model variants, the no-dead-parameters
gate, the source counterfactual gate, the parameter-budget match for the
three baselines, shard resume/corruption, and the matched-triplet gate.

## Data

The v2 dataset is regenerated from Galaxy10 DECals (2.54 GB, downloaded once
into `data/raw/`). Each triplet shares one macro lens and one noise
realisation; only the substructure differs.

```bash
# full dataset (train/val/test), resumable
python -m gravilens.simulate --config configs/ablation.yaml

# 3-triplet pilot
python -m gravilens.simulate --data-root data/pilot --splits test \
    --limit 3 --shard-size 9
```

Shards are atomic `.npz` files under `data/<root>/<split>/`; rerunning the
same command resumes. A real triplet costs ~5.4 s on CPU. Inspect a split with:

```bash
python -m gravilens.dataset --data-root data/gravilens-v2 --split test
```

`data/` is gitignored. The Galaxy10 selection is `ans == 6` and
`redshift < 0.02` (58 source galaxies); the same pool is used for every class.

## Train and evaluate

Training never loads the test split; only `evaluate` does, once per run.

```bash
# one run (GPU recommended)
python -m gravilens train --config configs/ablation.yaml \
    --model gravilens --variant full --seed 42

# test metrics for that checkpoint
python -m gravilens evaluate --run runs/gravilens_full_seed42

# full grid: 4 variants x 3 seeds + 3 baselines x 3 seeds = 21 runs
python -m gravilens ablate --config configs/ablation.yaml --dry-run
python -m gravilens ablate --config configs/ablation.yaml
```

Each run writes `manifest.json`, `history.json`, `val_metrics.json`, `best.pt`,
and then `metrics.json`. `ablate` skips runs that already have `metrics.json`,
so it resumes. The test split is evaluated only after checkpoint selection.

## Model

```
image -> tokenizer -> physics stack -> theta_E -> differentiable SIS inverse
      -> source tokens -> cross-attention at the front of the classifier stack
      -> classifier stack -> pooled head -> logits
```

`GraviLensP` has four variants: `full`, `no_sup` (theta_E loss weight 0),
`no_sis` (lensed image replaces the reconstructed source), and `dist`
(distortion tokens added at the input). `no_sis` differs from `full` in exactly
one tensor.

## Layout

```
gravilens/       package (config, physics, models, simulate, dataset, train, evaluate, cli)
tests/           CPU gates
configs/         YAML run configurations
data/            generated shards (gitignored)
notebooks/       thin Colab driver
third_party/     vendored reference copy of DeepLenseSim (not imported)
docs/            roadmap, method, limitations
```

## Roadmap

See `docs/roadmap-v1.md` for the scientific plan and
`docs/architecture-v2.md` for the implementation plan.
