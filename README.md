# GraviLens

Physics-informed lens modelling for dark-matter substructure classification.

**Status:** architecture v2, Phases 0–5 code complete. The full 21-run
ablation has not been executed yet; no result numbers are claimed.

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
three baselines, shard resume/corruption, the matched-triplet gate, tiny-overfit
training, train/val-only isolation, evaluation metrics, and the 21-run audit.

## Data

The v2 dataset is regenerated from Galaxy10 DECals (2.54 GB, downloaded once
into `data/raw/`). Each triplet shares one macro lens and one noise
realisation; only the substructure differs.

```bash
# full dataset (train/val/test), resumable
python -m gravilens generate --config configs/data.yaml

# 3-triplet pilot
python -m gravilens generate --config configs/data.yaml --data-root data/pilot \
    --splits test --limit 3 --shard-size 9
```

Shards are atomic `.npz` files under `data/<root>/<split>/`; rerunning the
same command resumes. A real triplet costs ~8 s on the current CPU; the
2,333-triplet train split takes ~5 h. `val` and `test` can run as separate
concurrent commands and finish much sooner. Inspect a split with:

```bash
python -m gravilens.dataset --data-root data/gravilens-v2 --split test
python -m gravilens.dataset --data-root data/gravilens-v2 --split all
```

`data/` is gitignored. The Galaxy10 selection is `ans == 6` and
`redshift < 0.02` (58 source galaxies); the same pool is used for every class.

## Train and evaluate

Training never loads test images or labels; only `evaluate` does, once per run.
The provenance hash scans all shard bytes, including test, but does not feed
them to training.

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
`last.pt`, and then `metrics.json`. `ablate` skips a run only when it used the
same recipe and the same dataset; a run from another recipe or an older
dataset is redone. The test split is evaluated only after checkpoint
selection.

## Results and audit

The full grid has **not** been run in this snapshot. No result numbers are
claimed in this README. When all 21 runs exist:

```bash
python -m gravilens audit    # exits non-zero if a run or required field is missing
python -m gravilens figures  # writes runs/figures/{main_table.md,summary.json,*.png}
```

`audit` checks the expected 4 variants × 3 seeds and 3 baselines × 3 seeds,
the pre-registered training recipe, parameter matching, test-metric fields,
data fingerprints, and physics references. `figures` reads only the JSON
artifacts.

## Model

```
image -> tokenizer -> physics stack -> theta_E -> differentiable SIS inverse
      -> source tokens -> cross-attention at the front of the classifier stack
      -> classifier stack -> pooled head -> logits
```

`GraviLensP` has four variants: `full`, `no_sup` (theta_E loss weight 0),
`no_sis` (lensed image replaces the reconstructed source, with no theta_E
auxiliary loss), and `dist` (distortion tokens added at the input). The
`no_sis` forward path swaps exactly the source tensor.

### Model card

- **Parameters:** 4,111,684 trainable.
- **Input:** one 64×64 single-channel lensed image, max-normalised.
- **Output:** 3-class logits, an explicit θE in arcsec, and a 64×64 source
  reconstruction.
- **Training data:** 6,999 simulated images (2,333 matched triplets) from
  Galaxy10 DECals; 1,500 validation and 1,500 test images.
- **Intended use:** research on matched-lens substructure classification. Not
  a survey pipeline or a calibrated dark-matter detector.
- **Known limits:** simulated SIS macro lens, 58-source pool, fixed
  substructure priors, axion fluctuations disabled, three seeds. See
  [`docs/limitations.md`](docs/limitations.md).

The implementation and the pre-registered decision rule are in
[`docs/method.md`](docs/method.md).

## Layout

```
gravilens/       package (config, physics, models, simulate, dataset, train, evaluate, figures, cli)
tests/           CPU gates
configs/         data and ablation YAML
data/            generated shards (gitignored)
notebooks/       thin Colab driver
third_party/     vendored reference copy of DeepLenseSim (not imported)
docs/            architecture, method, limitations, roadmap
```

## Roadmap

See `docs/roadmap-v1.md` for the scientific plan and
`docs/architecture-v2.md` for the implementation plan. The next step is the
21-run ablation on a GPU runtime, followed by `audit` and `figures`.
