# Method

This document describes the v2 pipeline as implemented. It is deliberately
short; the approved design and the risk register are in
[`architecture-v2.md`](architecture-v2.md).

## Data

The dataset is simulated from Galaxy10 DECals cutouts selected with
`ans == 6` and `redshift < 0.02` (58 galaxies). The same galaxy pool is used for
every class. Each **triplet** is one macro lens rendered three times:
`no_sub` (SIE + shear), `cdm` (same lens + pyHalo CDM subhaloes), and `axion`
(same lens + pyHalo ULDM haloes). Within a triplet the following are shared
exactly: θE, shear, lens-light orientation, source galaxy, source position,
source angle, source magnitude, and one noise realisation. Only the
substructure population differs, so a classifier cannot use the macro lens or
noise to identify the class.

The macro lens is an SIE with `e1=e2=0`, a shear of magnitude 0.05 at a random
orientation, and θE drawn from a clipped `TruncNormal(1.4, 0.35)` on
`[0.5, 2.0]` arcsec. Lens-light orientations are random. Cosmology is
`FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.049)` at `z_lens=0.5` and
`z_source=2.5`, identical for lenstronomy and pyHalo. The source is an
`INTERPOL` Galaxy10 cutout with `scale=0.015`; observations use the Euclid VIS
band with the lenstronomy noise model.

Substructure uses `sigma_sub=0.025`, `log_mlow=7`, `log_mhigh=10`, and a
6.5 arcsec cone. The resolution floor keeps about 800 CDM haloes per
realisation instead of ~16,000 and reduces generation from ~32 s to ~2 s per
CDM image. The axion class uses `log10_m_uldm=-21.5`, the same `sigma_sub`,
and `flucs=False`. The main-deflector fluctuation field is therefore disabled;
this is a limitation, not a claim that the axion model is complete.

Default split sizes are 6,999/1,500/1,500 images (2,333/500/500 triplets).
The measured CPU cost is ~6 s per triplet when a split runs alone and ~8 s
with three splits running concurrently (the 3-triplet pilot was 5.4 s). Shards
are atomic `.npz` files named by first triplet; rerunning the generator
resumes completed shards. A fixed stratified slice of the test split
(167 images per class, seed 42, 501 images) is the physics evaluation set; the
source ground truth is stored for every generated image, so no separate
folder is needed.

The dataset that the recorded runs were trained on has fingerprint
`a92ba64977d930d3` (train 6,999 / val 1,500 / test 1,500 images; 2,333/500/500
matched triplets, equal class counts, θE in [0.5, 2.0]). Generating it took
3.5 h for the train split and 1.1 h each for `val` and `test` on 8 CPU cores.

Two sanity checks on the completed dataset, both reproducible from the shards
(4-fold cross-validation grouped by triplet, so no triplet crosses a fold):

- **No brightness shortcut.** Total image flux is not a class cue: a
  flux-only classifier reaches AUC 0.53 (cdm vs no_sub), 0.50 (axion vs
  no_sub) and 0.53 (axion vs cdm). The macro lens dominates the pixel values.
- **The class signal is present.** A linear probe on raw pixels is at chance
  (0.32 macro-F1) for the same reason, but the same probe on triplet
  residuals — each image minus the mean of its own triplet, which removes the
  macro lens — reaches 0.68 macro-F1. The mean pixel difference between a
  class image and its `no_sub` partner is 0.0096, about 5% of the mean pixel
  intensity.

The residual probe is a diagnostic, not a result: it uses the other two
classes of the same triplet, which a classifier never sees at test time. It
bounds how much information the images carry; it is not a performance the
model can reach.

## Model

`GraviLensP` has 4,111,684 trainable parameters:

```
image -> shifted-patch tokenizer -> 4-block physics stack -> theta_E head
      -> differentiable bilinear SIS inverse -> source tokenizer
      -> cross-attention (query = image tokens, key/value = source tokens)
      -> 4-block classifier stack -> mean-pool -> 3 logits
```

The SIS inverse samples the source plane from the image plane with bilinear
interpolation, so gradients flow from the source through θE to the physics
stack. The source cross-attention is at the **front** of the classifier stack,
so the reconstruction conditions every classifier block. θE is an explicit
return value, never hidden state.

The ablation variants change a controlled part of the forward path or loss:

| Variant | Source tensor | θE aux loss | Distortion tokens |
|---|---|---|---|
| `full` | differentiable SIS inverse | yes | no |
| `no_sup` | differentiable SIS inverse | no | no |
| `no_sis` | max-normalised lensed image | no | no |
| `dist` | differentiable SIS inverse | yes | yes |

The three classifier baselines (ViT, CaiT, ViTSD) use their standard patch
embeddings inside the same training recipe, with `dim=288` and depths 4/3/4.
Their trainable counts are
4,035,459/4,035,747/4,035,459 (98.1–98.2% of the GraviLens budget), within the
pre-registered ±10% window.

## Training

One loop trains GraviLens variants and baselines. AdamW uses lr `3e-4`, weight
decay `1e-3`, five warmup epochs, and cosine decay to `1e-6`; training runs 50
epochs at batch 64. The loss is cross-entropy with label smoothing 0.05, plus
`0.1 * SmoothL1(theta_E)` for `full` and `dist`. EMA is retained as the one
stabiliser and is a config flag; decay 0.999 is an ~9-epoch window at 109
steps/epoch, so it can affect a 50-epoch run. Flips and ±15° rotations start at epoch 10;
they are on-manifold because shear and lens-light orientations are isotropic.

Checkpoint selection uses **validation** macro-F1 (lower validation loss
breaks ties); early stopping is patience 25 on the **validation loss**.
`train.py` calls `load_split` only for `train` and `val`; a pytest spy asserts
this. The provenance fingerprint scans every shard, including test, but no test
image or label enters training. `best.pt` is the selected checkpoint; `last.pt`
is written every epoch for inspection.

The grid is GPU-bound by design: measured on 8 CPU cores, one step (batch 64)
costs ~1.8 s, so a 50-epoch run takes ~2.7 h and the 21-run grid ~57 h. A
single GPU runtime is the intended host.

## Evaluation

`evaluate.py` is the only module that loads the test split. Classification
metrics use all 1,500 test images: accuracy, macro-F1 (primary), macro OvR
AUC, per-class F1, and a fixed-order 3×3 confusion matrix. Physics metrics use
the fixed 501-image stratified slice: θE MAE/RMSE/bias/Pearson/fraction within
10%, always alongside the train-mean constant-guess MAE; and source SSIM for
the model, oracle-bilinear, oracle-scatter, and lensed-image references.

Seeds are 42/123/777. Results are reported as seed-level mean ± SD; raw
per-seed values remain in each `metrics.json`.

## Pre-registered decision rule

If the three-seed mean of `full − no_sis` test macro-F1 is **≤ 0**, the result
is reported as a null result: the reconstruction is differentiable and
physically interpretable but does not improve three-class classification on
this dataset. θE and SSIM diagnostics are reported separately. No Route C
model is trained as part of this result.

## Provenance

Each run writes `manifest.json` (git SHA and dirty flag, library versions,
model/training config, data root and content fingerprint, parameter count,
seeds) and `metrics.json` (test metrics, physics slice hash, references). The
dataset fingerprint (`dataset.fingerprint()`) hashes the array contents and the
data-defining manifest fields, not the `.npz` container bytes, so two
independent generations of the same data hash identically and a copied dataset
can be verified with one command
(`python -m gravilens.dataset --split all` prints the fingerprint). Rendering
is bit-for-bit deterministic for a given triplet seed:
`tests/test_pipeline.py` renders a real triplet twice and asserts identical
pixels.
`python -m gravilens audit` checks all 21 runs, enforces the recipe above
(epochs, batch size, learning rate, EMA, augmentation, seed) and refuses to
generate figures from an incomplete grid; `python -m gravilens figures` writes
the main table and figures from those JSON artifacts only. `ablate` reuses an
existing run only when its manifest records the same recipe and the same
dataset fingerprint.
