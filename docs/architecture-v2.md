# GraviLens — Architecture v2: 5 → 9

**Repo/package:** `gravilens` (GitHub: `https://github.com/<handle>/gravilens`, import name `gravilens`)
**Status:** design approved, not yet implemented.
**Baseline for comparison:** current state = 3 notebooks, 6.6k lines, monolithic, unversioned (no git).

---

## 1. What "9/10" means (acceptance criteria)

A reviewer can clone the repo, `pip install -e .`, run `pytest` in under 2 minutes on CPU, regenerate the dataset from seeds, reproduce every number in the paper and poster, and find no claim that the code does not support.

**10/10 is not a target.** Reserve it for "we tried to break it and couldn't". The concrete definition of done is §10.

### Non-goals

- Chasing a higher macro-F1. Route C (richer mass model) is a *follow-up triggered by* the Route B result, not a parallel bet.
- Reimplementing Lensformer / LensPINN. Our own ablation matrix is the physics-informed comparison.
- MLflow / W&B / DVC / Hydra / Docker / src-layout. See §9.

---

## 2. The eight moves

| # | Move | Kills |
|---|---|---|
| 1 | One model: `GraviLensP` v2 | inert `key/value` cross-attention, non-differentiable scatter inversion, dead `center_branch`, hidden `_theta_E` state |
| 2 | One dataset: matched macro lens, per-sample seeds | shear confound, `…-Eins` vs `…-Eins-Phys` misalignment, 2-class/3-class fallback |
| 3 | One training loop, val-only selection | Lookahead, adaptive clipping, grad-accum(=1), per-epoch test evaluation |
| 4 | One artifact surface: `runs/<run>/metrics.json` + `manifest.json` | stale `RECORDED_RUN` fallback, untraceable poster numbers |
| 5 | Package: `pyproject.toml` + `gravilens/` + `pytest` | 3 notebook copies, 6.6k-line monolith, hardcoded `/content/drive` and `/home/deansama` paths |
| 6 | Parameter-matched baselines (ViT/CaiT/ViTSD) | 192-dim vs 1024-dim comparison; ViT's hobbled `mlp_dim=16` |
| 7 | Gates as pytest tests, not notebook cells | the bug class where a branch silently disconnects |
| 8 | Figures from artifacts only | every poster number that cannot be traced to a run |

---

## 3. Target repo layout

```
gravilens/
  pyproject.toml
  README.md
  LICENSE
  .gitignore
  gravilens/
    __init__.py
    config.py        # dataclasses + YAML loading, path resolution
    physics.py       # SIS deflection, differentiable inverse, oracle reference
    models.py        # GraviLensP v2 + variants + 3 baselines + param budget
    simulate.py      # shared-macro-lens simulators, sharded resumable generation
    dataset.py       # Dataset, augmentation, loaders, manifest reading
    train.py         # single training loop, checkpointing, resume
    evaluate.py      # classification + physics metrics, gates, ablation runner
    figures.py       # poster figures + claim audit, artifacts only
    cli.py           # python -m gravilens {generate,train,ablate,evaluate,figures,audit,gates}
  tests/
    test_physics.py
    test_models.py
    test_pipeline.py
  configs/
    data.yaml
    ablation.yaml
    final.yaml
  notebooks/
    gravilens_colab.ipynb     # thin driver: pip install + calls
  docs/
    method.md
    limitations.md
```

Target size: ~2.2k lines of package code + ~450 lines of tests, down from 6.6k lines of notebook.

### Files removed from the tree (git history retains them)

- `gravilens_finals_poster_resumable_patched.ipynb`
- `gravilens_monolithic_patched.ipynb`
- `gravilens_monolithic_colab.ipynb`
- `ttv_eins_final.ipynb`, `ttv_eins (1).ipynb` (identical 932-line duplicates)
- `*.ipynb:Zone.Identifier`, `*:Zone.Identifier` (Windows NTFS stream artifacts — junk)
- `DeepLenseSim/` → replaced by an upstream commit citation + `third_party/NOTICE.md` (pyHalo/lenstronomy APIs are pinned; the vendored copy is not imported). Rationale: it is the reference for the axion simulator, which we port into `simulate.py` and verify against the pinned upstream commit.

---

## 4. Model v2 specification

Decision taken: **keep the two stacks (physics stack + classifier stack), move the source cross-attention to the front of the classifier stack** so the source conditions every classifier block.

```python
def forward(self, images, return_aux=False):
    init = self.tokenizer(images)                              # [B, 1+64, D]
    enc  = self.physics_stack(init)                             # 4 window-attention blocks
    theta = self.theta_head(enc[:, 1:].mean(1))                 # [B], mapped to [0.5, 2.0]
    src   = sis_invert_bilinear(images, theta)                  # differentiable, one call
    src_t = self.source_tokenizer(src)                          # [B, 1+64, D]
    x     = init + self.cross_attn(q=init, kv=src_t)            # source injected FIRST
    out   = self.classifier_stack(x)                            # 4 window-attention blocks
    logits = self.class_head(out)
    return (logits, theta, src) if return_aux else logits
```

### 4.1 Required changes vs the current notebook

| Current | v2 |
|---|---|
| `TransformerBlock.forward(x, H, W, key, value)` accepts and ignores `key`/`value` | remove the parameters; source enters through the explicit `cross_attn` at the front of the classifier stack |
| `SimpleSISPredictor.forward` computes a κ-map + `center_branch` | delete both; `theta_head` is a single MLP on mean-pooled patch tokens, returns θE explicitly |
| `self._theta_E` hidden mutable state | gone; `theta` is a return value everywhere |
| `SimpleSISPredictor` called twice in `GraviLensP.forward` (once for `_theta_E`, once for `predict_theta`) | one call |
| old `SISPhysicsInformedEncoder` source path (`.round().long()` + `scatter_add`, non-differentiable) | deleted; only `sis_invert_bilinear` survives |
| `GraviLens` (legacy) class | deleted; `GraviLensP` v2 is the only physics model |
| `Lensformer`, `LensPINN` re-implementations | deleted (not used as baselines) |
| `ViTModel`, `CaiTModel`, `ViTSDModel` | kept, retuned, `mlp_dim = 4*dim` (was 16) |

### 4.2 Variants (the ablation matrix)

All variants share the same architecture and the same recipe; each differs in exactly one thing.

| Variant | Source tensor | θE aux loss | Distortion tokens |
|---|---|---|---|
| `full` | differentiable SIS inverse | yes | no |
| `no_sup` | differentiable SIS inverse | no (λ=0) | no |
| `no_sis` | max-normalised lensed image | no | no |
| `dist` | differentiable SIS inverse | yes | yes (added at tokenizer input) |

`no_sis` swaps only the source tensor → the cleanest possible one-variable ablation.

### 4.3 Parameter budget

Measured GraviLensP v2 trainable params (image 64, patch 8, D=192, 8 heads, 4+4 blocks, window 4, 3 classes):

| Component | Params |
|---|---|
| 2 × ShiftedPatchTokenization | 148,224 |
| 8 × TransformerBlock (2 stacks) | 3,562,048 |
| cross-attention + norm | 148,608 |
| θE head (MLP 192→512→256→1) | 231,937 |
| classifier head (mean-pool → 192→64→64→64→3) | 20,867 |
| **total** | **4,111,684** |

The head is mean-pooled rather than flattened: the flattened variant measured in the notebook costs ~0.8M params in a single linear layer, which is both the largest parameter sink in the model and the easiest place to memorise. Matching the baselines to 4.11M is therefore also a stricter comparison.

Measured baseline matches (search grid, ±10% tolerance):

| Baseline | dim | depth | params | vs target |
|---|---|---|---|---|
| ViT | 288 | 4 | 4,035,459 | 98.1% |
| CaiT | 288 | 3 | 4,035,747 | 98.2% |
| ViTSD | 288 | 4 | 4,035,459 | 98.1% |

`models.py` ships `count_trainable_params()` as the authority. `match_budget(target, tolerance=0.10)` searches `(dim, depth)` over a small grid for each baseline and writes the chosen config into `configs/ablation.yaml`. Baselines are reported with their exact param count; no comparison sentence is allowed without it.

---

## 5. Data v2 specification

### 5.1 One shared macro lens (the confound fix)

All three classes share the **same distribution** of macro-lens parameters. Class identity is carried **only** by substructure.

| Component | Spec (identical for all classes) |
|---|---|
| Main halo | SIE, `e1=e2=0`, centre (0,0) |
| θE | `TruncNormal(mu=1.4, sd=0.35)` clipped to [0.5, 2.0] arcsec |
| Shear | magnitude 0.05, **orientation random** (see 5.2) — currently `no_sub` has none at all |
| Lens light | 2 × SERSIC_ELLIPSE, mag 40, `(R=0.4, n=2.3)` and `(R=1.5, n=1.2)`, orientations random |
| Source | INTERPOL, mag ~ N(16, 0.1), position ~ U(-0.4, 0.4)², angle ~ U(-π, π), scale 0.015 |
| Observation | Euclid VIS band, Gaussian PSF, 6-year coadd, lenstronomy noise model |
| Cosmology | FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.049), z_lens 0.5, z_source 2.5 — **identical for lenstronomy and pyHalo** (Ob0 must be non-zero: Colossus refuses the Eisenstein & Hu transfer function at Ob0=0) |
| Substructure | pyHalo `sigma_sub=0.025`, `log_mlow=7`, `log_mhigh=10`, cone 6.5 arcsec (covers the 6.36 arcsec image). `log_mlow=7` is the resolution floor; it cuts ~15k mostly unresolvable halos per realisation to ~800 and reduces generation from ~32 s to ~2 s per CDM image. |

Class-specific part:
- `no_sub` — SIE + shear only, no subhaloes.
- `cdm` — + pyHalo CDM realisation.
- `axion` — + pyHalo ULDM halo population (`log10_m_uldm=-21.5`). The optional main-deflector fluctuation field is disabled (`flucs=False`): the pinned pyHalo 1.4.6 requires image-plane apertures for it, and the halo population is the class signal. Ported from `DeepLenseSim/Model_IV/sim_axion.py`; this deviation is disclosed in `docs/method.md`.

**Two bugs fixed here, both of which currently leak class identity:**
1. `no_sub` currently renders with no shear while `cdm`/`axion` carry `gamma1=0.05` — a classifier can separate the classes on shear, not substructure.
2. `no_sub` draws θE from N(1.24, 0.19) while `cdm`/`axion` draw N(1.5, 0.5) — a classifier can separate the classes on macro-lens size, not substructure.

### 5.2 Orientation randomisation (AUG-1)

With a fixed `gamma1=0.05`, horizontal flip / vertical flip / ±15° rotation are **not** physics-preserving: they produce images whose shear or lens-light orientation is outside the training distribution. The current augmentation therefore trains on off-manifold images.

Fix: sample the shear and lens-light orientation randomly (same prior for all classes), which makes the dataset isotropic and the existing augmentation valid. The alternative (drop augmentation) stays available as a config flag, decided on validation only.

### 5.3 Reproducibility

- Per-sample seed: `seed = f(master_seed, class, split, index)`; all stochastic calls in lenstronomy/pyHalo/numpy run under it.
- Stored per sample: `images`, `labels`, `theta_e`, `sources`, `seeds`, `class_ids`, `sample_ids`.
- Split layout: `data/gravilens-v2/{train,val,test}/shard_XXX.npz` + `manifest.json` per split containing: per-class counts, θE statistics, sha256 per shard, config hash, seed, git commit, library versions, timestamp.
- Sizes default 6999/1500/1500, equal thirds per class (7,000 is not divisible by 3; the generator always writes whole triplets). Configurable (e.g. 4000/1000/1000) without code change.

### 5.4 Physics evaluation set — dissolved

There is no separate `…-Eins-Phys` folder. The physics evaluation set is a **fixed stratified slice of the same test split** (e.g. 500 images, stratified by class), because the source ground truth is produced for every generated image anyway. Image / label / θE / source alignment is then guaranteed by construction, and the `…-Eins` vs `…-Eins-Phys` question disappears instead of being documented.

---

## 6. Training specification

One loop for every variant and every baseline (`train.py`).

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr 3e-4, weight decay 1e-3 |
| Schedule | 5 warmup epochs → cosine to 1e-6 |
| Epochs / batch | 50 / 64 |
| Loss | CE with label smoothing 0.05 (+ `λ·SmoothL1(θE)` where `λ>0`) |
| EMA | kept, config flag `use_ema` (default True) — documented as the one retained stabiliser |
| Deleted | Lookahead, `adaptive_gradient_clip`, `GradientAccumulator` (`accumulation_steps=1` already) |
| Augmentation | flips + ±15° rotation, intensity ramp from epoch 10 (valid under 5.2) |
| Selection | best **val** macro-F1; tie-break on lower val loss; early stop on val loss, patience 25 |
| Test set | never touched inside `train.py`; a test asserts this with a spy loader |
| Determinism | `seed_everything` per run; cudnn deterministic where cheap; non-deterministic ops recorded in the manifest |
| Checkpoints | `last.pt` and `best.pt`, atomic write (tmp + rename), full optimizer/scheduler/epoch/history state |
| Resume | restores exactly; a test asserts resume does not duplicate epochs and reproduces the same weights |

Rationale for deleting the stabilisers: with `accumulation_steps=1` the accumulator is a no-op, and Lookahead + EMA + adaptive clipping together make it impossible to attribute a metric change to the physics branch. One stabiliser, documented, is better science than three undocumented ones.

---

## 7. Evaluation specification

### 7.1 Metrics

- **Classification:** macro-F1 (primary), multiclass AUC (OvR, macro — the method is stated in every table), accuracy, 3×3 confusion matrix in fixed class order, per-class F1.
- **Physics:** θE MAE / RMSE / bias / Pearson / fraction within 10%, always against the **train-mean constant-guess baseline**; source SSIM for four references — model, oracle-bilinear, oracle-scatter, lensed-image — so the model SSIM is read against its ceiling.
- **Three seeds (42, 123, 777):** mean ± SD computed from seed-level values; raw per-seed metrics saved; AUC seed basis stated.

### 7.2 Gates (enforced in `pytest`, not notebook cells)

| Gate | Assertion |
|---|---|
| G2 gradient | classification loss (λ=0) produces non-zero grads on θE-head params and on source-token / cross-attention params; grid coords stay connected to θE |
| G3 functional use | zeroing / shuffling the source tokens changes logits on a representative batch |
| G4 physical utility | model SSIM and θE error beat the lensed-image and constant-guess references where claimed |
| G5 ablation | `full` vs `no_sis` vs `no_sup` present, matched, 3 seeds |
| G6 reproducibility | every run has `metrics.json` + `manifest.json` with hashes, seeds, versions, param count |
| G7 artifact integrity | every poster number maps to an artifact; `figures.py audit` exits non-zero otherwise |

### 7.3 Pre-registered decision rule (decided before Phase 4 runs)

If `full − no_sis` ≤ 0 macro-F1 averaged over 3 seeds, the result is reported as a **null result**: *"the reconstruction is differentiable and physically interpretable but does not improve 3-class classification on this dataset"*. θE and SSIM diagnostics are reported separately. Route C (SIE+shear inverse) becomes the follow-up, scoped from this result.

### 7.4 Baselines

`ViTModel`, `CaiTModel`, `ViTSDModel` (re-implemented in `models.py`, `mlp_dim = 4*dim`), retuned to GraviLensP ±10% trainable params, trained with the **identical** recipe, seeds, and augmentation, 3 seeds each. Throughput benchmarked with warm-up, `torch.cuda.synchronize()`, fixed batch sweep, and end-to-end timing including source reconstruction.

---

## 8. Work plan

| Phase | Work | Gate |
|---|---|---|
| 0 | `git init`; `pyproject.toml`; `gravilens/{config,physics,models}.py`; `tests/test_physics.py`, `tests/test_models.py`; delete legacy notebooks; move roadmap into `docs/` | `pytest` green on CPU <2 min; no dead parameters; no `_theta_E`; import path works |
| 1 | `simulate.py` (shared macro lens, per-sample seeds, sharded + resumable) + `dataset.py` + `tests/test_pipeline.py` | 50-image pilot: θE in bounds, 3 classes, source GT aligned, identical macro-lens distribution across classes |
| 2 | Full dataset generation (7k/1.5k/1.5k) | manifest hashes; equal class counts; zero NaNs; physics slice is a subset of test |
| 3 | `train.py` + `evaluate.py` + gates + `cli.py` | tiny-overfit passes; G2/G3 green on a real checkpoint; train.py provably never sees test |
| 4 | 3 seeds × 4 variants + 3 matched baselines × 3 seeds (21 runs) | every run has `metrics.json` + `manifest.json`; decision rule evaluated |
| 5 | `figures.py`, `figures.py audit`, README, `docs/method.md`, `docs/limitations.md`, release tag, QR target | `audit` exits 0; poster numbers all traceable |
| 6 | Thin Colab notebook (`notebooks/gravilens_colab.ipynb`) | a clean Colab runtime can `pip install` from the tagged release and run the pipeline end to end |

Measured pilot (3 real triplets, CPU, 58-galaxy pool): **5.42 s/triplet**. Full 3,333 triplets
≈ 5.0 h single-process; train/val/test shards can run concurrently. The pilot shard is
`data/pilot/test/shard_000000.npz` (9 images, 3 matched triplets).

Phase 0 is CPU-only. Phases 1–2 (generation) run concurrently with 0–3 on separate Colab runtimes.

### 8.1 Run priority (if GPU time runs short)

1. `full`, `no_sis`, `no_sup` × 3 seeds — the reviewer-critical comparisons.
2. 3 matched baselines × 3 seeds — the "we beat data-driven" sentence.
3. `dist` × 3 seeds — only if the distortion question is still open.
4. `base` is **not rerun**; its historical numbers are disclosed in text only.

### 8.2 Parameter matching procedure

`models.match_budget(target, tolerance=0.10)` sweeps `(dim, depth, mlp_dim=4*dim)` over a small grid, builds each model on CPU, counts trainable params, and picks the closest configuration. The chosen values are written into `configs/ablation.yaml` and echoed into every run's `manifest.json`, so a reviewer can reproduce the budget claim without rerunning the search.

---

## 9. Explicitly skipped (YAGNI)

- Hydra / OmegaConf → dataclasses + YAML.
- MLflow / W&B → `metrics.json` + `manifest.json` + a `figures.py audit` gate.
- DVC → per-sample seeds + manifest hashes.
- Docker → `pyproject.toml` with a `sim` extra.
- `src/` layout → flat package.
- Plugin/model registry → two functions (`build_model`, `build_config`).
- Distributed training → single GPU.
- CI config → `pytest` runs locally in <2 min.
- Pre-registered experiment tracker → the decision rule is a paragraph in `docs/method.md`.
- A second physics inverse model (SIE + shear) → deferred to Route C, triggered by the §7.3 result.

Kept deliberately: EMA flag, augmentation intensity ramp, the `dist` variant. `base` is historical only and is not rerun.

---

## 10. Definition of done (the 9/10 checklist)

1. `git clone && pip install -e . && pytest` passes on CPU in <2 minutes.
2. `python -m gravilens generate --config configs/data.yaml` reproduces a 50-image sample bit-for-bit across two runs (hash equality).
3. `python -m gravilens ablate --config configs/ablation.yaml` produces 21 runs, each with `metrics.json` + `manifest.json`; test metrics appear exactly once per run, after checkpoint selection.
4. Every parameter of every model receives a gradient in the smoke backward pass (no dead parameters).
5. G2 and G3 pass on the final checkpoint and are enforced by pytest.
6. No test-set access inside `train.py` (asserted by a test using a spy loader).
7. The three classes share one macro-lens prior (asserted by a dataset test comparing θE / shear / lens-light distributions across classes).
8. Baselines are within ±10% trainable params, trained with the identical recipe, 3 seeds each.
9. Every poster number traces to an artifact; `figures.py audit` exits non-zero on any missing artifact.
10. `README.md` covers install, data regeneration, reproduction of the main table, the model card, and known limitations (SIS assumption, simulated data, 3 seeds).
11. Legacy notebooks are gone from the tree; git history retains them.
12. One thin Colab notebook installs the package and runs the pipeline end to end.

---

## 11. Risk register

| Risk | Mitigation |
|---|---|
| R1 — generation cost unknown (pyHalo per-image raytraces) | 50-image pilot measures cost and extrapolates before committing; shards + resume make it interruptible; split sizes are config, not code |
| R2 — the matched macro lens makes results worse | that is the point; the §7.3 null rule is pre-registered; worse-but-honest beats better-but-confounded |
| R3 — lenstronomy/pyHalo/astropy version drift breaks the simulator | pin exact versions; the astropy `angular_diameter_distance` shim stays but is version-guarded; manifest records versions |
| R4 — retuned baselines are stronger, weakening our F1 claim | accepted; param-matched comparison is the only defensible one |
| R5 — 21 runs exceed the GPU budget | §8.1 priority order |
| R6 — effort split between generation and code | phases 1–2 run on a separate Colab runtime from phases 0–3 |

---

## 12. Open items resolved

- Repo/package name: `gravilens` (GitHub `gravilens`, import `gravilens`).
- Model shape: two stacks, cross-attention moved to the front of the classifier stack.
- Baselines: 3 data-driven, parameter-matched, no re-implemented physics baselines.
- Legacy notebooks: deleted from the tree, retained in git history.
- Compute budget: full — matched dataset regeneration + 3 seeds × 4 variants + 3 parameter-matched baselines × 3 seeds.
- Delivery: public GitHub package + thin Colab driver.
- Success criterion: reviewer-proof repo + honest results, not a higher F1.
