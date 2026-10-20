# GraviLens — Comprehensive Research, Implementation, Evaluation, and Poster Roadmap

**Primary source:** `gravilens_finals_poster_resumable_patched.ipynb`  
**Purpose:** Make the reconstructed source genuinely influence classification logits, validate the physics path, run controlled experiments, establish reproducible metrics, and then finalize the conference poster.

---

## Executive decision

The best next approach is to use **`GraviLensP` as the candidate model**, then prove three separate claims:

1. **Computational dependence:** the reconstructed source participates in the forward path to classification logits.
2. **Physical validity:** the Einstein-radius predictor and SIS source reconstruction behave meaningfully on controlled simulations and held-out examples.
3. **Empirical value:** the source-aware model improves or otherwise provides a demonstrable benefit over carefully matched controls.

Do not preserve the original benchmark's reported `0.9840 ± 0.0018` macro-F1 as evidence of the physics branch's contribution. The official notebook's Section 19 documents that, in the original benchmark model, replacing the source image with noise or zeros leaves logits unchanged, the CLS token has zero variance across images, gradients to the predictor/encoder transformer are zero, and the rounded-index mapping is not differentiable in the predicted Einstein radius. The original benchmark result can remain a historical baseline, but the poster's proposed-model claim should come from the corrected model and its own evaluation.

**Critical principle:** a connected computation graph does not prove that the model uses useful physics, and useful-looking source images do not prove better classification. We need functional tests, physical tests, and matched empirical comparisons.

---

# Phase 0 — Freeze and document the source of truth

## Tasks

- [ ] Keep an untouched copy of the current official notebook and poster files.
- [ ] Create a versioned experiment branch or notebook copy for the final candidate.
- [ ] Record the exact code revision, configuration, data-folder paths, class order, image normalization, and split membership.
- [ ] Identify whether each result is from the old benchmark `GraviLens` or corrected `GraviLensP`.
- [ ] Do not overwrite or relabel old benchmark metrics as `GraviLensP` results.
- [ ] Decide whether the target variant is:
  - `full`: `GraviLensP`, `use_sis=True`, `use_dist=False`, with θE supervision;
  - `no_sup`: same corrected source branch without θE auxiliary loss;
  - `no_sis`: matched architecture with the lensed image used in place of the reconstructed source;
  - `dist`: corrected source branch plus distortion tokens (`use_dist=True`);
  - `base`: original benchmark, retained as a historical baseline rather than proof of functioning physics.

## Required source-of-truth record

Create a run manifest with:
- Git commit or archived notebook hash.
- Model class and exact flags (`use_sis`, `use_dist`, λθ).
- Dataset file hashes, shapes, class counts, and split membership.
- Seed and initialization details.
- Training config and checkpoint-selection rule.
- Test metrics, parameter count, throughput environment, and artifact paths.

### Exit criteria

- [ ] Anyone can tell which model version produced every displayed metric.
- [ ] The poster's QR destination resolves to the correct versioned code and experiment artifacts.

---

# Phase 1 — Correct the training/evaluation protocol before long runs

The official `train_gravilens_phys()` currently constructs a test loader and evaluates test metrics during the epoch loop. Even if those values are not used for early stopping, repeatedly inspecting test results creates an unnecessary risk of test-set feedback.

## Required changes

- [ ] Remove per-epoch test-set evaluation from the development training loop.
- [ ] Use training metrics for optimization diagnostics and validation metrics for checkpoint selection/early stopping.
- [ ] Save the checkpoint selected according to the prespecified validation criterion.
- [ ] Evaluate the held-out test set only after model selection and configuration tuning are complete.
- [ ] Keep test evaluation in a separate final-evaluation function that loads the selected checkpoint.
- [ ] If historic test curves are retained for comparison, label them clearly as retrospective and verify that they did not influence model selection.
- [ ] Confirm no data augmentation, normalization, or preprocessing is fitted using validation/test data.
- [ ] Confirm source and θE test labels are never used to tune a model after viewing test performance.

### Exit criteria

- [ ] The training loop cannot use test performance for model decisions.
- [ ] A clean final test evaluation can be run without altering the trained checkpoint.

---

# Phase 2 — Validate the physical data and coordinate conventions

The patched SIS reconstruction uses `grid_sample` with bilinear sampling and a predicted θE. The physical generator and reconstruction code must agree about coordinates, units, source normalization, lens center, and boundary behavior.

## 2.1 Dataset alignment

The notebook uses an `...-Eins` folder for the main ablation data and an `...-Eins-Phys` folder for physics test assets (`data_test`, `er_test`, `src_test`). Do not assume these are identical or aligned just because they share naming conventions.

- [ ] Compare shapes and hashes for image/label/θE splits.
- [ ] Confirm class counts and class ordering across folders.
- [ ] Confirm that the physics test images and source/θE labels correspond to the exact same images and order.
- [ ] If the physics test set is a separately generated sample, document it as a separate physics-evaluation set rather than silently treating it as the classification test set.
- [ ] Keep a stable sample ID/index mapping between image, label, θE, and source.
- [ ] Verify that source targets use the same 64×64 grid and intensity convention assumed by the reconstruction and SSIM metric.

## 2.2 Coordinate and SIS checks

- [ ] Confirm the pixel scale (`0.101 arcsec/pixel`) is appropriate for the generated dataset and matches the simulator.
- [ ] Confirm the assumed lens center is consistent with how the images were generated.
- [ ] Verify the mapping from image coordinates to `grid_sample` normalized coordinates, including x/y order and `align_corners=True`.
- [ ] Check that boundary masking and zero padding behave as intended.
- [ ] Inspect the `core`/radius regularization around the SIS singularity.
- [ ] Test θE values spanning the generator's stated range of 0.5–2.0 arcsec, not only a typical example.
- [ ] Verify the direction of each mapping: source-to-image rendering is not the same operation as image-to-source reconstruction.
- [ ] Ensure the two-image SIS reconstruction logic and the weighting inside the caustic are consistent with the intended simplified physical model.
- [ ] Test edge cases: θE bounds, center pixels, low intensity, off-grid samples, and images with no valid second-image sample.

## 2.3 Oracle inversion and physical reference checks

The notebook already has functions to compare bilinear inversion with the benchmark's scatter inversion and the true θE.

- [ ] Run the oracle inversion using the known true θE.
- [ ] Compare bilinear and scatter inversions against the ground-truth source.
- [ ] Report source SSIM for:
  - The observed lensed image used as a naive reference.
  - Bilinear inversion with true θE.
  - Scatter inversion with true θE.
  - Model-predicted θE inversion.
- [ ] Interpret source SSIM relative to those references; do not present SSIM alone as proof that a reconstruction is physically accurate.
- [ ] Test whether the reconstruction method remains valid under the actual simulator assumptions (e.g. PSF, noise, shear or other lens components if present).
- [ ] If the inverse approximation is only valid for centered SIS simulations, state that limitation plainly.

### Exit criteria

- [ ] Dataset arrays, labels, θE values, and source images are aligned.
- [ ] Oracle reconstruction behaves as expected over the supported simulation range.
- [ ] Coordinate conventions and assumptions are documented.

---

# Phase 3 — Make sure the source genuinely affects classification

This is the immediate implementation milestone.

## 3.1 Simplify the θE predictor

The official notebook notes that the original predictor's center-related branch is unused by the corrected reconstruction, which assumes a centered lens.

- [ ] Use a dedicated scalar θE head based on mean-pooled patch tokens (as `GraviLensP` intends).
- [ ] Remove unused center-offset parameters unless an off-center lens model is explicitly implemented and validated.
- [ ] Predict θE within the simulator's valid interval [0.5, 2.0] arcsec using the documented constraint or parameterization.
- [ ] Return θE explicitly from the forward pass/auxiliary output; avoid reliance on hidden mutable state such as `self._theta_E`.
- [ ] Add shape, bounds, and finite-value checks.
- [ ] Confirm that the reported model parameter count is computed from the exact final configuration.

## 3.2 Gradient-path tests

Create automated tests for a representative batch. Run the classification-loss test with the auxiliary θE loss set to zero so the auxiliary supervision cannot mask a broken source-to-classification path.

- [ ] `source_hat` has the expected `[B, 1, H, W]` shape and finite values.
- [ ] `theta_hat` has shape `[B]`, is finite, and respects its bounds.
- [ ] Classification logits have shape `[B, 3]` and finite values.
- [ ] Classification loss alone produces finite, nonzero gradient with respect to `theta_hat` on nondegenerate test batches.
- [ ] Relevant θE predictor parameters receive nonzero gradients from classification loss.
- [ ] Source-token and cross-attention parameters receive gradients from classification loss.
- [ ] Bilinear grid coordinates remain connected to θE in autograd.
- [ ] The tests fail intentionally if the source path is detached or replaced by rounded integer indices.
- [ ] Check more than one batch; a zero gradient for one degenerate sample should not be treated as a universal result.

## 3.3 Counterfactual source-sensitivity tests

With the same trained checkpoint and same lensed inputs, compare logits when only the source-token input is changed:

- [ ] Correct reconstructed source.
- [ ] Zero source.
- [ ] Source tokens shuffled across the batch.
- [ ] A lensed-image placeholder or other carefully chosen control.

Record the change in logits/probabilities and predicted class. A change establishes functional sensitivity, not scientific usefulness by itself. Repeat the comparison over a representative held-out subset and report aggregate effect sizes, not a cherry-picked example.

## 3.4 Small-subset overfit test

- [ ] Train on a deliberately tiny subset to verify the end-to-end training pipeline can overfit.
- [ ] Confirm loss decreases and predictions respond to the labels.
- [ ] Confirm the physical path stays differentiable during the small run.
- [ ] Keep smoke-test metrics out of poster claims.

### Exit criteria

- [ ] A classification-only backward pass reaches the θE predictor through the source reconstruction.
- [ ] The source representation can change logits when counterfactually altered.
- [ ] The small-subset sanity test behaves normally.

---

# Phase 4 — Define a clean, controlled ablation study

Do not ask only “does the corrected model train?” Ask whether each proposed component contributes something measurable.

## Primary ablation matrix

| Variant | SIS reconstruction | θE auxiliary loss | Distortion feature | Purpose |
|---|---|---|---|---|
| `base` | Original benchmark path | No effective supervision | Original benchmark input | Historical reference only |
| `full` | Differentiable bilinear reconstruction | Yes | Off | Main corrected candidate |
| `no_sup` | Differentiable reconstruction | No | Off | Isolate θE supervision |
| `no_sis` | No inversion; substitute lensed image in source pathway | No | Off | Matched control for source inversion |
| `dist` | Differentiable reconstruction | Yes | On | Test whether the distortion tokens add value |

## Improve experimental control

- [ ] Keep data, class mapping, split, optimizer recipe, augmentation, and seed identical within each matched comparison.
- [ ] Train each variant from scratch; do not inherit a favorable checkpoint unless transfer is an explicit experimental variable.
- [ ] Keep the same architecture wherever possible when testing the value of SIS inversion.
- [ ] Clearly distinguish `base` (different/known-broken historical architecture) from matched `GraviLensP` ablations.
- [ ] Make `no_sis` differ from `full` only in the source-generation operation, as intended by the notebook.
- [ ] Make `no_sup` differ only by setting the θE auxiliary loss weight to zero.
- [ ] Make `dist` differ only by enabling the documented distortion feature path.
- [ ] Recalculate trainable parameter counts for all variants.
- [ ] Use validation data to select loss weight and training choices. Start at `ABL_LAM=1.0` but test a small, prespecified range if needed; do not select a weight with test performance.
- [ ] If you alter architecture to remove unused parameters, record that as a new model version instead of calling it a pure ablation.

## Minimum useful comparison order

To control compute while reducing uncertainty:

1. Smoke tests for `full` and `no_sis`.
2. One-seed validation/development runs for `full`, `no_sis`, and `no_sup`.
3. Fix issues and freeze the training recipe.
4. Run three-seed training for the final comparison.
5. Add `dist` if the distortion feature is a real research question; otherwise do not burden the core story with it.
6. Retain `base` as historical context only if the comparison is disclosed honestly.

### Exit criteria

- [ ] The main experiment isolates source reconstruction from other changes.
- [ ] Results are not selected using test scores.
- [ ] The poster's physics-benefit claim is supported by a fair comparison.

---

# Phase 5 — Training, checkpointing, and reproducibility

## 5.1 Training recipe

- [ ] Freeze the training config only after small validation runs.
- [ ] Use deterministic seeds as far as practical and record nondeterministic operations.
- [ ] Keep optimizer, scheduler, batch size, label smoothing, augmentation, and early-stopping rule consistent between matched ablations.
- [ ] Use validation loss/performance for checkpoint selection.
- [ ] Save the selected checkpoint, full config, random seeds, history, and model parameter count.
- [ ] Confirm resuming a checkpoint does not change the optimizer/scheduler state or accidentally duplicate epochs.
- [ ] Record hardware, framework versions, precision, and run duration.

## 5.2 Final seeds

The notebook currently sets `ABL_SEEDS = [42]`; this is suitable for early debugging, not for a three-seed mean ± SD.

- [ ] After the recipe is frozen, run the final selected model and key controls on seeds 42, 123, and 777.
- [ ] Save each seed's raw metrics individually.
- [ ] Compute macro-F1 mean and standard deviation directly from those three seed-level values.
- [ ] Compute AUC consistently across seeds if possible. Otherwise label AUC as seed 42 throughout and do not imply it is a multi-seed average.
- [ ] Keep test performance out of model selection.
- [ ] Ensure the final reported run uses the exact same three-class dataset/split as the final baselines.

## 5.3 Data and test alignment

- [ ] Compare `...-Eins` and `...-Eins-Phys` data paths and determine whether the physics evaluation set corresponds to the main classification test set or a distinct sample.
- [ ] If distinct, report those as separate evaluations.
- [ ] Use stable sample IDs to align lensed images, source targets, labels, and θE values.
- [ ] Do not use a separate physics dataset's source/θE target alignment to make claims about the classification test set unless the images truly correspond.
- [ ] Confirm no information from test labels or targets leaks into training data, normalization, split creation, or feature construction.

### Exit criteria

- [ ] The final metrics can be regenerated from saved runs.
- [ ] Three-seed variability is traceable to the raw seed-level results.
- [ ] The final held-out test set is used only for evaluation.

---

# Phase 6 — Evaluate classification and physics jointly

Classification F1/AUC alone cannot establish that the physics branch works.

## Classification metrics

- [ ] Macro-F1.
- [ ] Multiclass AUC with its exact averaging method and seed basis.
- [ ] Accuracy as a secondary metric, if helpful.
- [ ] 3×3 confusion matrix in a consistent class order.
- [ ] Per-class precision/recall or class-wise error rates if they support the scientific question.
- [ ] Three-seed summary for the final model and primary ablation(s).

## Physics metrics from the notebook

- [ ] θE MAE.
- [ ] θE RMSE.
- [ ] θE bias.
- [ ] Pearson correlation, compared with the constant-guess baseline.
- [ ] Source SSIM for predicted θE reconstruction.
- [ ] Source SSIM for true-θE oracle inversion.
- [ ] Source SSIM for the lensed-image reference.
- [ ] Bilinear vs scatter inversion comparison, where scientifically appropriate.

## Interpretation rules

- [ ] Compare θE prediction against a train-set constant-guess baseline rather than reporting MAE in isolation.
- [ ] Interpret source SSIM relative to the lensed-image baseline and oracle inversion.
- [ ] A good oracle reconstruction does not prove the predicted-θE model reconstructs well.
- [ ] A nonzero source gradient does not prove the model uses the source beneficially.
- [ ] Better F1 in a single seed is not enough to claim reliable improvement.
- [ ] If physics metrics improve but classification does not, report that distinction honestly.
- [ ] If classification improves but θE prediction is poor, do not claim successful physical inference without qualification.

### Exit criteria

- [ ] We can state separately what improved in classification and what improved in physical reconstruction.
- [ ] Every conclusion corresponds to a measurable result.

---

# Phase 7 — Throughput and parameter accounting

- [ ] Recompute parameter counts from the exact final model instance using `requires_grad` parameters; state whether totals are trainable parameters.
- [ ] Benchmark throughput with a consistent warm-up, synchronization, timed iteration count, batch size, device, and precision.
- [ ] Confirm the reported T4 values (including the current 2,807 images/s at batch 64 and 54 images/s at batch 1) on the final model; do not reuse old-model numbers for `GraviLensP`.
- [ ] Include preprocessing and source reconstruction in the timed path if the claim is end-to-end inference. If only model forward time is measured, say so.
- [ ] Compare against baselines under equivalent conditions.
- [ ] Specify which physics-informed baselines support any “~3× faster” statement.
- [ ] Use measured values—not derived visual estimates—in the final chart.

### Exit criteria

- [ ] Parameter count and throughput correspond to the exact checkpoint/model configuration reported.

---

# Phase 8 — Generate trustworthy artifacts and figures

The official notebook's Section 18 `DATA_SOURCE='auto'` can use live data, a cache, or `RECORDED_RUN`. The recorded fallback is an older two-class run, while the poster presents a three-class experiment. Do not let this fallback silently generate the final figures.

## Tasks

- [ ] Make the final export explicitly use results from the selected final experiment.
- [ ] Avoid using the stale two-class `RECORDED_RUN` for any current three-class figure.
- [ ] Run the poster-claim audit and inspect every `CHECK`/failed comparison.
- [ ] Confirm three classes and the intended class order.
- [ ] Confirm confusion matrices are 3×3.
- [ ] Check the manifest for data-source markers, model names, parameter counts, seeds, and output paths.
- [ ] Generate figures from live final results or a cache created from those exact results.
- [ ] Save table source data as CSV/JSON alongside vector/raster figures.
- [ ] Keep a run manifest so every poster number maps to a result artifact.

### Exit criteria

- [ ] No plot or metric comes from an unidentified/stale fallback.
- [ ] The complete poster can be rebuilt from documented source data.

---

# Phase 9 — Revise the poster only after results are locked

## 9.1 Title and headline

- [ ] Use the actual final architecture name; do not keep `ViT` if the evaluated encoder should specifically be named differently.
- [ ] Keep the title concise and use a short scientific takeaway under it.
- [ ] Include headline metrics only after the final benchmark validates them.

Provisional takeaway (replace metric/model wording as appropriate):
> **A differentiable SIS reconstruction lets the classifier condition its predictions on a physics-derived source representation.**

Use this claim only if the final model and ablation results establish the intended behavior. If the empirical advantage is modest or uncertain, state the demonstrated result more cautiously.

## 9.2 Architecture panel

- [ ] Draw the actual forward pass of the final selected `GraviLensP` configuration.
- [ ] For `use_dist=False`, omit distortion tokens from the final architecture diagram.
- [ ] For `use_dist=True`, show how the distortion representation is added to the lensed tokens, as implemented; do not invent a separate third transformer stack unless the implementation is changed.
- [ ] Show lensed tokens as cross-attention query and source tokens as key/value if that is the selected path.
- [ ] Distinguish trainable components from fixed physical operations accurately.
- [ ] Align the equation and labels across the embedded architecture PDF and the method text.
- [ ] Avoid the phrase “lensed-minus-source residual” unless feature subtraction actually occurs in the final code.

## 9.3 Results panel

Preferred visual hierarchy:
1. Large verified macro-F1 and AUC with unambiguous seed labels.
2. Parameter count.
3. Main comparison: macro-F1 vs parameter count or table.
4. A matched ablation result demonstrating the contribution of source reconstruction.
5. Physics diagnostics: θE error/source SSIM, if they strengthen the central story.
6. Throughput as supporting evidence.

- [ ] Consider a parameter-count vs macro-F1 scatter plot instead of the current narrow training-curves panel.
- [ ] Include an ablation summary if it directly tests the physics contribution.
- [ ] Do not fill space with a graph that doesn't support the main claim.
- [ ] Ensure all labels and axes are readable at A0 print size.

## 9.4 Dataset and method

- [ ] State the 64×64 image specification once.
- [ ] Keep representative images as actual model/dataset outputs; don't fabricate or generatively enhance scientific details.
- [ ] Show dataset split sizes compactly.
- [ ] Keep the SIS equation and distortion expression faithful to the final code.
- [ ] Reduce text repeated by the architecture diagram.
- [ ] Describe simulation assumptions and limitations clearly.

## 9.5 Conclusion and limitations

- [ ] Summarize only results of the selected final model version.
- [ ] Distinguish simulated-data performance from future real-survey evaluation.
- [ ] Keep limitations visible: simulated images, fixed SIS mass model if applicable, seed count, and the inverse-mapping approximation.
- [ ] Future work can show `fixed SIS → general mass model → Rubin/Euclid`, but make clear these are future steps, not completed evaluations.

## 9.6 Layout and print polish

- [ ] Rebalance the Results region rather than leaving unused space beside a small plot.
- [ ] Replace hard-coded large vertical spacers after content is locked.
- [ ] Fix footer/software-logo/reference alignment.
- [ ] Add a label to the QR code and point it to the exact final model release.
- [ ] Ensure consistent figure caption sizes, axes, legends, panel headings, border treatment, and margins.
- [ ] Preserve the burgundy/cream visual identity; no wholesale redesign needed.
- [ ] Inspect full-page, close-up, and actual print-scale views.

### Exit criteria

- [ ] The poster's claims, model diagram, equations, metrics, and QR destination all describe the same final experiment.

---

# Phase 10 — Final decision gates

Do not proceed to the next stage until the gate is passed.

| Gate | Must demonstrate | If it fails |
|---|---|---|
| **G1: Data/physics** | Data, labels, θE, source, and coordinates align; oracle inversion behaves as intended | Fix data/coordinate pipeline before training |
| **G2: Gradient** | Classification loss reaches θE predictor through differentiable source reconstruction | Fix the graph/parameterization; no long run |
| **G3: Functional source use** | Source interventions change logits on representative samples | Inspect cross-attention, masks, normalization, and pathway; no physics-use claim |
| **G4: Physical utility** | Predicted θE and reconstruction outperform defined baselines where claimed | Revise physics claim or method; don't overstate |
| **G5: Ablation** | Full model compared with matched no-inversion and no-supervision controls | Keep only claims supported by evidence |
| **G6: Reproducibility** | Final three-seed metrics and matching baselines are saved and traceable | Do not publish multi-seed mean/SD |
| **G7: Artifact integrity** | Figures are from final three-class runs; audit passes | Regenerate figures/data |
| **G8: Poster preflight** | Model, text, equations, numbers, and layout are consistent | Do not print |

---

# Suggested practical order of work

1. **Patch the trainer first:** remove per-epoch test evaluation; add explicit final evaluation.
2. **Fix and test `GraviLensP`:** simplify θE head, verify forward outputs and gradients.
3. **Run smoke tests:** finite tensors, nonzero classification-to-θE gradients, source counterfactual test, tiny-subset overfit.
4. **Validate physics:** oracle source inversion and θE error against baselines.
5. **Run development ablations:** `full`, `no_sis`, `no_sup`; add `dist` if needed.
6. **Freeze the recipe:** choose θE loss weight and checkpoint policy using validation only.
7. **Run the final three-seed comparison:** selected model and matched controls.
8. **Re-evaluate baselines if data or split changed.**
9. **Rebuild figures and tables:** bypass the stale two-class fallback and audit all poster claims.
10. **Update architecture and equations:** exactly match selected `forward()` path.
11. **Redesign Results:** use verified performance/parameter trade-off and essential ablations/physics diagnostics.
12. **Final typography/layout/QR review.**
13. **Generate and inspect the final A0 PDF.**

---

# What success looks like

A strong final outcome should let us make these claims separately and honestly:

- **The reconstruction is differentiable:** demonstrated by automated gradient tests.
- **The model uses the source representation:** demonstrated by source interventions and architecture tests.
- **The source reconstruction is physically meaningful:** supported by θE prediction and source metrics against clear reference baselines.
- **The physics branch improves classification:** supported by matched ablations and held-out, multi-seed results.
- **The results are reproducible:** every table/plot value maps to a known run, configuration, dataset split, and checkpoint.
- **The poster is scientifically coherent:** equations, architecture diagram, captions, metrics, and public code all describe the same evaluated model.

The objective is not merely to recover the old F1 score. It is to develop a model whose **physics pathway actually works, whose contribution is experimentally demonstrated, and whose poster communicates the evidence clearly**.
