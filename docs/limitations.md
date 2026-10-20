# Limitations

These are the limitations a reviewer should assume unless a result explicitly
says otherwise.

## Simulation, not survey data

- Every image is simulated. There is no Euclid, HST, or other real-survey
  image in the training or evaluation set, so the numbers describe the
  simulator, not the sky.
- The macro lens is an SIS with zero ellipticity plus a constant shear. Real
  deflectors have elliptical mass distributions, external convergence,
  line-of-sight structure, and lens-light that is not a simple two-Sérsic
  profile. A classifier can learn simulator artefacts; matched triplets remove
  the macro-lens confound but not every simulator-specific cue.
- The source pool is only 58 Galaxy10 cutouts, selected with
  `ans == 6, redshift < 0.02`. Sources therefore repeat across thousands of
  triplets. This is a source-diversity ceiling and can encourage source
  memorisation even though the source is shared across classes.

## Substructure priors

- The CDM and axion populations are pyHalo presets, not a measurement. The
  result depends on `sigma_sub=0.025`, `log_mlow=7`, `log_mhigh=10`, the 6.5
  arcsec cone, and the chosen ULDM mass `log10_m_uldm=-21.5`.
- `log_mlow=7` is a resolution floor, not a physical cutoff. Lower-mass haloes
  are omitted for cost; the class signal is therefore a particular,
  resolvable substructure population.
- The axion model uses `flucs=False`: the main-deflector fluctuation field is
  disabled. The class signal is the ULDM halo population only. This deviation
  from a full ULDM treatment is deliberate and disclosed.
- The axion preset's default `log10_sigma_sub` is disabled so the axion and
  CDM populations use the same `sigma_sub`; without that, the axion class would
  differ in overall substructure abundance as well as in halo physics.

## Matched triplets reduce the effective sample size

- 6,999 training images are 2,333 independent macro lenses, and 1,500 test
  images are 500 independent lenses. The three images in a triplet are not
  independent observations of the sky. Confidence intervals that treat every
  image as independent would be too narrow; the seed-level mean ± SD is the
  honest unit used here.
- One noise realisation is shared across the three classes in a triplet. This
  is the correct matching choice, but it also means the three images are
  correlated.

## Statistics and scope

- Three seeds are enough to expose gross seed sensitivity, not to support
  precise claims about small differences. The pre-registered decision rule is
  a sign test on the three-seed mean, not a significance test.
- The classification signal is subtle by construction: the macro lens, source,
  and noise are matched, so only substructure remains. A null result is a
  legitimate outcome and is not grounds for changing the model after seeing
  the test set.
- The parameter-matched baselines are a tighter comparison than the original
  notebook's 192-dim models, but they are not exhaustively tuned. They use the
  identical recipe and seeds.
- `no_sis` replaces the reconstructed source with the observed image and also
  drops the θE auxiliary loss, so `full - no_sis` changes two things at once;
  `no_sup` isolates the auxiliary loss on its own. The comparison is a
  forward-path ablation, not a proof that no physics information exists
  elsewhere in the network.

## Reproducibility

- CUDA non-determinism and library version drift can change the last digits of
  a run. The manifest records the git SHA, dirty flag, library versions, data
  fingerprint, seed, and parameter count; exact bitwise reproduction across
  hardware is not claimed.
- The full 21-run ablation has **not** been executed in this repository
  snapshot. Until it is, `python -m gravilens audit` exits non-zero and
  `figures` refuses to write. No result numbers are claimed in the README.
- Training does not automatically resume mid-run. `last.pt` is saved each
  epoch, but `ablate` resumes at run granularity by skipping runs that already
  have `metrics.json`.

## Intended use

This is a research artefact for testing whether differentiable SIS
reconstruction helps substructure classification under a matched macro lens.
It is not a dark-matter detector, a survey pipeline, or a calibrated
probability model.
