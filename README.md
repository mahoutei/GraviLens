# GraviLens

Physics-informed lens modelling for dark-matter substructure classification.

**Status:** architecture v2, Phase 0. The API and the data format are not frozen
until Phase 4.

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
gate, the source counterfactual gate, and the parameter-budget match for the
three baselines.

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
gravilens/       package (config, physics, models; data/train/eval land next)
tests/           CPU gates
configs/         YAML run configurations
notebooks/       thin Colab driver
third_party/     vendored reference copy of DeepLenseSim (not imported)
docs/            roadmap, method, limitations
```

## Roadmap

See `docs/roadmap-v1.md` for the scientific plan and
`docs/architecture-v2.md` for the implementation plan.
