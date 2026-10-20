# Third-party notices

## DeepLenseSim

`third_party/DeepLenseSim/` is a vendored reference copy of
[ML4SCI/DeepLenseSim](https://github.com/ML4SCI/DeepLenseSim) (MIT, Copyright
(c) 2022 Michael Toomey). It predates this repository's git history, so the
upstream commit is not recorded here.

It is **not imported** by `gravilens`. It is kept only as the reference
implementation for the axion simulator ported into `gravilens/simulate.py`.
The runtime dependencies are pinned in `pyproject.toml`; the reference copy
exists so the port can be diffed against its source.

License: `third_party/DeepLenseSim/LICENSE`.
