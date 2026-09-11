# 🦋 Parvaneh

### Accelerated and reproducible two-dimensional phase unwrapping

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-263466.svg)](https://www.python.org/)
[![Release: alpha](https://img.shields.io/badge/release-0.1.0a1-D94F88.svg)](https://github.com/mohseniaref/Parvaneh/releases)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-D94F88.svg)](LICENSE)
[![Tests](https://github.com/mohseniaref/Parvaneh/actions/workflows/tests.yml/badge.svg)](https://github.com/mohseniaref/Parvaneh/actions/workflows/tests.yml)

**Parvaneh** (Persian: پروانه, *butterfly*) is a research and teaching package
for two-dimensional phase unwrapping. It combines independently written Python
implementations, accelerated backends, synthetic validation scenes,
reproducible benchmarks, and technical documentation.

## From a wrapped caterpillar to an unwrapped butterfly

Imagine a caterpillar curled tightly inside a cocoon. From the outside, we can
see where each turn lies on the circle, but not how many complete turns were
made. Wrapped phase has the same ambiguity: every value is observed modulo
$2\pi$, so neighboring cycles overlap.

Phase unwrapping follows those turns and restores a continuous surface. In the
project's metaphor, the curled, worm-like caterpillar is the wrapped
observation; **Parvaneh** is the butterfly revealed when the folds are opened.
The transformation is not magic: residues, noise, masks, and discontinuities
can make the path uncertain, so every algorithm is paired with synthetic tests
and explicit quality measures.

```text
wrapped phase                         unwrapped phase
  ↻  ↻  ↻       continuity model        ╱╲
 (caterpillar)  ─────────────────▶      ╱  ╲   (butterfly)
  modulo 2π                         continuous surface
```

The project is inspired by the mathematical framework of Ghiglia and Pritt's
*Two-Dimensional Phase Unwrapping: Theory, Algorithms, and Software*. It does
not distribute the book, its original source code, or its supplied datasets.

## Highlights

- Weighted and unweighted least-squares unwrapping
- NumPy/SciPy, explicit BLAS, Numba, optional Cython, and optional CuPy backends
- Quality-guided path following
- Goldstein branch cuts and residue detection
- Quality-guided mask cuts
- Flynn minimum-discontinuity unwrapping
- Robust minimum-$L^p$ reconstruction with IRLS
- Synthetic InSAR-like tests and correctness metrics
- Warmed identical-input performance benchmarks

## Installation

```bash
git clone https://github.com/mohseniaref/Parvaneh.git
cd Parvaneh
python -m pip install -e '.[test,numba]'
```

CuPy is optional and must match the CUDA runtime installed on the machine. The
package reports CuPy as available only when CuPy and a working CUDA device are
both present.

## Quick example

```python
import matplotlib.pyplot as plt

from accelerated_unwrap import make_synthetic, rmse_aligned, unwrap

truth, wrapped, quality = make_synthetic(shape=(256, 320), noise=0.25, seed=7)
estimate = unwrap(
    wrapped,
    quality,
    backend="numba",
    workers=-1,
    max_iter=200,
    tol=1e-7,
)

print(f"aligned RMSE: {rmse_aligned(estimate, truth):.4f} rad")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
for axis, image, title in zip(
    axes,
    (truth, wrapped, estimate),
    ("continuous truth", "wrapped observation", "reconstruction"),
):
    artist = axis.imshow(image, cmap="twilight_shifted")
    axis.set_title(title)
    axis.axis("off")
    fig.colorbar(artist, ax=axis, shrink=0.75, label="phase (rad)")
plt.show()
```

An unwrapped result has an arbitrary additive constant. Quantitative
comparisons must use a reference pixel, stable region, or an offset-aligned
metric such as `rmse_aligned`.

## Algorithms

| Family | Public API | Status |
|---|---|---|
| Unweighted least squares | `unwrap(phase)` | Implemented and reference-tested |
| Weighted least squares | `unwrap(phase, weight)` | Implemented and backend-tested |
| Quality-guided path following | `quality_guided_unwrap` | Implemented and reference-tested |
| Residue detection | `phase_residues` | Implemented and unit-tested |
| Goldstein branch cuts | `goldstein_unwrap` | Implemented and reference-tested |
| Quality-guided mask cuts | `mask_cut_unwrap` | Implemented and reference-tested |
| Flynn minimum discontinuity | `flynn_unwrap` | Implemented and reference-tested |
| Minimum-$L^p$ norm | `unwrap_lp` | Implemented and synthetically tested |
| Multigrid families | — | Not yet ported |

Exact porting and validation status is recorded in
[`docs/porting_status.md`](docs/porting_status.md).

## Notebooks and documentation

The notebooks are stored without execution outputs and run without protected
input rasters, external executables, or embedded third-party figures.

- [`independent_synthetic_examples.ipynb`](notebooks/independent_synthetic_examples.ipynb)
  runs all public algorithm families on one deterministic synthetic experiment.
- [`chapter_01_introduction.ipynb`](notebooks/chapter_01_introduction.ipynb)
  introduces wrapping, integer ambiguity, Itoh's condition, and failure cases.
- [`chapter_02_line_integrals_residues.ipynb`](notebooks/chapter_02_line_integrals_residues.ipynb)
  develops path independence, discrete curl, and residues.
- [`phase_unwrapping_history_and_theory.md`](docs/phase_unwrapping_history_and_theory.md)
  provides a historical and theoretical overview.

The longer historical-data notebooks are intentionally kept outside the
independent public release. Their concepts are covered by the synthetic
notebook and publication-figure generator.

## Validation and benchmarks

```bash
pytest -q
python benchmark.py --rows 256 --cols 320
python benchmarks/benchmark_backends.py --rows 256 --cols 256
python benchmarks/benchmark_path_following.py
```

All benchmarks generate deterministic inputs in memory. See
[`docs/performance.md`](docs/performance.md) and
[`docs/binary_formats.md`](docs/binary_formats.md).

## Repository and copyright boundaries

This public repository contains only independently written software, original
documentation, synthetic examples, tests, and original figures. It excludes:

- the Ghiglia--Pritt book PDF;
- the book's original C or MATLAB programs;
- supplied phase, surface, correlation, and mask rasters;
- third-party ZIP archives and document files; and
- PDF exports, locally compiled binaries, generated extensions, reports, and
  build logs.

The BSD license applies only to files distributed as part of Parvaneh. It does
not grant rights to separately obtained third-party material.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The
foundational reference is:

> D. C. Ghiglia and M. D. Pritt, *Two-Dimensional Phase Unwrapping: Theory,
> Algorithms, and Software*. New York: Wiley, 1998.

Release metadata for Zenodo are provided in [`.zenodo.json`](.zenodo.json).

## License

Parvaneh's independently written source and documentation are released under
the [BSD 3-Clause License](LICENSE).
