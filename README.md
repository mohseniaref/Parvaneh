# 🦋 Parvaneh

### Accelerated and reproducible phase unwrapping

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-263466.svg)](https://www.python.org/)
[![Release: alpha](https://img.shields.io/badge/release-0.1.0a1-D94F88.svg)](https://github.com/mohseniaref/Parvaneh/releases)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-D94F88.svg)](LICENSE)
[![Tests](https://github.com/mohseniaref/Parvaneh/actions/workflows/tests.yml/badge.svg)](https://github.com/mohseniaref/Parvaneh/actions/workflows/tests.yml)

**Parvaneh** (Persian: پروانه, *butterfly*) is a Python package for phase
unwrapping, written to be used for research, for teaching, and for everyday
processing work. It combines independently written implementations, accelerated
backends, synthetic validation scenes, reproducible benchmarks, and technical
documentation.

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
- A `parvaneh` command line with method selection and machine-readable reports
- Direct reading and writing of GDAL rasters — GeoTIFF, COG, ENVI — when a GDAL
  binding is installed
- NumPy/SciPy, explicit BLAS, Numba, optional Cython, and optional CuPy backends
- Quality-guided path following
- Goldstein branch cuts and residue detection
- Quality-guided mask cuts
- Flynn minimum-discontinuity unwrapping
- Robust minimum-$L^p$ reconstruction with IRLS
- Reliability sorting into a maximum-reliability tree
- Least-squares and reliability solvers accept any number of axes, so a stack or
  a volume can be unwrapped in one piece instead of slice by slice
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

GeoTIFF and other GDAL rasters need an optional binding, either
[`rasterio`](https://rasterio.readthedocs.io/) or
[`GDAL`](https://gdal.org/) itself:

```bash
python -m pip install -e '.[test,numba,geo]'
```

Everything else works without it; only raster input and output print an install
hint when neither binding is present.

## Quick example

```python
import matplotlib.pyplot as plt

from parvaneh import make_synthetic, rmse_aligned, unwrap

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

## Command line

The same algorithms are available without writing Python. `parvaneh` reads a
wrapped phase raster, unwraps it with the method you name, and writes the result
back out; `python -m parvaneh` is an equivalent spelling.

```bash
parvaneh --method list                                  # available methods
parvaneh unwrap wrapped.npy -o unwrapped.npy --method goldstein --info
parvaneh wrapped.npy -o unwrapped.npy --method ls \
    --weight quality.npy --backend numba --workers -1
```

If the package is not installed yet, the repository root holds a launcher that
does the same thing with `src/` already on the import path:

```bash
python unwrap.py unwrap wrapped.tif -o unwrapped.tif --method flynn
```

Input and output are `.npy` or `.npz` files, GeoTIFF and similar GDAL rasters, or
headerless raw rasters described by `--shape`, `--dtype`, `--order`, `--scale`,
and `--offset`. The command line also handles masks, weights, per-method tuning,
backend selection, and a machine-readable `--info` summary.

See [`docs/cli.md`](docs/cli.md) for the complete option reference,
[`docs/raster_io.md`](docs/raster_io.md) for the raster workflow, and
[`docs/algorithms.md`](docs/algorithms.md) for what each method does.

## Validation

```bash
pytest -q                                        # the full test suite
python benchmark.py --rows 256 --cols 320        # warmed backend timings
```

Every algorithm is checked against a mathematical property or a synthetic scene
with a known answer, and every benchmark generates its input in memory, so no
external dataset is required. See [`docs/validation.md`](docs/validation.md) for
the methodology, [`docs/performance.md`](docs/performance.md) for measured
timings, and [`docs/porting_status.md`](docs/porting_status.md) for the exact
status of each algorithm.

## Documentation

| Page | Contents |
|---|---|
| [`docs/algorithms.md`](docs/algorithms.md) | the algorithm families, tuning, and offset invariance |
| [`docs/cli.md`](docs/cli.md) | command-line manual: files, masks, backends, exit codes |
| [`docs/notebooks.md`](docs/notebooks.md) | the notebooks and how to run them |
| [`docs/validation.md`](docs/validation.md) | how correctness is tested and benchmarked |
| [`docs/performance.md`](docs/performance.md) | measured timings and scaling |
| [`docs/binary_formats.md`](docs/binary_formats.md) | raw raster conventions and I/O |
| [`docs/raster_io.md`](docs/raster_io.md) | reading and writing GeoTIFF and other GDAL rasters |
| [`docs/porting_status.md`](docs/porting_status.md) | per-algorithm porting and validation status |
| [`docs/phase_unwrapping_history_and_theory.md`](docs/phase_unwrapping_history_and_theory.md) | historical and theoretical background |
| [`docs/repository_scope.md`](docs/repository_scope.md) | what the repository contains and what the license covers |
| [`docs/release_checklist.md`](docs/release_checklist.md) | the release process |

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The
foundational reference is:

> D. C. Ghiglia and M. D. Pritt, *Two-Dimensional Phase Unwrapping: Theory,
> Algorithms, and Software*. New York: Wiley, 1998.

Release metadata for Zenodo are provided in [`.zenodo.json`](.zenodo.json).

## License

Parvaneh's independently written source and documentation are released under
the [BSD 3-Clause License](LICENSE). The license covers the files distributed as
part of this repository; it does not grant rights to separately obtained
third-party material. The boundary is documented in
[`docs/repository_scope.md`](docs/repository_scope.md).
