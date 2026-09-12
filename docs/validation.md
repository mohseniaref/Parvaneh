# Validation

Parvaneh makes quantitative claims — "this method recovers the truth on smooth
phase", "these backends agree", "this noise model is the physics of a
single-look interferogram" — and this page explains how those claims are
checked. The short version: every check compares against a **known
mathematical property** or a synthetic scene with an exact answer key. No test
depends on a stored output of the implementation itself, and no test needs an
external dataset, an external executable, or a network connection.

## Running the tests

From a clone, with nothing installed:

```bash
pytest -q
```

The root [`conftest.py`](../conftest.py) adds `src/` to `sys.path` when the
package is not installed, so a bare clone works. After
`python -m pip install -e '.[test,numba]'` the same command runs against the
installed copy.

```bash
pytest -q                                        # the whole suite
pytest -q tests/test_unwrap.py                   # one module
pytest -q -k "flynn"                             # one algorithm
pytest -q --doctest-modules src/parvaneh/synthetic   # docstring examples
```

Continuous integration runs `python -m pip install ".[test,numba]"` followed by
`python -m pytest -q` on Python 3.10 and 3.12
([`.github/workflows/tests.yml`](../.github/workflows/tests.yml)), so the suite
must pass on a clean machine with no Cython build step.

## What each test module covers

| Module | Subject | Nature of the checks |
|---|---|---|
| [`tests/test_unwrap.py`](../tests/test_unwrap.py) | the unwrapping algorithms and the raw-raster I/O | recovery of smooth synthetic phase, phase congruence, backend equivalence, offset invariance, deterministic iteration counts for Flynn, mask/weight plumbing, binary round-trips |
| [`tests/test_synthetic.py`](../tests/test_synthetic.py) | the InSAR scene generator | wrapping identity, exact reconstruction from the integer ambiguity, zero-deformation behaviour, line-of-sight projection, Mogi symmetry and decay, boundary-circulation identity, aliasing detection, seeded reproducibility |
| [`tests/test_savage.py`](../tests/test_savage.py) | the Savage–Burford interseismic model | far-field limits $\pm V/2$, vanishing velocity on the fault trace, antisymmetry, monotone growth away from the fault, strike dependence, velocity × interval = displacement |
| [`tests/test_noise.py`](../tests/test_noise.py) | the phase-noise model | the simulated histogram matches a closed-form single-look phase density bin by bin, plus independent moment checks |
| [`tests/test_coherence.py`](../tests/test_coherence.py) | coherence-map generators | range $[0,1]$, multiplicative combining rule, windowed shape behaviour, seeded reproducibility |
| [`tests/test_cli.py`](../tests/test_cli.py) | the `parvaneh` command line | exit status 1 and no traceback on every expected failure, stdout reserved for what was asked for, input and output guards, agreement between `--backend` and `--quality`, the JSON report, the offset convention, routing of raster inputs and outputs |
| [`tests/test_raster.py`](../tests/test_raster.py) | GDAL raster input and output | round-tripping a GeoTIFF preserves geotransform and coordinate reference, no-data becomes `NaN` on read and is written back on write, band and dtype selection, refused output drivers, and an actionable error when no GDAL binding is installed |

Three habits run through all of them:

- **Closed forms, not snapshots.** Where a test compares a number, it computes
  the expected value from an independently written formula — for example
  $v(x) = (V/\pi)\arctan(x/D)$ for the Savage model — never from a saved run of
  the code under test. A snapshot would only detect change, not error.
- **Properties, not pictures.** A test asserts that a map is antisymmetric or
  decays like $1/r$; it does not assert that a plot looks right.
- **Seeded randomness.** Every random input comes from a seeded generator, so a
  failure is always reproducible.

## Comparing a result with the truth

An unwrapped image is defined up to an additive constant, so a naive
`estimate - truth` is meaningless and the metric has to remove that one degree
of freedom. Parvaneh's metric is `rmse_aligned`, which subtracts the mean of
the difference before taking the root mean square:

```python
from parvaneh import make_synthetic, rmse_aligned, unwrap

truth, wrapped, quality = make_synthetic(shape=(256, 320), noise=0.25, seed=7)
estimate = unwrap(wrapped, quality, backend="numba", workers=-1)

print(f"{rmse_aligned(estimate, truth):.4f} rad")
```

`rmse_aligned` is therefore a fair comparison between methods even though their
raw outputs sit at different offsets — which is exactly why the command line
offers `--center circular` and reports which convention was used. See
[`algorithms.md`](algorithms.md) for the two unobservable quantities and
[`cli.md`](cli.md) for the runtime behaviour.

For visual and structural checks the suite additionally uses
`surface_difference`, `discontinuity_map`, `phase_residues` and `wrap_phase`,
all of which are exported from the package.

## Benchmarks

Benchmarks are separate from the tests and every one of them generates its
input in memory, so no dataset is required:

```bash
python benchmark.py --rows 256 --cols 320
python benchmarks/benchmark_backends.py --rows 256 --cols 256
python benchmarks/benchmark_path_following.py
python benchmarks/profile_solver.py
```

Every script warms the code up (JIT compilation for Numba, DCT plans for
SciPy) before timing, repeats the measurement (`--repeat`, `--warmup`) and
reports a summary such as the median or the minimum. The two `benchmarks/`
scripts also write a JSON report and `profile_solver.py` writes a `pstats`
summary; all three default to `reports/`, which is gitignored, and every script
takes `--report` to put them elsewhere. Measured numbers for this project are
collected in [`performance.md`](performance.md). Timings are
machine-dependent, so treat them as a way to compare backends *on your own
hardware*, not as a promise.

## Notebooks

The notebooks are validated the same way as any other code:

- they are stored **with the outputs of the last successful run**, so a reader
  sees exactly what the committed code produced and any change to a figure or a
  printed number shows up in the diff instead of rotting silently;
- each starts from a seeded generator, so a run is repeatable;
- they are executed end to end from the repository root
  (`jupyter nbconvert --to notebook --execute`), so a rename or a moved import
  cannot survive unnoticed.

If you change a notebook, run it end to end and commit the fresh outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

See [`notebooks.md`](notebooks.md) for what each notebook demonstrates.

## What is deliberately not covered

- **No third-party datasets.** Validation uses synthetic scenes with an exact
  answer key. Real interferograms have no ground truth and cannot be used as a
  correctness oracle.
- **No comparison against the historical C/MATLAB programs.** Those are not
  distributed here and are not executed. Mathematical provenance is cited, and
  the algorithms are validated against their published properties instead.
- **No GPU tests.** The CuPy backend is reported as available only when CuPy and
  a working CUDA device are both present, and CI has neither, so it is exercised
  only on machines that do.
- **Cython is optional.** The extension is not built in CI; the test suite
  treats `cython` as a backend that may legitimately be missing.
- **Timing assertions are avoided.** Runtime is benchmarked, not asserted, so a
  slow or busy machine cannot fail the build.

## Adding a test

1. Check a *property*, not the current output. If you need a number, derive it
   from an independently written formula.
2. Seed every random input.
3. If the new test pins iteration counts or a checksum-like quantity, add a
   comment saying which mathematical invariant makes that value reproducible.
4. Keep it runnable without Numba, Cython, CuPy, network access, or extra data.
