# Python API

Parvaneh accepts a wrapped phase array whose values are normally in
$[-\\pi,\\pi)$. The public entry points are imported from `parvaneh`, so a
typical program starts with:

```python
from parvaneh import unwrap, reliability_unwrap, wrap_phase
```

## Core N-dimensional solvers

These functions work on any rank (1-D, 2-D, 3-D, or higher), provided every
axis has at least two samples. They couple neighboring samples along every
axis, so passing a 3-D volume performs a joint solve rather than silently
processing slices independently.

### `unwrap`

```python
unwrap(phase, weight=None, *, backend="numpy", max_iter=100, tol=1e-8,
       workers=1, return_info=False)
```

Solves weighted or unweighted least squares. `weight` is a nonnegative array
with the same shape as `phase`; use zero weights to exclude samples. `backend`
selects an available numerical implementation (`numpy`, `blas`, `numba`, or
`cython`, depending on the input and installation). The result is an array of
unwrapped phase. With `return_info=True`, it returns `(result, info)`.

The additive constant is not observable. Compare results with
`rmse_aligned` or subtract the value at a chosen reference pixel.

### `reliability_unwrap`

```python
reliability_unwrap(phase, mask=None, weight=None, *, backend="python",
                   return_info=False)
```

Ranks neighboring edges by local reliability and builds a spanning tree. It is
often a useful fast alternative when preserving sharp features matters. The
same `mask` and `weight` conventions apply. With `return_info=True`, it
returns `(result, ReliabilityInfo)`; the report includes accepted merges,
discarded loop-closing edges, and connected components.

### `multigrid_unwrap`

The multigrid least-squares solver has the same N-dimensional input contract
and supports `mask`, `weight`, iteration controls, and `return_info`. Consult
`help(parvaneh.multigrid_unwrap)` for the installed signature because optional
solver controls evolve independently of the core API.

## Two-dimensional algorithms

The following functions intentionally require a 2-D array: they operate on
$2\\times2$ cells, image paths, branch cuts, or the 2-D dual residue network.

| Function | Main idea |
|---|---|
| `quality_guided_unwrap` | grow from the most reliable pixel |
| `phase_residues` | measure non-zero wrapped circulation in each cell |
| `goldstein_unwrap` | place branch cuts between residues, then integrate |
| `mask_cut_unwrap` | turn low-quality regions into traversal barriers |
| `flynn_unwrap` | minimize discontinuities using region/path decisions |
| `unwrap_lp` | robust iteratively reweighted $L^p$ reconstruction, $1\\le p\\le2$ |
| `network_flow_unwrap` | pair residue charges with minimum-cost integer flow |
| `puma_unwrap` | solve a discrete phase-label energy with graph cuts |
| `stat_cost_unwrap` | statistical-cost residue optimization |
| `space_time_unwrap` | use temporal priors for a 2-D interferogram sequence |

For exact parameters and dataclass fields, use Python's built-in help, for
example `help(network_flow_unwrap)`, or inspect the corresponding module
docstring. Invalid higher-rank input is rejected rather than interpreted as a
stack, which prevents accidental changes in algorithm meaning. To use a 2-D
method on a stack, call it explicitly for each slice.

## Supporting utilities

- `wrap_phase(phase)`: wrap values into the package's principal interval.
- `rmse_aligned(estimate, truth)`: RMSE after removing the best constant
  offset.
- `pixel_reliability(phase, weight=None, mask=None)`: return the reliability
  map used by the sorting solver.
- `available_backends()`: list numerical backends available in the current
  installation.
- `read_raster` and `write_raster`: optional GDAL/rasterio-backed raster I/O.

## A small complete example

```python
import numpy as np
from parvaneh import make_synthetic, rmse_aligned, unwrap

truth, wrapped, quality = make_synthetic(shape=(64, 80), noise=0.2, seed=7)
estimate = unwrap(wrapped, weight=quality, max_iter=200)
print(rmse_aligned(estimate, truth))
```

For a volume, change `shape` to something like `(8, 64, 80)` and pass the
result directly to `unwrap`; the third axis is treated as another neighbor
axis in the objective.
