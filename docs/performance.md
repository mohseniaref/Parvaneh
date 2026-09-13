# Performance method and current measurements

Measurements below were made on 2026-08-31 on a 4-core / 8-thread x86-64 CPU
with a 256x256 weighted synthetic input. Input generation
was excluded, every backend received the same arrays and tolerance, one warm-up
call was discarded, and five subsequent in-process calls were timed. Full raw
samples and convergence fields are in `reports/backend_benchmark.json` locally.

| Backend | DCT workers | Median time |
|---|---:|---:|
| NumPy | all (`-1`) | 0.0781 s |
| BLAS dot products | all (`-1`) | 0.0845 s |
| Numba stencil | all (`-1`) | 0.0851 s |
| Cython stencil | all (`-1`) | 0.0879 s |

These are whole-solver results, not isolated kernel timings. With a single DCT
worker, the earlier three-run medians were 0.1159 s (NumPy), 0.2742 s (BLAS),
0.1017 s (Numba), and 0.1181 s (Cython). The ranking is machine- and size-
dependent; users should rerun the provided benchmark rather than quote these
numbers for other systems.

Profiling identified the repeated DCT/IDCT preconditioner as about 40% of the
warmed runtime and dot products as about 30%. The NumPy stencil was about 13%;
Numba reduced it to about 5%, but the whole solver gained much less. Enabling
multi-worker DCTs addressed the measured dominant cost and reduced the NumPy
median by roughly one third. Explicit BLAS and compiled stencil labels did not
make the complete algorithm faster in the multi-worker measurement.

The machine used above has no NVIDIA CUDA device, so CuPy is correctly reported
unavailable. An integrated GPU cannot be made CUDA-capable through a driver
change; a CPU path or a separately implemented non-CUDA GPU backend is required.

The quality-guided benchmark now creates its input with `make_synthetic` and
compares only the independently distributed Python and Numba implementations.
It records the seed, shape, noise level, warm-up policy, raw timing samples, and
aligned RMSE. Run it on the target machine rather than comparing results from
different hardware:

```bash
python benchmarks/benchmark_path_following.py --rows 257 --cols 257 --repeat 5
```

## Volumes and reliability sorting

`benchmarks/benchmark_nd.py` covers the two families that accept an
$n$-dimensional array. It builds a wrapped volume in memory, then unwraps it
twice: once as a whole and once slice by slice with the same solver. It reports
aligned RMSE, the fraction of samples a whole fringe off, the spread of
per-slice offsets, local error, and the measured wall-clock ratio between the
two runs. The reliability merge is timed in both its compiled and its Python
form. Needs no raster and no GPU:

```bash
python benchmarks/benchmark_nd.py --rows 64 --cols 64 --slices 12 --noise 0.9
```

What it shows, at the sizes in the third notebook:

- The joint solve costs about **twice** the slice-by-slice loop, not ten times.
  The work is the same with one more axis of neighbours: a 48x48x8 volume has
  52 224 candidate edges versus 36 096 for the eight independent slices, so the
  factor follows from the extra inter-slice pairs and from sorting one long list
  instead of eight short ones.
- The error improves out of all proportion to that cost once noise exists: at
  $\sigma = 0.9$ rad the joint reliability solve is roughly 3x more accurate
  than per-slice reliability sorting, and the per-slice offsets that plague the
  independent runs disappear (spread 2.7 rad down to 0.02 rad).
- The compiled reliability merge is roughly an order of magnitude faster than
  the Python reference loop on the same volume, and the two are bit-identical.

Absolute numbers here are small (milliseconds for a volume of this size) and
dominated by Numba's compile time on the first call, so the benchmark discards
one warm-up call before timing. Rerun it on your own hardware rather than
quoting these figures.

## Minimum-cost flow

`benchmarks/benchmark_mcf.py` sweeps the square sizes below. Every scene comes
from `make_synthetic` with noise 1.0 and seed 11, so it carries hundreds to
thousands of units of residue charge, and the reported time is the median of
warmed runs. The solver is the single-core Python reference implementation, so
these numbers contain no GPU and no compiled-kernel component.

| Size | Charge | Nodes | Arcs | Augmentations | Linear cost | Quadratic cost | Aligned RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32x32 | 134 | 962 | 1 984 | 67 | 0.040 s | 0.045 s | 1.243 rad |
| 48x48 | 267 | 2 210 | 4 512 | 134 | 0.157 s | 0.158 s | 1.247 rad |
| 64x64 | 440 | 3 970 | 8 064 | 220 | 0.422 s | 0.473 s | 1.190 rad |
| 96x96 | 975 | 9 026 | 18 240 | 489 | 1.890 s | -- | 1.307 rad |
| 128x128 | 1 674 | 16 130 | 32 512 | 838 | 5.358 s | -- | 1.346 rad |
| 256x256 | 6 585 | 65 026 | 130 560 | 3 295 | 74.076 s | -- | 1.289 rad |

The first three rows are medians of three runs, the next two medians of two, and
the last row is a single run, all on the same machine as above (Intel Core
i7-8650U, 4 cores / 8 threads at 1.90 GHz). Repeating the same cell on that
laptop moves the median by 10 to 20 percent, so read these as orders of
magnitude, not as thresholds.

Three things to read out of the table:

- **Charge sets the work, not pixels.** A clean scene has no residues, the
  network arrives already balanced, and the solver returns without performing a
  single augmentation. Otherwise each augmentation balances one positive unit of
  charge against one negative unit, so each augmentation retires two units and
  the augmentation column is half the charge column in every row above, where
  `charge` counts both signs of every dipole. The halving is a property of these
  scenes, not a law: the charges of a closed array sum to the net number of
  whole turns the measured gradient accumulates around the outer boundary, and
  the ground node absorbs that leftover, so a scene with 492 units of charge and
  2 units of ground imbalance needs 247 augmentations rather than 246. The
  `augmentations` field is the one that predicts the run time of the next
  bullet.
- **Time follows charge times nodes.** The products of those two columns are
  1.3e5, 5.9e5, 1.8e6, 8.8e6, 2.7e7 and 4.3e8, and the measured times are
  0.040, 0.157, 0.422, 1.890, 5.358 and 74.076 s. From row to row the two
  series agree to within about 15 percent, which is what the algorithm
  predicts: every augmentation runs one Dijkstra search over the node set, so
  the total cost is (number of augmentations) x (search cost per node).
- **Quadratic costs are not more expensive in time.** They use the same network
  and the same number of augmentations, and only the arithmetic that computes a
  marginal cost changes, so they came out 0 to 12 percent slower than the linear
  mode at these sizes.

Passing a weight leaves the network unchanged -- the same 3 970 nodes, 8 064
arcs and 220 augmentations at 64x64 -- but it makes the shortest-path search
itself slower, because the arc costs stop being uniform: with the build and the
solve timed separately on that scene, the solve grew from 0.33 s to 0.49 s,
about half again as long, while the build, including the edge-cost table, stayed
under a millisecond either way. The end-to-end medians in the next table
(0.331 s and 0.411 s) come from a separate session and show the same direction
with a smaller gap, which is the run-to-run spread described above.

The same 64x64 weighted scene shows where the flow solver sits among the other
families (median of three warmed runs each):

| Method | Median time | Aligned RMSE |
|---|---:|---:|
| Least squares | 0.001 s | 3.117 rad |
| Least squares + weights | 0.006 s | 2.971 rad |
| Reliability sorting | 0.023 s | 4.572 rad |
| Quality-guided | 0.107 s | 2.261 rad |
| Goldstein | 0.113 s | 3.745 rad |
| Minimum-$L^p$ | 0.258 s | 1.134 rad |
| Minimum-cost flow (linear) | 0.331 s | 1.329 rad |
| Minimum-cost flow (linear, weighted) | 0.411 s | 1.190 rad |
| Flynn | 0.414 s | 1.247 rad |
| Mask cuts | 0.460 s | 8.037 rad |

The RMSE column measures the scene, not the code, and it is here to keep the
timings honest: on this noisy scene the three methods that minimise an explicit
discrete or robust objective (minimum-$L^p$, weighted flow, Flynn) land within
about 10 percent of each other and closest to the truth, while the fast
least-squares solve is roughly two and a half times further away. Which of
those two trade-offs is right depends on the application, which is why both
families are offered. Section 17 of [`mathematics.md`](mathematics.md) repeats
the accuracy comparison on small scenes where every method can be run many times
and the individual whole-turn failures can be counted.

The benchmark also records `max_wrapped_departure_rad`, the largest departure of
the output from the period of the input. It sits at 1e-14 for every size above,
which is the benchmark's own check that unwrapping changes only whole turns.

The 256x256 row is the practical limit of this reference solver: about a minute
and a quarter for a quarter-megapixel scene whose residue charge is 6 585. The
scaling rule above says what that means for larger scenes -- a megapixel scene
with the same density of residues would need hours -- so the Python
implementation is aimed at small and medium scenes, at exact reference answers,
and at validating a faster or GPU solver. Nothing in the formulation is
Python-specific; the network and the objective do not change if the
augmenting-path search is replaced by a compiled or parallel one.

## Multigrid

`benchmarks/benchmark_multigrid.py` builds each scene in memory
(`make_synthetic`, noise 0.6 rad, seed 11), unwraps it with the multigrid solver
and with the transform-preconditioned conjugate-gradient solver on the same
weight field, and records each solver's own work counters next to its time. It
needs no raster and no GPU. The tolerance is 1e-10, each V-cycle smooths twice
before and twice after, the cycle cap is 200, and every cell is one timed run on
the same Intel Core i7-8650U as the sections above:

| Size | Weight field | MG cycles | MG time | CG iterations | CG time | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| 16x16 | uniform | 12 | 0.062 s | 1 | under 0.001 s | 129x |
| 16x16 | mask | 16 | 0.083 s | 16 | 0.002 s | 41x |
| 128x128 | uniform | 13 | 0.479 s | 1 | 0.005 s | 96x |
| 128x128 | mask | 79 | 1.924 s | 39 | 0.044 s | 44x |
| 256x256 | uniform | 13 | 0.864 s | 1 | 0.018 s | 47x |
| 256x256 | mask | 155 | 9.580 s | 56 | 0.675 s | 14x |
| 512x512 | uniform | 12 | 1.914 s | 1 | 0.051 s | 37x |
| 512x512 | mask | 200, stalled at 8.0e-09 | 31.933 s | 72 | 3.104 s | 10x |

Read the two columns together, because neither means much alone.

* **The cycle count is flat, and that is what the hierarchy buys.** The uniform
  field takes 12, 13, 13 and 12 cycles at 16x16, 128x128, 256x256 and 512x512:
  the same answer for a scene holding a thousand times more samples. Plain
  Gauss--Seidel on the finest grid alone, with the very same smoother and a
  500-sweep budget, is nowhere near the tolerance -- 1.8e-03 on the 128x128
  uniform scene and 3.9e-04 on the 512x512 one. The flat count is the coarse
  grids doing the long-wavelength work, not a particularly good smoother.

* **A cycle is not a sweep.** One V-cycle visits every level, so it is worth
  about 5.3 sweeps of the finest grid; the benchmark stores that number per cell
  as `fine_sweep_equivalents`. Thirteen cycles is therefore some 70 fine sweeps
  at any size, which is also the honest way to compare multigrid with plain
  relaxation: per cycle the hierarchy contracts by 0.17 to 0.23 on the uniform
  field, per sweep plain relaxation manages only 0.95 to 0.99, so it creeps
  rather than converges.

* **The transform solver is faster here, by 10 to 130 times.** With a uniform
  weight field conjugate gradient needs a single iteration, because the cosine
  transform then inverts the operator exactly, and no hierarchy can compete with
  that. On the other weight fields it needs 38 to 72 iterations, and it still
  wins everywhere in the table. The hierarchy's own case is the one where that
  transform is unavailable: on an irregular domain, under a mask that is not a
  rectangle, or in three dimensions, where one smoothed stencil and one smoother
  still suffice, while a transform of the same kind does not exist.

* **The mask is where the count grows.** It runs a cut across the grid, which
  leaves four islands that share no data and slows every transfer between the
  levels: 16, 79, 155 cycles as the grid grows from 16x16 to 256x256, and a
  stall just short of the tolerance at 512x512 after 200 cycles. Smoothing more
  per cycle buys most of that back -- 125, 79, 58 and 40 cycles at 128x128 for
  one, two, four and ten sweeps a side, in 3.068, 1.924, 2.232 and 3.231 s --
  which says the smoother, not the hierarchy, is doing the work there.

The two high-contrast fields are the failure case the module's docstring warns
about. Neither converges within 200 cycles, and the stalled answer is materially
worse, not merely less well converged:

| Size | Weight field | Cycles | Relative residual | MG RMSE | CG RMSE |
|---|---:|---:|---:|---:|---:|
| 128x128 | thin lines at 1e-2 | 200 | 2.2e-06 | 1.939 rad | 1.601 rad |
| 128x128 | 16x16 blocks at 2e-3 | 200 | 2.1e-07 | 3.580 rad | 1.619 rad |
| 256x256 | thin lines at 1e-2 | 200 | 1.5e-06 | 1.762 rad | 1.342 rad |
| 256x256 | 16x16 blocks at 2e-3 | 200 | 1.3e-07 | 3.162 rad | 1.135 rad |

The stalls are not a rounding detail. At 128x128 the stalled thin-line answer is
1.94 rad RMS from the truth against 1.60 for the converging solver, and the
blocky one 3.58 against 1.62; at 256x256 the pattern repeats, 1.76 against 1.34
and 3.16 against 1.14. A radian of RMS error is a different answer, not a less
converged one, and one sweep a side on the blocky field does not even stall: it
diverges, reaching 9.8e+43 after 200 cycles.
Coarsening the operator itself (Galerkin, $L_{\ell+1} = R L_\ell P$) instead of
re-deriving the coarse weights is the remedy multigrid theory prescribes
(Trottenberg et al. 2001, section 7); it costs the five-point stencil on the
coarse levels, and is left to later work.

On the four measured fields the multigrid family is thus the more predictable
solver, not the faster one: its work per cycle is bounded, its failures are
visible in the counters, and its cost grows with the number of cycles rather than
with the grid. Where such a transform applies, the solver described in
[`algorithms.md`](algorithms.md) is the recommendation, and where the weights
have high contrast it is the requirement.
The JSON reports behind these tables are reproduced by the commands in the
docstring of
[`benchmarks/benchmark_multigrid.py`](../benchmarks/benchmark_multigrid.py);
the derivation is in section 18 of [`mathematics.md`](mathematics.md).
