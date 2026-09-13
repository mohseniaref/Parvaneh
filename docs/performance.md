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
