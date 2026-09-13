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
