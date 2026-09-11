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
