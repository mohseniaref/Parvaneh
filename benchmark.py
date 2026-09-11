#!/usr/bin/env python
import argparse, time
from accelerated_unwrap import available_backends, make_synthetic, rmse_aligned, unwrap

p = argparse.ArgumentParser(description="Benchmark accelerated 2-D weighted phase unwrapping")
p.add_argument("--rows", type=int, default=256); p.add_argument("--cols", type=int, default=320)
p.add_argument("--repeat", type=int, default=3); p.add_argument("--backends", nargs="*", default=["numpy","blas","numba","cython","cupy"])
a = p.parse_args(); truth, wrapped, weight = make_synthetic((a.rows, a.cols))
available = available_backends()
print(f"shape={wrapped.shape}  available={available}")
for backend in a.backends:
    if not available.get(backend, False): print(f"{backend:8s} SKIP (unavailable)"); continue
    unwrap(wrapped, weight, backend=backend, max_iter=2)  # warm-up/JIT
    times=[]
    for _ in range(a.repeat):
        start=time.perf_counter(); result, info=unwrap(wrapped, weight, backend=backend, return_info=True); times.append(time.perf_counter()-start)
    print(f"{backend:8s} {min(times):.4f}s  iterations={info.iterations:3d} residual={info.relative_residual:.2e} rmse={rmse_aligned(result,truth):.4f}")
