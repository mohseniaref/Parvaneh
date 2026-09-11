#!/usr/bin/env python3
"""Profile a warmed solver call and save the largest cumulative costs."""

import argparse
import cProfile
import io
import pstats
from pathlib import Path

from accelerated_unwrap import make_synthetic, unwrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="numpy",
                        choices=("numpy", "blas", "numba", "cython"))
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=512)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report", type=Path, default=Path("reports/profile_solver.txt"))
    args = parser.parse_args()
    _, phase, weight = make_synthetic((args.rows, args.cols), noise=0.02)
    unwrap(phase, weight, backend=args.backend, workers=args.workers)
    profiler = cProfile.Profile()
    profiler.enable()
    result, info = unwrap(phase, weight, backend=args.backend,
                          workers=args.workers, return_info=True)
    profiler.disable()
    stream = io.StringIO()
    stream.write("backend={} shape={} workers={} iterations={} residual={}\n".format(
        args.backend, phase.shape, args.workers, info.iterations, info.relative_residual))
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(args.limit)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(stream.getvalue())
    print(stream.getvalue())
    print("wrote {}".format(args.report))


if __name__ == "__main__":
    main()
