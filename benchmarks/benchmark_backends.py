#!/usr/bin/env python3
"""Warmed, identical-input benchmark for package compute backends."""

import argparse
import json
import statistics
import time
from pathlib import Path

from accelerated_unwrap import available_backends, make_synthetic, rmse_aligned, unwrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--workers", type=int, default=-1,
                        help="SciPy DCT workers; -1 uses all CPU cores")
    parser.add_argument("--report", type=Path, default=Path("reports/backend_benchmark.json"))
    args = parser.parse_args()
    truth, phase, weight = make_synthetic((args.rows, args.cols), noise=0.02)
    records = []
    for backend, enabled in available_backends().items():
        if not enabled:
            records.append({"backend": backend, "status": "unavailable"})
            continue
        for _ in range(args.warmup):
            unwrap(phase, weight, backend=backend, max_iter=100, tol=1e-8,
                   workers=args.workers)
        samples = []
        result = None
        for _ in range(args.repeat):
            start = time.perf_counter()
            result, info = unwrap(phase, weight, backend=backend, max_iter=100,
                                  tol=1e-8, workers=args.workers, return_info=True)
            samples.append(time.perf_counter() - start)
        records.append({
            "backend": backend, "status": "ok", "median_seconds": statistics.median(samples),
            "minimum_seconds": min(samples), "samples_seconds": samples,
            "iterations": info.iterations, "relative_residual": info.relative_residual,
            "truth_aligned_rmse_rad": rmse_aligned(result, truth),
        })
        print("{:<8} median {:.6f} s".format(backend, statistics.median(samples)))
    document = {"schema_version": 1, "shape_rows_cols": [args.rows, args.cols],
                "warmup_runs": args.warmup, "measured_runs": args.repeat,
                "dct_workers": args.workers,
                "timing_scope": "in-process unwrap call; input creation excluded", "results": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(document, indent=2) + "\n")
    print("wrote {}".format(args.report))


if __name__ == "__main__":
    main()
