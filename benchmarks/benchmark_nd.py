#!/usr/bin/env python3
"""Measure whether a third axis helps unwrapping, and what it costs.

The input is a volume whose truth is a ramp, a Gaussian hill and a slow drift
along the third axis, wrapped and then corrupted with wrapped Gaussian noise
of a chosen strength.  Six ways of using that volume are compared:

==================  ===============================================
``2d-ls``           least squares on every slice on its own
``3d-ls``           least squares on the whole volume at once
``2d-reliability``  reliability sorting on every slice on its own
``3d-reliability``  reliability sorting on the whole volume at once
``2d-skimage``      ``skimage.restoration.unwrap_phase`` per slice
``3d-skimage``      ``skimage.restoration.unwrap_phase`` on the volume
==================  ===============================================

The scikit-image rows appear only when that package is importable.  They are
given a fixed seed because that solver follows a random configuration, but be
warned that the seed does not make the compiled 2-D/3-D solver repeatable: its
own stream advances with every call, so its rows carry a jitter of order
1e-3 rad from run to run.  Comparisons of those rows are meaningful at the
scale of the differences reported here, not in their last digits.

Four numbers describe each answer.  ``aligned_rmse_rad`` is the distance from
the truth after one constant offset is removed from the whole volume, which is
the honest overall score.  ``wrong_fringe_fraction`` counts the pixels that
ended a whole fringe away, the mistake a user notices because a patch of the
surface jumps by 2*pi.  ``slice_offset_spread_rad`` keeps only the offsets
between slices, so it is large when the slices were never tied together.
``local_rmse_rad`` removes every slice's own offset first, so it measures the
shape inside the slices alone.

Run it with, for example::

    python benchmarks/benchmark_nd.py --noise 0 0.4 0.9 1.5
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from parvaneh import available_backends, reliability_unwrap, unwrap, wrap_phase


DEFAULT_NOISE = [0.0, 0.2, 0.4, 0.6, 0.9, 1.5]

# The scikit-image solver grows its answer from a random configuration, so the
# API asks for a seed and a fixed one is passed.  It does not, however, make the
# compiled solver repeatable: consecutive calls on identical input differ, so
# its rows are reproducible only to about three significant figures.
SKIMAGE_SEED = 20240101


def _unwrap_phase_named(seed=SKIMAGE_SEED):
    """Return ``unwrap_phase`` called with ``seed`` on every input."""
    from skimage.restoration import unwrap_phase

    def solve(phase):
        return unwrap_phase(phase, seed=seed)

    return solve


def build_volume(rows, cols, slices, sigma, seed, drift=0.15, height=10.0,
                 width=7.0):
    """Return the truth and its wrapped, noisy version.

    The truth rises by 0.12 radians per column and 0.05 radians per row, plus
    a Gaussian hill in the middle and a drift of ``drift`` radians per slice.
    The noise is Gaussian with standard deviation ``sigma``, and the sum is
    wrapped into ``(-pi, pi]``, exactly as a measured interferogram would be.
    """
    y, x = np.mgrid[0:rows, 0:cols].astype(float)
    hill = height * np.exp(-(((x - (cols - 1) / 2.0) / width) ** 2
                             + ((y - (rows - 1) / 2.0) / width) ** 2))
    truth = ((0.12 * x + 0.05 * y)[:, :, None] + hill[:, :, None]
             + drift * np.arange(slices)[None, None, :])
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=sigma, size=truth.shape) if sigma > 0 else 0.0
    return truth, wrap_phase(truth + noise)


def per_slice(volume, solver):
    """Unwrap every slice on its own and stack the answers."""
    return np.stack([solver(volume[:, :, k])
                     for k in range(volume.shape[2])], axis=2)


def metrics(estimate, truth):
    """Compare a volume estimate with the truth slice by slice."""
    error = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    error = error - error.mean()
    offsets = error.mean(axis=(0, 1))
    local = error - offsets
    return {
        "aligned_rmse_rad": float(np.sqrt(np.mean(error ** 2))),
        "wrong_fringe_fraction": float((np.abs(error) > np.pi).mean()),
        "slice_offset_spread_rad": float(offsets.std()),
        "local_rmse_rad": float(np.sqrt(np.mean(local ** 2))),
    }


def strategies(reliability_backend):
    """Return the ``(name, function)`` pairs this benchmark compares."""
    result = [
        ("2d-ls", lambda volume: per_slice(volume, unwrap)),
        ("3d-ls", lambda volume: unwrap(volume)),
        ("2d-reliability",
         lambda volume: per_slice(volume, reliability_unwrap)),
        ("3d-reliability",
         lambda volume: reliability_unwrap(volume,
                                           backend=reliability_backend)),
    ]
    try:
        solve = _unwrap_phase_named()
    except ImportError:
        return result
    result.append(("2d-skimage", lambda volume: per_slice(volume, solve)))
    result.append(("3d-skimage", lambda volume: solve(volume)))
    return result


def timed(function, repeat):
    """Run ``function`` ``repeat`` times and keep the median wall time."""
    samples = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return result, statistics.median(samples)


def improvements(records):
    """Compare each joint three-dimensional row with its per-slice partner."""
    summary = []
    for sigma in sorted({record["noise"] for record in records}):
        block = {record["method"]: record for record in records
                 if record["noise"] == sigma}
        for family in ("ls", "reliability", "skimage"):
            pair = ("2d-%s" % family, "3d-%s" % family)
            if pair[0] not in block or pair[1] not in block:
                continue
            flat, joint = block[pair[0]], block[pair[1]]
            reference = flat["aligned_rmse_rad"]
            summary.append({
                "noise": sigma,
                "family": family,
                "aligned_rmse_2d_rad": reference,
                "aligned_rmse_3d_rad": joint["aligned_rmse_rad"],
                "rmse_reduction": ((reference - joint["aligned_rmse_rad"])
                                   / reference if reference > 0 else 0.0),
                "offset_spread_2d_rad": flat["slice_offset_spread_rad"],
                "offset_spread_3d_rad": joint["slice_offset_spread_rad"],
                "local_rmse_2d_rad": flat["local_rmse_rad"],
                "local_rmse_3d_rad": joint["local_rmse_rad"],
            })
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--slices", type=int, default=12)
    parser.add_argument("--noise", type=float, nargs="+",
                        default=DEFAULT_NOISE,
                        help="noise standard deviation; pass several values")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--reliability-backend", default="auto",
                        choices=("auto", "python", "numba", "cython"),
                        help="implementation used by the reliability rows")
    parser.add_argument("--report", type=Path,
                        default=Path("reports/nd_benchmark.json"))
    args = parser.parse_args()
    if args.rows < 2 or args.cols < 2 or args.slices < 2:
        parser.error("rows, cols and slices must all be at least 2")
    if args.repeat < 1:
        parser.error("repeat must be at least 1")
    if any(sigma < 0 for sigma in args.noise):
        parser.error("noise must be nonnegative")

    backend = args.reliability_backend
    if backend == "auto":
        backend = "numba" if available_backends().get("numba") else "python"
    print("reliability rows use the %s engine" % backend)

    rows = [("noise", "method", "rmse", "wrong", "offsets", "local",
             "seconds")]
    records = []
    for sigma in args.noise:
        truth, volume = build_volume(args.rows, args.cols, args.slices,
                                     sigma, args.seed)
        for name, function in strategies(backend):
            function(volume)  # warm up the compiled paths
            estimate, seconds = timed(lambda f=function: f(volume),
                                     args.repeat)
            record = {"noise": sigma, "method": name, "seconds": seconds}
            record.update(metrics(estimate, truth))
            records.append(record)
            rows.append((
                "%.2f" % sigma, name,
                "%.3f" % record["aligned_rmse_rad"],
                "%.3f" % record["wrong_fringe_fraction"],
                "%.3f" % record["slice_offset_spread_rad"],
                "%.3f" % record["local_rmse_rad"],
                "%.3f" % record["seconds"]))

    widths = [max(len(row[column]) for row in rows)
              for column in range(len(rows[0]))]
    for index, row in enumerate(rows):
        print("  ".join(value.ljust(widths[column])
                        for column, value in enumerate(row)))
        if index == 0:
            print("  ".join("-" * width for width in widths))

    summary = improvements(records)
    for entry in summary:
        print("%.2f  %-12s RMSE %.3f -> %.3f rad (%+.0f%%), offsets "
              "%.3f -> %.3f rad" % (
                  entry["noise"], entry["family"],
                  entry["aligned_rmse_2d_rad"], entry["aligned_rmse_3d_rad"],
                  100.0 * entry["rmse_reduction"],
                  entry["offset_spread_2d_rad"],
                  entry["offset_spread_3d_rad"]))

    document = {
        "schema_version": 1,
        "input": "ramp, Gaussian hill, drift and wrapped Gaussian noise",
        "shape_rows_cols_slices": [args.rows, args.cols, args.slices],
        "noise": args.noise,
        "seed": args.seed,
        "reliability_backend": backend,
        "warmup_runs": 1,
        "measured_runs": args.repeat,
        "timing_scope": "in-process compute, median of the measured runs",
        "results": records,
        "summary": summary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(document, indent=2) + "\n")
    print("wrote {}".format(args.report))


if __name__ == "__main__":
    main()
