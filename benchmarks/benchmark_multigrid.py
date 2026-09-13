#!/usr/bin/env python3
"""Benchmark the multigrid family against the conjugate-gradient solver.

Both methods minimise the same weighted least-squares objective, so they can be
compared on the same scene and the same weight field.  What differs is how the
answer is reached: :func:`parvaneh.multigrid_unwrap` runs V-cycles over a
hierarchy of grids, while :func:`parvaneh.core.unwrap` runs a conjugate-gradient
iteration preconditioned by a fast cosine transform.

The sweep therefore reports, for every size, both solvers on the same weights,
plus the multigrid counters (levels, V-cycles, sweeps) that explain its time.
Four weight fields are offered because they behave very differently:
``uniform`` (no zero or tiny weights), ``mask`` (a validity mask that cuts the
image into weakly connected islands), ``thin`` (thin lines of near-zero weight)
and ``blocky`` (square blocks of low weight).  The last two are the cases the
hierarchy struggles with, and the ones the caveat in
:mod:`parvaneh.multigrid` is about.

Two baselines come with every cell, because a cycle count means nothing on its
own.  The first is plain Gauss--Seidel relaxation of the finest grid with no
hierarchy at all; that is the method multigrid is supposed to beat, and the
contraction rate of the two is what makes the comparison meaningful.  A cycle
costs several fine sweeps, so the record also stores ``fine_sweep_equivalents``,
the number of finest-grid sweeps one V-cycle is worth; without it a rate per
cycle cannot be compared with a rate per sweep.  The second baseline is
:func:`parvaneh.core.unwrap`, the transform-preconditioned conjugate-gradient
solver the package recommends when the weights have high contrast.
``--smoothing`` repeats the whole sweep at several values of
``pre_smooth = post_smooth``, because the number of cycles depends on how much
smoothing each cycle does.

One property of the ``mask`` field needs care, and it turned out to matter in a
second way as well.  Its cut runs right across the grid, so the weight map falls
into four islands that share no data at all; a least-squares fit assigns each
island a constant, and nothing in the objective constrains those four constants.
Adding any amount to a whole island changes the residual by nothing, so two
solvers can return different representatives of the same solution set, and a
plain root-mean-square distance between them or to the truth measures that
freedom rather than accuracy.  Every distance reported here therefore removes the
mean difference island by island first, and the record says how many islands the
weight map had.  With a single island this is the ordinary constant-removed
distance.

The cells *inside* the cut are the same kind of freedom: their weight is zero,
so the objective says nothing about them either, and their value is whatever the
solver interpolated.  On the ``mask`` field they dominate a whole-grid distance
(``aligned_rmse_rad``, ``solver_difference_rad``) while the cells that carry data
agree to the convergence tolerance.  Both are therefore reported: the ``known_*``
columns average over the cells of nonzero weight only, and those are the numbers
to quote when the question is accuracy rather than appearance.

Every scene is generated deterministically in memory; nothing is read from disk.

The numbers quoted in the package documentation come from these four commands,
which between them give every cell exactly one source.

    python benchmarks/benchmark_multigrid.py --sizes 128 256 \
        --fields uniform mask thin blocky --smoothing 2 \
        --report reports/mg_fields.json
    python benchmarks/benchmark_multigrid.py --sizes 128 256 \
        --fields uniform mask --smoothing 1 4 10 \
        --report reports/mg_smoothing.json
    python benchmarks/benchmark_multigrid.py --sizes 16 512 \
        --fields uniform mask --smoothing 2 --report reports/mg_scales.json
    python benchmarks/benchmark_multigrid.py --sizes 128 \
        --fields thin blocky --smoothing 1 \
        --report reports/mg_contrast_s1.json
    python benchmarks/benchmark_multigrid.py --modes 128 \
        --report reports/mg_modes.json

The fourth command is the only source for the claim that one smoothing sweep a
side is not enough: the ``blocky`` field diverges there, and a diverging run has
no distance worth quoting, so it is measured once, on its own, rather than
repeated in the other three sweeps.

``--modes`` is not a solver comparison at all, so it replaces the sweep instead
of extending it.  It applies the hierarchy's own smoother to the Fourier modes
of the uniform-weight Laplacian with a zero right-hand side, and the number it
reports is therefore the amplification factor of one sweep on that mode.  Those
factors are the reason the levels exist: the roughest mode is almost annihilated
by a single sweep while the smoothest one is barely touched, so a single grid
spends its whole budget on an error the coarser grids could remove cheaply.

They were run on an idle Intel Core i7-8650U (4 cores / 8 threads at 1.90 GHz),
the machine quoted in ``docs/performance.md``.  The scenes are deterministic, so
a reader reproduces every cycle count exactly and every time within machine
noise.
"""

import argparse
import json
import math
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage

from parvaneh import make_synthetic
from parvaneh.core import unwrap
from parvaneh.multigrid import _build, _residual, _smooth, multigrid_unwrap
from parvaneh.synthetic import wrap_phase

#: Weight fields the sweep can build, described in the module docstring.
FIELDS = ("uniform", "mask", "thin", "blocky")

#: Weight given to the thin weak lines of the ``thin`` field.
THIN_WEIGHT = 1e-2

#: Weight given to the low-weight blocks of the ``blocky`` field.
BLOCK_WEIGHT = 2e-3

#: Side of one block of the ``blocky`` field, in samples.
BLOCK_SIDE = 16

#: Coarse-grid sweep budget of :func:`parvaneh.multigrid_unwrap` by default.
COARSE_SWEEPS = 50

#: Sweeps :func:`sweep_mode_decay` applies to one mode before reading it back.
MODE_SWEEPS = 1

#: Smallest grid :func:`measure_modes` accepts, below which it says nothing.
MIN_MODE_SIZE = 8


def weight_field(phase, name):
    """Return the weight map ``name`` describes, on ``phase``'s grid."""
    if name == "uniform":
        return np.ones(phase.shape)
    if name == "mask":
        mask = np.ones(phase.shape, dtype=bool)
        middle = phase.shape[0] // 2
        mask[:, middle:middle + 2] = False
        mask[middle, :] = False
        return mask.astype(np.float64)
    if name == "thin":
        weight = np.ones(phase.shape)
        weight[::32, :] = THIN_WEIGHT
        weight[:, ::32] = THIN_WEIGHT
        return weight
    if name == "blocky":
        weight = np.ones(phase.shape)
        blocks = np.indices(phase.shape) // BLOCK_SIDE
        weight[np.sum(blocks, axis=0) % 2 == 0] = BLOCK_WEIGHT
        return weight
    raise ValueError("unknown field %r" % name)


def timed(function, repeat):
    """Run ``function`` ``repeat`` times and return its last result, samples."""
    samples = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return result, samples


def finite_or_none(value):
    """Return ``value`` as a float, or ``None`` when it is not finite.

    A run that diverges reaches values that no JSON reader should be asked to
    parse, and a missing number is a more honest record than ``Infinity``.
    """
    value = float(value)
    return value if np.isfinite(value) else None


def rad_text(value):
    """Format a distance in radians that may be missing, for the log line."""
    return "       n/a" if value is None else "{0:>10.2e}".format(value)


def contraction_rate(residuals):
    """Average factor by which the residual falls per iteration.

    The geometric mean of the successive ratios is the right summary when the
    ratios themselves settle, as they do for a stationary iteration; a rate of
    ``1.0`` means the iteration is not moving, and a rate above ``1.0`` means it
    is diverging.  One iteration is one V-cycle for multigrid and one sweep for
    the hierarchy-free baseline, which is why the two rates are only comparable
    after ``fine_sweep_equivalents`` has converted a cycle into sweeps.
    """
    finite = [value for value in residuals if np.isfinite(value)]
    if len(finite) < 2 or finite[0] <= 0.0:
        return None
    return finite[-1] / finite[0] if len(finite) == 2 else float(
        (finite[-1] / finite[0]) ** (1.0 / (len(finite) - 1)))


def component_labels(weight):
    """Label the islands of nonzero weight, and count them.

    Cells that share a face are in the same island.  Each island is one free
    constant in the fit, so a weight map that falls apart into several islands
    describes a least-squares problem whose solution is a set, not a point.
    """
    known = np.asarray(weight) > 0.0
    structure = ndimage.generate_binary_structure(known.ndim, 1)
    labels, count = ndimage.label(known, structure=structure)
    return labels, int(count)


def component_aligned_rmse(estimate, truth, labels, known=None):
    """Distance between two fields with every island's offset removed.

    See the module docstring: the constants attached to disconnected islands are
    not part of the fit, so subtracting the mean difference island by island
    leaves only the part the objective constrains.  For a connected weight map
    this is exactly :func:`parvaneh.rmse_aligned`.

    With ``known`` set to the cells of nonzero weight the average is taken over
    those cells alone, which drops the zero-weight cells: nothing in the data
    fixes them, so their value is whatever the solver interpolated and a
    distance that includes them measures that choice rather than accuracy.
    """
    delta = np.asarray(estimate, dtype=np.float64) - np.asarray(truth)
    for index in range(1, int(labels.max()) + 1):
        inside = labels == index
        if np.any(inside):
            delta[inside] -= delta[inside].mean()
    if known is not None:
        delta = delta[known]
    return float(np.sqrt(np.mean(delta * delta)))


def fine_sweep_equivalents(levels, pre_smooth, post_smooth, coarse_sweeps,
                           ndim):
    """Work of one V-cycle, counted in sweeps of the finest grid.

    A sweep on level ``l`` visits ``2 ** (-l * ndim)`` as many points as a sweep
    on the finest grid, so the sweeps a cycle spends on the coarse grids can be
    added up in units of one finest-grid sweep.  This is the conversion that
    keeps a cycle count from being read as if a cycle were a sweep.
    """
    per_level = float(2 ** ndim)
    cost = sum((pre_smooth + post_smooth) * per_level ** -level
               for level in range(levels - 1))
    return float(cost + coarse_sweeps * per_level ** -(levels - 1))


def plain_relaxation(phase, weight, tol, max_sweeps, sweeps_per_pass):
    """Relax the finest grid alone, the baseline a V-cycle has to beat.

    This is the same red-black Gauss--Seidel smoother the hierarchy uses, run on
    the finest grid with no coarser grid to correct it.  It shares the operator,
    so the comparison isolates what the hierarchy contributes.
    """
    level = _build(phase, np.asarray(weight, dtype=np.float64), levels=1)[0]
    norm = float(np.linalg.norm(level.rhs))
    residuals = []
    sweeps = 0
    while sweeps < max_sweeps:
        _smooth(level, sweeps_per_pass)
        sweeps += sweeps_per_pass
        residuals.append(float(np.linalg.norm(_residual(level))) / norm
                         if norm > 0.0 else 0.0)
        if residuals[-1] <= tol:
            break
    return {
        "sweeps": sweeps,
        "sweeps_per_pass": sweeps_per_pass,
        "converged": bool(residuals[-1] <= tol),
        "relative_residual": finite_or_none(residuals[-1]),
        "rate_per_sweep": contraction_rate(residuals),
    }


def mode_indices(size):
    """Wavenumbers :func:`measure_modes` probes, from smoothest to roughest.

    Powers of two up to half the grid, then the last wavenumber, which is the
    closest a grid of ``size`` samples can come to alternating sample by sample.
    """
    indices = [1]
    while 2 * indices[-1] <= size // 2:
        indices.append(2 * indices[-1])
    return indices + [size - 1]


def sweep_mode_decay(size, index, sweeps=MODE_SWEEPS):
    """Amplification of one Fourier mode by plain sweeps of the finest grid.

    The mode is the one-dimensional cosine of section 6 of
    ``docs/mathematics.md``, held constant along the other axis, so it is an
    eigenvector of the uniform-weight Laplacian.  The sweep has a zero
    right-hand side, so the operator being applied is the one the error
    obeys, and reading the mode's coefficient before and after gives the factor
    by which one smoother pass multiplies it.  A factor near one means the error
    survives; near zero means it dies.
    """
    theta = math.pi * index / size
    cosines = np.cos(theta * (np.arange(size) + 0.5)).reshape(size, 1)
    mode = cosines * np.ones((1, size))
    level = _build(np.zeros((size, size)), np.ones((size, size)), 1)[0]
    level.rhs[...] = 0.0
    level.phi[...] = mode
    squared = float((mode * mode).sum())
    before = float((level.phi * mode).sum()) / squared
    _smooth(level, sweeps)
    after = float((level.phi * mode).sum()) / squared
    return {
        "index": index,
        "theta_over_pi": theta / math.pi,
        "decay": finite_or_none(after / before),
    }


def measure_modes(size, sweeps=MODE_SWEEPS):
    """Decay of every mode of :func:`mode_indices` on one grid size."""
    return {
        "size": size,
        "sweeps": sweeps,
        "indices": mode_indices(size),
        "decay": [sweep_mode_decay(size, index, sweeps)
                  for index in mode_indices(size)],
    }


def measure(size, field, noise, seed, repeat, max_cycles, tol, smoothing,
            plain_max_sweeps):
    """Time both solvers on one scene and collect their own work counters."""
    truth, phase, _ = make_synthetic((size, size), noise=noise, seed=seed)
    weight = weight_field(phase, field)
    labels, islands = component_labels(weight)
    known = np.asarray(weight) > 0.0

    mg_result, mg_samples = timed(
        lambda: multigrid_unwrap(phase, weight, max_cycles=max_cycles, tol=tol,
                                 pre_smooth=smoothing, post_smooth=smoothing,
                                 return_info=True), repeat)
    cg_result, cg_samples = timed(
        lambda: unwrap(phase, weight, max_iter=1000, tol=tol,
                       return_info=True), repeat)
    mg_phase, mg_info = mg_result
    cg_phase, cg_info = cg_result
    # Unwrapping only adds whole turns, so each output must still live in the
    # same period as its input.  A nonzero value here would be a real bug.
    return {
        "shape_rows_cols": [size, size],
        "field": field,
        "pre_smooth": smoothing,
        "post_smooth": smoothing,
        # Islands of nonzero weight: each one is a constant the fit cannot see.
        "weight_islands": islands,
        "multigrid": {
            "levels": mg_info.levels,
            "coarsest": list(mg_info.coarsest),
            "cycles": mg_info.cycles,
            "sweeps": mg_info.sweeps,
            "converged": bool(mg_info.converged),
            "relative_residual": finite_or_none(mg_info.relative_residual),
            "rate_per_cycle": contraction_rate(mg_info.residuals),
            "fine_sweep_equivalents_per_cycle": fine_sweep_equivalents(
                mg_info.levels, smoothing, smoothing, COARSE_SWEEPS,
                np.ndim(phase)),
            "fine_sweep_equivalents": fine_sweep_equivalents(
                mg_info.levels, smoothing, smoothing, COARSE_SWEEPS,
                np.ndim(phase)) * mg_info.cycles,
            "median_seconds": statistics.median(mg_samples),
            "samples_seconds": mg_samples,
            "aligned_rmse_rad": component_aligned_rmse(mg_phase, truth,
                                                       labels),
            "known_rmse_rad": component_aligned_rmse(mg_phase, truth, labels,
                                                     known),
            "max_wrapped_departure_rad": float(
                np.abs(wrap_phase(mg_phase) - phase).max())
            if np.all(np.isfinite(mg_phase)) else None,
        },
        "plain_relaxation": plain_relaxation(phase, weight, tol,
                                             plain_max_sweeps, smoothing),
        "conjugate_gradient": {
            "iterations": cg_info.iterations,
            "converged": bool(cg_info.converged),
            "median_seconds": statistics.median(cg_samples),
            "samples_seconds": cg_samples,
            "aligned_rmse_rad": component_aligned_rmse(cg_phase, truth,
                                                       labels),
            "known_rmse_rad": component_aligned_rmse(cg_phase, truth, labels,
                                                     known),
        },
        # The two methods solve the same normal equations, so they have to
        # agree to their convergence tolerance once the free island constants
        # are removed; this column is the check.  The pair is whole-grid and
        # nonzero-weight-only, which differ only on the zero-weight cells.
        "solver_difference_rad": component_aligned_rmse(mg_phase, cg_phase,
                                                        labels)
        if np.all(np.isfinite(mg_phase)) else None,
        "solver_difference_known_rad": component_aligned_rmse(
            mg_phase, cg_phase, labels, known)
        if np.all(np.isfinite(mg_phase)) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256],
                        help="square scene sizes to sweep (default: 128 256)")
    parser.add_argument("--fields", nargs="+", choices=FIELDS, default=FIELDS,
                        help="weight fields to sweep (default: all of %s)"
                             % " ".join(FIELDS))
    parser.add_argument("--noise", type=float, default=0.6,
                        help="generator noise scale (default: 0.6)")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-cycles", type=int, default=200,
                        help="V-cycle budget per run (default: 200); a run "
                             "that hits it is reported as unconverged")
    parser.add_argument("--tol", type=float, default=1e-10,
                        help="relative residual target for both solvers "
                             "(default: 1e-10)")
    parser.add_argument("--smoothing", nargs="+", type=int, default=[2],
                        help="values of pre_smooth = post_smooth to repeat the "
                             "sweep at (default: 2); the cycle count depends "
                             "on how much smoothing each cycle does")
    parser.add_argument("--plain-max-sweeps", type=int, default=500,
                        help="sweep budget for the hierarchy-free baseline "
                             "(default: 500); on uniform weights a single grid "
                             "needs hundreds of sweeps to reach the tolerance")
    parser.add_argument("--repeat", type=int, default=1,
                        help="timed runs per cell (default: 1); raise this on "
                             "a quiet machine, since a thin-line run is slow")
    parser.add_argument("--modes", type=int, default=None, metavar="SIZE",
                        help="instead of the solver sweep, measure the decay of "
                             "single Fourier modes under sweeps of the smoother "
                             "on a SIZE x SIZE uniform grid")
    parser.add_argument("--report", type=Path,
                        default=Path("reports/multigrid_benchmark.json"))
    args = parser.parse_args()
    if (any(size < 2 for size in args.sizes) or args.repeat < 1
            or args.noise < 0 or args.max_cycles < 1 or args.tol <= 0
            or any(value < 1 for value in args.smoothing)
            or args.plain_max_sweeps < 1):
        parser.error("sizes must be >= 2, repeat >= 1, max-cycles >= 1, tol > 0 "
                     "and noise, smoothing and plain-max-sweeps positive")
    if args.modes is not None and args.modes < MIN_MODE_SIZE:
        parser.error("modes must be >= %d" % MIN_MODE_SIZE)

    if args.modes is not None:
        modes = measure_modes(args.modes)
        print("mode decay on a {0} x {0} uniform grid, {1} sweep{2}".format(
            modes["size"], modes["sweeps"],
            "" if modes["sweeps"] == 1 else "s"))
        for entry in modes["decay"]:
            print("  k {0:>5}  theta/pi {1:>7.5f}  decay {2:>10.5f}".format(
                entry["index"], entry["theta_over_pi"], entry["decay"]))
        document = {
            "schema_version": 4,
            "input": "in-memory uniform-weight grid, no scene",
            "mode_sweeps": modes["sweeps"],
            "note": "amplification of single Fourier modes of the "
                    "uniform-weight Laplacian by the red-black Gauss-Seidel "
                    "smoother the hierarchy uses, applied with a zero "
                    "right-hand side; the roughest mode dies in one sweep and "
                    "the smoothest one survives it, which is why the coarser "
                    "grids exist (section 18 of docs/mathematics.md)",
            "modes": modes,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2) + "\n")
        print("wrote {}".format(args.report))
        return

    records = []
    for size in args.sizes:
        for field in args.fields:
            for smoothing in args.smoothing:
                with warnings.catch_warnings():
                    # A divergent run overflows; the record already says so.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    record = measure(size, field, args.noise, args.seed,
                                     args.repeat, args.max_cycles, args.tol,
                                     smoothing, args.plain_max_sweeps)
                records.append(record)
                mg = record["multigrid"]
                cg = record["conjugate_gradient"]
                plain = record["plain_relaxation"]
                islands = record["weight_islands"]
                print("{0:>4} x {1:<4} {2:<8} s={3:<2} mg {4:>7.3f} s "
                      "({5:>3} cyc, {6:>2} lv, {7}, {8:>5.1f} fsw)  "
                      "plain {9:>4} sw{10}  cg {11:>6.3f} s ({12:>4} it)  "
                      "ratio {13:>6.1f}x{14}".format(
                          size, size, field, smoothing,
                          mg["median_seconds"], mg["cycles"], mg["levels"],
                          "conv" if mg["converged"] else "stall",
                          mg["fine_sweep_equivalents"],
                          plain["sweeps"],
                          "" if plain["converged"] else "*",
                          cg["median_seconds"], cg["iterations"],
                          mg["median_seconds"] / max(cg["median_seconds"], 1e-9),
                          "" if islands == 1 else "  %d islands" % islands))
                print("      rmse mg {0:>6.3f}|{1:>6.3f}  cg {2:>6.3f}|{3:>6.3f}  "
                      "rad (all|known)   mg vs cg {4}|{5} rad".format(
                          mg["aligned_rmse_rad"], mg["known_rmse_rad"],
                          cg["aligned_rmse_rad"], cg["known_rmse_rad"],
                          rad_text(record["solver_difference_rad"]),
                          rad_text(record["solver_difference_known_rad"])))

    document = {
        "schema_version": 4,
        "input": "generated by parvaneh.make_synthetic",
        "noise": args.noise,
        "seed": args.seed,
        "max_cycles": args.max_cycles,
        "tol": args.tol,
        "thin_weight": THIN_WEIGHT,
        "block_weight": BLOCK_WEIGHT,
        "block_side": BLOCK_SIDE,
        "coarse_sweeps": COARSE_SWEEPS,
        "plain_max_sweeps": args.plain_max_sweeps,
        "measured_runs": args.repeat,
        "note": "both solvers minimise the same weighted least-squares "
                "objective; multigrid may stop at its cycle budget far from "
                "the target on the thin and blocky fields, which is the "
                "measured limit documented in parvaneh.multigrid.  A plain "
                "relaxation marked unconverged (*) spent its whole sweep "
                "budget.  Distances are measured after removing the free "
                "constant of every island of nonzero weight; the known_* "
                "columns average over the cells of nonzero weight only, since "
                "the zero-weight cells are not constrained by the data and "
                "their value is an interpolation choice.  One V-cycle is worth "
                "fine_sweep_equivalents_per_cycle sweeps of the finest grid, "
                "so a rate per cycle is not a rate per sweep",
        "results": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(document, indent=2) + "\n")
    print("wrote {}".format(args.report))


if __name__ == "__main__":
    main()
