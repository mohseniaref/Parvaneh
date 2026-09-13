"""The ``parvaneh unwrap`` command: choose an algorithm and unwrap an image.

Chosen with ``--method``, this command reads a wrapped phase image, unwraps it
and writes the result::

    parvaneh unwrap wrapped.raw --shape 1024 1024 --method goldstein -o out.npy

Because unwrapping is what the command is for, the subcommand name may be left
out::

    parvaneh wrapped.raw --shape 1024 1024 --method goldstein -o out.npy

Headerless raw rasters need explicit ``--shape`` and ``--dtype`` flags; ``.npy``
and ``.npz`` inputs carry their own metadata and need nothing extra, and a
georeferenced raster (GeoTIFF and friends) is read through GDAL, which brings its
own geotransform, projection and nodata value.

Input and output are two-dimensional by default, but a stack (a cube of
interferograms, say) is welcome as ``.npy``/``.npz`` and is unwrapped jointly
along every axis::

    parvaneh unwrap cube.npy --method reliability -o cube-u.npy

Headerless raw rasters and georeferenced rasters stay two-dimensional, because
neither format carries an axis count.

All progress and diagnostic chatter goes to stderr, so the ``--info`` JSON on
stdout stays clean for piping::

    parvaneh unwrap wrapped.raw --shape 1024 1024 -o out.npy --info | jq .seconds
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from .. import __version__
from ..core import available_backends, unwrap
from ..flynn import flynn_unwrap
from ..goldstein import goldstein_unwrap, mask_cut_unwrap
from ..io import read_raw_raster, write_raw_raster
from ..minimum_norm import unwrap_lp
from ..multigrid import multigrid_unwrap
from ..network_flow import COST_MODES, network_flow_unwrap
from ..path_following import quality_guided_unwrap
from ..quality import (derivative_variance_quality, max_gradient_quality,
                       pseudocorrelation_quality)
from ..reliability import reliability_unwrap
from ..raster import RASTER_SUFFIXES, RasterError, read_raster, write_raster
from .base import CommandParser, log


#: Algorithm names accepted by ``--method``, with one-line descriptions shown by
#: ``--method list``.
METHODS = (
    ("ls", "weighted least squares (Ghiglia-Romero); smooth, globally optimal"),
    ("quality-guided", "best-first traversal from the most reliable edges"),
    ("reliability", "reliability sorting into a maximum-reliability tree"),
    ("goldstein", "Goldstein expanding-box branch cuts"),
    ("mask-cut", "branch cuts placed from an unwrapped quality-guided mask"),
    ("flynn", "Flynn minimum-discontinuity network"),
    ("mcf", "minimum-cost flow on the dual network (Costantini)"),
    ("lp", "minimum-Lp, iteratively reweighted least squares"),
    ("multigrid", "V-cycles over a hierarchy of grids; the ls answer again"),
)

#: Backends each method can actually run on, best first.  An empty tuple means
#: the method has a single fixed implementation and ``--backend`` has no effect.
#: Note that the path-following traversal calls its pure-Python engine
#: ``"python"``, not ``"numpy"``, so it must not be offered ``--backend numpy``.
METHOD_BACKENDS = {
    "ls": ("numba", "cython", "blas", "numpy", "cupy"),
    "quality-guided": ("numba", "python"),
    "reliability": ("numba", "python"),
    "goldstein": (),
    "mask-cut": (),
    "flynn": (),
    "mcf": (),
    "lp": (),
    "multigrid": (),
}

#: Methods that unwrap a stack of any rank, not only a single two-dimensional
#: image.  Everything else is a two-dimensional algorithm and is told so when
#: handed a cube, rather than failing somewhere deep inside.
ND_METHODS = ("ls", "reliability", "multigrid")

#: Methods that can use ``--weight``.  The others build their own quality map,
#: so a weight handed to them would be silently dropped.
WEIGHT_METHODS = ("ls", "reliability", "mcf", "multigrid")


def build_parser(add_help=True):
    """Build the parser for ``parvaneh unwrap`` and its implicit shorthand."""
    parser = CommandParser(
        prog="parvaneh unwrap",
        description="Unwrap a wrapped phase image or a stack of them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=add_help,
        epilog="""\
examples:
  # raw float32 raster, least squares, all CPU cores
  parvaneh unwrap wrapped.raw --shape 1024 1024 -o unwrapped.raw

  # NumPy archive in and out, with timings on stdout
  parvaneh unwrap wrapped.npy --method goldstein -o unwrapped.npy --info

  # low-coherence data: hide unreliable pixels behind a validity mask
  parvaneh unwrap wrapped.npy --method quality-guided --mask valid.npy -o out.npy

  # a noisy image that carries residues: reliability sorting avoids them
  parvaneh unwrap noisy.npy --method reliability -o unwrapped.npy

  # the same residues handled exactly, as a minimum-cost flow on the dual net
  parvaneh unwrap noisy.npy --method mcf --cost linear -o unwrapped.npy

  # the least-squares answer again, from a multigrid hierarchy
  parvaneh unwrap noisy.npy --method multigrid -o unwrapped.npy

  # a cube of interferograms, unwrapped jointly along all three axes
  parvaneh unwrap cube.npy --method ls -o cube-unwrapped.npy

  # georeferenced raster in and out, keeping the geotransform and the CRS
  parvaneh unwrap wrapped.tif --method goldstein -o unwrapped.tif

  # cheapest way to get a quick look at a big raster
  parvaneh unwrap big.raw --shape 4096 4096 --tol 1e-6 --max-iter 40 -o big-u.raw

  # the same as the first example: "unwrap" is optional
  parvaneh wrapped.raw --shape 1024 1024 -o unwrapped.raw
""")
    parser.add_argument("input", nargs="?", help="wrapped phase image to read")
    parser.add_argument("-o", "--output", help="file to write the unwrapped phase to")
    parser.add_argument("--method", default="ls",
                        help="unwrapping algorithm (default: ls); "
                             "use '--method list' to see them all")

    io_group = parser.add_argument_group("input/output")
    io_group.add_argument("--shape", nargs="+", type=int, metavar="DIM",
                          help="shape of a headerless raw raster, ROWS COLS (not "
                               "needed for .npy, .npz or a georeferenced raster, "
                               "which know their own shape)")
    io_group.add_argument("--dtype", default="<f4",
                          help="dtype of a headerless raw input (default: <f4)")
    io_group.add_argument("--band", type=int, default=1,
                          help="band to read from a georeferenced raster, "
                               "counting from 1 (default: 1)")
    io_group.add_argument("--out-dtype", default=None,
                          help="dtype of the output; raw output falls back to "
                               "--dtype, raster output to the input raster "
                               "(default: see those)")
    io_group.add_argument("--order", choices=("C", "F"), default="C",
                          help="memory order of headerless raw rasters (default: C)")
    io_group.add_argument("--scale", type=float, default=1.0,
                          help="multiply raw or raster input by this before "
                               "unwrapping")
    io_group.add_argument("--offset", type=float, default=0.0,
                          help="add this to raw or raster input after scaling")
    io_group.add_argument("--npz-key", default="phase",
                          help="array name inside .npz files (default: phase)")
    io_group.add_argument("--invalid-fill", type=float, default=None,
                          help="value used to replace non-finite output pixels "
                               "when the output format cannot store NaN "
                               "(default: refuse and explain)")

    aux_group = parser.add_argument_group("mask and weight")
    aux_group.add_argument("--mask", help="validity mask; non-zero means valid "
                                          "(in a georeferenced raster, a nodata "
                                          "pixel is never valid)")
    aux_group.add_argument("--mask-dtype", default="<f4",
                           help="dtype of a headerless raw --mask (default: <f4)")
    aux_group.add_argument("--weight", help="nonnegative weight raster "
                                            "(--method ls, reliability, mcf "
                                            "or multigrid)")
    aux_group.add_argument("--weight-dtype", default="<f4",
                           help="dtype of a headerless raw --weight (default: <f4)")

    speed_group = parser.add_argument_group("speed")
    speed_group.add_argument("--workers", type=int, default=-1,
                             help="threads for the DCT preconditioner; -1 uses "
                                  "every core (default: -1)")
    speed_group.add_argument("--max-iter", type=int, default=100,
                             help="least-squares iteration cap, or V-cycles for "
                                  "--method multigrid (default: 100)")
    speed_group.add_argument("--tol", type=float, default=1e-8,
                             help="least-squares relative residual target; larger "
                                  "is faster and still usually invisible "
                                  "(default: 1e-8)")

    method_group = parser.add_argument_group("method options")
    method_group.add_argument("--backend", default="auto",
                              help="execution backend: auto, numpy, blas, numba, "
                                   "cython or cupy (default: auto)")
    method_group.add_argument("--quality", default="min_gradient",
                              choices=("min_gradient", "max_gradient",
                                       "pseudocorrelation", "derivative_variance"),
                              help="quality map for quality-guided and flynn "
                                   "(default: min_gradient)")
    method_group.add_argument("--window", type=int, default=None,
                              help="window size for windowed quality maps")
    method_group.add_argument("--p", type=float, default=1.2,
                              help="norm for --method lp, in [1, 2] (default: 1.2)")
    method_group.add_argument("--outer-iter", type=int, default=12,
                              help="outer reweighting iterations for --method lp "
                                   "(default: 12)")
    method_group.add_argument("--inner-iter", type=int, default=100,
                              help="inner least-squares iterations for --method lp "
                                   "(default: 100)")
    method_group.add_argument("--epsilon", type=float, default=1e-3,
                              help="reweighting softening in radians for "
                                   "--method lp (default: 1e-3)")
    method_group.add_argument("--max-cut-length", type=int, default=None,
                              help="branch-cut length cap for --method goldstein")
    method_group.add_argument("--cost", choices=COST_MODES, default="linear",
                              help="edge penalty for --method mcf: 'linear' "
                                   "counts 2*pi jumps, 'quadratic' squares them "
                                   "(default: linear)")

    parser.add_argument("--center", choices=("circular", "none"), default="circular",
                        help="output offset convention: 'circular' (default) pins "
                             "the result to the wrapped input so results from "
                             "different methods are directly comparable; 'none' "
                             "keeps whatever offset the algorithm produced")
    parser.add_argument("--info", action="store_true",
                        help="print a JSON summary to stdout")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="suppress progress messages on stderr")
    parser.add_argument("-V", "--version", action="store_true",
                        help="print the package version and exit")
    return parser


def print_methods():
    """Print the method table to stdout.

    Unlike progress chatter, this listing *is* what the user asked for, so it
    goes to stdout where it can be piped or searched.
    """
    available = available_backends()

    def present(name):
        return available.get(name, name == "python")

    lines = ["available methods:"]
    width = max(len(name) for name, _ in METHODS)
    for name, description in METHODS:
        choices = METHOD_BACKENDS[name]
        usable = [b for b in choices if present(b)]
        tag = ", ".join(usable) if usable else "no backend choice"
        lines.append("  %-*s  %s" % (width, name, description))
        lines.append("  %-*s    backends: %s" % (width, "", tag))
    lines.append("")
    lines.append("compiled kernels on this machine:")
    for name, is_present in available.items():
        lines.append("  %-8s %s"
                     % (name, "available" if is_present else "not available"))
    print("\n".join(lines))


def _load(path, *, shape, dtype, order, npz_key, what, scale=1.0, offset=0.0,
          band=1):
    """Read an array from ``.npy``, ``.npz``, a raster or a headerless file.

    Any rank is accepted; whether a particular method can use a rank other than
    two is decided by the caller, which can name the alternatives.

    Returns ``(array, meta)``.  ``meta`` is a
    :class:`~parvaneh.raster.RasterMeta` for georeferenced input, which is what
    lets the output inherit the same grid, and ``None`` for the other formats.
    """
    source = Path(path)
    if not source.exists():
        raise SystemExit("error: %s file not found: %s" % (what, source))

    suffix = source.suffix.lower()
    meta = None
    if suffix in RASTER_SUFFIXES:
        try:
            array, meta = read_raster(source, band=band, nodata_fill=np.nan)
        except RasterError as exc:
            raise SystemExit("error: cannot read %s: %s" % (source, exc))
        if scale != 1.0 or offset != 0.0:
            array = array * scale + offset
    elif suffix == ".npy":
        array = np.load(str(source))
    elif suffix == ".npz":
        with np.load(str(source)) as bundle:
            if npz_key not in bundle.files:
                raise SystemExit(
                    "error: %s archive %s has no %r array (contains: %s)"
                    % (what, source, npz_key, ", ".join(bundle.files) or "nothing"))
            array = bundle[npz_key]
    else:
        if shape is None:
            raise SystemExit(
                "error: %s is not .npy/.npz, so --shape ROWS COLS is required"
                % source)
        try:
            array = read_raw_raster(source, shape, dtype,
                                    scale=scale, offset=offset, order=order)
        except ValueError as exc:
            raise SystemExit("error: cannot read %s: %s" % (source, exc))

    array = np.asarray(array)
    if array.ndim == 0:
        raise SystemExit("error: %s is a single number, not an image or a stack"
                         % what)
    return array, meta


def _save(path, array, *, shape, dtype, order, npz_key, invalid_fill, like=None):
    """Write an array to ``.npy``, ``.npz``, a raster or a headerless file.

    Only the first two formats can hold a stack: a raster band and a headerless
    raster are both a single two-dimensional image by construction.
    """
    target = Path(path)
    suffix = target.suffix.lower()

    if suffix in RASTER_SUFFIXES:
        if np.ndim(array) != 2:
            raise SystemExit(
                "error: a georeferenced raster holds one two-dimensional band, "
                "so %s cannot store a %d-dimensional result.\n"
                "       Write a cube to .npy or .npz instead."
                % (target, np.ndim(array)))
        return _save_raster(target, array, dtype, invalid_fill, like)
    if suffix == ".npy":
        np.save(str(target), array)
        return
    if suffix == ".npz":
        np.savez(str(target), **{npz_key: array})
        return

    if np.ndim(array) != 2:
        raise SystemExit(
            "error: a headerless raster holds one two-dimensional image, so %s "
            "cannot store a %d-dimensional result.\n"
            "       Write a cube to .npy or .npz instead."
            % (target, np.ndim(array)))

    finite = np.isfinite(array).all()
    if not finite:
        # A headerless raster has no way to carry NaN, so make the user pick a
        # sentinel rather than silently writing corrupted-looking numbers.
        if invalid_fill is None or not np.isfinite(invalid_fill):
            raise SystemExit(
                "error: the unwrapped phase contains non-finite pixels, which a "
                "raw raster cannot store.\n"
                "       Pass --invalid-fill VALUE to substitute a sentinel, or "
                "write to .npy/.npz instead.")
        array = np.where(np.isfinite(array), array, invalid_fill)

    if shape is not None:
        array = array.reshape(shape)
    try:
        write_raw_raster(target, array, dtype, order=order)
    except ValueError as exc:
        raise SystemExit("error: cannot write %s: %s" % (target, exc))


def _save_raster(target, array, dtype, invalid_fill, like):
    """Write ``array`` as a raster, inheriting the grid of ``like`` if given."""
    if dtype is None:
        # An unwrapped phase is real-valued, so an integer input raster (a
        # quality map, say) cannot be copied; float32 keeps the file small and
        # still resolves a fringe many times over.
        dtype = (like.dtype if like is not None
                 and np.issubdtype(np.dtype(like.dtype), np.floating)
                 else "float32")
    floating = np.issubdtype(np.dtype(dtype), np.floating)
    if invalid_fill is not None and not np.isfinite(invalid_fill):
        raise SystemExit("error: --invalid-fill must be a finite number")
    if not np.isfinite(array).all():
        if not floating and invalid_fill is None:
            raise SystemExit(
                "error: the unwrapped phase contains non-finite pixels, which "
                "the %s output dtype cannot store.\n"
                "       Pass --invalid-fill VALUE to substitute a sentinel, or "
                "use a floating-point --out-dtype such as float32."
                % np.dtype(dtype).name)
        if invalid_fill is not None:
            array = np.where(np.isfinite(array), array, invalid_fill)

    if invalid_fill is not None:
        nodata = invalid_fill
    else:
        # NaN is the natural "no data" marker in a floating-point band; an
        # integer band has no NaN, and a band with no invalid pixel has nothing
        # to declare at all.
        nodata = np.nan if floating and not np.isfinite(array).all() else None
    try:
        write_raster(target, array, like=like, dtype=dtype, nodata=nodata)
    except RasterError as exc:
        raise SystemExit("error: cannot write %s: %s" % (target, exc))


def _resolve_backend(requested, method):
    """Resolve ``--backend`` for ``method``.

    Returns the concrete backend name to use, or ``None`` when the method has a
    single fixed implementation and no backend choice exists.
    """
    supported = METHOD_BACKENDS[method]
    if not supported:
        if requested not in ("auto", "numpy"):
            log("note: --method %s has no backend choice, ignoring --backend %s"
                 % (method, requested))
        return None

    available = available_backends()
    # ``python`` is the path-following engine and is always present, even though
    # available_backends() reports only the compiled kernels.
    def present(name):
        return available.get(name, name == "python")

    if requested != "auto":
        if requested not in supported:
            raise SystemExit("error: --method %s supports %s, not --backend %s"
                             % (method, " or ".join(supported), requested))
        if not present(requested):
            usable = [name for name in supported if present(name)]
            raise SystemExit("error: backend %r is not available on this machine "
                             "(usable here: %s)"
                             % (requested, ", ".join(usable) or "none"))
        return requested

    for candidate in supported:
        if present(candidate):
            return candidate
    raise SystemExit("error: no usable backend for --method %s" % method)


def center_circular(result, phase):
    """Align the output offset with the wrapped input.

    The absolute phase an unwrapper returns is arbitrary: adding any constant to
    a valid solution leaves it equally valid at every pixel.  Different
    algorithms happen to pick different constants, so without this step two
    methods can disagree by a constant shift and look broken when they are both
    correct.  Subtracting the circular mean of ``result - phase`` pins the output
    offset to the input, which makes results from different methods directly
    comparable.  Non-finite pixels (for example a masked-out region) are left
    alone.
    """
    valid = np.isfinite(result)
    if not valid.any():
        return result
    turn = np.exp(1j * (result[valid] - phase[valid]))
    offset = float(np.angle(np.mean(turn)))
    # Every cycle-counting method (Goldstein, mask cut, Flynn) returns
    # ``phase + pi`` modulo 2 pi, so its offset is exactly pi and the circular
    # mean lands on the negative real axis.  There the imaginary part is only
    # rounding noise, and np.angle answers +pi or -pi for the same image
    # depending on the input dtype.  Both answers describe the same solution,
    # but a pipeline that compares two runs should not see a whole fringe of
    # difference, so the ambiguous case is pinned to +pi.
    if abs(abs(offset) - np.pi) <= 1e-3:
        offset = np.pi
    centered = result.copy()
    centered[valid] = result[valid] - offset
    return centered


def _load_mask(args):
    """Read ``--mask`` as a boolean validity map, or ``None`` when unset."""
    if args.mask is None:
        return None
    mask, meta = _load(args.mask, shape=args.shape, dtype=args.mask_dtype,
                       order=args.order, npz_key=args.npz_key, what="mask",
                       band=args.band)
    mask = np.asarray(mask)
    if meta is not None:
        # A raster can mark "no data" with a sentinel that became NaN, or with a
        # flag value; either way the pixel is not valid.
        return np.isfinite(mask) & (mask != 0)
    return mask != 0


def _load_weight(args):
    """Read ``--weight`` as a nonnegative float map, or ``None`` when unset."""
    if args.weight is None:
        return None
    weight, meta = _load(args.weight, shape=args.shape, dtype=args.weight_dtype,
                         order=args.order, npz_key=args.npz_key, what="weight",
                         band=args.band)
    weight = np.asarray(weight, dtype=np.float64)
    if meta is not None:
        # Outside the valid footprint a raster weight is undefined, and zero
        # weight means "ignore this pixel", which is exactly right.
        weight = np.where(np.isfinite(weight), weight, 0.0)
    if not np.all(np.isfinite(weight)) or np.any(weight < 0):
        raise SystemExit("error: weight must be finite and nonnegative")
    return weight


def _window_kwargs(args):
    return {} if args.window is None else {"window": args.window}


def _output_dtype(args):
    """Dtype to hand :func:`_save`, or ``None`` to let it choose.

    A headerless raw raster has no metadata at all, so the fallback there is the
    input dtype.  A georeferenced raster does have metadata, and the output
    should keep the input grid's dtype whenever that can hold a phase.
    """
    if args.out_dtype is not None:
        return args.out_dtype
    if args.output is not None and Path(args.output).suffix.lower() in RASTER_SUFFIXES:
        return None
    return args.dtype


def _raster_summary(meta):
    """Describe an input raster for ``--info``.

    ``json.dumps`` writes a bare ``NaN`` that no JSON reader accepts, so a NaN
    nodata is reported as ``null`` instead.
    """
    nodata = meta.nodata
    if nodata is not None and not np.isfinite(nodata):
        nodata = None
    return {"driver": meta.driver, "dtype": meta.dtype, "shape": list(meta.shape),
            "band": meta.band, "bands": meta.count,
            "crs": meta.crs, "transform": (None if meta.transform is None
                                           else list(meta.transform)),
            "nodata": nodata}


def _quality_from_args(args, phase, method, backend):
    """Return the ``quality`` argument the requested method expects.

    Only ``quality-guided`` understands the string ``"min_gradient"``: there it
    selects a specialised edge-priority traversal that works in phase cycles.
    Every other consumer wants an explicit weight map, so the name is turned
    into one in that case.
    """
    if args.quality == "min_gradient" and method == "quality-guided":
        # The edge-priority formulation, and the only mode the compiled path
        # backend implements.
        return "min_gradient"
    if backend == "numba":
        raise SystemExit(
            "error: --backend numba only supports --quality min_gradient; "
            "use --backend python for --quality %s" % args.quality)
    if args.quality in ("min_gradient", "max_gradient") and args.window is None:
        return None                       # the module's own default
    if args.quality in ("min_gradient", "max_gradient"):
        return max_gradient_quality(phase, **_window_kwargs(args))
    if args.quality == "pseudocorrelation":
        return pseudocorrelation_quality(phase, **_window_kwargs(args))
    return derivative_variance_quality(phase, **_window_kwargs(args))


def _run_method(args, phase, backend, mask, weight):
    """Dispatch to the requested algorithm; return ``(result, detail)``."""
    method = args.method

    if method == "ls":
        # A validity mask is exactly a 0/1 weight for the least-squares solver.
        if weight is None and mask is not None:
            weight = mask.astype(np.float64)
        result, info = unwrap(phase, weight, backend=backend,
                              max_iter=args.max_iter, tol=args.tol,
                              workers=args.workers, return_info=True)
        return result, {"weights": "custom" if args.weight is not None
                        else ("mask" if args.mask is not None else "uniform"),
                        **dataclasses.asdict(info)}

    if method == "quality-guided":
        quality = _quality_from_args(args, phase, method, backend)
        return quality_guided_unwrap(phase, quality, mask,
                                     backend=backend), {"quality": args.quality}

    if method == "reliability":
        # The reliability rating already folds in a mask and a weight map, so
        # both options keep their usual meaning here.
        result, info = reliability_unwrap(phase, mask, weight, backend=backend,
                                          return_info=True)
        return result, dataclasses.asdict(info)

    if method == "goldstein":
        result, cuts = goldstein_unwrap(phase, mask,
                                        max_cut_length=args.max_cut_length,
                                        return_cuts=True)
        return result, {"cut_pixels": int(np.count_nonzero(cuts))}

    if method == "mask-cut":
        result, cuts = mask_cut_unwrap(phase, mask, return_cuts=True)
        return result, {"cut_pixels": int(np.count_nonzero(cuts))}

    if method == "flynn":
        quality = _quality_from_args(args, phase, method, "python")
        result, iterations = flynn_unwrap(phase, quality, mask,
                                          return_iterations=True)
        return result, {"iterations": int(iterations)}

    if method == "mcf":
        # The solver's own iteration count is a property of the network, not a
        # convergence knob, so --max-iter is deliberately left unused here.
        result, info = network_flow_unwrap(phase, weight, mask=mask,
                                           cost=args.cost, return_info=True)
        return result, dataclasses.asdict(info)

    if method == "multigrid":
        # The same normal equations as "ls", solved by V-cycles instead of by
        # conjugate gradients.  --max-iter caps the cycles, which are cheap but
        # not unlimited: see docs/performance.md for when "ls" is the better
        # choice of the two.
        result, info = multigrid_unwrap(phase, weight, mask=mask,
                                        max_cycles=args.max_iter, tol=args.tol,
                                        return_info=True)
        if not info.converged:
            log("note     multigrid stopped after %d V-cycles at a relative "
                "residual of %.1e; sharp-edged weights suit --method ls better"
                % (info.cycles, info.relative_residual))
        detail = dataclasses.asdict(info)
        # Under "--info" the key "seconds" is the wall time of the whole run,
        # which is what every other method reports there, so the solver's own
        # timing does not belong under it.  MultigridInfo.seconds stays
        # available to library users.
        detail.pop("seconds")
        return result, {"weights": "custom" if args.weight is not None
                        else ("mask" if args.mask is not None else "uniform"),
                        **detail}

    result, info = unwrap_lp(phase, p=args.p, epsilon=args.epsilon,
                             outer_iter=args.outer_iter,
                             inner_iter=args.inner_iter, tol=args.tol,
                             return_info=True)
    return result, dataclasses.asdict(info)


def run(args):
    """Run ``parvaneh unwrap`` with parsed ``args``; return a process exit code."""
    if args.version:
        print(__version__)
        return 0

    if args.method == "list":
        print_methods()
        return 0

    if args.input is None:
        raise SystemExit("error: an input file is required (or --method list)")

    known = [name for name, _ in METHODS]
    if args.method not in known:
        raise SystemExit("error: unknown method %r (choose from: %s)"
                         % (args.method, ", ".join(known)))

    if args.max_iter < 1:
        raise SystemExit("error: --max-iter must be at least 1")

    if args.shape is not None and len(args.shape) != 2:
        raise SystemExit(
            "error: --shape takes two numbers, ROWS COLS, because a headerless "
            "raw raster is a single two-dimensional image (got %d numbers).\n"
            "       Save a stack as .npy or .npz, which record their own shape."
            % len(args.shape))

    phase, meta = _load(args.input, shape=args.shape, dtype=args.dtype,
                        order=args.order, npz_key=args.npz_key, what="input",
                        scale=args.scale, offset=args.offset, band=args.band)
    if phase.ndim != 2 and args.method not in ND_METHODS:
        raise SystemExit(
            "error: --method %s unwraps a two-dimensional image, but %s has "
            "shape %s.\n"
            "       The methods that accept a stack of any rank are: %s."
            % (args.method, args.input,
               "x".join(str(size) for size in phase.shape),
               ", ".join(ND_METHODS)))
    input_mask = None
    if not np.all(np.isfinite(phase)):
        if meta is None:
            raise SystemExit("error: the input phase contains NaN or infinity; "
                             "unwrapping needs a fully defined image")
        # A georeferenced raster carries its own nodata mask, so invalid pixels
        # are a footprint to exclude rather than an error to report.
        input_mask = np.isfinite(phase)
        if not input_mask.any():
            raise SystemExit("error: every pixel of %s is nodata, so there is "
                             "nothing to unwrap" % args.input)
        if not args.quiet:
            log("note     %d nodata pixel(s) of %s excluded from unwrapping"
                % (int(input_mask.size - input_mask.sum()), args.input))
        phase = np.where(input_mask, phase, 0.0)

    mask = _load_mask(args)
    weight = _load_weight(args)
    for name, array in (("mask", mask), ("weight", weight)):
        if array is not None and array.shape != phase.shape:
            raise SystemExit("error: %s has shape %s but the input has shape %s"
                             % (name, array.shape, phase.shape))
    if weight is not None and args.method not in WEIGHT_METHODS and not args.quiet:
        # Dropping a weight silently would let a user believe coherence steered
        # the result when it did not.  The run still makes sense without it, so
        # this is a note rather than a refusal.
        log("note     --method %s builds its own quality map and ignores "
            "--weight (weights are used by: %s)"
            % (args.method, ", ".join(WEIGHT_METHODS)))
    if input_mask is not None:
        mask = input_mask if mask is None else (mask & input_mask)

    backend = _resolve_backend(args.backend, args.method)
    # The compiled least-squares kernels are two-dimensional, and core.unwrap()
    # quietly runs the NumPy engine for a stack instead of failing.  Report the
    # engine that really runs, so this line agrees with --info's "backend".
    fallback = ("numba", "cython", "cupy")
    running = ("numpy" if phase.ndim != 2 and backend in fallback else backend)
    if not args.quiet:
        log("input    %s  shape=%s" % (args.input, "x".join(map(str, phase.shape))))
        log("method   %s (backend %s, workers %d)"
            % (args.method, running or "-", args.workers))

    started = time.perf_counter()
    result, detail = _run_method(args, phase, backend, mask, weight)
    if mask is not None:
        # Not every algorithm blanks out the pixels it was told to ignore, but
        # the CLI promises one convention, and a raster output needs it to mark
        # the same footprint as the input.
        result = np.asarray(result, dtype=np.float64)
        result[~mask] = np.nan
    if args.center == "circular":
        result = center_circular(result, phase)
    seconds = time.perf_counter() - started
    result = np.asarray(result, dtype=np.float64)

    if args.output is not None:
        _save(args.output, result,
              shape=args.shape, dtype=_output_dtype(args),
              order=args.order, npz_key=args.npz_key,
              invalid_fill=args.invalid_fill, like=meta)

    if not args.quiet:
        log("elapsed  %.3f s" % seconds)
        if args.output is not None:
            log("wrote    %s" % args.output)

    if args.info:
        # ``detail`` carries a "backend" of its own for the methods that return a
        # report; where it does not, the engine that ran is the one above.
        summary = {"method": args.method, "backend": running,
                   "center": args.center, "shape": list(result.shape),
                   "seconds": seconds, "input": args.input, "output": args.output}
        if meta is not None:
            summary["raster"] = _raster_summary(meta)
        summary.update(detail)
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
