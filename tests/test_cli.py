"""Tests for the ``parvaneh`` command line.

The command line is the only part of the package whose contract is not a Python
signature: it is argument parsing, exit codes, and the split between stdout and
stderr.  Those are what these tests pin down, because scripts and CI pipelines
depend on them.

Two rules are checked again and again:

* a failure is an ``error: ...`` message with exit status 1, never a traceback;
* stdout carries only what the user asked for, so ``--info`` JSON and
  ``--method list`` stay pipeable while progress notes go to stderr.
"""

import json
import sys

import numpy as np
import pytest
from parvaneh import __version__, make_synthetic, wrap_phase
from parvaneh.cli import main
from parvaneh.cli.unwrap import (METHOD_BACKENDS, METHODS, ND_METHODS,
                                 build_parser, center_circular)
from parvaneh.raster import raster_backend, read_raster

METHOD_NAMES = [name for name, _ in METHODS]


@pytest.fixture
def scene(tmp_path):
    """A small synthetic interferogram on disk, with its answer key."""
    truth, wrapped, quality = make_synthetic(shape=(24, 32), noise=0.02, seed=11)
    path = tmp_path / "wrapped.npy"
    np.save(str(path), wrapped.astype(np.float32))
    return truth, wrapped, path


def run(argv, capsys):
    """Run the command line and return ``(status, stdout, stderr)``.

    ``main`` reports every expected failure with :class:`SystemExit`, exactly as
    the console script does.  A non-integer exit code carries a message, which
    the interpreter would print to stderr, so it is printed here too and the
    status becomes 1 -- the behaviour a user actually sees.
    """
    status = 0
    message = None
    try:
        status = main(argv)
    except SystemExit as exc:
        if exc.code is None:
            status = 0
        elif isinstance(exc.code, int):
            status = exc.code
        else:
            message, status = str(exc.code), 1
    captured = capsys.readouterr()
    err = captured.err
    if message is not None:
        # The interpreter prints the message of a SystemExit for us; a test that
        # catches the exception has to do it itself.
        print(message, file=sys.stderr)
        err += message + "\n"
    return status, captured.out, err


# --------------------------------------------------------------------------
# dispatch: no arguments, help, version
# --------------------------------------------------------------------------

def test_no_arguments_prints_help(capsys):
    status, out, err = run([], capsys)
    assert status == 0
    assert "usage: parvaneh" in out
    assert "unwrap" in out
    assert err == ""


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_exits_zero(flag, capsys):
    status, out, _ = run([flag], capsys)
    assert status == 0
    assert "usage: parvaneh" in out


@pytest.mark.parametrize("flag", ["-V", "--version"])
def test_version_goes_to_stdout(flag, capsys):
    status, out, err = run([flag], capsys)
    assert status == 0
    assert out.strip() == __version__
    assert err == ""


def test_missing_input_is_an_error(capsys):
    status, out, err = run(["--info"], capsys)
    assert status == 1
    assert "error: an input file is required" in err
    assert out == ""


def test_unknown_method_is_an_error(capsys):
    status, _, err = run(["wrapped.npy", "--method", "magic"], capsys)
    assert status == 1
    assert "error: unknown method 'magic'" in err
    for name in METHOD_NAMES:
        assert name in err


def test_unknown_flag_exits_one_not_two(capsys):
    """``CommandParser.error`` must report status 1, unlike bare argparse."""
    status, _, err = run(["wrapped.npy", "--not-a-flag"], capsys)
    assert status == 1
    assert "error:" in err
    assert "usage:" in err


def test_subcommand_name_is_optional(tmp_path, capsys):
    """``parvaneh FILE`` and ``parvaneh unwrap FILE`` must agree exactly."""
    _, wrapped, _ = make_synthetic(shape=(16, 20), noise=0.02, seed=3)
    path = tmp_path / "wrapped.npy"
    np.save(str(path), wrapped.astype(np.float32))
    explicit = tmp_path / "explicit.npy"
    implicit = tmp_path / "implicit.npy"

    assert run(["unwrap", str(path), "-o", str(explicit), "-q"], capsys)[0] == 0
    assert run([str(path), "-o", str(implicit), "-q"], capsys)[0] == 0
    assert np.array_equal(np.load(str(explicit)), np.load(str(implicit)))


# --------------------------------------------------------------------------
# --method list answers a question, so it belongs on stdout
# --------------------------------------------------------------------------

def test_method_list_uses_stdout(capsys):
    status, out, err = run(["--method", "list"], capsys)
    assert status == 0
    for name in METHOD_NAMES:
        assert name in out
    assert "backends:" in out
    assert err == ""


def test_every_method_declares_its_backends():
    assert set(METHOD_BACKENDS) == set(METHOD_NAMES)


# --------------------------------------------------------------------------
# input and output guards
# --------------------------------------------------------------------------

def test_raw_input_needs_a_shape(tmp_path, capsys):
    path = tmp_path / "wrapped.raw"
    path.write_bytes(b"\0" * 64)
    status, _, err = run([str(path), "-o", str(tmp_path / "out.npy")], capsys)
    assert status == 1
    assert "--shape ROWS COLS is required" in err


def test_missing_file_is_an_error(tmp_path, capsys):
    status, _, err = run([str(tmp_path / "nope.npy")], capsys)
    assert status == 1
    assert "file not found" in err


def test_non_finite_input_is_rejected(tmp_path, capsys):
    phase = np.zeros((8, 8), dtype=np.float32)
    phase[3, 4] = np.nan
    path = tmp_path / "wrapped.npy"
    np.save(str(path), phase)
    status, _, err = run([str(path)], capsys)
    assert status == 1
    assert "contains NaN or infinity" in err


def test_shape_mismatch_is_an_error(scene, tmp_path, capsys):
    _, _, path = scene
    np.save(str(tmp_path / "mask.npy"), np.ones((5, 5), dtype=bool))
    status, _, err = run([str(path), "--mask", str(tmp_path / "mask.npy")], capsys)
    assert status == 1
    assert "mask has shape (5, 5)" in err


def test_negative_weight_is_rejected(scene, tmp_path, capsys):
    _, _, path = scene
    weight = np.ones((24, 32))
    weight[0, 0] = -1.0
    np.save(str(tmp_path / "weight.npy"), weight)
    status, _, err = run([str(path), "--weight", str(tmp_path / "weight.npy")], capsys)
    assert status == 1
    assert "weight must be finite and nonnegative" in err


def test_zero_max_iter_is_an_error(scene, capsys):
    _, _, path = scene
    status, _, err = run([str(path), "--max-iter", "0"], capsys)
    assert status == 1
    assert "--max-iter must be at least 1" in err


def test_masked_pixels_stay_nan_in_the_output(tmp_path, capsys):
    """A non-zero mask entry means valid, so the rest of the image stays NaN."""
    phase = np.zeros((12, 12), dtype=np.float32)
    np.save(str(tmp_path / "wrapped.npy"), phase)
    mask = np.zeros((12, 12), dtype=bool)
    mask[:, 6:] = True
    np.save(str(tmp_path / "mask.npy"), mask)
    out_path = tmp_path / "out.npy"

    status, _, _ = run([str(tmp_path / "wrapped.npy"), "--method", "quality-guided",
                        "--mask", str(tmp_path / "mask.npy"),
                        "-o", str(out_path), "-q"], capsys)
    assert status == 0
    result = np.load(str(out_path))
    assert np.isnan(result[:, :6]).all()
    assert np.isfinite(result[:, 6:]).all()


def test_raw_output_refuses_non_finite_without_fill(tmp_path, capsys):
    """A headerless raster cannot store NaN, so the user must choose a sentinel."""
    phase = np.zeros((12, 12), dtype=np.float32)
    np.save(str(tmp_path / "wrapped.npy"), phase)
    mask = np.zeros((12, 12), dtype=bool)
    mask[:, 6:] = True
    np.save(str(tmp_path / "mask.npy"), mask)
    common = ["--method", "quality-guided", "--mask", str(tmp_path / "mask.npy")]

    status, _, err = run([str(tmp_path / "wrapped.npy")] + common + [
        "-o", str(tmp_path / "out.raw"), "--shape", "12", "12"], capsys)
    assert status == 1
    assert "non-finite pixels" in err
    assert "--invalid-fill VALUE" in err

    status, _, _ = run([str(tmp_path / "wrapped.npy")] + common + [
        "-o", str(tmp_path / "out.raw"), "--shape", "12", "12",
        "--invalid-fill", "-9999"], capsys)
    assert status == 0
    written = np.fromfile(str(tmp_path / "out.raw"), dtype="<f4").reshape(12, 12)
    assert (written[:, :6] == -9999).all()


# --------------------------------------------------------------------------
# georeferenced rasters (GeoTIFF and other GDAL formats)
#
# These need a GDAL binding, so each test skips when neither rasterio nor
# osgeo.gdal is installed.  That skip is the documented behaviour of the
# package as well: without a binding the command line still handles .npy, .npz
# and headerless rasters, and reports an install hint for everything else.
# --------------------------------------------------------------------------

#: A north-up 30 m grid, so a test can prove the geometry survives a round trip.
GRID = (30.0, 0.0, 412345.0, 0.0, -30.0, 4645110.0)

raster_only = pytest.mark.skipif(
    raster_backend() is None,
    reason="no GDAL binding: neither rasterio nor osgeo.gdal is installed")


def write_tif(path, array, *, nodata=None, transform=GRID, bands=None):
    """Write ``array`` (or a list of bands) as a GeoTIFF, backend directly.

    ``write_raster`` only creates single-band files, so the multi-band fixture
    has to be built here.
    """
    rasterio = pytest.importorskip("rasterio")
    stack = np.asarray(array) if bands is None else np.stack(bands)
    count = 1 if stack.ndim == 2 else stack.shape[0]
    height, width = stack.shape[-2:]
    with rasterio.open(str(path), "w", driver="GTiff", height=height,
                       width=width, count=count, dtype=stack.dtype.name,
                       nodata=nodata, transform=rasterio.Affine(*transform)) as dst:
        dst.write(stack, 1) if count == 1 else dst.write(stack)
    return path


@raster_only
def test_raster_input_and_output_round_trip(scene, tmp_path, capsys):
    """A .tif in, a .tif out: same grid, same numbers as the .npy run."""
    _, wrapped, _ = scene
    source = write_tif(tmp_path / "wrapped.tif", wrapped.astype(np.float32))

    status, out, err = run([str(source), "-o", str(tmp_path / "out.npy"), "-q"],
                           capsys)
    assert status == 0, err

    status, _, err = run([str(source), "-o", str(tmp_path / "out.tif"), "-q"],
                         capsys)
    assert status == 0, err

    values, meta = read_raster(tmp_path / "out.tif")
    assert meta.driver == "GTiff"
    assert meta.transform == GRID
    assert values.shape == wrapped.shape
    assert np.allclose(values, np.load(str(tmp_path / "out.npy")), atol=1e-5)


@raster_only
def test_raster_nodata_pixels_are_excluded(tmp_path, capsys):
    """Nodata is a validity declaration: those pixels are not unwrapped."""
    phase = np.zeros((16, 20), dtype=np.float32)
    phase[2:6, 14:19] = np.nan
    source = write_tif(tmp_path / "hole.tif", phase, nodata=np.nan)

    # no -q: the note that reports how many pixels were dropped is progress
    # chatter, and progress chatter is exactly what -q silences.
    status, _, err = run([str(source), "-o", str(tmp_path / "out.tif")], capsys)
    assert status == 0, err
    assert "20 nodata pixel(s)" in err

    values, meta = read_raster(tmp_path / "out.tif")
    assert np.isnan(values[2:6, 14:19]).all()
    assert np.isfinite(values).sum() == phase.size - 20
    assert meta.nodata is not None


@raster_only
def test_band_selects_which_image_is_unwrapped(scene, tmp_path, capsys):
    """A phase band stored next to other bands can be picked with --band."""
    _, wrapped, path = scene
    source = write_tif(tmp_path / "stack.tif", None,
                       bands=[(wrapped * 0.5).astype(np.float32), wrapped])

    results = {}
    for band in ("1", "2"):
        target = tmp_path / ("band%s.npy" % band)
        status, _, err = run([str(source), "--band", band,
                              "-o", str(target), "-q"], capsys)
        assert status == 0, err
        results[band] = np.load(str(target))

    # band 2 must unwrap to exactly what the .npy route produces from the same
    # array, and it must not be the answer for band 1.
    status, _, err = run([str(path), "-o", str(tmp_path / "direct.npy"), "-q"],
                         capsys)
    assert status == 0, err
    assert np.allclose(results["2"], np.load(str(tmp_path / "direct.npy")))
    assert not np.allclose(results["1"], results["2"])

    status, _, err = run([str(source), "--band", "3", "-o",
                          str(tmp_path / "oops.npy")], capsys)
    assert status == 1
    assert "band 3 does not exist" in err
    assert "the file has 2 bands" in err


@raster_only
def test_scale_applies_to_raster_input(tmp_path, capsys):
    """--scale multiplies phase read from a raster, before unwrapping."""
    rows, cols = 20, 24
    yy, xx = np.mgrid[0:rows, 0:cols]
    smooth = (0.25 * xx + 0.15 * yy) / 4.0
    source = write_tif(tmp_path / "smooth.tif",
                       wrap_phase(smooth).astype(np.float32))

    outputs = []
    for factor in ("1", "2"):
        target = tmp_path / ("scale%s.npy" % factor)
        status, _, err = run([str(source), "--scale", factor, "-o", str(target),
                              "-q"], capsys)
        assert status == 0, err
        outputs.append(np.load(str(target)))

    assert np.allclose(outputs[1], 2.0 * outputs[0], atol=1e-3)


@raster_only
def test_integer_raster_keeps_its_dtype_and_sentinel(tmp_path, capsys):
    """An int16 input with a -9999 sentinel can be unwrapped back to int16."""
    values = np.arange(12 * 12, dtype="int16").reshape(12, 12)
    values[0:3, 0:3] = -9999
    source = write_tif(tmp_path / "int16.tif", values, nodata=-9999)

    status, _, err = run([str(source), "--out-dtype", "int16",
                          "--invalid-fill", "-9999",
                          "-o", str(tmp_path / "out.tif"), "-q"], capsys)
    assert status == 0, err

    back, meta = read_raster(tmp_path / "out.tif", nodata_fill=None)
    assert meta.dtype == "int16"
    assert meta.nodata == -9999
    assert (back[0:3, 0:3] == -9999).all()
    assert meta.transform == GRID


@raster_only
def test_raster_mask_and_weight_files_are_accepted(scene, tmp_path, capsys):
    """--mask and --weight take rasters as well as .npy arrays."""
    _, wrapped, _ = scene
    source = write_tif(tmp_path / "wrapped.tif", wrapped.astype(np.float32))
    mask = np.ones(wrapped.shape, dtype=np.float32)
    mask[:, :8] = 0.0
    write_tif(tmp_path / "mask.tif", mask)
    write_tif(tmp_path / "weight.tif", mask)

    status, _, err = run([str(source), "--method", "quality-guided",
                          "--mask", str(tmp_path / "mask.tif"),
                          "--weight", str(tmp_path / "weight.tif"),
                          "-o", str(tmp_path / "out.tif"), "-q"], capsys)
    assert status == 0, err

    values, _ = read_raster(tmp_path / "out.tif")
    assert np.isnan(values[:, :8]).all()
    assert np.isfinite(values[:, 8:]).all()


@raster_only
def test_unwritable_raster_format_is_refused(tmp_path, capsys):
    """Only the formats this package can create are allowed as output."""
    phase = np.zeros((8, 8), dtype=np.float32)
    source = write_tif(tmp_path / "wrapped.tif", phase)

    status, _, err = run([str(source), "-o", str(tmp_path / "out.vrt")], capsys)
    assert status == 1
    assert "writing '.vrt' files is not supported" in err


@raster_only
def test_info_reports_the_raster_metadata(scene, tmp_path, capsys):
    _, wrapped, _ = scene
    source = write_tif(tmp_path / "wrapped.tif", wrapped.astype(np.float32))

    status, out, err = run([str(source), "--info", "-o",
                            str(tmp_path / "out.tif"), "-q"], capsys)
    assert status == 0, err
    report = json.loads(out)["raster"]
    assert report["driver"] == "GTiff"
    assert report["transform"] == list(GRID)
    assert report["band"] == 1
    assert report["shape"] == list(wrapped.shape)


def test_without_a_binding_a_raster_input_explains_how_to_install(
        monkeypatch, tmp_path, capsys):
    """The no-GDAL path is supported, so it must fail with a helpful message."""
    import parvaneh.raster as raster

    monkeypatch.setattr(raster, "_import_rasterio", lambda: None)
    monkeypatch.setattr(raster, "_import_gdal", lambda: None)
    source = tmp_path / "wrapped.tif"
    source.write_bytes(b"\0" * 8)

    status, out, err = run([str(source), "-o", str(tmp_path / "out.tif")], capsys)

    assert status == 1
    assert out == ""
    assert "neither rasterio nor osgeo.gdal is installed" in err
    assert "pip install 'parvaneh[geo]'" in err
    assert "python3-gdal" in err


# --------------------------------------------------------------------------
# backend and quality agreement
# --------------------------------------------------------------------------

def test_backend_without_a_choice_is_reported_but_harmless(scene, capsys):
    """``goldstein`` has one implementation, so ``--backend`` only produces a note."""
    _, _, path = scene
    status, _, err = run([str(path), "--method", "goldstein",
                          "--backend", "numba"], capsys)
    assert status == 0
    assert "has no backend choice, ignoring --backend numba" in err

    # "auto" is the default and is accepted silently.
    status, _, err = run([str(path), "--method", "goldstein", "--backend", "auto"],
                         capsys)
    assert status == 0
    assert "ignoring --backend" not in err


def test_unsupported_backend_is_an_error(scene, capsys):
    _, _, path = scene
    status, _, err = run([str(path), "--method", "quality-guided",
                          "--backend", "numpy"], capsys)
    assert status == 1
    assert "supports numba or python" in err


def test_numba_requires_the_min_gradient_quality(scene, capsys):
    _, _, path = scene
    status, _, err = run([str(path), "--method", "quality-guided",
                          "--backend", "numba", "--quality", "pseudocorrelation"],
                         capsys)
    assert status == 1
    assert "only supports --quality min_gradient" in err


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

def test_unwrap_writes_the_right_shape(scene, tmp_path, capsys):
    truth, _, path = scene
    out_path = tmp_path / "unwrapped.npy"
    status, _, err = run([str(path), "-o", str(out_path)], capsys)
    assert status == 0
    result = np.load(str(out_path))
    assert result.shape == truth.shape
    assert result.dtype == np.float64
    assert np.isfinite(result).all()
    assert "wrote" in err


def test_info_json_is_valid_and_keeps_stdout_clean(scene, tmp_path, capsys):
    _, _, path = scene
    status, out, err = run([str(path), "-o", str(tmp_path / "out.npy"),
                            "--info", "--backend", "numpy"], capsys)
    assert status == 0
    summary = json.loads(out)          # stdout must be JSON and nothing else
    assert summary["method"] == "ls"
    assert summary["backend"] == "numpy"
    assert summary["center"] == "circular"
    assert summary["shape"] == [24, 32]
    assert summary["weights"] == "uniform"
    assert summary["converged"] is True
    assert err != ""                   # the progress notes went to stderr


def test_quiet_silences_progress_but_not_info(scene, tmp_path, capsys):
    _, _, path = scene
    status, out, err = run([str(path), "-o", str(tmp_path / "out.npy"),
                            "--info", "-q"], capsys)
    assert status == 0
    assert err == ""
    assert json.loads(out)["seconds"] >= 0


def test_center_convention_explains_the_half_turn_between_methods(scene, tmp_path,
                                                                 capsys):
    """The offset convention, not the algorithm, decides the residual shift.

    Cycle-counting methods return ``phase + pi``, so ``--center none`` keeps that
    half turn while ``--center circular`` removes it and puts every method on the
    input's own offset.
    """
    _, _, path = scene
    outputs = {}
    for name in ("goldstein", "flynn"):
        for center in ("circular", "none"):
            target = tmp_path / ("%s-%s.npy" % (name, center))
            status, _, _ = run([str(path), "--method", name, "--center", center,
                                "-o", str(target), "-q"], capsys)
            assert status == 0
            outputs[(name, center)] = np.load(str(target))

    shift = outputs[("goldstein", "none")] - outputs[("goldstein", "circular")]
    assert np.abs(shift - np.pi).max() < 1e-3

    # With circular centering the two methods share one phase field: they agree
    # modulo a whole fringe everywhere.  They are not identical pixel by pixel,
    # because Goldstein and Flynn minimise different objectives and may resolve
    # the boundary differently; one corner pixel of this scene carries such a
    # branch difference.
    difference = outputs[("goldstein", "circular")] - outputs[("flynn", "circular")]
    assert np.abs(wrap_phase(difference)).max() < 1e-3


# --------------------------------------------------------------------------
# the +-pi tie-break inside center_circular
# --------------------------------------------------------------------------

def test_center_circular_pins_the_ambiguous_half_turn():
    """An offset of exactly pi must resolve the same way for every dtype.

    There the circular mean is a unit complex number on the negative real axis,
    whose imaginary part is only rounding noise, so ``np.angle`` answers +pi or
    -pi for the same image depending on the input dtype.  Both answers describe
    the same solution, but a comparison between two runs must not see a whole
    fringe of difference.
    """
    phase = make_synthetic(shape=(20, 24), noise=0.0, seed=5)[1]
    answers = []
    for dtype in (np.float32, np.float64):
        result = phase.astype(dtype) + np.array(np.pi, dtype=dtype)
        centered = center_circular(result, phase)
        answers.append(centered)
        residual = np.angle(np.exp(1j * (centered - phase)))
        assert np.abs(residual).max() < 1e-4      # congruent to the input

    assert np.abs(answers[0] - answers[1]).max() < 1e-4


def test_center_circular_removes_a_plain_offset():
    phase = np.linspace(0.0, 1.0, 60).reshape(6, 10)
    centered = center_circular(phase + 2.0, phase)
    assert np.abs(centered - phase).max() < 1e-12


def test_center_circular_ignores_non_finite_pixels():
    phase = np.zeros((4, 5))
    result = phase + 0.25
    result[0, 0] = np.nan
    centered = center_circular(result, phase)
    assert np.isnan(centered[0, 0])
    assert np.abs(centered[np.isfinite(centered)]).max() < 1e-12


def test_parser_defaults_match_the_documented_interface():
    args = build_parser().parse_args(["wrapped.npy"])
    assert args.method == "ls"
    assert args.center == "circular"
    assert args.quality == "min_gradient"
    assert args.backend == "auto"
    assert args.workers == -1
    assert args.max_iter == 100
    assert args.tol == 1e-8
    assert args.cost == "linear"
    assert args.dtype == "<f4"
    assert args.out_dtype is None
    assert args.order == "C"
    assert args.npz_key == "phase"
    assert args.invalid_fill is None
    assert args.shape is None
    parsed = build_parser().parse_args(["wrapped.raw", "--shape", "4", "5"])
    assert parsed.shape == [4, 5]


# --------------------------------------------------------------------------
# minimum-cost flow is wired like any other method
# --------------------------------------------------------------------------

def test_mcf_agrees_with_the_answer_key(scene, tmp_path, capsys):
    """The command line reaches the solver and returns a consistent surface."""
    _, wrapped, path = scene
    out_path = tmp_path / "mcf.npy"
    status, _, err = run([str(path), "--method", "mcf", "-o", str(out_path)],
                         capsys)
    assert status == 0, err
    result = np.load(str(out_path))
    # The contract of any unwrapping is that it wraps back to its input; the
    # scene's own noise is what separates the answer key from the input.
    assert np.abs(wrap_phase(result - wrapped)).max() < 1e-5


def test_mcf_reports_the_network_it_solved(scene, tmp_path, capsys):
    _, _, path = scene
    status, out, err = run([str(path), "--method", "mcf", "-o",
                            str(tmp_path / "mcf.npy"), "--info"], capsys)
    assert status == 0, err
    summary = json.loads(out)
    assert summary["method"] == "mcf"
    assert summary["cost"] == "linear"
    assert summary["nodes"] == 23 * 31 + 1        # one node per valid 2x2 cell
    assert summary["edges"] == 23 * 32 + 24 * 31  # one arc per pixel edge
    assert summary["total_cost"] == 0             # a scene this smooth is exact


def test_mcf_uses_a_weight_map_instead_of_dropping_it(scene, tmp_path, capsys):
    """``mcf`` is in WEIGHT_METHODS, so coherence must reach the solver."""
    _, wrapped, path = scene
    np.save(str(tmp_path / "weight.npy"), np.ones(wrapped.shape))
    status, _, err = run([str(path), "--method", "mcf",
                          "--weight", str(tmp_path / "weight.npy"),
                          "-o", str(tmp_path / "out.npy")], capsys)
    assert status == 0, err
    assert "ignores" not in err


@pytest.mark.parametrize("cost", ["linear", "quadratic"])
def test_mcf_cost_choice_is_accepted(cost, scene, tmp_path, capsys):
    _, _, path = scene
    status, out, err = run([str(path), "--method", "mcf", "--cost", cost,
                            "-o", str(tmp_path / "out.npy"), "--info"], capsys)
    assert status == 0, err
    assert json.loads(out)["cost"] == cost


def test_mcf_rejects_an_unknown_cost(scene, capsys):
    _, _, path = scene
    status, _, err = run([str(path), "--method", "mcf", "--cost", "cubic"],
                         capsys)
    assert status == 1
    assert "invalid choice: 'cubic'" in err


# --------------------------------------------------------------------------
# multigrid is wired like any other method
# --------------------------------------------------------------------------

def test_multigrid_agrees_with_the_answer_key(scene, tmp_path, capsys):
    """The command line reaches the hierarchy solver and returns a surface."""
    _, wrapped, path = scene
    out_path = tmp_path / "multigrid.npy"
    status, _, err = run([str(path), "--method", "multigrid",
                          "-o", str(out_path)], capsys)
    assert status == 0, err
    result = np.load(str(out_path))
    assert np.abs(wrap_phase(result - wrapped)).max() < 1e-5


def test_multigrid_reports_the_hierarchy_it_solved(scene, tmp_path, capsys):
    _, _, path = scene
    status, out, err = run([str(path), "--method", "multigrid", "-o",
                            str(tmp_path / "multigrid.npy"), "--info"], capsys)
    assert status == 0, err
    summary = json.loads(out)
    assert summary["method"] == "multigrid"
    assert summary["backend"] == "numpy"
    assert summary["weights"] == "uniform"
    assert summary["converged"] is True
    assert summary["cycles"] == len(summary["residuals"])
    assert summary["coarsest"] == [2, 2]        # 24x32, halved four times
    assert summary["levels"] == 5
    assert summary["relative_residual"] < 1e-8


def test_multigrid_solves_the_same_equations_as_ls(scene, tmp_path, capsys):
    """Both methods minimise the same weighted least-squares objective."""
    _, wrapped, path = scene
    np.save(str(tmp_path / "weight.npy"), np.ones(wrapped.shape))
    outputs = []
    for method in ("ls", "multigrid"):
        out_path = tmp_path / ("%s.npy" % method)
        status, _, err = run([str(path), "--method", method, "--weight",
                              str(tmp_path / "weight.npy"),
                              "-o", str(out_path), "--tol", "1e-12"], capsys)
        assert status == 0, err
        outputs.append(np.load(str(out_path)))
    # Two independent solvers, so they agree to their convergence tolerance
    # rather than bit for bit.
    assert np.abs(wrap_phase(outputs[0] - outputs[1])).max() < 1e-4


def test_multigrid_uses_a_mask_instead_of_dropping_it(scene, tmp_path, capsys):
    """``multigrid`` is in WEIGHT_METHODS, so a mask must reach the solver."""
    _, wrapped, path = scene
    mask = np.ones(wrapped.shape, dtype=bool)
    mask[:, :4] = False
    np.save(str(tmp_path / "mask.npy"), mask)
    out_path = tmp_path / "out.npy"
    status, out, err = run([str(path), "--method", "multigrid", "--mask",
                            str(tmp_path / "mask.npy"), "-o", str(out_path),
                            "--info"], capsys)
    assert status == 0, err
    assert json.loads(out)["weights"] == "mask"
    result = np.load(str(out_path))
    assert np.isnan(result[:, :4]).all()
    assert np.isfinite(result[:, 4:]).all()


def test_multigrid_accepts_a_weight_map(scene, tmp_path, capsys):
    _, wrapped, path = scene
    np.save(str(tmp_path / "weight.npy"), np.ones(wrapped.shape))
    status, out, err = run([str(path), "--method", "multigrid", "--weight",
                            str(tmp_path / "weight.npy"), "-o",
                            str(tmp_path / "out.npy"), "--info"], capsys)
    assert status == 0, err
    assert "ignores" not in err
    assert json.loads(out)["weights"] == "custom"


def test_multigrid_max_iter_caps_the_cycles_and_says_so(scene, tmp_path, capsys):
    """--max-iter is the cycle budget here, and a starved run is reported."""
    _, _, path = scene
    status, out, err = run([str(path), "--method", "multigrid", "-o",
                            str(tmp_path / "out.npy"), "--max-iter", "1",
                            "--info"], capsys)
    assert status == 0, err
    summary = json.loads(out)
    assert summary["cycles"] == 1
    assert summary["converged"] is False
    assert "stopped after 1 V-cycles" in err


# --------------------------------------------------------------------------
# stacks: .npy/.npz may hold more than two dimensions
# --------------------------------------------------------------------------

@pytest.fixture
def cube(tmp_path):
    """A small deterministic stack of images, and the ramp they came from."""
    zz, yy, xx = np.meshgrid(np.arange(4), np.arange(12), np.arange(16),
                             indexing="ij")
    truth = 0.35 * xx + 0.25 * yy + 0.6 * zz
    path = tmp_path / "cube.npy"
    np.save(str(path), wrap_phase(truth).astype(np.float32))
    return truth, path


def test_ls_unwraps_a_stack_jointly(cube, tmp_path, capsys):
    """A cube is one solve over every axis, and --info reports the rank."""
    truth, path = cube
    target = tmp_path / "out.npy"
    status, out, err = run([str(path), "--method", "ls", "--info",
                            "-o", str(target), "-q"], capsys)
    assert status == 0, err
    report = json.loads(out)
    assert report["shape"] == [4, 12, 16]
    assert report["method"] == "ls"
    result = np.load(str(target))
    assert result.shape == truth.shape
    # The steps are all below pi, so the answer is exact up to the one global
    # multiple of 2 pi that no unwrapper can pin down.
    assert np.abs(wrap_phase(result - truth)).max() < 1e-5


def test_reliability_unwraps_a_stack(cube, tmp_path, capsys):
    truth, path = cube
    target = tmp_path / "out.npy"
    status, out, err = run([str(path), "--method", "reliability", "--info",
                            "-o", str(target), "-q"], capsys)
    assert status == 0, err
    report = json.loads(out)
    assert report["shape"] == [4, 12, 16]
    result = np.load(str(target))
    assert np.abs(wrap_phase(result - truth)).max() < 1e-5


def test_multigrid_unwraps_a_stack(cube, tmp_path, capsys):
    truth, path = cube
    target = tmp_path / "out.npy"
    status, out, err = run([str(path), "--method", "multigrid", "--info",
                            "-o", str(target), "-q"], capsys)
    assert status == 0, err
    report = json.loads(out)
    assert report["shape"] == [4, 12, 16]
    assert [int(size) for size in report["coarsest"]] == [2, 6, 8]  # halved once
    result = np.load(str(target))
    assert np.abs(wrap_phase(result - truth)).max() < 1e-5


def test_two_dimensional_method_refuses_a_stack(cube, capsys):
    _, path = cube
    status, _, err = run([str(path), "--method", "goldstein"], capsys)
    assert status == 1
    assert "unwraps a two-dimensional image" in err
    assert "has shape 4x12x16" in err
    assert ", ".join(ND_METHODS) in err


def test_shape_with_three_numbers_is_an_error(cube, tmp_path, capsys):
    """--shape describes one raw image, so it never takes three numbers."""
    _, path = cube
    status, _, err = run([str(path), "--shape", "4", "12", "16",
                          "-o", str(tmp_path / "out.npy")], capsys)
    assert status == 1
    assert "--shape takes two numbers" in err
    assert "got 3 numbers" in err


def test_weight_for_a_quality_method_prints_a_note(scene, tmp_path, capsys):
    """A dropped weight is announced rather than silently ignored."""
    _, wrapped, path = scene
    np.save(str(tmp_path / "weight.npy"), np.ones(wrapped.shape))
    status, _, err = run([str(path), "--method", "goldstein",
                          "--weight", str(tmp_path / "weight.npy"),
                          "-o", str(tmp_path / "out.npy")], capsys)
    assert status == 0, err
    assert "ignores" in err
    assert "--weight" in err
    # -q hides the note; the run is identical either way.
    status, _, quiet = run([str(path), "--method", "goldstein",
                           "--weight", str(tmp_path / "weight.npy"),
                           "-o", str(tmp_path / "out.npy"), "-q"], capsys)
    assert status == 0
    assert quiet == ""


def test_a_stack_reports_the_engine_that_ran(cube, tmp_path, capsys):
    """Compiled kernels are two-dimensional, so a stack runs the general path.

    The report must say what really ran, because a claim of "numba" on an array
    Numba never touched is a wrong answer to a fair question.
    """
    _, path = cube
    status, out, err = run([str(path), "--method", "ls", "--backend", "numpy",
                            "--info", "-o", str(tmp_path / "out.npy")], capsys)
    assert status == 0, err
    assert json.loads(out)["backend"] == "numpy"
    assert "backend numpy" in err

    pytest.importorskip("numba")
    status, out, err = run([str(path), "--method", "ls", "--backend", "numba",
                            "--info", "-o", str(tmp_path / "out.npy")], capsys)
    assert status == 0, err
    assert json.loads(out)["backend"] == "numpy"
    assert "backend numpy" in err


def test_a_two_dimensional_image_keeps_its_compiled_backend(scene, tmp_path,
                                                            capsys):
    """The same request on an image does reach the compiled kernel."""
    _, _, path = scene
    pytest.importorskip("numba")
    status, out, err = run([str(path), "--method", "ls", "--backend", "numba",
                            "--info", "-o", str(tmp_path / "out.npy")], capsys)
    assert status == 0, err
    assert json.loads(out)["backend"] == "numba"
    assert "backend numba" in err


def test_a_stack_cannot_be_written_as_a_raw_raster(cube, tmp_path, capsys):
    _, path = cube
    status, _, err = run([str(path), "--method", "ls", "--shape", "4", "12",
                          "-o", str(tmp_path / "out.raw")], capsys)
    assert status == 1
    assert "headerless raster holds one two-dimensional image" in err
    assert "Write a cube to .npy or .npz" in err


@raster_only
def test_a_stack_cannot_be_written_as_a_raster(cube, tmp_path, capsys):
    _, path = cube
    status, _, err = run([str(path), "--method", "ls",
                          "-o", str(tmp_path / "out.tif")], capsys)
    assert status == 1
    assert "georeferenced raster holds one two-dimensional band" in err
    assert "Write a cube to .npy or .npz" in err


def test_a_stack_survives_an_npz_round_trip(cube, tmp_path, capsys):
    """The stack formats are the ones that record their own shape."""
    _, path = cube
    target = tmp_path / "out.npz"
    status, _, err = run([str(path), "--method", "reliability",
                          "-o", str(target), "-q"], capsys)
    assert status == 0, err
    with np.load(str(target)) as archive:
        result = archive["phase"]
    assert result.shape == (4, 12, 16)
