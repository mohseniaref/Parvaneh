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
from parvaneh.cli.unwrap import (METHOD_BACKENDS, METHODS, build_parser,
                                 center_circular)

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
    assert args.dtype == "<f4"
    assert args.out_dtype is None
    assert args.order == "C"
    assert args.npz_key == "phase"
    assert args.invalid_fill is None
