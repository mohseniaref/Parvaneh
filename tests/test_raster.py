"""Tests for :mod:`parvaneh.raster`, the GeoTIFF and GDAL bridge.

Everything here needs a GDAL binding, so the whole module is skipped when
neither ``rasterio`` nor ``osgeo.gdal`` is importable.  That is the intended
user experience as well: the unwrapping algorithms themselves need only NumPy,
and the command line falls back to ``.npy``, ``.npz`` and headerless rasters.

Two environment details shape these tests.

* The CRS is written as a full WKT literal rather than an authority code such
  as ``"EPSG:32633"``.  Resolving an authority code needs a working PROJ
  database, which some installations lack, while parsing WKT text does not.
* GDAL normalises WKT the first time it writes it (it expands the datum name
  and adds authority codes), so a CRS is compared semantically with
  ``CRS.from_wkt(...)`` rather than by string equality.
"""

import numpy as np
import pytest
from parvaneh import wrap_phase
from parvaneh.raster import (RASTER_SUFFIXES, NoRasterBackendError, RasterError,
                             RasterMeta, raster_backend, read_raster,
                             write_raster)

pytestmark = pytest.mark.skipif(
    raster_backend() is None,
    reason="no GDAL binding: neither rasterio nor osgeo.gdal is installed")

#: WGS 84 / UTM zone 33N, spelled out.  See the module docstring for why the
#: tests avoid the shorter ``"EPSG:32633"`` form.
UTM33N = (
    'PROJCS["WGS 84 / UTM zone 33N",GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],'
    'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
    'PARAMETER["false_northing",0],UNIT["metre",1]]')

#: A north-up 30 m grid starting at an arbitrary point inside zone 33N.
TRANSFORM = (30.0, 0.0, 412345.0, 0.0, -30.0, 4645110.0)


def same_crs(left, right):
    """Compare two WKT strings through GDAL, which normalises their spelling."""
    import rasterio
    return rasterio.crs.CRS.from_wkt(left) == rasterio.crs.CRS.from_wkt(right)


@pytest.fixture
def phase():
    """A wrapped ramp: smooth, so unwrapping it is unambiguous."""
    rows, cols = 24, 32
    yy, xx = np.mgrid[0:rows, 0:cols]
    return wrap_phase(0.30 * xx + 0.17 * yy).astype(np.float32)


def test_round_trip_preserves_values_and_grid(tmp_path, phase):
    """Write and read back: pixels, geotransform, CRS and dtype all survive."""
    path = tmp_path / "wrapped.tif"
    write_raster(path, phase, crs=UTM33N, transform=TRANSFORM, nodata=np.nan)

    values, meta = read_raster(path)

    assert values.shape == phase.shape
    assert np.allclose(values, phase, atol=1e-6)
    assert meta.driver == "GTiff"
    assert meta.dtype == "float32"
    assert meta.shape == phase.shape
    assert meta.georeferenced
    assert meta.transform == TRANSFORM
    assert same_crs(meta.crs, UTM33N)


def test_plain_array_reports_no_georeferencing(tmp_path, phase):
    """A GeoTIFF with no transform and no CRS must not invent either."""
    path = tmp_path / "plain.tif"
    write_raster(path, phase, nodata=None)

    _, meta = read_raster(path)

    assert meta.transform is None
    assert meta.crs is None
    assert not meta.georeferenced


def test_nodata_becomes_nan(tmp_path, phase):
    """The file's nodata pixels are validity declarations, not measurements."""
    marked = phase.copy()
    marked[:4, :5] = -9999.0
    path = tmp_path / "marked.tif"
    write_raster(path, marked, nodata=-9999.0)

    values, meta = read_raster(path)

    assert meta.nodata == -9999.0
    assert np.isnan(values[:4, :5]).all()
    assert np.isfinite(values[4:, 5:]).all()


def test_nodata_fill_none_keeps_the_sentinel(tmp_path, phase):
    """``nodata_fill=None`` is the escape hatch for inspecting raw values."""
    marked = phase.copy()
    marked[0, 0] = -9999.0
    path = tmp_path / "marked.tif"
    write_raster(path, marked, nodata=-9999.0)

    values, _ = read_raster(path, nodata_fill=None)

    assert values[0, 0] == pytest.approx(-9999.0)


def test_zero_is_not_treated_as_nodata(tmp_path, phase):
    """Only the declared sentinel means "no data"; a real zero is a value."""
    marked = phase.copy()
    marked[0, 0] = 0.0
    path = tmp_path / "zero.tif"
    write_raster(path, marked, nodata=-9999.0)

    values, _ = read_raster(path, nodata_fill=None)

    assert values[0, 0] == pytest.approx(0.0)


def test_integer_output_needs_a_finite_sentinel(tmp_path):
    """An integer band cannot hold NaN, so it must be told what to write."""
    values = np.zeros((8, 8))
    values[0, 0] = np.nan

    with pytest.raises(RasterError, match="no nodata value was given"):
        write_raster(tmp_path / "int.tif", values, dtype="int16", nodata=None)

    with pytest.raises(RasterError, match="cannot store"):
        write_raster(tmp_path / "int.tif", values, dtype="int16", nodata=np.nan)


def test_integer_output_substitutes_the_sentinel(tmp_path):
    """Casting NaN to an integer is undefined, so the sentinel goes in instead."""
    values = np.arange(64, dtype="float64").reshape(8, 8)
    values[0, 0] = np.nan
    path = tmp_path / "int.tif"
    write_raster(path, values, dtype="int16", nodata=-9999)

    back, meta = read_raster(path, nodata_fill=None)

    assert meta.dtype == "int16"
    assert meta.nodata == -9999
    assert back[0, 0] == -9999
    assert np.array_equal(back[1:, :], values[1:, :])


def test_integer_raster_round_trips_with_its_sentinel(tmp_path):
    values = np.arange(64, dtype="int16").reshape(8, 8)
    values[0, 0] = -9999
    path = tmp_path / "int.tif"
    write_raster(path, values, dtype="int16", nodata=-9999)

    back, meta = read_raster(path, nodata_fill=None)

    assert meta.dtype == "int16"
    assert meta.nodata == -9999
    assert np.array_equal(back, values)


def test_like_reuses_the_reference_grid(tmp_path, phase):
    """``like=`` is how an unwrapped result inherits the input's geometry."""
    source = tmp_path / "wrapped.tif"
    write_raster(source, phase, crs=UTM33N, transform=TRANSFORM, nodata=np.nan)
    _, meta = read_raster(source)

    target = tmp_path / "unwrapped.tif"
    write_raster(target, phase + 1.0, like=meta, dtype=meta.dtype)
    _, written = read_raster(target)

    assert written.transform == meta.transform
    assert same_crs(written.crs, meta.crs)
    assert written.dtype == "float32"


def test_like_rejects_a_different_shape(tmp_path, phase):
    """A geotransform only fits the grid it was computed for."""
    source = tmp_path / "wrapped.tif"
    write_raster(source, phase, crs=UTM33N, transform=TRANSFORM)
    _, meta = read_raster(source)

    with pytest.raises(RasterError, match="geotransform only fits"):
        write_raster(tmp_path / "other.tif", phase[:-1, :-1], like=meta)


def test_three_dimensional_input_is_refused(tmp_path):
    with pytest.raises(RasterError, match="must be two-dimensional"):
        write_raster(tmp_path / "stack.tif", np.zeros((2, 4, 4)))


def test_missing_band_names_the_band_count(tmp_path, phase):
    path = tmp_path / "wrapped.tif"
    write_raster(path, phase)

    with pytest.raises(RasterError, match="band 2 does not exist"):
        read_raster(path, band=2)


def test_band_number_must_be_one_or_greater(tmp_path, phase):
    path = tmp_path / "wrapped.tif"
    write_raster(path, phase)

    with pytest.raises(RasterError, match="band must be 1 or greater"):
        read_raster(path, band=0)


def test_multiband_file_can_select_a_band(tmp_path, phase):
    """A phase band next to a coherence band is the ordinary InSAR layout.

    ``write_raster`` only creates single-band files, so the two-band fixture
    is built with the backend directly.
    """
    rasterio = pytest.importorskip("rasterio")
    path = tmp_path / "both.tif"
    stack = np.stack([phase, phase * 2.0])
    with rasterio.open(str(path), "w", driver="GTiff",
                       height=phase.shape[0], width=phase.shape[1],
                       count=2, dtype="float32", crs=UTM33N,
                       transform=rasterio.Affine(*TRANSFORM)) as dst:
        dst.write(stack)

    second, meta = read_raster(path, band=2)

    assert meta.count == 2
    assert meta.band == 2
    assert np.allclose(second, phase * 2.0, atol=1e-5)
    assert meta.transform == TRANSFORM
    assert same_crs(meta.crs, UTM33N)


def test_missing_file_is_a_clean_error(tmp_path):
    with pytest.raises(RasterError, match="raster file not found"):
        read_raster(tmp_path / "nope.tif")


def test_corrupt_file_is_a_raster_error_not_a_backend_error(tmp_path):
    """Backend exception types must not leak into user-facing messages."""
    path = tmp_path / "garbage.tif"
    path.write_bytes(b"this is not a TIFF")
    with pytest.raises(RasterError) as caught:
        read_raster(path)
    assert not isinstance(caught.value, NoRasterBackendError)
    assert "could not read" in str(caught.value)


def test_writable_suffixes_produce_a_file(tmp_path, phase):
    """The suffixes the command line recognises as writable really are."""
    for suffix in (".tif", ".tiff", ".gtif", ".img"):
        path = tmp_path / ("phase" + suffix)
        write_raster(path, phase, transform=TRANSFORM, crs=UTM33N)
        values, meta = read_raster(path)
        assert values.shape == phase.shape, suffix
        assert meta.transform == TRANSFORM, suffix


def test_read_only_suffix_is_refused_for_writing(tmp_path, phase):
    """A format GDAL can read but this package cannot create must say so."""
    assert ".vrt" in RASTER_SUFFIXES
    with pytest.raises(RasterError, match="writing '.vrt' files is not supported"):
        write_raster(tmp_path / "phase.vrt", phase)


def test_non_raster_suffix_defaults_to_geotiff(tmp_path, phase):
    path = tmp_path / "phase.dat"
    write_raster(path, phase)
    _, meta = read_raster(path)
    assert meta.driver == "GTiff"


def test_metadata_is_immutable_and_comparable(phase, tmp_path):
    meta = RasterMeta(path="x.tif", driver="GTiff", height=4, width=5,
                      count=1, dtype="float32")
    assert meta.shape == (4, 5)
    assert not meta.georeferenced
    with pytest.raises(Exception):
        meta.driver = "HFA"


def test_backend_name_matches_what_is_importable():
    name = raster_backend()
    assert name in ("rasterio", "gdal")
    import importlib
    if name == "rasterio":
        assert importlib.import_module("rasterio") is not None
    else:
        assert importlib.import_module("osgeo.gdal") is not None


def test_without_a_backend_the_hint_explains_how_to_install(monkeypatch, tmp_path,
                                                            phase):
    """The no-GDAL path is a normal, supported state, so it needs a real hint."""
    import parvaneh.raster as raster

    monkeypatch.setattr(raster, "_import_rasterio", lambda: None)
    monkeypatch.setattr(raster, "_import_gdal", lambda: None)

    assert raster.raster_backend() is None

    source = tmp_path / "any.tif"
    source.write_bytes(b"\0" * 8)   # so the failure is the backend, not the path
    for call in (lambda: read_raster(source),
                 lambda: write_raster(tmp_path / "any.tif", phase)):
        with pytest.raises(NoRasterBackendError) as caught:
            call()
        message = str(caught.value)
        assert "neither rasterio nor osgeo.gdal is installed" in message
        assert "pip install 'parvaneh[geo]'" in message
        assert "python3-gdal" in message
        assert ".npy/.npz" in message


def test_identity_transform_reads_as_ungoereferenced(tmp_path, phase):
    """GDAL answers with an identity matrix for a file it cannot locate."""
    path = tmp_path / "identity.tif"
    write_raster(path, phase, transform=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))

    _, meta = read_raster(path)

    assert meta.transform is None
    assert not meta.georeferenced
