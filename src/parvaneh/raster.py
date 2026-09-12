"""Georeferenced raster input/output through GDAL.

Wrapped phase in a real InSAR workflow almost never arrives as a bare array of
numbers.  It arrives as a *raster file*: a rectangular grid of pixels plus a
header that records where on the Earth those pixels are.  The usual format is
GeoTIFF, and the usual library for reading it is GDAL.

This module is the bridge between those files and the plain arrays the rest of
the package works with.  It is deliberately separate from :mod:`parvaneh.io`:
``parvaneh.io`` reads *headerless* rasters whose shape and dtype the caller has
to state, while this module reads *self-describing* files whose header carries
the shape, the dtype, the georeferencing, and the "no data" convention.

GDAL is optional
================

The unwrapping algorithms need only NumPy.  Georeferenced rasters need a GDAL
binding, and none is required by a plain ``pip install parvaneh``.  Two
bindings are understood, and whichever is importable is used:

``rasterio``
    The default choice: ``pip install parvaneh[geo]``.  It is a thin,
    well-maintained Python layer over GDAL, and it bundles a GDAL build.
``osgeo.gdal``
    The official GDAL Python bindings, used when ``rasterio`` is absent.  This
    is what a system package such as ``python3-gdal`` provides.

When neither is importable, :func:`read_raster` and :func:`write_raster` raise
:class:`NoRasterBackendError` with instructions.  Headerless input keeps
working either way, because it never touches GDAL.

What the header contains
========================

Three pieces of metadata matter for phase unwrapping, and all three are kept:

``transform``
    The affine geotransform, six numbers :math:`(a, b, c, d, e, f)` that map
    pixel coordinates to map coordinates:
    :math:`x = a \\cdot \\mathrm{col} + b \\cdot \\mathrm{row} + c` and
    :math:`y = d \\cdot \\mathrm{col} + e \\cdot \\mathrm{row} + f`.  For the
    usual north-up raster :math:`a` is the positive pixel width, :math:`e` is
    the negative pixel height, and :math:`b = d = 0`.  Unwrapping does not
    change the grid, so the transform is copied unchanged to the output.
``crs``
    The coordinate reference system, stored as WKT text.  It says what the
    map coordinates mean (for example UTM zone 33N on WGS 84).
``nodata``
    The value the file uses for pixels that hold no measurement.  Those pixels
    are a *validity declaration*, not data: this module turns them into
    ``NaN``, which is the value the package already uses for "invalid".

Units are whatever the band stores.  InSAR phase rasters hold radians, but a
"wrapped fringe" raster that counts cycles needs a scale factor; see the
command-line documentation for ``--scale``.
"""

import contextlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["RASTER_SUFFIXES", "RasterError", "RasterMeta",
           "NoRasterBackendError", "raster_backend", "read_raster",
           "write_raster"]


#: File extensions the command line treats as georeferenced rasters.  GDAL,
#: not this tuple, decides whether a file really is a raster, so a suffix that
#: is missing here can still be opened by calling the functions directly.
RASTER_SUFFIXES = (
    ".tif", ".tiff", ".gtif", ".gtiff", ".cog", ".img", ".vrt", ".ers",
    ".hgt", ".grd", ".bil", ".bsq", ".bip", ".flt", ".dem", ".dt0", ".dt1",
    ".dt2", ".asc", ".nc", ".hdf", ".h5", ".he5", ".jp2", ".png", ".bmp",
)

#: GDAL driver to use when creating a file, chosen from the output suffix.
#: Suffixes that name a readable but non-writable format are reported instead
#: of silently producing a GeoTIFF under a misleading name.
_WRITE_DRIVERS = {
    ".tif": "GTiff", ".tiff": "GTiff", ".gtif": "GTiff", ".gtiff": "GTiff",
    ".cog": "COG", ".img": "HFA",
}

#: GDAL's geotransform is ``(c, a, b, f, d, e)``; ours is ``(a, b, c, d, e, f)``.
#: ``(1, 0, 0, 0, 1, 0)`` is the identity, which GDAL reports for a raster it
#: cannot locate (a plain array saved as a GeoTIFF, for instance), and
#: ``(0, 0, 0, 0, 0, 0)`` is what an unset geotransform comes back as.  Both
#: mean "no georeferencing", so both are reported as ``None``.
_DEFAULT_TRANSFORMS = ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                       (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

#: GDAL type names, which are needed when creating a file through osgeo.gdal.
#: ``GDALGetDataTypeName`` spells them differently from NumPy, and a few are
#: actively misleading: NumPy reads ``"byte"`` as ``int8``, not ``uint8``.
_NUMPY_TO_GDAL = {
    "uint8": "Byte", "uint16": "UInt16", "int16": "Int16",
    "uint32": "UInt32", "int32": "Int32", "float32": "Float32",
    "float64": "Float64",
}

_GDAL_TO_NUMPY = {name: numpy_name for numpy_name, name in _NUMPY_TO_GDAL.items()}

_INSTALL_HINT = (
    "this file needs a GDAL binding, and neither rasterio nor osgeo.gdal is "
    "installed.\n"
    "Install the optional extra, which pulls in rasterio:\n"
    "    pip install 'parvaneh[geo]'\n"
    "or the official GDAL bindings from your package manager "
    "(python3-gdal).\n"
    "Headerless data does not need GDAL: use .npy/.npz, or a raw raster with "
    "--shape and --dtype."
)


class RasterError(RuntimeError):
    """Raised when a raster cannot be read, understood or written."""


class NoRasterBackendError(RasterError):
    """Raised when a georeferenced raster is used but no GDAL binding exists."""


@dataclass(frozen=True)
class RasterMeta:
    """What a raster file says about itself.

    Instances are immutable; use :func:`dataclasses.replace` to derive one.

    Attributes
    ----------
    path : str
        File the metadata came from.
    driver : str
        GDAL driver short name, for example ``"GTiff"``.
    height, width : int
        Raster size in pixels, ``rows`` and ``columns``.
    count : int
        Number of bands in the file, whether or not all of them were read.
    dtype : str
        NumPy name of the dtype of the band that was read, for example
        ``"float32"``.
    band : int
        One-based index of the band that was read.
    transform : tuple or None
        Affine geotransform ``(a, b, c, d, e, f)``.  ``None`` when the file
        carries no usable georeferencing.
    crs : str or None
        Coordinate reference system as WKT text.
    nodata : float or None
        The value the file declares as "no data".
    """

    path: str
    driver: str
    height: int
    width: int
    count: int
    dtype: str
    band: int = 1
    transform: tuple = None
    crs: str = None
    nodata: float = None

    @property
    def shape(self):
        """``(rows, columns)`` of the raster."""
        return (self.height, self.width)

    @property
    def georeferenced(self):
        """True when the raster has a coordinate reference system or transform."""
        return self.crs is not None or self.transform is not None


def _import_rasterio():
    try:
        import rasterio
    except ImportError:
        return None
    return rasterio


def _import_gdal():
    try:
        from osgeo import gdal
    except ImportError:
        return None
    return gdal


def _backend():
    """Return ``(name, rasterio_module, gdal_module)`` for the binding in use."""
    rasterio = _import_rasterio()
    if rasterio is not None:
        return "rasterio", rasterio, None
    gdal = _import_gdal()
    if gdal is not None:
        return "gdal", None, gdal
    raise NoRasterBackendError(_INSTALL_HINT)


def raster_backend():
    """Name of the GDAL binding that will be used, or ``None`` if there is none.

    The command line reports this through ``parvaneh unwrap --methods``; tests
    use it to skip when no binding is installed.
    """
    if _import_rasterio() is not None:
        return "rasterio"
    if _import_gdal() is not None:
        return "gdal"
    return None


def _band_number(band, count=None):
    """Validate a one-based band index, against ``count`` when it is known."""
    if isinstance(band, bool) or not isinstance(band, (int, np.integer)):
        raise RasterError("band must be a whole number, got %r" % (band,))
    band = int(band)
    if band < 1:
        raise RasterError("band must be 1 or greater, got %d" % band)
    if count is not None and band > count:
        raise RasterError("band %d does not exist: the file has %d band%s"
                          % (band, count, "" if count == 1 else "s"))
    return band


def _transform_or_none(values):
    """Drop GDAL's placeholder geotransforms so that they report as "unknown"."""
    if values is None:
        return None
    numbers = tuple(float(value) for value in values[:6])
    if numbers in _DEFAULT_TRANSFORMS:
        return None
    return numbers


def _as_float(value):
    if value is None:
        return None
    return float(value)


def _fill_nodata(values, nodata, fill):
    """Replace the file's nodata value with ``fill`` (``NaN`` by default)."""
    if nodata is None or fill is None:
        return values
    if np.isnan(nodata):
        invalid = np.isnan(values)
    else:
        invalid = values == nodata
    if not invalid.any():
        return values
    filled = np.asarray(values, dtype=np.float64).copy()
    filled[invalid] = fill
    return filled


def _backend_call(action, path, function, *args):
    """Run a backend call, reporting any failure as a :class:`RasterError`.

    GDAL and rasterio raise their own exception types for a broken file, a
    missing driver, or a bad nodata value.  Leaking those would force callers to
    know which backend happens to be installed, so they are folded into one
    error type with the original message kept in the text.
    """
    try:
        return function(*args)
    except RasterError:
        raise
    except Exception as exc:
        raise RasterError("could not %s %s: %s: %s"
                          % (action, path, type(exc).__name__, exc)) from None


@contextlib.contextmanager
def _quiet_georeference_warnings(rasterio):
    """Hide rasterio's complaint about a raster with no geotransform.

    A plain TIFF of wrapped phase has no georeferencing, and that is a normal,
    supported case here: :class:`RasterMeta` reports it honestly as
    ``transform=None``, ``crs=None``.  rasterio warns about it on every open and
    every write, which would drown the command line's own progress lines.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            getattr(getattr(rasterio, "errors", None),
                    "NotGeoreferencedWarning", Warning))
        yield


def _read_rasterio(rasterio, path, band):
    with _quiet_georeference_warnings(rasterio):
        with rasterio.open(str(path)) as dataset:
            count = int(dataset.count)
            _band_number(band, count)
            values = np.asarray(dataset.read(band))
            meta = RasterMeta(
                path=str(path),
                driver=dataset.driver,
                height=int(dataset.height),
                width=int(dataset.width),
                count=count,
                dtype=np.dtype(dataset.dtypes[band - 1]).name,
                band=band,
                transform=_transform_or_none(tuple(dataset.transform)),
                crs=dataset.crs.to_wkt() if dataset.crs is not None else None,
                nodata=_as_float(dataset.nodata),
            )
    return values, meta


def _read_gdal(gdal, path, band):
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise RasterError("GDAL could not open %s as a raster" % path)
    try:
        count = int(dataset.RasterCount)
        _band_number(band, count)
        source = dataset.GetRasterBand(band)
        values = np.asarray(source.ReadAsArray())
        rows, cols = int(source.YSize), int(source.XSize)
        if values.shape != (rows, cols):
            values = values.reshape((rows, cols))
        geotransform = dataset.GetGeoTransform()
        if geotransform is None:
            transform = None
        else:
            transform = _transform_or_none((geotransform[1], geotransform[2],
                                            geotransform[0], geotransform[4],
                                            geotransform[5], geotransform[3]))
        gdal_name = gdal.GetDataTypeName(source.DataType)
        if gdal_name not in _GDAL_TO_NUMPY:
            raise RasterError("unsupported GDAL data type %r in %s"
                              % (gdal_name, path))
        meta = RasterMeta(
            path=str(path),
            driver=dataset.GetDriver().ShortName,
            height=rows,
            width=cols,
            count=count,
            dtype=_GDAL_TO_NUMPY[gdal_name],
            band=band,
            transform=transform,
            crs=dataset.GetProjection() or None,
            nodata=_as_float(source.GetNoDataValue()),
        )
    finally:
        # Closing the GDAL dataset is done by dropping the last reference.
        dataset = None
    return values, meta


def read_raster(path, band=1, *, dtype=np.float64, nodata_fill=np.nan):
    """Read one band of a georeferenced raster.

    Parameters
    ----------
    path : str or pathlib.Path
        Raster file.  GeoTIFF is the common case; anything GDAL can open with
        a two-dimensional first band will do.
    band : int
        One-based band index.  Use it for a file that keeps several images,
        such as a phase band next to a coherence band.
    dtype : numpy dtype
        Dtype of the returned array (default ``float64``).  Phase is computed
        in double precision, so there is rarely a reason to change this.
    nodata_fill : float or None
        Value written where the file says "no data" (default ``NaN``).  Pass
        ``None`` to leave the sentinel values untouched, which is only useful
        for inspecting a file.

    Returns
    -------
    values : numpy.ndarray
        ``(rows, columns)`` array of the band's values, with nodata replaced.
    meta : RasterMeta
        Driver, size, dtype, geotransform, CRS and nodata of the file.

    Raises
    ------
    NoRasterBackendError
        Neither ``rasterio`` nor ``osgeo.gdal`` is installed.
    RasterError
        The file is missing, cannot be opened, or has no such band.

    Examples
    --------
    The returned metadata is what makes an unwrapped result reusable, because
    it can be handed straight back to :func:`write_raster`::

        phase, meta = read_raster("wrapped.tif")
        write_raster("unwrapped.tif", unwrap(phase), like=meta, dtype=meta.dtype)
    """
    source = Path(path)
    if not source.exists():
        raise RasterError("raster file not found: %s" % source)
    band = _band_number(band)
    _, rasterio, gdal = _backend()
    if rasterio is not None:
        values, meta = _backend_call("read", source, _read_rasterio,
                                     rasterio, source, band)
    else:
        values, meta = _backend_call("read", source, _read_gdal,
                                     gdal, source, band)
    values = _fill_nodata(values, meta.nodata, nodata_fill)
    return np.asarray(values, dtype=np.dtype(dtype)), meta


def _driver_for(path, driver):
    if driver:
        return driver
    suffix = path.suffix.lower()
    if suffix in _WRITE_DRIVERS:
        return _WRITE_DRIVERS[suffix]
    if suffix in RASTER_SUFFIXES:
        raise RasterError(
            "writing %r files is not supported; write a GeoTIFF (.tif) or "
            "pass driver= explicitly" % suffix)
    return "GTiff"


def _prepare_write(path, array, like, dtype, driver, transform, crs, nodata):
    """Validate a write request and resolve every value it leaves implicit."""
    values = np.asarray(array)
    if values.ndim != 2:
        raise RasterError("a raster band must be two-dimensional, got shape %s"
                          % (values.shape,))
    rows, cols = values.shape

    if like is not None:
        if not isinstance(like, RasterMeta):
            raise RasterError("like must be a RasterMeta, got %r" % (type(like),))
        if like.shape != (rows, cols):
            raise RasterError(
                "the output has shape %s but the reference raster is %s; a "
                "geotransform only fits the grid it was computed for"
                % ((rows, cols), like.shape))
        if transform is None:
            transform = like.transform
        if crs is None:
            crs = like.crs

    target = Path(path)
    driver = _driver_for(target, driver)
    if dtype is None:
        dtype = like.dtype if like is not None else values.dtype
    dtype = np.dtype(dtype).name
    if dtype not in _NUMPY_TO_GDAL:
        raise RasterError("GDAL cannot store %r rasters; choose one of: %s"
                          % (dtype, ", ".join(sorted(_NUMPY_TO_GDAL))))

    floating = np.issubdtype(np.dtype(dtype), np.floating)
    finite = bool(np.isfinite(values).all())
    nodata = _as_float(nodata)
    if not finite:
        if nodata is None:
            raise RasterError(
                "the array contains NaN or infinity but no nodata value was "
                "given, so the invalid pixels could not be marked; pass a "
                "sentinel, for example nodata=-9999")
        if not floating:
            if not np.isfinite(nodata):
                raise RasterError(
                    "the array contains NaN or infinity, which a %s raster "
                    "cannot store; write a floating-point raster or pass a "
                    "finite sentinel with nodata=" % dtype)
            # An integer band has no NaN, so the invalid pixels are stored as
            # the declared sentinel; casting NaN to an integer would otherwise
            # put an arbitrary number in the file.
            values = np.where(np.isfinite(values), values, nodata)
    if nodata is not None and np.isnan(nodata) and not floating:
        # NaN can only mark "no data" inside a floating-point file.  An integer
        # array cannot hold a NaN, so there is nothing to mark and the default
        # sentinel is dropped instead of being passed to a backend that would
        # reject it.
        nodata = None

    transform = _transform_or_none(transform)
    return values, dtype, driver, transform, crs, nodata


def _write_rasterio(rasterio, path, values, dtype, driver, transform, crs,
                    nodata, compress):
    options = {"driver": driver, "height": int(values.shape[0]),
               "width": int(values.shape[1]), "count": 1, "dtype": dtype}
    if transform is not None:
        options["transform"] = rasterio.Affine(*transform)
    if crs is not None:
        try:
            options["crs"] = rasterio.crs.CRS.from_user_input(crs)
        except Exception as exc:  # rasterio raises its own CRS errors
            raise RasterError("cannot interpret the CRS %r: %s" % (crs, exc))
    if nodata is not None:
        options["nodata"] = nodata
    if compress:
        options["compress"] = compress
    with _quiet_georeference_warnings(rasterio):
        with rasterio.open(str(path), "w", **options) as dataset:
            dataset.write(np.asarray(values, dtype=dtype), 1)
    return Path(path)


def _write_gdal(gdal, path, values, dtype, driver, transform, crs, nodata,
                compress):
    handler = gdal.GetDriverByName(driver)
    if handler is None:
        raise RasterError("this GDAL build has no %r driver" % driver)
    creation = []
    if compress and driver in ("GTiff", "COG"):
        creation.append("COMPRESS=%s" % compress.upper())
    dataset = handler.Create(str(path), int(values.shape[1]),
                             int(values.shape[0]), 1,
                             gdal.GetDataTypeByName(_NUMPY_TO_GDAL[dtype]),
                             options=creation)
    if dataset is None:
        raise RasterError("GDAL could not create %s with the %r driver"
                          % (path, driver))
    try:
        if transform is not None:
            a, b, c, d, e, f = transform
            dataset.SetGeoTransform((c, a, b, f, d, e))
        if crs is not None:
            dataset.SetProjection(crs)
        target = dataset.GetRasterBand(1)
        if nodata is not None:
            target.SetNoDataValue(nodata)
        target.WriteArray(np.asarray(values, dtype=dtype))
        target.FlushCache()
    finally:
        dataset = None
    return Path(path)


def write_raster(path, array, *, like=None, dtype=None, driver=None,
                 transform=None, crs=None, nodata=np.nan, compress="deflate"):
    """Write a 2-D array as a single-band georeferenced raster.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file.  The suffix selects the format: ``.tif``/``.tiff`` for a
        GeoTIFF, ``.cog`` for a cloud-optimized GeoTIFF, ``.img`` for ERDAS
        Imagine.  Any other raster suffix is refused rather than silently
        written as a GeoTIFF.
    array : numpy.ndarray
        Two-dimensional array to write.  Non-finite entries become the nodata
        pixels, so the dtype has to be a floating-point one unless ``nodata``
        is finite.
    like : RasterMeta, optional
        Metadata of the raster this output is derived from.  Its geotransform
        and CRS are reused, and its dtype becomes the default output dtype.
        The array shape must match that raster.
    dtype : numpy dtype, optional
        Output dtype.  Defaults to the reference raster's dtype when ``like``
        is given, and to the array's own dtype otherwise.
    driver : str, optional
        GDAL driver name, overriding the choice made from the suffix.
    transform, crs : optional
        Explicit geometry, overriding whatever ``like`` supplies.  ``transform``
        is ``(a, b, c, d, e, f)`` and ``crs`` is WKT text or any other string
        GDAL understands, such as ``"EPSG:32633"``.
    nodata : float or None
        Value marking pixels without data (default ``NaN``).  When the output
        dtype cannot hold NaN, the non-finite pixels are written as this
        sentinel instead, so it has to be finite.
    compress : str or None
        Compression for formats that support it (default ``"deflate"``).

    Returns
    -------
    pathlib.Path
        The file that was written.

    Raises
    ------
    NoRasterBackendError
        Neither ``rasterio`` nor ``osgeo.gdal`` is installed.
    RasterError
        The request is inconsistent: wrong dimensionality, a shape that does
        not fit ``like``, an unsupported dtype, or non-finite values with no
        way to mark them.

    Notes
    -----
    Only single-band output is written, and no resampling, reprojection or
    clipping is done: the pixels you pass in are the pixels that come out.
    Use ``gdalwarp`` or ``gdal_translate`` first when the grid has to change.
    """
    values, dtype, driver, transform, crs, nodata = _prepare_write(
        path, array, like, dtype, driver, transform, crs, nodata)
    _, rasterio, gdal = _backend()
    if rasterio is not None:
        return _backend_call("write", path, _write_rasterio, rasterio, path,
                             values, dtype, driver, transform, crs, nodata,
                             compress)
    return _backend_call("write", path, _write_gdal, gdal, path, values, dtype,
                         driver, transform, crs, nodata, compress)
