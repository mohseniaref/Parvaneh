# Georeferenced rasters: GeoTIFF and friends

Most real interferometric phase does not arrive as a `.npy` file. It arrives as
a **GeoTIFF** — a picture of numbers that also knows *where on Earth* those
numbers belong. This page explains what such a file contains, how to unwrap one
with Parvaneh, and how to get a result that still knows where it belongs.

You do not need to know anything about geospatial formats to follow this page,
only that a wrapped phase image can be stored as an ordinary image file with a
little extra bookkeeping attached to it.

## Contents

- [What a GeoTIFF actually is](#what-a-geotiff-actually-is)
- [Installing the raster support](#installing-the-raster-support)
- [Reading a raster in Python](#reading-a-raster-in-python)
- [Writing a raster in Python](#writing-a-raster-in-python)
- [Unwrapping a raster from the command line](#unwrapping-a-raster-from-the-command-line)
- [No-data pixels are a mask](#no-data-pixels-are-a-mask)
- [Choosing the output data type](#choosing-the-output-data-type)
- [Which formats can be read and written](#which-formats-can-be-read-and-written)
- [A complete worked example](#a-complete-worked-example)
- [Problems and their messages](#problems-and-their-messages)
- [What Parvaneh deliberately does not do](#what-parvaneh-deliberately-does-not-do)

## What a GeoTIFF actually is

A GeoTIFF is a TIFF image — the same kind of file a scanner produces — with two
extra pieces of information stored inside it.

1. **A geotransform.** Six numbers, `(a, b, c, d, e, f)`, that turn a pixel
   index into a map coordinate. Read the first row as "take column number $j$,
   multiply by `a`, add the easting offset `c`", and the second row the same way
   with rows, `e`, and the northing offset `f`. For the usual north-up image,
   `b = d = 0`, `a` is the pixel width in metres, and `e` is *negative* because
   row 0 is the northernmost row. Parvaneh reports the transform in this
   reading order even though GDAL stores the same numbers as
   `(c, a, b, f, d, e)`; converting between the two is the single most common
   source of transposed results in this field.

2. **A coordinate reference system (CRS).** A description of which map the
   coordinates refer to, for example "WGS 84 / UTM zone 33N". Without it, a
   northing of 4 645 110 is just a number.

Two more things are worth knowing, because Parvaneh reacts to both.

- **No-data value.** One value in the band is declared to mean "there is no
  measurement here". Common choices are `-9999`, `0`, and `NaN`. It is exactly
  the same idea as a mask, and Parvaneh treats it that way.
- **Bands.** A TIFF can hold several images of the same size. InSAR products
  often keep phase in one band and coherence in another, so a band must be
  chosen before anything can be unwrapped.

The *values* are what get unwrapped. The georeferencing is not involved in the
mathematics at all — it only travels with the array so the result stays usable
in a GIS. That is why the same algorithm runs on a GeoTIFF and on a plain
`.npy` array, and why you can process one and inherit the other's grid.

## Installing the raster support

Parvaneh reads rasters through a **GDAL** binding. GDAL is the library almost
every geospatial tool is built on; it is large, it is optional, and it is the
only dependency here that is not part of a normal scientific Python install.

```bash
python -m pip install -e '.[geo]'     # installs rasterio, which bundles GDAL
```

`rasterio` is the friendly Python face of GDAL and is what the package prefers
when both are present. If you already have the official bindings from your
distribution — usually the `python3-gdal` package — those work too and need no
extra install. Check what the package sees:

```python
from parvaneh import raster_backend
print(raster_backend())     # "rasterio", "gdal", or None
```

**Nothing else in Parvaneh needs GDAL.** Without it the package still imports,
still unwraps `.npy`, `.npz`, and headerless rasters, and still passes its own
test suite; only the raster functions refuse to run, with a message telling you
what to install. The raster tests skip themselves in that situation rather than
failing, which is why the count of tests reported by `pytest` depends on your
machine.

## Reading a raster in Python

```python
from parvaneh import read_raster

values, meta = read_raster("wrapped.tif")
```

Two things come back, and the second one matters as much as the first.

- `values` is a `(rows, columns)` array. It is returned as `float64` so that
  unwrapping arithmetic does not lose precision, whatever the file stored.
- `meta` is a `RasterMeta` describing the file:

| Attribute | Meaning |
|---|---|
| `driver` | GDAL's short name for the format, for example `"GTiff"` |
| `height`, `width` | raster size in pixels; `shape` is `(height, width)` |
| `count` | how many bands the file has |
| `band` | which band was read (1-based) |
| `dtype` | the dtype *of the file*, as a NumPy name such as `"float32"` |
| `transform` | the six geotransform numbers, or `None` |
| `crs` | the CRS as WKT text, or `None` |
| `nodata` | the no-data value the file declares, or `None` |
| `georeferenced` | `True` if the file has a CRS or a transform |

`meta` is immutable — derive a new one with `dataclasses.replace` if you need to
change something.

### Choosing a band

```python
phase, meta = read_raster("interferogram.tif", band=1)
coherence, _ = read_raster("interferogram.tif", band=2)
```

`band` is 1-based, matching every GIS tool you will meet. If the file has fewer
bands, the error message tells you how many it does have, which is far more
useful than an index error three lines later.

### What happens to the no-data pixels

By default they become `NaN`:

```python
values, meta = read_raster("wrapped.tif")            # nodata -> NaN
```

`NaN` is the right default because it cannot be confused with a measurement:
no real phase is `NaN`, whereas a sentinel such as `-9999` is a perfectly legal
number that a later average would happily include. The original value is still
available in `meta.nodata`, so nothing is lost. If you need the raw sentinel
instead — for example to hand the array to code that does not understand `NaN`:

```python
values, meta = read_raster("wrapped.tif", nodata_fill=None)   # keep -9999
```

You can also substitute your own marker with `nodata_fill=-9999`, and if the
declared no-data value appears as a genuine measurement elsewhere in the band,
it is replaced like every other occurrence — the file's declaration is believed
as written.

A file with **no** declared no-data value is read exactly as stored; Parvaneh
does not guess that `-9999` or `0` looks suspicious.

## Writing a raster in Python

```python
import numpy as np
from parvaneh import read_raster, unwrap, write_raster

phase, meta = read_raster("wrapped.tif")        # nodata pixels are NaN
valid = np.isfinite(phase)                      # ... so they are also the mask

# unwrap() refuses non-finite input rather than inventing a value for a pixel
# that has none, so no-data pixels get a finite placeholder and zero weight,
# which deletes their edges from the fit.
unwrapped = unwrap(np.where(valid, phase, 0.0), valid.astype(float))
unwrapped[~valid] = np.nan                      # NaN in, NaN out

write_raster("unwrapped.tif", unwrapped, like=meta, dtype="float32")
```

That pair of lines is the whole no-data convention, and it is worth reading
carefully:

- `np.where(valid, phase, 0.0)` replaces each missing pixel with a placeholder
  so that the array is finite. `unwrap` rejects `NaN` and infinities with a
  `ValueError` instead of propagating them through the iterations, because a
  pixel with no value cannot be unwrapped — and because `0 * NaN` is `NaN`, so
  a zero weight on its own would not have saved it.
- `valid.astype(float)` is the weight. A weight of `0` removes every edge that
  touches the pixel, so the placeholder never enters the fit. It is a mask, not
  a number to be averaged: the values the solver returns *inside* the mask are
  an artefact of the linear solve, so overwrite them with `NaN` (the CLI does
  this for you) before the array is used or written.

Two arguments are worth explaining.

- `like=meta` copies the geotransform and the CRS from the reference raster, so
  the output lands on the same grid as the input. It also supplies the default
  output dtype. The array shape must match the reference — a helper that moves
  five pixels of geotransform onto a differently sized grid would be producing
  quietly wrong coordinates, so that combination is refused.
- `dtype=` chooses the encoding. Specialised writers such as the
  [backends](performance.md) are not involved here; this is single-precision or
  double-precision floating point, nothing exotic.

If you do not pass `like`, pass `transform=` and `crs=` explicitly, or the file
is written as a plain image with no location. That is legal and occasionally
what you want, but it is not what a downstream GIS expects, and the usual
symptom is a result that cannot be overlaid on anything.

`write_raster` writes **one band**. Multi-band output, band reordering, and
band-by-band stacking in one file are jobs for a GIS tool, not for an
unwrapping routine; see
[What Parvaneh deliberately does not do](#what-parvaneh-deliberately-does-not-do).

Non-finite entries in the array become the no-data pixels, which imposes one
rule: **either the dtype is floating-point, or you must give a finite sentinel.**
This is accepted, and marks the invalid pixels with `-9999`:

```python
import numpy as np
array = np.array([[1.0, np.nan], [3.0, 4.0]])
write_raster("out.tif", array, dtype="int16", nodata=-9999)
```

This is refused, because `int16` has no way to express "no data" and the writer
will not invent a number for you:

```python
write_raster("out.tif", array, dtype="int16")      # RasterError
```

```text
the array contains NaN or infinity, which a int16 raster cannot store;
write a floating-point raster or pass a finite sentinel with nodata=
```

`compress="deflate"` is the default: GeoTIFFs are compressed losslessly, so a
smooth unwrapped phase usually shrinks several-fold with no effect on the
values. Pass `compress=None` if a downstream tool objects.

## Unwrapping a raster from the command line

The command line recognises a raster by its file extension, so nothing has to be
declared. This is the whole thing:

```bash
parvaneh unwrap wrapped.tif -o unwrapped.tif
```

```text
input    wrapped.tif  shape=1024x1024
method   ls (backend numba, workers -1)
elapsed  0.842 s
wrote    unwrapped.tif
```

The geometry is carried through automatically: `unwrapped.tif` has the same
transform, the same CRS, and the same size as `wrapped.tif`, because the input's
metadata is used as the reference for the output. There is no flag to turn this
on.

| Option | Effect on raster input |
|---|---|
| `--band N` | unwrap band `N` (default `1`) |
| `--scale S`, `--offset O` | apply `value * S + O` to the phase before unwrapping |
| `--mask FILE` | a second raster (or `.npy`): non-zero and finite means valid |
| `--weight FILE` | a second raster of nonnegative weights; no-data becomes zero weight |
| `--info` | print the raster's metadata as JSON |
| `-o FILE.tif` | write a raster; the suffix picks the format |
| `--out-dtype` | force the output encoding, overriding the automatic choice |
| `--invalid-fill VALUE` | value written where the result is undefined |

`--shape`, `--dtype` and `--order` do not apply to a raster: the file states its
own size and encoding, and reading a GeoTIFF with `--shape` silently ignored
would be worse than saying so. They apply only to headerless input, which has no
metadata at all.

### Inspecting the input

```bash
parvaneh unwrap wrapped.tif --info -o unwrapped.tif
```

```json
{
  "raster": {
    "band": 1,
    "bands": 1,
    "crs": "PROJCS[\"WGS 84 / UTM zone 33N\", ...]",
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": null,
    "shape": [1024, 1024],
    "transform": [30.0, 0.0, 412345.0, 0.0, -30.0, 4645110.0]
  }
}
```

This is the fastest way to answer "is my file georeferenced?" before wondering
why the output will not overlay. A `null` transform or CRS means the file is a
plain image wearing a `.tif` extension. `gdalinfo wrapped.tif` prints the same
information in more detail if you have GDAL's command-line tools installed.

## No-data pixels are a mask

A raster that declares no-data has already told you which pixels are valid, so
Parvaneh does not ask twice and does not make it an error:

```text
note     112 nodata pixel(s) of wrapped.tif excluded from unwrapping
```

Those pixels are removed from the problem, the remaining ones are unwrapped,
and the same footprint is written back as no-data in the output. Round-tripping
a mask costs nothing and needs no flags:

```bash
parvaneh unwrap wrapped.tif -o unwrapped.tif      # invalid pixels stay invalid
```

This is different from a `.npy` input containing `NaN`, which is still an error
(`error: the input phase contains NaN or infinity`). The difference is
deliberate: a bare array of numbers cannot explain *why* a pixel is `NaN`, while
a raster declares it, and a declaration is data to be honoured rather than a
mistake to be reported.

If you also pass `--mask`, the two are combined: a pixel is unwrapped only if it
is inside the raster's footprint **and** non-zero in your mask. A mask supplied
as a raster works the same way as the input — finite and non-zero means valid —
so a mask GeoTIFF that marks the invalid areas with `-9999` behaves the way you
expect without any conversion.

The note is progress chatter, so `-q` suppresses it, exactly as it suppresses
the other lines. The exclusion itself happens either way.

In Python the same rule is expressed as a weight rather than a flag; see
[Writing a raster in Python](#writing-a-raster-in-python) for the three lines
that do it.

## Choosing the output data type

There is no single right answer, so the rule is short and mechanical:

| Input raster dtype | Default output dtype | Why |
|---|---|---|
| floating point (`float32`, `float64`) | the same | the file's precision is the best guess at the precision the user wants |
| integer (`int16`, `uint8`, …) | `float32` | an unwrapped phase is real-valued and cannot be an integer; a count raster was never going to be copied exactly |

An integer input is not an error. It is the normal shape of a product whose
phase was stored as a scaled integer, which is exactly what `--scale` is for:

```bash
parvaneh unwrap phase.int16.tif --scale 0.001 -o unwrapped.tif
```

If the result contains non-finite pixels and the output dtype cannot store them,
the command stops rather than writing a wrong file:

```text
error: the unwrapped phase contains non-finite pixels, which the int16 output dtype cannot store.
       Pass --invalid-fill VALUE to substitute a sentinel, or use a floating-point --out-dtype such as float32.
```

Follow either piece of advice:

```bash
parvaneh unwrap wrapped.tif -o out.tif --out-dtype int16 --invalid-fill -9999
```

With `--invalid-fill`, that value becomes both the substituted pixel value and
the no-data value declared in the output header, so the next tool to read the
file knows which pixels to skip.

## Which formats can be read and written

Reading is GDAL's job, so anything GDAL can open can be unwrapped: GeoTIFF
(`.tif`, `.tiff`), ungeoreferenced TIFF, ENVI (`.img`, `.bil`, `.bsq`), ERDAS
Imagine, USGS DEM (`.dem`, `.dt0`, `.dt1`), AAIGrid (`.asc`), NetCDF (`.nc`),
HDF5 (`.h5`, `.he5`), JPEG 2000 (`.jp2`), plain PNG and BMP, and more. The
suffix only has to be recognised so the command line can tell "a raster" from "a
headerless float array"; `read_raster` itself will try any path you give it.

Writing is deliberately narrower, because a format you cannot write correctly is
worse than one you refuse:

| Output suffix | Driver | Notes |
|---|---|---|
| `.tif`, `.tiff`, `.gtif`, `.gtiff` | GTiff | the default and the safe choice |
| `.cog` | COG | Cloud-Optimized GeoTIFF, for tiled web/cloud delivery |
| `.img` | HFA | ERDAS Imagine |
| anything else in the raster list | — | refused: `.vrt`, `.png`, `.nc`, … |

```text
error: writing '.vrt' files is not supported; write a GeoTIFF (.tif) or pass driver= explicitly
```

Passing `driver=` to `write_raster` overrides the table, which is the escape
hatch if you really do need another GDAL driver.

## A complete worked example

A synthetic interferogram, written to a GeoTIFF with a hole in it, then
unwrapped, with every number checked along the way.

```python
import numpy as np
from parvaneh import make_synthetic, read_raster, unwrap, wrap_phase, write_raster

# 1. A deterministic test scene, and a grid to place it on.
truth, wrapped, _ = make_synthetic(shape=(64, 80), noise=0.02, seed=3)
transform = (30.0, 0.0, 412345.0, 0.0, -30.0, 4645110.0)   # 30 m pixels, UTM 33N

# 2. Declare a hole: the lake in the middle of the scene has no measurement.
wrapped = wrapped.astype(np.float32)
wrapped[20:28, 30:50] = np.nan
write_raster("wrapped.tif", wrapped, transform=transform, nodata=-9999)

# 3. What did the file end up saying about itself?
_, meta = read_raster("wrapped.tif")
print(meta)
```

```text
RasterMeta(path='wrapped.tif', driver='GTiff', height=64, width=80, count=1, dtype='float32', band=1, transform=(30.0, 0.0, 412345.0, 0.0, -30.0, 4645110.0), crs=None, nodata=-9999.0)
```

Every field survived the round trip: the 30 m geotransform, the `float32` storage
type, and the `-9999` sentinel. `crs` is `None` because the example never gave
one.

```bash
$ parvaneh unwrap wrapped.tif -o unwrapped.tif
note     160 nodata pixel(s) of wrapped.tif excluded from unwrapping
input    wrapped.tif  shape=64x80
method   ls (backend numba, workers -1)
elapsed  0.380 s
wrote    unwrapped.tif
```

The `elapsed` line is machine-dependent — this machine reported `0.373 s` and
`0.380 s` on two consecutive runs of the same job, and a slower algorithm or a
larger scene takes much longer. What is *not* machine-dependent is the `note`
line: the 160 pixels that were flagged `-9999` in the file were turned into
`NaN` and left out of the fit.

```python
# 4. Read the result back and check it against the truth we never wrote down.
result, meta = read_raster("unwrapped.tif")

print(meta.transform == transform)              # True: same grid
print(np.isnan(result[20:28, 30:50]).all())     # True: hole preserved
print(np.isfinite(result).sum())                # 4960 of 5120 pixels

valid = np.isfinite(result)
offset = np.median((truth - result)[valid])     # one arbitrary constant
aligned = np.abs(truth - result - offset)
print(aligned[valid].max())                     # 0.297
print(np.median(aligned[valid]))                # 0.015
```

```text
True
True
4960
0.2973791994296473
0.014667428106959424
```

The first three facts are exact: the grid survived, the invalid footprint
survived, and the 160 hole pixels are still missing. The two error figures are
the CLI's, and the last digits depend on the backend (`numpy` and `numba` agree
to about seven significant digits here), so treat them as `0.297` and `0.015`
rather than as a golden value. They also need a closer look, because the naive
reading — "the hole costs me 0.297 rad" — is wrong, and the measurements that
show why are cheap to make.

Start with the wrapped field itself. Before the hole is punched, it is globally
consistent:

```python
from parvaneh import phase_residues

clean, _, _ = make_synthetic(shape=(64, 80), noise=0.02, seed=3)
print(np.count_nonzero(phase_residues(clean)))     # 0
```

Zero residues is the condition under which unwrapping is well posed: every loop
integral of the wrapped gradients agrees. Now remove the noise and unwrap the
*holed* scene:

```python
truth0, wrapped0, _ = make_synthetic(shape=(64, 80), noise=0.0, seed=3)
valid0 = np.ones(wrapped0.shape, dtype=bool)
valid0[20:28, 30:50] = False
exact = unwrap(np.where(valid0, wrapped0, 0.0), valid0.astype(float))
offset0 = np.median((truth0 - exact)[valid0])
print(np.abs(truth0 - exact - offset0)[valid0].max())   # 9.1e-09
```

So a hole with zero weight costs nothing at all: it deletes edges from the
graph, and the rest of the scene is still solved exactly. The 0.297 rad is
therefore the *input's* noise, not the hole's doing. Unwrapping the same scene
with **no hole at all** gives `max 0.297276`, the same maximum at the same pixel
`(34, 39)`:

```python
truth, wrapped, _ = make_synthetic(shape=(64, 80), noise=0.02, seed=3)
plain = unwrap(wrapped)
offset = np.median(truth - plain)
error = np.abs(truth - plain - offset)
print(error.max())                        # 0.2972760962...
print(np.unravel_index(np.argmax(error), error.shape))   # (34, 39)
```

The synthetic noise field added by `make_synthetic` has `max 0.297434`, rms
`0.0261`, and median `0.0149`; the unwrapped result reproduces it almost exactly
(median error `0.0149`). That is the real story: this noise is spatially
smoothed rather than pixel-independent, so it survives the least-squares fit
instead of being averaged away, and its largest excursion is ten times its
median. A useful rule when you read such numbers is to compare the error
against the noise you put in, not against the noise's standard deviation alone.

The moral for a raster with excluded footprints is narrow and practical:

- zero-weight pixels are removed from the fit, so they cannot bias their
  neighbours' differences; the cost of a hole is that the deformation *under*
  the hole is unknown, not that the surrounding values are wrong;
- the values the solver returns inside the hole are a smooth continuation
  because of the iterative solve, not a measurement, so mask them out (`NaN`)
  before writing or interpreting the result;
- filling a hole with a placeholder like `0.0` and forgetting the weight does
  invent information: it creates four spurious residues at the corners of the
  rectangle, at `(19, 33)`, `(19, 45)`, `(27, 29)`, and `(27, 40)`. Those four
  are an artefact of the substitute value, which is exactly why the placeholder
  is paired with a zero weight.


## Problems and their messages

| Message | Cause | Fix |
|---|---|---|
| `this file needs a GDAL binding, and neither rasterio nor osgeo.gdal is installed.` | no GDAL binding | `pip install -e '.[geo]'`, or use `python3-gdal` |
| `raster file not found: …` | wrong path | check the path; this is reported even when GDAL is missing |
| `could not read …` | the file is not a raster GDAL can open, or is truncated | `gdalinfo FILE`, or `parvaneh … --info` if it opens |
| `band 3 does not exist in …; the file has 2 bands` | wrong `--band` | count the bands with `--info` |
| `band numbers start at 1` | `--band 0` | band indices are 1-based |
| `the file has no usable georeferencing` | a plain TIFF | harmlessly nothing: unwrapping needs no coordinates |
| `writing '.vrt' files is not supported` | unwritable output format | write `.tif`, or pass `driver=` |
| `could not write …` | permissions, a missing directory, or a driver that is not built | write somewhere writable |
| unwrapped values all wrong by a multiple of $2\pi$ | the input was **not** wrapped phase | unwrapping assumes every input value is already in $(-\pi, \pi]$ |

## What Parvaneh deliberately does not do

Each of these is a real job, done well by GDAL's own tools, and a poor use of
unwrapping code:

- **Reprojecting, resampling, cropping or mosaicking.** Use `gdalwarp`. The
  geometry must be settled before you unwrap, or the phase gradient changes
  meaning underneath the algorithm.
- **Writing multi-band rasters.** Unwrap one band at a time and stack the
  results with `gdal_merge.py`, `gdal_translate -b`, or `rasterio` directly.
- **Reading a band that is not two-dimensional**, such as a time series stacked
  into a three-dimensional array. Split it first.
- **Converting a raster to another format just to unwrap it.** No conversion is
  needed: any GDAL-readable file can be unwrapped as it is.
- **Guessing a missing no-data value.** If the file does not declare one, none
  is assumed. Supply `--mask` if you know better than the file does.

## Related documentation

- [`cli.md`](cli.md) — the full command-line manual, including masks, weights
  and backends
- [`binary_formats.md`](binary_formats.md) — the headerless raw conventions for
  data that has no header at all
- [`algorithms.md`](algorithms.md) — which unwrapping algorithm to choose
