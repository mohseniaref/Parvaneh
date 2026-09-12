# Generic binary-raster conventions

Parvaneh does not assume a vendor or book-specific filename convention. Raw
headerless rasters are loaded by stating all information that cannot be inferred
from the bytes:

```python
from parvaneh import read_raw_raster

phase = read_raw_raster(
    "interferogram.f32",
    shape=(2048, 4096),       # rows, columns
    dtype="<f4",             # little-endian float32
)
```

Unsigned integer phase codes can be converted to radians explicitly:

```python
import numpy as np
from parvaneh import read_raw_raster

phase = read_raw_raster(
    "wrapped_phase.u8",
    shape=(1024, 1536),
    dtype=np.uint8,
    scale=2 * np.pi / 256,
    offset=-np.pi,
)
```

Use `">f4"` for big-endian float32 or `"<f4"` for little-endian float32.
Storage order defaults to row-major (`order="C"`); column-major input can be
read with `order="F"`. The reader rejects a file whose value count does not
exactly equal `rows * columns`.

Writing is equally explicit:

```python
from parvaneh import write_raw_raster

write_raw_raster("result.f32", unwrapped_phase, dtype="<f4")
```

For self-describing research products, prefer formats such as GeoTIFF, NetCDF,
HDF5, or NumPy `.npy` and preserve coordinate reference, geotransform, nodata,
units, and processing metadata alongside the phase array. Parvaneh reads and
writes GDAL rasters directly when a GDAL binding is installed; the details,
including no-data handling, are in [Reading and writing GDAL rasters](raster_io.md).

