"""Regular sampling grids in a local East-North-Up (ENU) Cartesian frame.

Why this module exists
----------------------
Every deformation model in this package is a function of horizontal position
``(x, y)``.  Before we can evaluate a model we must decide *where* to evaluate
it.  That decision sounds trivial, but it is the single most common source of
confusing results in InSAR modelling, because three different conventions get
mixed together in practice:

1. **Which way is up?**  Geodesy uses East-North-Up (ENU).  Seismology often
   uses North-East-Down (NED).  Getting this wrong flips the sign of the
   vertical displacement, which then flips the sign of the wrapped phase.
2. **Which way do rows run?**  Mathematics textbooks put *y* increasing along
   the rows of a matrix.  Images and rasters put row 0 at the *top*, so the
   row index increases as *y* decreases.
3. **Do we name pixel centres or pixel edges?**  A 512-pixel grid with 30 m
   spacing is 15330 m wide if you count centre to centre, but 15360 m wide if
   you count edges.

This module fixes one choice for each question and states it explicitly.  All
other modules in :mod:`accelerated_unwrap.synthetic` inherit that choice, so if
you understand this file you understand the geometry of the whole package.

The conventions
---------------
* Axes are **ENU**: ``x`` is East [m], ``y`` is North [m], ``z`` is Up [m].
* All internal lengths are in **metres** and all angles in **radians**, unless
  a parameter name ends in ``_deg``.
* Arrays are indexed ``[row, column]`` with shape ``(ny, nx)``.
* **Row 0 is the northernmost row.**  The row index increases southwards, so
  ``y`` *decreases* as the row index increases.  This matches the raster and
  ``matplotlib.imshow(origin="upper")`` convention, so a model field can be
  plotted directly without flipping it.
* **Column 0 is the westernmost column**, so ``x`` increases with the column
  index.
* The grid stores **pixel centre** coordinates.  Pixel *edges* are available
  through :attr:`Grid.x_edges` and :attr:`Grid.y_edges` for plotting and area
  integration.

A worked example
----------------
>>> from accelerated_unwrap.synthetic.grid import Grid
>>> grid = Grid(nx=4, ny=3, spacing=30.0, x_min=0.0, y_max=0.0)
>>> grid.x.tolist()               # west -> east
[0.0, 30.0, 60.0, 90.0]
>>> grid.y.tolist()               # north -> south
[0.0, -30.0, -60.0]
>>> grid.shape
(3, 4)
>>> grid.Y.shape
(3, 4)
>>> grid.center
(45.0, -30.0)
"""

import numpy as np

from ._checks import finite_float as _finite_float
from ._checks import positive_float as _positive_float
from ._checks import positive_int as _positive_int

DEFAULT_NX = 512
DEFAULT_NY = 512
DEFAULT_SPACING = 30.0

#: Largest grid this package will build without the caller opting in.  A
#: 4096 x 4096 float64 array is already 128 MB, so silently allocating one
#: because of a typo (``nx=5120`` instead of ``nx=512``) is unkind.
MAX_PIXELS = 4096 * 4096


class Grid:
    """A regular Cartesian ENU grid of pixel centres.

    Parameters
    ----------
    nx, ny : int
        Number of pixels along East (columns) and North (rows).  Default 512.
    spacing : float
        Pixel spacing in metres, applied in both directions.  Default 30 m,
        which is the ground sampling distance of Sentinel-1 IW products.
    x_min : float
        East coordinate of the **centre** of column 0.  Defaults to ``0.0``.
    y_max : float
        North coordinate of the **centre** of row 0 (the northernmost row).
        Defaults to ``0.0``.
    allow_large : bool
        Set to ``True`` to bypass the :data:`MAX_PIXELS` safety check.

    Attributes
    ----------
    x : numpy.ndarray
        Pixel-centre East coordinates, shape ``(nx,)``, increasing.
    y : numpy.ndarray
        Pixel-centre North coordinates, shape ``(ny,)``, decreasing.
    X, Y : numpy.ndarray
        Meshgrids of shape ``(ny, nx)`` holding the East and North coordinate
        of every pixel centre.
    """

    def __init__(self, nx=DEFAULT_NX, ny=DEFAULT_NY, spacing=DEFAULT_SPACING,
                 x_min=0.0, y_max=0.0, allow_large=False):
        self.nx = _positive_int(nx, "nx")
        self.ny = _positive_int(ny, "ny")
        self.spacing = _positive_float(spacing, "spacing")
        self.x_min = _finite_float(x_min, "x_min")
        self.y_max = _finite_float(y_max, "y_max")
        if not allow_large and self.nx * self.ny > MAX_PIXELS:
            raise ValueError(
                "grid of {} x {} pixels exceeds the {} pixel safety limit; "
                "pass allow_large=True if this is intentional".format(
                    self.ny, self.nx, MAX_PIXELS))

        # indexing="xy" makes X vary along columns and Y along rows, which is
        # exactly the [row, column] = [north, east] convention we want.
        self.x = self.x_min + np.arange(self.nx, dtype=np.float64) * self.spacing
        self.y = self.y_max - np.arange(self.ny, dtype=np.float64) * self.spacing
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------
    @classmethod
    def centered(cls, nx=DEFAULT_NX, ny=DEFAULT_NY, spacing=DEFAULT_SPACING,
                 allow_large=False):
        """Return a grid whose pixel centres are symmetric about ``(0, 0)``.

        The grid is *centred on the origin*, meaning the mean of the pixel
        centres is exactly ``(0, 0)``.  This is convenient for point sources
        (Mogi) and fault models (Okada) that are naturally described relative
        to a local origin.
        """
        return cls(nx=nx, ny=ny, spacing=spacing,
                   x_min=-0.5 * (nx - 1) * spacing,
                   y_max=0.5 * (ny - 1) * spacing,
                   allow_large=allow_large)

    @classmethod
    def from_bounds(cls, nx, ny, x_first, x_last, y_first, y_last):
        """Build a grid from the coordinates of its first and last centres.

        ``x_first``/``y_first`` are the coordinates of the centre of pixel
        ``(row 0, column 0)``; ``x_last``/``y_last`` are the coordinates of
        the centre of pixel ``(row ny-1, column nx-1)``.  The spacing implied
        by each axis is computed and checked for consistency, which catches
        off-by-one errors such as asking for 512 pixels to span 512 spacings.
        """
        nx = _positive_int(nx, "nx")
        ny = _positive_int(ny, "ny")
        spacing_x = (float(x_last) - float(x_first)) / (nx - 1) if nx > 1 else 0.0
        spacing_y = (float(y_first) - float(y_last)) / (ny - 1) if ny > 1 else 0.0
        if nx > 1 and ny > 1:
            if not np.isclose(spacing_x, spacing_y, rtol=1e-9):
                raise ValueError(
                    "x and y spans imply different spacings "
                    "({:.6g} m vs {:.6g} m)".format(spacing_x, spacing_y))
            spacing = spacing_x
        else:
            spacing = spacing_x if nx > 1 else spacing_y
        return cls(nx=nx, ny=ny, spacing=spacing, x_min=x_first, y_max=y_first)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @property
    def shape(self):
        """Array shape ``(ny, nx)`` of any model field on this grid."""
        return (self.ny, self.nx)

    @property
    def size(self):
        """Total number of pixels, ``ny * nx``."""
        return self.ny * self.nx

    @property
    def pixel_area(self):
        """Area of one pixel in square metres, ``spacing ** 2``."""
        return self.spacing ** 2

    @property
    def x_max(self):
        """East coordinate of the centre of the easternmost column."""
        return float(self.x[-1])

    @property
    def y_min(self):
        """North coordinate of the centre of the southernmost row."""
        return float(self.y[-1])

    @property
    def center(self):
        """``(x, y)`` of the geometric centre of the pixel-centre lattice."""
        return (0.5 * (self.x_min + self.x_max), 0.5 * (self.y_max + self.y_min))

    @property
    def origin(self):
        """``(x, y)`` of the centre of pixel ``(0, 0)``, the north-west pixel."""
        return (self.x_min, self.y_max)

    @property
    def width(self):
        """East-west extent in metres, measured centre to centre."""
        return float((self.nx - 1) * self.spacing)

    @property
    def height(self):
        """North-south extent in metres, measured centre to centre."""
        return float((self.ny - 1) * self.spacing)

    @property
    def x_edges(self):
        """Pixel-edge East coordinates, shape ``(nx + 1,)``, increasing."""
        half = 0.5 * self.spacing
        return np.concatenate(([self.x[0] - half], self.x + half))

    @property
    def y_edges(self):
        """Pixel-edge North coordinates, shape ``(ny + 1,)``, decreasing."""
        half = 0.5 * self.spacing
        return np.concatenate(([self.y[0] + half], self.y - half))

    @property
    def extent(self):
        """``(x_min, x_max, y_min, y_max)`` pixel-edge bounds for ``imshow``.

        The values are the outer edges of the footprint, which is what
        ``matplotlib.pyplot.imshow`` expects, so that an image plotted with
        ``imshow(field, extent=grid.extent, origin="upper")`` lines up with the
        ENU coordinates of the model that produced it.
        """
        half = 0.5 * self.spacing
        return (self.x_min - half, self.x_max + half,
                self.y_min - half, self.y_max + half)

    def radial_distance(self, x0, y0):
        """Planar radial distance from ``(x0, y0)`` to every pixel centre.

        Returns an array of shape ``(ny, nx)`` in metres.  This is the ``r``
        that appears in the Mogi solution,
        ``r**2 = (x - x0)**2 + (y - y0)**2``.
        """
        dx = self.X - float(x0)
        dy = self.Y - float(y0)
        return np.sqrt(dx * dx + dy * dy)

    def nearest_column(self, x):
        """Column index of the pixel centre nearest to East coordinate ``x``."""
        return int(np.clip(np.round((float(x) - self.x_min) / self.spacing),
                           0, self.nx - 1))

    def nearest_row(self, y):
        """Row index of the pixel centre nearest to North coordinate ``y``."""
        return int(np.clip(np.round((self.y_max - float(y)) / self.spacing),
                           0, self.ny - 1))

    def describe(self):
        """Return a one-line human-readable summary of the grid."""
        return ("Grid {} x {} pixels, {:.6g} m spacing, "
                "x in [{:.6g}, {:.6g}] m, y in [{:.6g}, {:.6g}] m".format(
                    self.nx, self.ny, self.spacing,
                    self.x_min, self.x_max, self.y_min, self.y_max))

    def __repr__(self):
        return ("Grid(nx={}, ny={}, spacing={!r}, x_min={!r}, y_max={!r})"
                .format(self.nx, self.ny, self.spacing, self.x_min, self.y_max))


def make_grid(nx=DEFAULT_NX, ny=DEFAULT_NY, spacing=DEFAULT_SPACING,
              x_min=0.0, y_max=0.0, centered=False, allow_large=False):
    """Build a :class:`Grid`, optionally centred on the origin.

    This is a thin convenience wrapper for callers who prefer a function to a
    constructor.  ``make_grid(nx=256, spacing=30.0, centered=True)`` is
    equivalent to ``Grid.centered(nx=256, spacing=30.0)``.

    Returns
    -------
    Grid
    """
    if centered:
        if x_min != 0.0 or y_max != 0.0:
            raise ValueError("x_min/y_max cannot be combined with centered=True")
        return Grid.centered(nx=nx, ny=ny, spacing=spacing,
                             allow_large=allow_large)
    return Grid(nx=nx, ny=ny, spacing=spacing, x_min=x_min, y_max=y_max,
                allow_large=allow_large)
