"""Accelerated and reproducible phase unwrapping."""

__version__ = "0.1.0a1"

from .core import available_backends, unwrap
from .synthetic import make_synthetic, rmse_aligned, wrap_phase
from .io import read_raw_raster, write_raw_raster
from .plotting import SAKURA, SAKURA_DIVERGING
from .path_following import phase_residues, quality_guided_unwrap
from .quality import (derivative_variance_quality, max_gradient_quality,
                      pseudocorrelation_quality, wrapped_gradients)
from .minimum_norm import unwrap_lp
from .goldstein import goldstein_unwrap, mask_cut_unwrap
from .diagnostics import discontinuity_map, surface_difference
from .flynn import flynn_unwrap
from .reliability import ReliabilityInfo, pixel_reliability, reliability_unwrap
from .raster import (RasterError, RasterMeta, NoRasterBackendError,
                     read_raster, raster_backend, write_raster)

__all__ = [
    "available_backends", "unwrap", "make_synthetic", "rmse_aligned",
    "wrap_phase", "read_raw_raster", "write_raw_raster", "SAKURA",
    "SAKURA_DIVERGING",
    "phase_residues", "quality_guided_unwrap", "wrapped_gradients",
    "max_gradient_quality", "pseudocorrelation_quality",
    "derivative_variance_quality",
    "unwrap_lp",
    "goldstein_unwrap",
    "mask_cut_unwrap",
    "surface_difference", "discontinuity_map",
    "flynn_unwrap",
    "pixel_reliability", "reliability_unwrap", "ReliabilityInfo",
    "read_raster", "write_raster", "raster_backend", "RasterMeta",
    "RasterError", "NoRasterBackendError",
]
