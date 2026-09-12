"""Project colormaps with phase-aware defaults."""

from matplotlib.colors import LinearSegmentedColormap


SAKURA = LinearSegmentedColormap.from_list(
    "sakura_sunset",
    ["#24164f", "#5d3a86", "#ad5f96", "#e990ae", "#f8c7bd", "#ffe69a", "#fff7d1"],
)

SAKURA_DIVERGING = LinearSegmentedColormap.from_list(
    "sakura_diverging",
    ["#394a8a", "#7889bd", "#d6d4df", "#fff3c4", "#f4a3b5", "#b83f71"],
)
