"""House figure style for the BernFac study (scientific-figure-making skill).

PALETTE / DEFAULT_COLORS   semantic colours (blue = proposed / selected model,
                           greens = improvements, reds = baselines / contrasts,
                           neutral = background, teal / violet = extra)
apply_publication_style()  rcParams (sans-serif, top/right spines off,
                           frameless legends, slide-legible font sizes)
finalize_figure()          PNG (dpi 200) + PDF export with a small pad and a
                           1.5 MB PNG size check
lambda_ramp() / k_ramp()   ordered sequential ramps for hyper-parameter lines
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
    # darker neutrals for baseline lines / annotations (design-theory neutrals)
    "grey": "#4D4D4D", "grey_mid": "#767676", "grey_dark": "#272727",
}
DEFAULT_COLORS = [PALETTE[c] for c in ["blue_main", "green_3", "red_strong", "teal", "violet", "neutral"]]

# font-size hierarchy (points); base 15 is legible when a ~16-18 in wide figure
# is placed full-width on a 16:9 slide
FS = 15          # axis labels
FS_TITLE = 15    # panel titles
FS_TICK = 13     # tick labels
FS_LEGEND = 12   # legends
FS_SMALL = 12    # in-panel annotations
FS_TINY = 10.5   # dense cell annotations (heatmap numbers)
FS_PANEL = 19    # bold panel letters

# sequential colormaps used across the study
CMAP_SEQ = "viridis"       # values (heatmaps, grids)
CMAP_SEQ_R = "viridis_r"   # "lower is better" grids
CMAP_PROB = "magma"        # fitted probabilities P
CMAP_Y = "Greys"           # binary data Y
CMAP_DENSITY = "Blues"     # hexbin densities

_LAMBDA_CMAP = LinearSegmentedColormap.from_list(
    "lambda_blue", ["#B7D2EE", "#6E9FD3", PALETTE["blue_secondary"], PALETTE["blue_main"]])
_K_CMAP = LinearSegmentedColormap.from_list(
    "k_teal", ["#B8DDD9", "#6FB6B5", PALETTE["teal"], "#1D5960"])


def lambda_ramp(n):
    """n ordered colours, light blue -> blue_main (small -> large lambda)."""
    return [_LAMBDA_CMAP(x) for x in np.linspace(0.0, 1.0, n)]


def k_ramp(n):
    """n ordered colours, light teal -> dark teal (small -> large rank k)."""
    return [_K_CMAP(x) for x in np.linspace(0.0, 1.0, n)]


@dataclass(frozen=True)
class FigureStyle:
    font_size: float = FS
    axes_linewidth: float = 2.0
    font_family: tuple = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style=None):
    s = style or FigureStyle()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": list(s.font_family),
        "font.size": s.font_size, "axes.titlesize": FS_TITLE, "axes.labelsize": s.font_size,
        "legend.fontsize": FS_LEGEND, "legend.title_fontsize": FS_LEGEND,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "figure.titlesize": s.font_size + 1,
        "axes.linewidth": s.axes_linewidth, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": s.axes_linewidth * 0.75, "ytick.major.width": s.axes_linewidth * 0.75,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "lines.linewidth": 2.2, "lines.markersize": 6,
        "legend.frameon": False, "legend.handlelength": 1.8,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "image.interpolation": "nearest", "axes.titlepad": 8,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def finalize_figure(fig, out_path, formats=("png", "pdf"), dpi=200, pad=0.05, close=True,
                    max_png_mb=1.5):
    """Save <out_path>.<fmt> for each format (tight bbox, small pad)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out_path.with_suffix(f".{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=pad)
        paths.append(p)
    if close:
        plt.close(fig)
    for p in paths:
        if p.suffix == ".png":
            size = p.stat().st_size / 1e6
            if size > max_png_mb:  # dense raster panels: lossless-looking 256-colour palette PNG
                from PIL import Image
                Image.open(p).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                                      dither=Image.Dither.NONE).save(p, optimize=True)
                size = p.stat().st_size / 1e6
            print(f"{p.name}  {size:.2f} MB")
            if size > max_png_mb:
                print(f"WARNING {p.name} is {size:.2f} MB (> {max_png_mb} MB)")
    return paths


def panel_label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=FS_PANEL, va="bottom", ha="left")
