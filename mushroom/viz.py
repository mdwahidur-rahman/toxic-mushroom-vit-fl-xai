"""Publication figure style for the mushroom ViT-FL-XAI study.

Single source of truth for colors, typography and figure geometry so every panel
in the manuscript reads as one system. Palette is CVD-validated (OKLab dE>=8 on
all pairs for the 3 primary series; >=9.1 adjacent for 4).
Outputs vector PDF (for LaTeX) + 600-dpi PNG (for preview).
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- palette -----------------------------------------------------------------
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", violet="#4a3aa7", red="#e34948", green="#008300")
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"]]        # fixed order, never cycled
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#ffffff"
CRIT = "#d03b3b"                                                  # status: critical only

# IEEE column geometry (inches)
COL1, COL2 = 3.5, 7.16


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "figure.facecolor": SURF, "axes.facecolor": SURF,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 4,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.35, "legend.borderpad": 0.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,                     # editable text in PDF
        "errorbar.capsize": 2,
    })


def save(fig, outdir, name):
    """Write both vector and raster copies; return the PDF path."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{name}.pdf"
    fig.savefig(pdf); fig.savefig(outdir / f"{name}.png")
    plt.close(fig)
    return pdf


def label_end(ax, x, y, text, color, dx=0.01, **kw):
    """Direct label at a line's end - identity is never color-alone."""
    ax.annotate(text, xy=(x, y), xytext=(4, 0), textcoords="offset points",
                color=color, fontsize=7.5, fontweight="bold",
                va="center", ha="left", **kw)


def panel_tag(ax, letter):
    ax.text(-0.02, 1.06, f"({letter})", transform=ax.transAxes, fontsize=9,
            fontweight="bold", color=INK, ha="right", va="bottom")
