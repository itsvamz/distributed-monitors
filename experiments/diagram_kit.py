"""Tiny helpers for drawing IEEE-style single-column flow diagrams (3.5 in wide)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

plt.rcParams.update({"font.family": ["Liberation Serif", "DejaVu Serif"], "font.size": 6.8,
                     "axes.linewidth": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42})
W = 3.5
PAL = {"node": "#DCE6F2", "rule": "#ECECEC", "attack": "#F7DEDE", "good": "#DFF0DF",
       "data": "#FFF3CF", "white": "#FFFFFF", "warn": "#FBE3C8"}


def canvas(h):
    fig = plt.figure(figsize=(W, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h / W)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc="white", bold=False, fs=6.6, ec="black", lw=0.6, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=PAL.get(fc, fc), ec=ec, lw=lw, ls=ls))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.15)


def diamond(ax, x, y, w, h, text, fc="white", fs=6.2):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                         closed=True, fc=PAL.get(fc, fc), ec="black", lw=0.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.1)


def arrow(ax, p, q, text=None, tpos=0.5, dx=0, dy=1.6, style="-|>", lw=0.6, ls="-", fs=6.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=6, lw=lw, color="black",
                                 linestyle=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text(p[0] + (q[0] - p[0]) * tpos + dx, p[1] + (q[1] - p[1]) * tpos + dy, text,
                ha="center", va="center", fontsize=fs, style="italic")


def save(fig, name):
    fig.savefig(f"docs/figures/{name}.png", dpi=300)
    plt.close(fig)
