"""P3 figure: probe AUROC at every layer, one panel per pairing, with the TA's
spoken score and chance as reference lines. Static PNG for the README.

Palette: categorical slots 1-2 of the dataviz reference palette (validated:
light surface #fcfcfb, all checks pass). Text in ink tokens, not series colors.

Usage:
    uv run python scripts/p3_figure.py [--condition rate_stripped]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

PANELS = [
    ("human_vs_backdoor", "Human honest vs backdoor"),
    ("haiku_vs_backdoor", "Haiku honest vs backdoor"),
    ("style", "Style control: human vs Haiku honest"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="rate_stripped")
    args = ap.parse_args()
    d = ROOT / "results" / "p3" / args.condition
    m = json.loads((d / "metrics.json").read_text())["pairings"]
    curves = np.load(d / "layer_curves.npz")

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURFACE)
    for ax, (key, title) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        layers = np.arange(len(curves[f"{key}_probe"]))
        ax.axhline(0.5, color=GRID, lw=1, zorder=1)
        ax.text(layers[-1], 0.505, "chance", color=INK2, fontsize=8, ha="right", va="bottom")
        spoken = m[key]["spoken"]["auroc"]
        ax.axhline(spoken, color=INK2, lw=1, zorder=1)
        ax.text(layers[-1], spoken - 0.008, f"TA's spoken score {spoken:.2f}", color=INK2, fontsize=8,
                ha="right", va="top")
        ax.plot(layers, curves[f"{key}_probe"], color=BLUE, lw=2, zorder=3, label="probe trained on this pairing")
        if key == "haiku_vs_backdoor" and "human_vs_backdoor_transfer_curve" in curves:
            ax.plot(layers, curves["human_vs_backdoor_transfer_curve"], color=ORANGE, lw=2, zorder=2,
                    label="probe trained on human vs backdoor")
            ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
        head = m[key]["probe"]
        ax.set_title(f"{title}\nheadline probe {head['auroc']:.2f} "
                     f"[{head['auroc_ci'][0]:.2f}, {head['auroc_ci'][1]:.2f}]",
                     color=INK, fontsize=10, loc="left")
        ax.set_xlabel("layer (0 = embeddings)")
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("held-out AUROC")
    axes[0].set_ylim(0.3, 1.0)
    fig.tight_layout()
    out = d / "layer_curves.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
