"""Figure for a same-author matched set: probe AUROC per layer, the two transfer
curves, and the monitor's spoken score as a reference line.

Usage:
    uv run python scripts/p3_matched_figure.py --set sonnet
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
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["haiku", "sonnet"], default="sonnet")
    args = ap.parse_args()
    suffix = "" if args.set == "haiku" else "_sonnet"
    d = ROOT / "results" / "p3" / f"rate_stripped_matched{suffix}"
    m = json.loads((d / "metrics.json").read_text())
    c = np.load(d / "layer_curves.npz")
    label = "Haiku" if args.set == "haiku" else "Sonnet"

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    layers = np.arange(len(c["matched"]))
    ax.axhline(0.5, color=GRID, lw=1, zorder=1)
    ax.text(layers[-1], 0.495, "chance", color=INK2, fontsize=8, ha="right", va="top")
    spoken = m["matched"]["spoken"]["auroc"]
    ax.axhline(spoken, color=INK2, lw=1, zorder=1)
    ax.text(layers[-1], spoken + 0.008, f"monitor's spoken score {spoken:.2f}", color=INK2, fontsize=8,
            ha="right", va="bottom")
    words = m["matched"]["word_counter"]["auroc"]
    ax.axhline(words, color=GRID, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(layers[-1], words - 0.008, f"word counter, no monitor {words:.2f}", color=INK2, fontsize=8,
            ha="right", va="top")
    ax.plot(layers, c["matched"], color=BLUE, lw=2, zorder=3, label=f"probe trained on {label} pairs (held-out)")
    ax.plot(layers, c["original_to_matched"], color=ORANGE, lw=2, zorder=2,
            label=f"probe trained on human vs dataset backdoor, tested on {label} pairs")
    ax.plot(layers, c["matched_to_original"], color=AQUA, lw=2, zorder=2,
            label=f"probe trained on {label} pairs, tested on human vs dataset backdoor")
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    head = m["matched"]["probe"]
    ax.set_title(f"{label} same-author pairs ({m['n_pairs']} pairs): held-out AUROC by layer\n"
                 f"headline probe {head['auroc']:.2f} [{head['auroc_ci'][0]:.2f}, {head['auroc_ci'][1]:.2f}], "
                 f"catch at 2% false alarms {100 * head['catch_at_2pct']:.0f}% vs spoken "
                 f"{100 * m['matched']['spoken']['catch_at_2pct']:.0f}%", color=INK, fontsize=10, loc="left")
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("held-out AUROC")
    ax.set_ylim(0.3, 1.0)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = d / "layer_curves.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
