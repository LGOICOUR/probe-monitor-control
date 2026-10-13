"""Figure 1 for the write-up: every monitor on every tier, AUROC with 95% CIs.
Reads the P3 and P4 metrics; writes results/fig_ladder.png."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"probe": "#2a78d6", "spoken": "#eb6834", "words": "#1baf7a", "judge": "#eda100"}
LABEL = {"probe": "probe on the monitor's residual stream", "spoken": "monitor's own rating",
         "words": "word counter, no monitor", "judge": "frontier judge (Opus 5.5)"}

def m(path): return json.loads((ROOT / "results" / path).read_text())
orig = m("p3/rate_stripped/metrics.json")["pairings"]["human_vs_backdoor"]
haiku = m("p3/rate_stripped_matched/metrics.json")["matched"]
sonnet = m("p3/rate_stripped_matched_sonnet/metrics.json")["matched"]
judge = m("p4_judge/metrics.json")
tiers = [
    ("Haiku pairs\n(102, monitor at ceiling)", {"probe": haiku["probe"], "spoken": haiku["spoken"], "words": haiku["word_counter"]}),
    ("Human honest vs\ndataset backdoor (150)", {"probe": orig["probe"], "spoken": orig["spoken"],
        "words": {"auroc": 0.73, "auroc_ci": [np.nan, np.nan]}, "judge": judge["original"]["Opus 5.5 judge"]}),
    ("Sonnet pairs\n(82, hardest)", {"probe": sonnet["probe"], "spoken": sonnet["spoken"], "words": sonnet["word_counter"],
        "judge": judge["sonnet"]["Opus 5.5 judge"]}),
]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(8.4, 5.0), facecolor=SURFACE); ax.set_facecolor(SURFACE)
keys = ["spoken", "words", "probe", "judge"]; w = 0.19
for ti, (name, vals) in enumerate(tiers):
    for ki, k in enumerate(keys):
        if k not in vals: continue
        b = vals[k]; x = ti + (ki - 1.5) * w
        a = b["auroc"]; lo, hi = b.get("auroc_ci", [np.nan, np.nan])
        ax.bar(x, a, width=w - 0.02, color=COL[k], zorder=3, label=LABEL[k] if ti == 2 or (ti == 0 and k != "judge") else None)
        if not np.isnan(lo):
            ax.plot([x, x], [lo, hi], color=INK, lw=1, zorder=4)
        ax.text(x, a + (0.012 if np.isnan(hi) else hi - a + 0.012) + 0.0, f"{a:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=5); ax.text(-0.42, 0.505, "chance", color=INK2, fontsize=8, ha="left", va="bottom", zorder=6)
ax.set_xticks(range(3)); ax.set_xticklabels([t[0] for t in tiers])
ax.set_ylim(0.3, 1.05); ax.set_ylabel("held-out AUROC (95% CI resamples problems)")
ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
h, l = ax.get_legend_handles_labels(); seen = {}
for hh, ll in zip(h, l): seen.setdefault(ll, hh)
ax.legend([seen[k] for k in [LABEL[x] for x in keys]], [LABEL[x] for x in keys], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=8, labelcolor=INK)
ax.set_title("Three difficulty tiers: the monitor's rating falls to chance; its residual stream does not", color=INK, fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(ROOT / "results" / "fig_ladder.png", dpi=160, facecolor=SURFACE); print("saved results/fig_ladder.png")
