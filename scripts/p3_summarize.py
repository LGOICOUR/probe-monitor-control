"""Lever: input-vs-judgment control.

The same papers were read twice: under the "rate" prompt (the monitor is asked
to judge suspicion) and under the "summarize" prompt (it is asked only to
describe the code). Probes are fit on each recording with the same grouped
folds, and each is also tested on the other recording (same problems held out).

Reading: if the summarize probe matches the rate probe, the signal is in how
the monitor reads code, not in any judgment about it. If it falls well short,
the rating instruction is doing work.

Usage:
    uv run python scripts/p3_summarize.py --set sonnet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_monitor.probe import catch_rate_at_fpr, grouped_bootstrap_ci, nested_probe  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def load(cond):
    d = ROOT / "results" / "p2" / cond
    rows = [json.loads(l) for l in (d / "ratings.jsonl").open()]
    A = np.load(d / "acts.npy").astype(np.float32)
    return rows, A


def block(name, y, s, g):
    lo, hi = grouped_bootstrap_ci(y, s, g, roc_auc_score)
    clo, chi = grouped_bootstrap_ci(y, s, g, catch_rate_at_fpr)
    return {"monitor": name, "auroc": float(roc_auc_score(y, s)), "auroc_ci": [lo, hi],
            "catch_at_2pct": catch_rate_at_fpr(y, s), "catch_ci": [clo, chi]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["haiku", "sonnet"], default="sonnet")
    args = ap.parse_args()
    suffix = "" if args.set == "haiku" else "_sonnet"
    label = "Haiku" if args.set == "haiku" else "Sonnet"
    rows_r, R = load(f"rate_stripped_matched{suffix}")
    rows_s, S = load(f"summarize_stripped_matched{suffix}")
    assert [(r["problem_id"], r["source"]) for r in rows_r] == [(r["problem_id"], r["source"]) for r in rows_s]
    y = np.array([r["label"] for r in rows_r]); g = np.array([r["problem_id"] for r in rows_r])
    spoken = np.array([r["expected_rating"] for r in rows_r])

    res_s = nested_probe(S, y, g, X_transfer=R, y_transfer=y, g_transfer=g)   # fit on summarize, test on rate
    res_r = nested_probe(R, y, g, X_transfer=S, y_transfer=y, g_transfer=g)   # fit on rate, test on summarize
    L = R.shape[1]
    curves = {
        "rate": np.array([roc_auc_score(y, res_r["layer_scores"][l]) for l in range(L)]),
        "summarize": np.array([roc_auc_score(y, res_s["layer_scores"][l]) for l in range(L)]),
        "rate_to_summarize": np.array([roc_auc_score(y, res_r["t_layer_scores"][l]) for l in range(L)]),
        "summarize_to_rate": np.array([roc_auc_score(y, res_s["t_layer_scores"][l]) for l in range(L)]),
    }
    metrics = {
        "set": args.set, "n_pairs": int(y.sum()),
        "rate_probe": block("probe on rate recording", y, res_r["headline"], g),
        "summarize_probe": block("probe on summarize recording", y, res_s["headline"], g),
        "rate_to_summarize": block("trained on rate, tested on summarize", y, res_r["t_headline"], g),
        "summarize_to_rate": block("trained on summarize, tested on rate", y, res_s["t_headline"], g),
        "spoken": block("monitor's spoken score (rate prompt)", y, spoken, g),
        "layers_chosen": {"rate": [c["layer"] for c in res_r["chosen"]], "summarize": [c["layer"] for c in res_s["chosen"]]},
        "layer0": {"rate": float(curves["rate"][0]), "summarize": float(curves["summarize"][0])},
    }
    out = ROOT / "results" / "p3" / f"summarize_control_matched{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    np.savez(out / "layer_curves.npz", **curves)

    def f(b):
        return (f"{b['auroc']:.2f} [{b['auroc_ci'][0]:.2f}, {b['auroc_ci'][1]:.2f}] | "
                f"{100 * b['catch_at_2pct']:.0f}% [{100 * b['catch_ci'][0]:.0f}, {100 * b['catch_ci'][1]:.0f}]")
    lines = [f"# Summarize control, {label} pairs ({metrics['n_pairs']})", "",
             "AUROC [95% CI] | catch at 2% [95% CI]", "", "| condition | AUROC | catch at 2% |", "|---|---|---|"]
    for k in ("rate_probe", "summarize_probe", "rate_to_summarize", "summarize_to_rate", "spoken"):
        a, c = f(metrics[k]).split(" | ")
        lines.append(f"| {metrics[k]['monitor']} | {a} | {c} |")
    lines += ["", f"- Layers chosen inside training: rate {metrics['layers_chosen']['rate']}, summarize {metrics['layers_chosen']['summarize']}",
              f"- Sanity, layer 0: rate {metrics['layer0']['rate']:.2f}, summarize {metrics['layer0']['summarize']:.2f}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    layers = np.arange(L)
    ax.axhline(0.5, color=GRID, lw=1, zorder=1)
    ax.text(layers[-1], 0.495, "chance", color=INK2, fontsize=8, ha="right", va="top")
    ax.plot(layers, curves["rate"], color=BLUE, lw=2, zorder=3, label='probe, "rate" prompt')
    ax.plot(layers, curves["summarize"], color=ORANGE, lw=2, zorder=3, label='probe, "summarize" prompt')
    ax.plot(layers, curves["rate_to_summarize"], color=AQUA, lw=1.5, ls=(0, (4, 2)), zorder=2,
            label="trained on rate, tested on summarize")
    ax.plot(layers, curves["summarize_to_rate"], color=YELLOW, lw=1.5, ls=(0, (4, 2)), zorder=2,
            label="trained on summarize, tested on rate")
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title(f"{label} pairs: does the probe need the rating instruction?\n"
                 f'probe AUROC: rate {metrics["rate_probe"]["auroc"]:.2f}, summarize {metrics["summarize_probe"]["auroc"]:.2f}',
                 color=INK, fontsize=10, loc="left")
    ax.set_xlabel("layer (0 = embeddings)"); ax.set_ylabel("held-out AUROC"); ax.set_ylim(0.3, 1.0)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "layer_curves.png", dpi=160, facecolor=SURFACE)
    print(f"saved {(out / 'layer_curves.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
