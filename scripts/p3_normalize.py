"""Lever "erase handwriting": the same papers with every user-defined identifier
renamed to a generic token, docstrings dropped, and formatting re-printed from
the syntax tree. Compares, on the same problems:
  * the TA's spoken score and the probe on the original (comment-stripped) view
  * the same on the normalized view
  * the word counter and length ruler on both
  * probes tested across the two recordings

Usage:
    uv run python scripts/p3_normalize.py --set sonnet
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
from probe_monitor.baselines import word_counter_scores  # noqa: E402
from probe_monitor.probe import catch_rate_at_fpr, grouped_bootstrap_ci, nested_probe  # noqa: E402
from probe_monitor.views import strip_comments  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SETS = {"sonnet": "_matched_sonnet", "haiku": "_matched"}


def load(cond):
    d = ROOT / "results" / "p2" / cond
    rows = [json.loads(l) for l in (d / "ratings.jsonl").open()]
    return rows, np.load(d / "acts.npy").astype(np.float32)


def block(name, y, s, g):
    lo, hi = grouped_bootstrap_ci(y, s, g, roc_auc_score)
    clo, chi = grouped_bootstrap_ci(y, s, g, catch_rate_at_fpr)
    return {"monitor": name, "auroc": float(roc_auc_score(y, s)), "auroc_ci": [lo, hi],
            "catch_at_2pct": catch_rate_at_fpr(y, s), "catch_ci": [clo, chi]}


def fmt(b):
    return (f"{b['auroc']:.2f} [{b['auroc_ci'][0]:.2f}, {b['auroc_ci'][1]:.2f}] | "
            f"{100 * b['catch_at_2pct']:.0f}% [{100 * b['catch_ci'][0]:.0f}, {100 * b['catch_ci'][1]:.0f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default="sonnet")
    args = ap.parse_args()
    suffix = SETS[args.set]
    rows_o, O = load(f"rate_stripped{suffix}")
    rows_n, N = load(f"rate_stripped{suffix}_norm")
    assert [(r["problem_id"], r["source"]) for r in rows_o] == [(r["problem_id"], r["source"]) for r in rows_n]
    papers_n = {(r["problem_id"], r["source"]): r for r in map(json.loads, (ROOT / "results" / "p1" / f"papers{suffix}_norm.jsonl").open())}
    y = np.array([r["label"] for r in rows_o]); g = np.array([r["problem_id"] for r in rows_o])
    code_o = [strip_comments(papers_n[(r["problem_id"], r["source"])]["code"]) for r in rows_o]
    code_n = [papers_n[(r["problem_id"], r["source"])]["code_norm"] for r in rows_o]

    res_o = nested_probe(O, y, g, X_transfer=N, y_transfer=y, g_transfer=g)
    res_n = nested_probe(N, y, g, X_transfer=O, y_transfer=y, g_transfer=g)
    L = O.shape[1]
    curves = {
        "original": np.array([roc_auc_score(y, res_o["layer_scores"][l]) for l in range(L)]),
        "normalized": np.array([roc_auc_score(y, res_n["layer_scores"][l]) for l in range(L)]),
        "original_to_normalized": np.array([roc_auc_score(y, res_o["t_layer_scores"][l]) for l in range(L)]),
        "normalized_to_original": np.array([roc_auc_score(y, res_n["t_layer_scores"][l]) for l in range(L)]),
    }
    metrics = {
        "set": args.set, "n_pairs": int(y.sum()),
        "probe_original": block("probe, original view", y, res_o["headline"], g),
        "probe_normalized": block("probe, normalized view", y, res_n["headline"], g),
        "spoken_original": block("TA spoken score, original view", y, np.array([r["expected_rating"] for r in rows_o]), g),
        "spoken_normalized": block("TA spoken score, normalized view", y, np.array([r["expected_rating"] for r in rows_n]), g),
        "words_original": block("word counter, original code", y, word_counter_scores(code_o, y, g), g),
        "words_normalized": block("word counter, normalized code", y, word_counter_scores(code_n, y, g), g),
        "length_original": block("length ruler, original", y, np.array([len(c) for c in code_o], float), g),
        "length_normalized": block("length ruler, normalized", y, np.array([len(c) for c in code_n], float), g),
        "transfer_original_to_normalized": block("probe trained original, tested normalized", y, res_o["t_headline"], g),
        "transfer_normalized_to_original": block("probe trained normalized, tested original", y, res_n["t_headline"], g),
        "layers_chosen": {"original": [c["layer"] for c in res_o["chosen"]], "normalized": [c["layer"] for c in res_n["chosen"]]},
    }
    out = ROOT / "results" / "p3" / f"normalize_control{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    np.savez(out / "layer_curves.npz", **curves)
    lines = [f"# Erase-handwriting control, {args.set} set ({metrics['n_pairs']} pairs)", "",
             "AUROC [95% CI] | catch at 2% [95% CI]", "", "| condition | AUROC | catch at 2% |", "|---|---|---|"]
    for k in ("probe_original", "probe_normalized", "spoken_original", "spoken_normalized", "words_original",
              "words_normalized", "length_original", "length_normalized", "transfer_original_to_normalized",
              "transfer_normalized_to_original"):
        a, c = fmt(metrics[k]).split(" | ")
        lines.append(f"| {metrics[k]['monitor']} | {a} | {c} |")
    lines += ["", f"- Layers chosen inside training: original {metrics['layers_chosen']['original']}, normalized {metrics['layers_chosen']['normalized']}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    layers = np.arange(L)
    ax.axhline(0.5, color=GRID, lw=1, zorder=1)
    ax.text(0.5, 0.495, "chance", color=INK2, fontsize=8, va="top")
    for key, ls, dy, va in (("words_original", "-", -0.008, "top"), ("words_normalized", (0, (4, 3)), 0.008, "bottom")):
        a = metrics[key]["auroc"]
        ax.axhline(a, color=GRID, lw=1, ls=ls, zorder=1)
        ax.text(layers[-1], a + dy, f"{metrics[key]['monitor']} {a:.2f}", color=INK2, fontsize=8, ha="right", va=va)
    ax.plot(layers, curves["original"], color=BLUE, lw=2, zorder=3, label="probe, original view")
    ax.plot(layers, curves["normalized"], color=ORANGE, lw=2, zorder=3, label="probe, normalized view")
    ax.plot(layers, curves["original_to_normalized"], color=AQUA, lw=1.5, ls=(0, (4, 2)), zorder=2,
            label="trained original, tested normalized")
    ax.plot(layers, curves["normalized_to_original"], color=YELLOW, lw=1.5, ls=(0, (4, 2)), zorder=2,
            label="trained normalized, tested original")
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title(f"{args.set} pairs: does the probe survive erased identifiers and formatting?\n"
                 f"probe {metrics['probe_original']['auroc']:.2f} → {metrics['probe_normalized']['auroc']:.2f}; "
                 f"word counter {metrics['words_original']['auroc']:.2f} → {metrics['words_normalized']['auroc']:.2f}; "
                 f"spoken {metrics['spoken_original']['auroc']:.2f} → {metrics['spoken_normalized']['auroc']:.2f}",
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
