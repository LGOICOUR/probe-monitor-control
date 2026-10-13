"""Lever 2b analysis: reasoning-first TA vs the immediate-answer baseline.

Compares, on the same papers:
  * the TA's spoken score when it answers immediately (baseline recording)
  * the TA's spoken score after writing a rationale (parsed digit, and the
    expected rating from its digit probabilities at the pre-digit position)
  * probes on each recording, and each probe tested on the other recording
  * a transcript judge: does the rationale's own wording separate the classes?
    (TF-IDF + logistic regression on the rationale text, grouped folds)

Usage:
    uv run python scripts/p3_reason.py --set sonnet
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SETS = {"sonnet": "_matched_sonnet", "haiku": "_matched", "original": ""}


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
    rows_b, B = load(f"rate_stripped{suffix}")
    rows_r, R = load(f"reason_stripped{suffix}")
    keys_r = {(r["problem_id"], r["source"]): i for i, r in enumerate(rows_r)}
    # align baseline to the reasoning rows (the original set also holds haiku_honest papers)
    idx_b = [i for i, r in enumerate(rows_b) if (r["problem_id"], r["source"]) in keys_r]
    rows_b = [rows_b[i] for i in idx_b]; B = B[idx_b]
    assert [(r["problem_id"], r["source"]) for r in rows_b] == [(r["problem_id"], r["source"]) for r in rows_r]
    y = np.array([r["label"] for r in rows_r]); g = np.array([r["problem_id"] for r in rows_r])
    spoken_imm = np.array([r["expected_rating"] for r in rows_b])
    spoken_reason = np.array([r["expected_rating"] for r in rows_r])
    parsed = np.array([r["parsed_rating"] if r["parsed_rating"] is not None else 0 for r in rows_r], dtype=float)
    n_unparsed = sum(r["parsed_rating"] is None for r in rows_r)

    res_r = nested_probe(R, y, g, X_transfer=B, y_transfer=y, g_transfer=g)
    res_b = nested_probe(B, y, g, X_transfer=R, y_transfer=y, g_transfer=g)
    L = B.shape[1]
    curves = {
        "immediate": np.array([roc_auc_score(y, res_b["layer_scores"][l]) for l in range(L)]),
        "after_reasoning": np.array([roc_auc_score(y, res_r["layer_scores"][l]) for l in range(L)]),
        "immediate_to_reasoning": np.array([roc_auc_score(y, res_b["t_layer_scores"][l]) for l in range(L)]),
        "reasoning_to_immediate": np.array([roc_auc_score(y, res_r["t_layer_scores"][l]) for l in range(L)]),
    }
    transcript = word_counter_scores([r["reasoning"] for r in rows_r], y, g)
    metrics = {
        "set": args.set, "n_pairs": int(y.sum()), "n_unparsed": n_unparsed,
        "spoken_immediate": block("TA spoken score, immediate answer", y, spoken_imm, g),
        "spoken_after_reasoning_probs": block("TA spoken score after reasoning (digit probabilities)", y, spoken_reason, g),
        "spoken_after_reasoning_parsed": block("TA spoken score after reasoning (written digit)", y, parsed, g),
        "probe_immediate": block("probe, immediate-answer recording", y, res_b["headline"], g),
        "probe_after_reasoning": block("probe, post-rationale recording", y, res_r["headline"], g),
        "transfer_immediate_to_reasoning": block("probe trained immediate, tested post-rationale", y, res_b["t_headline"], g),
        "transfer_reasoning_to_immediate": block("probe trained post-rationale, tested immediate", y, res_r["t_headline"], g),
        "transcript_words": block("word counter on the rationale text", y, transcript, g),
        "layers_chosen": {"immediate": [c["layer"] for c in res_b["chosen"]], "after_reasoning": [c["layer"] for c in res_r["chosen"]]},
        "parsed_hist": {str(k): v for k, v in sorted(Counter(int(x) for x in parsed).items())},
        "mean_generated_tokens": float(np.mean([r["n_generated"] for r in rows_r])),
    }
    out = ROOT / "results" / "p3" / f"reason_control{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    np.savez(out / "layer_curves.npz", **curves)

    lines = [f"# Reasoning-first TA, {args.set} set ({metrics['n_pairs']} pairs)", "",
             "AUROC [95% CI] | catch at 2% [95% CI]", "", "| condition | AUROC | catch at 2% |", "|---|---|---|"]
    for k in ("spoken_immediate", "spoken_after_reasoning_probs", "spoken_after_reasoning_parsed",
              "probe_immediate", "probe_after_reasoning", "transfer_immediate_to_reasoning",
              "transfer_reasoning_to_immediate", "transcript_words"):
        a, c = fmt(metrics[k]).split(" | ")
        lines.append(f"| {metrics[k]['monitor']} | {a} | {c} |")
    lines += ["", f"- Unparsed final digits: {n_unparsed}; written-digit histogram {metrics['parsed_hist']}; "
                  f"mean rationale length {metrics['mean_generated_tokens']:.0f} tokens",
              f"- Layers chosen inside training: immediate {metrics['layers_chosen']['immediate']}, "
              f"post-rationale {metrics['layers_chosen']['after_reasoning']}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    layers = np.arange(L)
    ax.axhline(0.5, color=GRID, lw=1, zorder=1)
    ax.text(0.5, 0.495, "chance", color=INK2, fontsize=8, va="top")
    for key, val, dy in (("spoken_immediate", spoken_imm, 0.008), ("spoken_after_reasoning_probs", spoken_reason, 0.008)):
        a = metrics[key]["auroc"]
        ax.axhline(a, color=INK2, lw=1, ls="-" if key == "spoken_immediate" else (0, (4, 3)), zorder=1)
        label = "spoken, immediate" if key == "spoken_immediate" else "spoken, after reasoning"
        ax.text(layers[-1], a + (0.008 if key != "spoken_immediate" else -0.008), f"{label} {a:.2f}", color=INK2,
                fontsize=8, ha="right", va="bottom" if key != "spoken_immediate" else "top")
    ax.plot(layers, curves["immediate"], color=BLUE, lw=2, zorder=3, label="probe, immediate-answer recording")
    ax.plot(layers, curves["after_reasoning"], color=ORANGE, lw=2, zorder=3, label="probe, post-rationale recording")
    ax.plot(layers, curves["immediate_to_reasoning"], color=AQUA, lw=1.5, ls=(0, (4, 2)), zorder=2,
            label="trained immediate, tested post-rationale")
    ax.plot(layers, curves["reasoning_to_immediate"], color=YELLOW, lw=1.5, ls=(0, (4, 2)), zorder=2,
            label="trained post-rationale, tested immediate")
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title(f"{args.set} pairs: does letting the TA reason first close the gap?\n"
                 f"spoken: immediate {metrics['spoken_immediate']['auroc']:.2f} vs after reasoning "
                 f"{metrics['spoken_after_reasoning_probs']['auroc']:.2f}; probe {metrics['probe_immediate']['auroc']:.2f} vs "
                 f"{metrics['probe_after_reasoning']['auroc']:.2f}", color=INK, fontsize=10, loc="left")
    ax.set_xlabel("layer (0 = embeddings)"); ax.set_ylabel("held-out AUROC"); ax.set_ylim(0.3, 1.0)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "layer_curves.png", dpi=160, facecolor=SURFACE)
    print(f"saved {(out / 'layer_curves.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
