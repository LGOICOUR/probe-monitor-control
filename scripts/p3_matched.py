"""P3 on the same-author matched set, plus transfer in both directions against the
original Option-A pairing (human honest vs dataset backdoor). Local, $0.

Same settings as the baseline P3: 5 grouped rounds,
layer and C chosen inside training, every paper scored once.

Usage:
    uv run python scripts/p3_matched.py                  # Haiku matched set
    uv run python scripts/p3_matched.py --set sonnet     # Sonnet matched set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_monitor.baselines import word_counter_scores  # noqa: E402
from probe_monitor.probe import catch_rate_at_fpr, grouped_bootstrap_ci, nested_probe  # noqa: E402
from probe_monitor.views import strip_comments  # noqa: E402


def load(cond: str, papers_file: str):
    d = ROOT / "results" / "p2" / cond
    rows = [json.loads(l) for l in (d / "ratings.jsonl").open()]
    A = np.load(d / "acts.npy").astype(np.float32)
    papers = {(r["problem_id"], r["source"]): r for r in map(json.loads, (ROOT / "results" / "p1" / papers_file).open())}
    code = [strip_comments(papers[(r["problem_id"], r["source"])]["code"]) for r in rows]
    return {"A": A, "source": np.array([r["source"] for r in rows]), "pid": np.array([r["problem_id"] for r in rows]),
            "spoken": np.array([r["expected_rating"] for r in rows]), "code": code}


def block(name, y, s, g):
    lo, hi = grouped_bootstrap_ci(y, s, g, roc_auc_score)
    clo, chi = grouped_bootstrap_ci(y, s, g, catch_rate_at_fpr)
    return {"monitor": name, "auroc": float(roc_auc_score(y, s)), "auroc_ci": [lo, hi],
            "catch_at_2pct": catch_rate_at_fpr(y, s), "catch_ci": [clo, chi]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["haiku", "sonnet"], default="haiku")
    args = ap.parse_args()
    suffix = "" if args.set == "haiku" else "_sonnet"
    M = load(f"rate_stripped_matched{suffix}", f"papers_matched{suffix}.jsonl")
    O = load("rate_stripped", "papers.jsonl")
    out = ROOT / "results" / "p3" / f"rate_stripped_matched{suffix}"
    out.mkdir(parents=True, exist_ok=True)

    ym, gm = (M["source"] == "sub_backdoor").astype(int), M["pid"]
    mo = np.isin(O["source"], ["human_honest", "backdoor"])
    yo, go = (O["source"][mo] == "backdoor").astype(int), O["pid"][mo]
    Ao = O["A"][mo]

    # matched pairing, with transfer to the original pairing (problems held out together)
    res_m = nested_probe(M["A"], ym, gm, X_transfer=Ao, y_transfer=yo, g_transfer=go)
    # original pairing, with transfer to the matched pairing
    res_o = nested_probe(Ao, yo, go, X_transfer=M["A"], y_transfer=ym, g_transfer=gm)

    L = M["A"].shape[1]
    curve_m = np.array([roc_auc_score(ym, res_m["layer_scores"][l]) for l in range(L)])
    t_mask_a = res_m["t_folds"] >= 0
    curve_m_to_o = np.array([roc_auc_score(yo[t_mask_a], res_m["t_layer_scores"][l][t_mask_a]) for l in range(L)])
    curve_o_to_m = np.array([roc_auc_score(ym, res_o["t_layer_scores"][l]) for l in range(L)])

    length = np.array([len(c) for c in M["code"]], dtype=float)
    words = word_counter_scores(M["code"], ym, gm)
    metrics = {
        "n_pairs": int(ym.sum()),
        "matched": {
            "probe": block("probe (layer chosen inside training)", ym, res_m["headline"], gm),
            "spoken": block("TA spoken score", ym, M["spoken"], gm),
            "length": block("length ruler", ym, length, gm),
            "word_counter": block("word counter (no TA)", ym, words, gm),
            "chosen_layers": [c["layer"] for c in res_m["chosen"]],
            "layer0_auroc": float(curve_m[0]),
            "hindsight": {"layer": int(np.argmax(curve_m)), "auroc": float(curve_m.max())},
        },
        "transfer_matched_to_original": block("trained matched, tested human vs dataset backdoor",
                                              yo[t_mask_a], res_m["t_headline"][t_mask_a], go[t_mask_a]),
        "transfer_original_to_matched": block("trained human vs dataset backdoor, tested matched",
                                              ym, res_o["t_headline"], gm),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    np.savez(out / "layer_curves.npz", matched=curve_m, matched_to_original=curve_m_to_o, original_to_matched=curve_o_to_m)

    def f(b):
        return (f"{b['auroc']:.2f} [{b['auroc_ci'][0]:.2f}, {b['auroc_ci'][1]:.2f}] | "
                f"{100 * b['catch_at_2pct']:.0f}% [{100 * b['catch_ci'][0]:.0f}, {100 * b['catch_ci'][1]:.0f}]")
    m = metrics["matched"]
    lines = [f"# P3 on the {args.set} matched set ({metrics['n_pairs']} same-author pairs)", "",
             "AUROC [95% CI] | catch rate at 2% false alarms [95% CI]. CIs resample problems.", "",
             "| monitor | AUROC | catch at 2% |", "|---|---|---|"]
    for key in ("probe", "spoken", "word_counter", "length"):
        a, c = f(m[key]).split(" | ")
        lines.append(f"| {m[key]['monitor']} | {a} | {c} |")
    for key in ("transfer_matched_to_original", "transfer_original_to_matched"):
        a, c = f(metrics[key]).split(" | ")
        lines.append(f"| transfer: {metrics[key]['monitor']} | {a} | {c} |")
    lines += ["", f"- Layers chosen inside training: {m['chosen_layers']}",
              f"- Hindsight layer: {m['hindsight']['layer']}, AUROC {m['hindsight']['auroc']:.2f}",
              f"- Sanity, layer 0: {m['layer0_auroc']:.2f}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
