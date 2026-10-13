"""Lever: is the probe direction load-bearing for the monitor's own rating?

On the human-honest vs dataset-backdoor set (the monitor's rating is 0.74 there,
so it has room to move), for each held-out fold:
  1. fit a probe on the training papers at each layer in scope -> a unit
     direction u in raw activation space, and the training honest mean mu;
  2. re-run the monitor on each held-out paper with a forward hook that, at
     the readout position of each in-scope layer, replaces the activation h by
     h - ((h - mu) . u) u   (project the paper onto the honest mean along u);
  3. read the digit probabilities -> spoken score after ablation;
  4. also record the final-layer activation to check whether a downstream
     probe (trained unablated, layer 36) still decodes the label.
Control: the same operation along a random unit direction of each layer.

Scopes: "single" = layer 22 (the headline layer); "band" = layers 18-26.

Usage:
    uv run python scripts/p4_ablate.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_monitor.probe import catch_rate_at_fpr, fit_probe, grouped_bootstrap_ci, grouped_folds, probe_scores  # noqa: E402
from probe_monitor.reader import build_prompt, digit_token_ids, load_ta  # noqa: E402

SCOPES = {"single": [22], "band": list(range(18, 27))}
FINAL = 36


def directions(M, y, g, layers, seed):
    """Per layer: probe unit direction (raw space), honest mean, random unit direction."""
    out = {}
    rng = np.random.default_rng(seed)
    for L in layers:
        sc, clf, _ = fit_probe(M, y, g, L)
        w = clf.coef_[0] / sc.scale_  # back to raw activation space
        u = w / np.linalg.norm(w)
        mu = M[y == 0, L].mean(0)
        r = rng.normal(size=u.shape); r /= np.linalg.norm(r)
        out[L] = (u.astype(np.float32), mu.astype(np.float32), r.astype(np.float32))
    return out


def make_hook(u, mu, device):
    u_t = torch.tensor(u, device=device); mu_t = torch.tensor(mu, device=device)

    def hook(_m, _a, out):
        h = out[0] if isinstance(out, tuple) else out
        last = h[0, -1].float()
        proj = torch.dot(last - mu_t, u_t)
        h[0, -1] = (last - proj * u_t).to(h.dtype)
        return out

    return hook


@torch.no_grad()
def read_with_ablation(tok, model, device, prompt_text, digit_ids, dirs, use_random):
    msgs = [{"role": "user", "content": prompt_text}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
    handles = []
    for L, (u, mu, r) in dirs.items():
        # our layer index L = hidden_states[L] = output of decoder layer L-1
        handles.append(model.model.layers[L - 1].register_forward_hook(make_hook(r if use_random else u, mu, device)))
    try:
        out = model(**enc, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()
    probs = out.logits[0, -1].float().softmax(-1)[digit_ids].cpu().numpy()
    probs = probs / probs.sum()
    final = out.hidden_states[FINAL][0, -1].float().cpu().numpy()
    return float(np.dot(np.arange(10), probs)), final


def block(y, s, g):
    lo, hi = grouped_bootstrap_ci(y, s, g, roc_auc_score)
    return {"auroc": float(roc_auc_score(y, s)), "auroc_ci": [lo, hi], "catch_at_2pct": catch_rate_at_fpr(y, s)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--scopes", nargs="+", choices=sorted(SCOPES), default=["single", "band"])
    args = ap.parse_args()
    d = ROOT / "results" / "p2" / "rate_stripped"
    rows_all = [json.loads(l) for l in (d / "ratings.jsonl").open()]
    A_all = np.load(d / "acts.npy").astype(np.float32)
    keep = [i for i, r in enumerate(rows_all) if r["source"] in ("human_honest", "backdoor")]
    rows = [rows_all[i] for i in keep]; A = A_all[keep]
    papers = {(r["problem_id"], r["source"]): r for r in map(json.loads, (ROOT / "results" / "p1" / "papers.jsonl").open())}
    if args.limit:
        rows, A = rows[: args.limit], A[: args.limit]
    y = np.array([r["label"] for r in rows]); g = np.array([r["problem_id"] for r in rows])
    spoken0 = np.array([r["expected_rating"] for r in rows])
    folds = grouped_folds(g, 5, 0)
    tok, model, device = load_ta()
    digit_ids = digit_token_ids(tok)

    results = {"n": len(rows), "baseline_spoken": block(y, spoken0, g)}
    t0 = time.time()
    for scope in args.scopes:
        layers = SCOPES[scope]
        spoken = {"probe": np.zeros(len(y)), "random": np.zeros(len(y))}
        final = {"probe": np.zeros((len(y), A.shape[2]), np.float32), "random": np.zeros((len(y), A.shape[2]), np.float32)}
        down = {"probe": np.zeros(len(y)), "random": np.zeros(len(y))}
        for f in range(5):
            tr, te = folds != f, np.where(folds == f)[0]
            dirs = directions(A[tr], y[tr], g[tr], layers, seed=f)
            sc36, clf36, _ = fit_probe(A[tr], y[tr], g[tr], FINAL)  # downstream decodability check
            for which in ("probe", "random"):
                for i in te:
                    r = rows[i]
                    s, fin = read_with_ablation(tok, model, device, build_prompt(papers[(r["problem_id"], r["source"])]["view"], "rate"),
                                                digit_ids, dirs, use_random=(which == "random"))
                    spoken[which][i] = s; final[which][i] = fin
                down[which][te] = probe_scores(sc36, clf36, final[which][te][:, None, :].repeat(FINAL + 1, axis=1), FINAL)
            print(f"{scope}: fold {f + 1}/5 done, {time.time() - t0:.0f}s", flush=True)
        results[scope] = {
            "layers": layers,
            "spoken_after_probe_ablation": block(y, spoken["probe"], g),
            "spoken_after_random_ablation": block(y, spoken["random"], g),
            "downstream_probe_layer36_after_probe_ablation": block(y, down["probe"], g),
            "downstream_probe_layer36_after_random_ablation": block(y, down["random"], g),
            "mean_rating_shift_probe": {"honest": float(np.mean(spoken["probe"][y == 0] - spoken0[y == 0])),
                                        "backdoor": float(np.mean(spoken["probe"][y == 1] - spoken0[y == 1]))},
            "mean_rating_shift_random": {"honest": float(np.mean(spoken["random"][y == 0] - spoken0[y == 0])),
                                         "backdoor": float(np.mean(spoken["random"][y == 1] - spoken0[y == 1]))},
        }
    out = ROOT / "results" / "p4_ablation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))

    def f(b):
        return f"{b['auroc']:.2f} [{b['auroc_ci'][0]:.2f}, {b['auroc_ci'][1]:.2f}] | {100 * b['catch_at_2pct']:.0f}%"
    lines = [f"# Probe-direction ablation, human honest vs dataset backdoor ({int(y.sum())} pairs)", "",
             "Spoken score = monitor's expected rating from its digit probabilities. Downstream check = a layer-36 probe trained on unablated training papers, applied to the ablated held-out papers.", "",
             "| condition | spoken AUROC [95% CI] | catch at 2% |", "|---|---|---|",
             f"| baseline, no ablation | {f(results['baseline_spoken'])} |"]
    for scope in args.scopes:
        R = results[scope]
        lines += [f"| {scope} (layers {R['layers'][0]}–{R['layers'][-1]}): probe direction removed | {f(R['spoken_after_probe_ablation'])} |",
                  f"| {scope}: random direction removed (control) | {f(R['spoken_after_random_ablation'])} |"]
    lines += ["", "| condition | downstream layer-36 probe AUROC | catch |", "|---|---|---|"]
    for scope in args.scopes:
        R = results[scope]
        lines += [f"| {scope}: after probe-direction ablation | {f(R['downstream_probe_layer36_after_probe_ablation'])} |",
                  f"| {scope}: after random ablation | {f(R['downstream_probe_layer36_after_random_ablation'])} |"]
    lines += [""]
    for scope in args.scopes:
        R = results[scope]
        lines.append(f"- {scope}: mean rating shift after probe ablation, honest {R['mean_rating_shift_probe']['honest']:+.2f}, "
                     f"backdoor {R['mean_rating_shift_probe']['backdoor']:+.2f}; random: honest {R['mean_rating_shift_random']['honest']:+.2f}, "
                     f"backdoor {R['mean_rating_shift_random']['backdoor']:+.2f}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
