"""Lever 2b: the TA reasons first, then rates. Local, $0.

Writes results/p2/reason_stripped<tag>/: ratings.jsonl (rationale text, parsed
digit, digit probabilities at the pre-digit position) and acts.npy (activation
at that same position, [n, n_layers + 1, width]).

Usage:
    uv run python scripts/p2_reason.py --papers papers_matched_sonnet.jsonl [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_monitor.reader import MODEL_ID, PROMPTS, build_prompt, digit_token_ids, load_ta, reason_one  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="papers_matched_sonnet.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    args = ap.parse_args()
    rows = [json.loads(l) for l in (ROOT / "results" / "p1" / args.papers).open()]
    if args.limit:
        rows = rows[: args.limit]
    tag = "" if args.papers == "papers.jsonl" else "_" + Path(args.papers).stem.replace("papers_", "")
    cond = f"reason_stripped{tag}" + (f"_limit{args.limit}" if args.limit else "")
    out_dir = ROOT / "results" / "p2" / cond
    out_dir.mkdir(parents=True, exist_ok=True)
    tok, model, device = load_ta()
    digit_ids = digit_token_ids(tok)
    acts, t0 = [], time.time()
    with (out_dir / "ratings.jsonl").open("w") as f:
        for i, r in enumerate(rows):
            res = reason_one(tok, model, device, build_prompt(r["view"], "reason"), digit_ids, args.max_new_tokens)
            acts.append(res.pop("acts"))
            f.write(json.dumps({"i": i, "problem_id": r["problem_id"], "source": r["source"], "label": r["label"], **res}) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(rows):
                print(f"{i + 1}/{len(rows)} papers, {time.time() - t0:.0f}s", flush=True)
    A = np.stack(acts)
    np.save(out_dir / "acts.npy", A)
    (out_dir / "config.json").write_text(json.dumps({
        "model": MODEL_ID, "prompt_kind": "reason", "prompt_template": PROMPTS["reason"], "view": "stripped",
        "position": "token before the final digit (after the rationale)", "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens, "n_papers": len(rows), "acts_shape": list(A.shape)}, indent=1))
    print(f"saved {out_dir.relative_to(ROOT)}  acts {A.shape}")


if __name__ == "__main__":
    main()
