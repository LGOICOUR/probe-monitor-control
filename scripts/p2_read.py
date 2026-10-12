"""P2: have the TA read every paper once and record it. Local, $0.

Baseline (default flags): prompt=rate, view=stripped. Each lever changes one
flag and writes to its own folder, so conditions never overwrite each other:
    --view raw            lever 3: comments left in
    --prompt summarize    lever 4: input-vs-judgment control

Outputs in results/p2/<prompt>_<view>/:
    ratings.jsonl   one row per paper: ids, label, rating probabilities, ...
    acts.npy        float16 [n_papers, n_layers + 1, width], same row order
    config.json     model, prompt text, view, position, counts

Usage:
    uv run python scripts/p2_read.py                 # baseline
    uv run python scripts/p2_read.py --limit 3       # quick check
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
from probe_monitor.reader import MODEL_ID, PROMPTS, build_prompt, digit_token_ids, load_ta, read_one  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="rate")
    ap.add_argument("--view", choices=["stripped", "raw"], default="stripped")
    ap.add_argument("--limit", type=int, default=None, help="first N papers only (for checks)")
    ap.add_argument("--papers", default="papers.jsonl", help="paper file in results/p1/")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / "results" / "p1" / args.papers).open()]
    if args.limit:
        rows = rows[: args.limit]
    tag = "" if args.papers == "papers.jsonl" else "_" + Path(args.papers).stem.replace("papers_", "")
    cond = f"{args.prompt}_{args.view}{tag}" + (f"_limit{args.limit}" if args.limit else "")
    out_dir = ROOT / "results" / "p2" / cond
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model, device = load_ta()
    digit_ids = digit_token_ids(tok)
    acts, t0 = [], time.time()
    with (out_dir / "ratings.jsonl").open("w") as f:
        for i, r in enumerate(rows):
            view = r["view"] if args.view == "stripped" else r["view_raw"]
            res = read_one(tok, model, device, build_prompt(view, args.prompt), digit_ids)
            acts.append(res.pop("acts"))
            f.write(json.dumps({"i": i, "problem_id": r["problem_id"], "source": r["source"],
                                "author": r["author"], "label": r["label"], **res}) + "\n")
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                print(f"{i + 1}/{len(rows)} papers, {time.time() - t0:.0f}s", flush=True)
    A = np.stack(acts)
    np.save(out_dir / "acts.npy", A)
    (out_dir / "config.json").write_text(json.dumps({
        "model": MODEL_ID, "prompt_kind": args.prompt, "prompt_template": PROMPTS[args.prompt],
        "view": args.view, "position": "last prompt token (after chat template's assistant header)",
        "n_papers": len(rows), "acts_shape": list(A.shape), "acts_dtype": "float16",
        "layer_index_note": "0 = embeddings, k = output of decoder layer k",
    }, indent=1))
    print(f"saved {out_dir.relative_to(ROOT)}  acts {A.shape}")


if __name__ == "__main__":
    main()
