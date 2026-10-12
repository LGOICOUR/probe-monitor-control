"""Lever: frontier judge. A frontier Claude model reads the same views with the
same 0-9 prompt as the local monitor and answers with one digit (extended
thinking disabled; the SDK no longer exposes temperature).

Spend guard: tokens are counted up front (chars/3.5 as a conservative
estimate), the run refuses to start if the estimate exceeds --cap dollars,
and every call records real usage so the final cost is printed. Results are
cached per (model, paper) so reruns never pay twice.

Usage:
    uv run python scripts/p4_judge.py --model claude-opus-5-5 --sets sonnet haiku original --cap 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_monitor.reader import build_prompt  # noqa: E402

# USD per million tokens, from platform.claude.com/docs/en/about-claude/pricing (read 2026-10-12)
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}
# Request shape per model. Opus 5.5 cannot run without thinking, so it reasons
# briefly (effort "low") before answering and its thinking tokens are billed as
# output; max_tokens must leave room for them. Sonnet 5.5 with thinking turned
# off wrote its reasoning into the visible answer instead of a digit, so it
# runs the same way as Opus. Neither accepts temperature.
REQUEST = {
    "claude-opus-5-5": dict(max_tokens=2048, output_config={"effort": "low"}),
    "claude-sonnet-5-5": dict(max_tokens=2048, output_config={"effort": "low"}),
    "claude-haiku-4-5-20251001": dict(max_tokens=8),
}
EXPECTED_OUT_TOKENS = {"claude-opus-5-5": 150, "claude-sonnet-5-5": 150, "claude-haiku-4-5-20251001": 4}


def parse_rating(text: str) -> int | None:
    """Accept only a standalone digit as the final line (markdown emphasis and
    trailing punctuation stripped); prose with digits inside is None."""
    last = text.strip().splitlines()[-1].strip() if text.strip() else ""
    last = re.sub(r"^[\s*_`#>]+|[\s*_`.)]+$", "", last)
    last = re.sub(r"^(?:rating|score|suspicion)\s*[:=]\s*", "", last, flags=re.I)
    return int(last) if re.fullmatch(r"\d", last) else None
SETS = {
    "sonnet": "papers_matched_sonnet.jsonl",
    "haiku": "papers_matched.jsonl",
    "original": "papers.jsonl",
}


def estimate_cost(model: str, prompts: list[str]) -> float:
    pin, pout = PRICES[model]
    tokens_in = sum(len(p) for p in prompts) / 3.5
    return tokens_in / 1e6 * pin + len(prompts) * EXPECTED_OUT_TOKENS[model] / 1e6 * pout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PRICES))
    ap.add_argument("--sets", nargs="+", choices=sorted(SETS), default=["sonnet"])
    ap.add_argument("--cap", type=float, required=True, help="refuse if the estimate exceeds this many dollars")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")
    import anthropic

    client = anthropic.Anthropic()
    out_dir = ROOT / "results" / "p4_judge" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.open():
            r = json.loads(line)
            cache[(r["set"], r["problem_id"], r["source"])] = r

    jobs = []
    for s in args.sets:
        rows = [json.loads(l) for l in (ROOT / "results" / "p1" / SETS[s]).open()]
        if args.limit:
            rows = rows[: args.limit]
        for r in rows:
            key = (s, r["problem_id"], r["source"])
            if key not in cache:
                jobs.append((s, r))
    prompts = [build_prompt(r["view"], "rate") for _, r in jobs]
    est = estimate_cost(args.model, prompts)
    print(f"{len(jobs)} uncached papers for {args.model}; estimated cost ${est:.2f} (cap ${args.cap:.2f})")
    if est > args.cap:
        raise SystemExit("refusing to run: estimate exceeds cap")

    pin, pout = PRICES[args.model]
    spent, t0 = 0.0, time.time()
    with cache_path.open("a") as f:
        for i, ((s, r), prompt) in enumerate(zip(jobs, prompts)):
            for attempt in range(4):
                try:
                    resp = client.messages.create(model=args.model, **REQUEST[args.model],
                                                  messages=[{"role": "user", "content": prompt}])
                    break
                except anthropic.RateLimitError:
                    time.sleep(5 * (attempt + 1))
            else:
                raise SystemExit("rate limited repeatedly; stopping (cache keeps progress)")
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            rating = parse_rating(text) if resp.stop_reason == "end_turn" else None
            u = resp.usage
            spent += u.input_tokens / 1e6 * pin + u.output_tokens / 1e6 * pout
            rec = {"set": s, "problem_id": r["problem_id"], "source": r["source"], "label": r["label"],
                   "rating": rating, "raw": text[-200:], "stop_reason": resp.stop_reason,
                   "in": u.input_tokens, "out": u.output_tokens}
            f.write(json.dumps(rec) + "\n"); f.flush()
            if spent > args.cap:
                raise SystemExit(f"stopping: real spend ${spent:.2f} exceeded cap")
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)}  spent ${spent:.2f}  {time.time() - t0:.0f}s", flush=True)
    print(f"done; this run spent ${spent:.2f}")


if __name__ == "__main__":
    main()
