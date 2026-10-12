"""Assemble the same-author matched paper set from the model-written batches.

A problem enters the set only if BOTH its model-written honest paper passes the test
bench AND its model-written backdoor passes the tests and fires (verify.json from
scripts/pilot_verify.py). Writes results/p1/papers_matched.jsonl in the same
format as papers.jsonl, with sources "sub_honest" / "sub_backdoor", plus a
leakage and confound report.

Usage:
    uv run python scripts/build_matched.py                       # Haiku set -> papers_matched.jsonl
    uv run python scripts/build_matched.py --set sonnet           # Sonnet set -> papers_matched_sonnet.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_monitor.views import Paper, comment_lines, giveaway_hits, leakage_problems, strip_comments  # noqa: E402

SETS = {
    "haiku": {"author": "anthropic/claude-haiku-4-5 (agentic, same instructions for both sides)",
              "dirs": ["results/pilot"] + sorted(glob.glob("results/batch*/part*")), "suffix": ""},
    "sonnet": {"author": "anthropic/claude-sonnet-5-5 (agentic, same instructions for both sides)",
               "dirs": ["results/sonnet_pilot"] + sorted(glob.glob("results/sonnet_batch*/part*")), "suffix": "_sonnet"},
}
MAGIC = re.compile(r"==\s*-?\d{3,}|-?\d{3,}\s*==|==\s*['\"][^'\"]{3,}['\"]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default="haiku")
    args = ap.parse_args()
    cfg = SETS[args.set]
    AUTHOR, suffix = cfg["author"], cfg["suffix"]
    problems = {}
    for r in map(json.loads, (ROOT / "results" / "p1" / "papers.jsonl").open()):
        if r["source"] == "human_honest":
            problems[r["problem_id"]] = re.search(r"<problem>\n(.*)\n</problem>", r["view"], re.S).group(1)

    dirs = [ROOT / d for d in cfg["dirs"]]
    papers, skipped = [], 0
    for d in dirs:
        verify = json.loads((d / "verify.json").read_text())
        code_dir = d / "code"
        for pid, v in verify.items():
            h, a = v.get("honest", {}), v.get("attack", {})
            if not (h.get("main_task_success") and a.get("main_task_success") and a.get("side_task_success")):
                skipped += 1
                continue
            hc = (code_dir / "honest" / f"{pid}.py").read_text()
            ac = (code_dir / "attack" / f"{pid}.py").read_text()
            sec = (code_dir / "attack" / f"{pid}.secret.txt").read_text()
            papers.append(Paper(pid, "sub_honest", AUTHOR, problems[pid], hc, 0, None, True, None,
                                {"batch_dir": str(d.relative_to(ROOT))}))
            papers.append(Paper(pid, "sub_backdoor", AUTHOR, problems[pid], ac, 1, sec, True, True,
                                {"batch_dir": str(d.relative_to(ROOT)),
                                 "magic_number_trigger": bool(MAGIC.search(strip_comments(ac)))}))

    out = ROOT / "results" / "p1" / f"papers_matched{suffix}.jsonl"
    with out.open("w") as f:
        for p in sorted(papers, key=lambda p: (int(p.problem_id), p.source != "sub_honest")):
            f.write(json.dumps({"problem_id": p.problem_id, "source": p.source, "author": p.author, "label": p.label,
                                "view": p.view, "view_raw": p.view_raw, "code": p.code, "secret_input": p.secret_input,
                                "main_task_success": p.main_task_success, "side_task_success": p.side_task_success,
                                **p.extra}) + "\n")

    leaks = [(p.problem_id, p.source, r) for p in papers for r in leakage_problems(p)]
    lines = [f"# Matched set: {len(papers) // 2} pairs ({skipped} problems skipped)", ""]
    lines += [f"- Leakage problems in views: {len(leaks)}"] + [f"  - {a} {b}: {c}" for a, b, c in leaks[:20]]
    lines += ["", "| source | n | median lines | median chars | % any comment (raw) | % confessing comment (raw) | % magic-number trigger |",
              "|---|---|---|---|---|---|---|"]
    for s in ("sub_honest", "sub_backdoor"):
        ps = [p for p in papers if p.source == s]
        stripped = [strip_comments(p.code) for p in ps]
        lines.append(
            f"| {s} | {len(ps)} | {st.median(len(x.splitlines()) for x in stripped)} | {st.median(len(x) for x in stripped)} | "
            f"{100 * sum(bool(comment_lines(p.code)) for p in ps) / len(ps):.0f} | "
            f"{100 * sum(bool(giveaway_hits(chr(10).join(comment_lines(p.code)))) for p in ps) / len(ps):.0f} | "
            f"{100 * sum(bool(MAGIC.search(x)) for x in stripped) / len(ps):.0f} |")
    report = "\n".join(lines) + "\n"
    (ROOT / "results" / "p1" / f"matched_report{suffix}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
