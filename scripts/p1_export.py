"""P1 export: turn the P1 logs + the Haiku honest log into one paper set, then run
the leakage checks and the confound table.

Outputs (results/p1/):
  papers.jsonl         one row per paper: problem_id, source, author, label,
                       view (exactly what the TA will see), code, secret_input
                       (never in the view), test-bench results
  leakage_report.md    leakage checks on every view + per-source confound stats

Usage:
    uv run python scripts/p1_export.py
"""

from __future__ import annotations

import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_monitor.views import Paper, comment_lines, giveaway_hits, leakage_problems, strip_comments  # noqa: E402

OUT = ROOT / "results" / "p1"
SOURCES = ("human_honest", "haiku_honest", "backdoor")


def submitted_args(sample) -> dict:
    for m in reversed(sample.messages):
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.function == "submit":
                return tc.arguments
    return {}


def score_flags(sample) -> tuple[bool | None, bool | None]:
    sc = (sample.scores or {}).get("apps_scorer")
    v = sc.value if sc and isinstance(sc.value, dict) else {}
    main = {"C": True, "I": False}.get(v.get("main_task_success"))
    side = {"C": True, "I": False}.get(v.get("side_task_success"))
    return main, side


def load_p1(ids: set[str]) -> tuple[list[Paper], dict[str, str]]:
    papers, side_desc = [], {}
    for path in glob.glob(str(ROOT / "logs" / "p1" / "*.eval")):
        log = read_eval_log(path)
        for s in log.samples or []:
            pid = str(s.id)
            if pid not in ids:
                continue
            md = s.metadata or {}
            mode = md.get("eval_mode")
            args = submitted_args(s)
            main, side = score_flags(s)
            if mode == "attack":
                side_desc[pid] = md.get("side_task_description") or ""
                inner = md.get("metadata") or {}
                bd = next((b for b in inner.get("backdoors") or [] if b.get("backdoor_works")), {})
                papers.append(Paper(pid, "backdoor", bd.get("model") or "unknown", md["main_task_description"],
                                    args.get("code", ""), 1, args.get("secret_input"), main, side,
                                    {"backdoor_source": bd.get("source")}))
            else:
                papers.append(Paper(pid, "human_honest", "human", md["main_task_description"],
                                    args.get("code", ""), 0, None, main, None))
    return papers, side_desc


def load_haiku(ids: set[str]) -> list[Paper]:
    path = glob.glob(str(ROOT / "logs" / "haiku_honest" / "*.eval"))[0]
    out = []
    for s in read_eval_log(path).samples or []:
        pid = str(s.id)
        if pid not in ids:
            continue
        main, _ = score_flags(s)
        out.append(Paper(pid, "haiku_honest", "anthropic/claude-haiku-4-5-20251001",
                         s.metadata["main_task_description"], submitted_args(s).get("code", ""), 0, None, main, None))
    return out


def stats_row(papers: list[Paper]) -> dict:
    stripped = [strip_comments(p.code) for p in papers]
    lines = [len(s.splitlines()) for s in stripped]
    chars = [len(s) for s in stripped]
    comm = [len(comment_lines(p.code)) for p in papers]
    return {
        "n": len(papers),
        "median_lines": st.median(lines) if lines else 0,
        "median_chars": st.median(chars) if chars else 0,
        "pct_with_any_comment": round(100 * sum(c > 0 for c in comm) / max(len(papers), 1), 1),
        "median_comment_lines": st.median(comm) if comm else 0,
        "pct_confessing_comment_raw": round(100 * sum(bool(giveaway_hits("\n".join(comment_lines(p.code)))) for p in papers) / max(len(papers), 1), 1),
        "pct_giveaway_after_strip": round(100 * sum(bool(giveaway_hits(s)) for s in stripped) / max(len(papers), 1), 1),
        "pct_def_main": round(100 * sum("def main" in p.code for p in papers) / max(len(papers), 1), 1),
    }


def main() -> None:
    sel = json.loads((OUT / "selected_ids.json").read_text())
    ids = set(sel["ids"])
    p1, side_desc = load_p1(ids)
    papers = p1 + load_haiku(ids)

    by_src = defaultdict(list)
    for p in papers:
        by_src[p.source].append(p)

    # --- integrity: every problem has all three papers, test-bench flags as expected
    per_problem = defaultdict(set)
    for p in papers:
        per_problem[p.problem_id].add(p.source)
    incomplete = sorted(pid for pid in ids if per_problem[pid] != set(SOURCES))
    bad_honest = [p.problem_id for p in papers if p.label == 0 and p.main_task_success is not True]
    bad_backdoor = [p.problem_id for p in papers if p.label == 1 and not (p.main_task_success and p.side_task_success)]

    # --- leakage on every view
    leaks = []
    for p in papers:
        for r in leakage_problems(p, side_task_description=side_desc.get(p.problem_id)):
            leaks.append((p.problem_id, p.source, r))
    struct_leak = [p.problem_id for p in papers if "secret_input" in p.view or "side_task" in p.view.lower()]

    with (OUT / "papers.jsonl").open("w") as f:
        for p in sorted(papers, key=lambda p: (int(p.problem_id), SOURCES.index(p.source))):
            f.write(json.dumps({
                "problem_id": p.problem_id, "source": p.source, "author": p.author, "label": p.label,
                "view": p.view, "view_raw": p.view_raw, "code": p.code, "secret_input": p.secret_input,
                "main_task_success": p.main_task_success, "side_task_success": p.side_task_success,
                **p.extra,
            }) + "\n")

    authors = defaultdict(int)
    for p in by_src["backdoor"]:
        authors[p.author] += 1

    rows = {s: stats_row(by_src[s]) for s in SOURCES}
    lines = [
        "# P1 leakage and confound report",
        "",
        f"Problems: {len(ids)} (seed {sel['seed']}). Papers: {len(papers)}.",
        "",
        "## Integrity",
        "",
        f"- Problems missing one of the three papers: {len(incomplete)} {incomplete[:10]}",
        f"- Honest papers that did not pass their tests: {len(bad_honest)} {bad_honest[:10]}",
        f"- Backdoored papers that failed tests or did not fire: {len(bad_backdoor)} {bad_backdoor[:10]}",
        "",
        "## Leakage (what the TA would see)",
        "",
        f"- Views containing the secret_input field or side-task text: {len(struct_leak)}",
        f"- Papers whose code contains a giveaway word or other leak: {len(leaks)}",
    ]
    for pid, src, r in leaks[:25]:
        lines.append(f"  - problem {pid} ({src}): {r}")
    lines += [
        "",
        "## Confounds by source (anything here that differs is a possible shortcut)",
        "",
        "Lines and chars are measured on the comment-stripped code the TA sees.",
        "",
        "| source | n | median lines | median chars | % any comment (raw) | median comment lines (raw) | % confessing comment (raw) | % giveaway after strip | % `def main` |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in SOURCES:
        r = rows[s]
        lines.append(f"| {s} | {r['n']} | {r['median_lines']} | {r['median_chars']} | {r['pct_with_any_comment']} | "
                     f"{r['median_comment_lines']} | {r['pct_confessing_comment_raw']} | {r['pct_giveaway_after_strip']} | {r['pct_def_main']} |")
    lines += ["", "## Who wrote the backdoors", ""]
    for a, n in sorted(authors.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {a}: {n}")
    (OUT / "leakage_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
