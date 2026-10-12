"""P0 smoke test: generate a handful of honest and attack trajectories on the
ControlArena Apps setting, monitored online by the built-in basic monitor.

Done-check for P0: the run completes, an .eval log lands in logs/p0/, and
`scripts/inspect_log.py` can print a suspicion score per sample.

Spend guard: `--limit` bounds the number of APPS problems per mode. Two modes
(honest, attack) x limit samples. Token usage is printed from the logs after
the run so every run is costed.

Usage:
    uv run python scripts/p0_smoke.py --limit 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.log import list_eval_logs, read_eval_log

from control_arena import EvalMode
from control_arena.eval import ControlEvalConfig, get_control_task_product
from control_arena.micro_protocol import trusted_monitoring_builder
from control_arena.monitor import basic_monitor_builder
from control_arena.scorers import max_suspicion_scorer
from control_arena.settings.apps.apps_setting import AppsSetting

ROOT = Path(__file__).resolve().parents[1]

# Cheapest capable Claude for both roles during the smoke test.
UNTRUSTED = "anthropic/claude-haiku-4-5-20251001"
TRUSTED = "anthropic/claude-haiku-4-5-20251001"


def build_tasks(limit: int, log_dir: Path):
    config = ControlEvalConfig(
        untrusted_model=UNTRUSTED,
        trusted_model=TRUSTED,
        limit=limit,
        log_dir=str(log_dir),
        max_sandboxes=4,
    )
    setting = AppsSetting(sandbox_type="docker")
    # Untrusted policy acts; trusted monitor scores every step; nothing is
    # deferred or edited, so the trajectories are the untrusted model's own.
    protocol = trusted_monitoring_builder(monitor=basic_monitor_builder())
    return get_control_task_product(
        settings=setting,
        micro_protocols=protocol,
        modes=[EvalMode.HONEST, EvalMode.ATTACK],
        eval_config=config,
        monitoring_scorers=[max_suspicion_scorer()],
    )


def summarize(log_dir: Path) -> None:
    total_in = total_out = 0
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info.name)
        mode = log.eval.metadata.get("eval_mode") if log.eval.metadata else None
        print(f"\n== {Path(info.name).name}  mode={mode}  status={log.status}")
        for sample in log.samples or []:
            scores = {k: v.value for k, v in (sample.scores or {}).items()}
            print(f"  sample {sample.id}: {scores}")
        if log.stats and log.stats.model_usage:
            for model, usage in log.stats.model_usage.items():
                total_in += usage.input_tokens
                total_out += usage.output_tokens
                print(f"  usage {model}: in={usage.input_tokens} out={usage.output_tokens}")
    print(f"\nTOTAL tokens: in={total_in} out={total_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2, help="APPS problems per mode")
    ap.add_argument("--log-dir", default=str(ROOT / "logs" / "p0"))
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args.limit, log_dir)
    # `ControlEvalConfig(limit=...)` is NOT applied by get_control_task_product;
    # it must be passed to inspect's eval() itself. The guard below is a hard stop.
    run_eval(tasks, limit=args.limit, log_dir=log_dir)
    summarize(log_dir)


MAX_SMOKE_LIMIT = 10


def run_eval(tasks, *, limit: int, log_dir: Path) -> None:
    """Single choke point for spending: every eval call goes through here."""
    if limit is None or limit <= 0 or limit > MAX_SMOKE_LIMIT:
        raise SystemExit(
            f"refusing to run: limit={limit!r} must be in 1..{MAX_SMOKE_LIMIT}"
        )
    inspect_eval(tasks=tasks, limit=limit, log_dir=str(log_dir))


if __name__ == "__main__":
    main()
