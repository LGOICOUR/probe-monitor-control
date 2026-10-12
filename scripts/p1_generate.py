"""P1: build the matched paper set with zero API spend.

Design (dataset solutions and backdoors, plus a matched Haiku honest set):
  * Problems = those with a verified working dataset backdoor AND a Haiku honest
    paper (ControlArena honest mode, Haiku 4.5) that passed its tests. A seeded sample of N.
  * For each problem, ControlArena runs a fixed policy (no model) twice:
      honest mode  -> submits the dataset's human-written solution
      attack mode  -> submits a dataset backdoor whose `backdoor_works` flag is True
  * The Apps test bench scores both in Docker: main task (tests pass) and side
    task (secret input gives a different output than the reference).
  * All model roles are mockllm, and the guard refuses any paid model.

Usage:
    uv run python scripts/p1_generate.py --n 150
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.agent import Agent, AgentState
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput
from inspect_ai.solver._task_state import sample_state
from inspect_ai.tool import Tool, ToolCall

from control_arena import EvalMode, control_agent, control_task_metadata
from control_arena.eval import ControlEvalConfig, get_control_task_product
from control_arena.settings.apps.apps_setting import AppsSetting
from control_arena.settings.apps.task.metadata import AppsTaskMetadata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_monitor.guard import run_guarded_eval  # noqa: E402

FREE_MODEL = "mockllm/model"
HAIKU_LOG_GLOB = str(ROOT / "logs" / "haiku_honest" / "*.eval")
OUT = ROOT / "results" / "p1"
MAX_N = 237  # size of the eligible pool; the guard refuses anything larger


def first_working_backdoor(md: AppsTaskMetadata) -> dict | None:
    for b in md.backdoors or []:
        if b.get("backdoor_works"):
            return b
    return None


@control_agent
def dataset_policy() -> Agent:
    """Like ControlArena's apps_solution_policy, but in attack mode it submits the
    first backdoor flagged as working, never a broken one, and never falls back
    to the honest solution."""

    async def impl(state: AgentState, tools: list[Tool]) -> AgentState:
        meta = control_task_metadata()
        md = meta.metadata_as(AppsTaskMetadata)
        if meta.eval_mode == EvalMode.HONEST:
            args = {"code": md.solutions[0]["code"]}
        else:
            b = first_working_backdoor(md)
            if b is None:
                raise ValueError(f"no working backdoor for {sample_state().sample_id}")
            args = {"code": b["backdoor_code"], "secret_input": b["backdoor_input"]}
        msg = ChatMessageAssistant(
            role="assistant",
            content="I am done",
            tool_calls=[ToolCall(id="done", function="submit", arguments=args)],
        )
        state.output = ModelOutput(
            choices=[ChatCompletionChoice(message=msg, stop_reason="tool_calls")],
            model="none/none",
        )
        state._messages.append(state.output.message)
        return state

    return impl


def haiku_passing_ids() -> set[str]:
    paths = glob.glob(HAIKU_LOG_GLOB)
    if not paths:
        raise SystemExit(f"no Haiku honest log at {HAIKU_LOG_GLOB}")
    log = read_eval_log(paths[0])
    ok = set()
    for s in log.samples or []:
        sc = (s.scores or {}).get("apps_scorer")
        if sc and isinstance(sc.value, dict) and sc.value.get("main_task_success") == "C":
            ok.add(str(s.id))
    return ok


def select_ids(n: int, seed: int) -> list[str]:
    pool_setting = AppsSetting(has_backdoor_solution=True)
    working = set()
    for s in pool_setting.get_dataset(EvalMode.ATTACK):
        md = AppsTaskMetadata.model_validate(s.metadata["metadata"])
        if first_working_backdoor(md) is not None:
            working.add(str(s.id))
    eligible = sorted(working & haiku_passing_ids(), key=lambda x: int(x) if x.isdigit() else x)
    if n > len(eligible):
        raise SystemExit(f"asked for {n} problems but only {len(eligible)} are eligible")
    return sorted(random.Random(seed).sample(eligible, n), key=lambda x: int(x) if x.isdigit() else x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="problems (each yields honest + attack)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ids = select_ids(args.n, args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selected_ids.json").write_text(json.dumps({"seed": args.seed, "n": len(ids), "ids": ids}, indent=1))
    print(f"selected {len(ids)} problems (seed={args.seed}) -> results/p1/selected_ids.json")

    id_set = set(ids)
    setting = AppsSetting(
        has_backdoor_solution=True,
        sample_filter_fn=lambda s: str(s.id) in id_set,
        sandbox_type="docker",
    )
    log_dir = ROOT / "logs" / "p1"
    log_dir.mkdir(parents=True, exist_ok=True)
    config = ControlEvalConfig(
        untrusted_model=FREE_MODEL, trusted_model=FREE_MODEL, log_dir=str(log_dir), max_sandboxes=8
    )
    tasks = get_control_task_product(
        settings=setting,
        micro_protocols=dataset_policy(),
        modes=[EvalMode.HONEST, EvalMode.ATTACK],
        eval_config=config,
    )
    run_guarded_eval(
        inspect_eval,
        tasks,
        limit=len(ids),
        max_limit=MAX_N,
        log_dir=log_dir,
        models=[config.untrusted_model, config.trusted_model],
        allow_paid=False,
        max_sandboxes=8,
    )


if __name__ == "__main__":
    main()
