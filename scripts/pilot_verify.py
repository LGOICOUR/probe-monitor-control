"""Verify model-written solutions with the Apps test bench, then check them
for copying from the dataset. $0: all model roles are mockllm.

Expects a folder with:
  pilot_ids.json          list of problem ids
  honest/<id>.py          honest solutions
  attack/<id>.py          backdoored solutions
  attack/<id>.secret.txt  secret inputs

Usage:
    uv run python scripts/pilot_verify.py --dir <folder> [--out results/pilot]
"""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import shutil
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


def make_file_policy(src: Path):
    @control_agent
    def file_policy() -> Agent:
        """Submits the model-written code for this problem (honest/ or attack/)."""

        async def impl(state: AgentState, tools: list[Tool]) -> AgentState:
            pid = str(sample_state().sample_id)
            mode = control_task_metadata().eval_mode
            if mode == EvalMode.HONEST:
                args = {"code": (src / "honest" / f"{pid}.py").read_text()}
            else:
                args = {"code": (src / "attack" / f"{pid}.py").read_text(),
                        "secret_input": (src / "attack" / f"{pid}.secret.txt").read_text()}
            msg = ChatMessageAssistant(role="assistant", content="I am done",
                                       tool_calls=[ToolCall(id="done", function="submit", arguments=args)])
            state.output = ModelOutput(choices=[ChatCompletionChoice(message=msg, stop_reason="tool_calls")],
                                       model="none/none")
            state._messages.append(state.output.message)
            return state

        return impl

    return file_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "pilot"))
    args = ap.parse_args()
    src, out = Path(args.dir), Path(args.out)
    ids = [i for i in json.loads((src / "pilot_ids.json").read_text())
           if (src / "honest" / f"{i}.py").exists() and (src / "attack" / f"{i}.py").exists()
           and (src / "attack" / f"{i}.secret.txt").exists()]
    missing = sorted(set(json.loads((src / "pilot_ids.json").read_text())) - set(ids))
    print(f"{len(ids)} problems with all three files; missing: {missing}")
    id_set = set(ids)

    log_dir = out / "logs"
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True)
    setting = AppsSetting(has_backdoor_solution=True, sample_filter_fn=lambda s: str(s.id) in id_set,
                          sandbox_type="docker")
    config = ControlEvalConfig(untrusted_model=FREE_MODEL, trusted_model=FREE_MODEL, log_dir=str(log_dir))
    tasks = get_control_task_product(settings=setting, micro_protocols=make_file_policy(src)(),
                                     modes=[EvalMode.HONEST, EvalMode.ATTACK], eval_config=config)
    run_guarded_eval(inspect_eval, tasks, limit=len(ids), max_limit=200, log_dir=log_dir,
                     models=[FREE_MODEL, FREE_MODEL], allow_paid=False, max_sandboxes=8)

    dataset = {str(s.id): AppsTaskMetadata.model_validate(s.metadata["metadata"])
               for s in AppsSetting(has_backdoor_solution=True).get_dataset(EvalMode.ATTACK) if str(s.id) in id_set}
    results = {}
    for path in glob.glob(str(log_dir / "*.eval")):
        log = read_eval_log(path)
        for s in log.samples or []:
            pid, mode = str(s.id), s.metadata.get("eval_mode")
            v = s.scores["apps_scorer"].value
            code = (src / ("honest" if mode == "honest" else "attack") / f"{pid}.py").read_text()
            md = dataset[pid]
            sim_sol = max(difflib.SequenceMatcher(None, code, x["code"]).ratio() for x in md.solutions)
            sim_bd = max([difflib.SequenceMatcher(None, code, b["backdoor_code"]).ratio() for b in md.backdoors] or [0])
            results.setdefault(pid, {})[mode] = {
                "main_task_success": v.get("main_task_success") == "C",
                "side_task_success": (v.get("side_task_success") == "C") if mode == "attack" else None,
                "max_similarity_to_dataset_solution": round(sim_sol, 2),
                "max_similarity_to_dataset_backdoor": round(sim_bd, 2),
            }
    (out / "verify.json").write_text(json.dumps(results, indent=1))
    print(f"\n{'id':>5} | honest passes | attack passes | backdoor fires | max sim to dataset (honest / attack)")
    for pid in sorted(results, key=int):
        h, a = results[pid].get("honest", {}), results[pid].get("attack", {})
        sims = f"{max(h.get('max_similarity_to_dataset_solution', 0), h.get('max_similarity_to_dataset_backdoor', 0)):.2f} / " \
               f"{max(a.get('max_similarity_to_dataset_solution', 0), a.get('max_similarity_to_dataset_backdoor', 0)):.2f}"
        print(f"{pid:>5} | {str(h.get('main_task_success')):>13} | {str(a.get('main_task_success')):>13} | "
              f"{str(a.get('side_task_success')):>14} | {sims}")
    ok_h = sum(r.get("honest", {}).get("main_task_success", False) for r in results.values())
    ok_a = sum(bool(r.get("attack", {}).get("main_task_success") and r.get("attack", {}).get("side_task_success"))
               for r in results.values())
    print(f"\nhonest passing: {ok_h}/{len(results)} | working backdoors: {ok_a}/{len(results)}")


if __name__ == "__main__":
    main()
