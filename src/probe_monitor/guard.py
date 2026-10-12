"""Spend guard: the single choke point for every inspect eval() call.

A limit set on ControlEvalConfig is only honoured when it is also passed to
inspect's eval(); forgetting that can run an entire dataset against a paid
model. Every script that runs an eval goes through run_guarded_eval().
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

# Model strings that cannot reach a paid API.
FREE_MODEL_PREFIXES = ("mockllm/", "none/")


def is_free_model(model: str | None) -> bool:
    return model is not None and model.startswith(FREE_MODEL_PREFIXES)


def run_guarded_eval(
    eval_fn: Callable,
    tasks,
    *,
    limit: int,
    max_limit: int,
    log_dir: Path,
    models: list[str | None],
    allow_paid: bool,
    **eval_kwargs,
):
    """Refuse to run unless the limit is explicit and in range, and (unless
    allow_paid) every model role is a free mock model."""
    if limit is None or not (1 <= limit <= max_limit):
        raise SystemExit(f"refusing to run: limit={limit!r} must be in 1..{max_limit}")
    if not allow_paid:
        paid = [m for m in models if not is_free_model(m)]
        if paid:
            raise SystemExit(f"refusing to run: paid model(s) {paid} in a free-only run")
    return eval_fn(tasks=tasks, limit=limit, log_dir=str(log_dir), **eval_kwargs)
