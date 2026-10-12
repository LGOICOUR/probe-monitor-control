"""Offline tests for the spend choke point in scripts/p0_smoke.py.

These tests assert (1) the limit is passed to inspect's eval() verbatim and
(2) out-of-range limits refuse to run. No network, no Docker.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p0_smoke.py"


def _load():
    spec = importlib.util.spec_from_file_location("p0_smoke", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p0_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_limit_reaches_inspect_eval(monkeypatch, tmp_path):
    mod = _load()
    seen = {}

    def fake_eval(tasks, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(mod, "inspect_eval", fake_eval)
    mod.run_eval(["task"], limit=2, log_dir=tmp_path)
    assert seen["limit"] == 2
    assert seen["log_dir"] == str(tmp_path)


@pytest.mark.parametrize("bad", [None, 0, -1, 11, 10_000])
def test_out_of_range_limit_refuses(monkeypatch, tmp_path, bad):
    mod = _load()
    called = []
    monkeypatch.setattr(mod, "inspect_eval", lambda *a, **k: called.append(1))
    with pytest.raises(SystemExit):
        mod.run_eval(["task"], limit=bad, log_dir=tmp_path)
    assert called == []
