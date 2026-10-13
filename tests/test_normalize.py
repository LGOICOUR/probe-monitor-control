"""Offline tests for the handwriting eraser."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from probe_monitor.normalize import normalize_code  # noqa: E402


SRC = '''import math
from collections import Counter as C

def digit_sum(n):
    """Sum the digits."""
    total = 0
    for ch in str(n):
        total += int(ch)
    return total

x = int(input())
best = x
for i in range(len(str(x))):
    cand = x - i
    if digit_sum(cand) > digit_sum(best):
        best = cand
print(best, math.sqrt(4), C("aab")["a"])
'''


def test_identifiers_renamed_and_builtins_kept():
    out, ok = normalize_code(SRC)
    assert ok
    for name in ("digit_sum", "total", "best", "cand", "ch"):
        assert name not in out
    for kept in ("int(", "input()", "range(", "len(", "print(", "math.sqrt", "str("):
        assert kept in out
    assert '"""' not in out  # docstring dropped


def test_normalized_program_still_runs_the_same():
    out, _ = normalize_code(SRC)
    def run(src):
        return subprocess.run([sys.executable, "-c", src], input="521\n", capture_output=True, text=True).stdout
    assert run(out) == run(SRC)


def test_unparsable_code_is_returned_unchanged():
    bad = "def f(:\n  pass"
    out, ok = normalize_code(bad)
    assert not ok and out == bad


def test_two_different_stylings_of_one_algorithm_converge():
    a = "numbers = list(map(int, input().split()))\nprint(max(numbers))\n"
    b = "xs=list(map(int,input().split()))\nprint( max(xs) )\n"
    assert normalize_code(a)[0] == normalize_code(b)[0]
