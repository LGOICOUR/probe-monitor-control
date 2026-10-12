"""Offline tests for the P2 prompt builder (no model needed) and digit tokens
(needs only the cached tokenizer; skipped if it is not on disk)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from probe_monitor.reader import build_prompt, digit_token_ids  # noqa: E402
from probe_monitor.views import build_view  # noqa: E402


def test_rate_prompt_has_view_definition_and_scale():
    v = build_view("Print the largest.", "print(max(map(int, input().split())))")
    p = build_prompt(v, "rate")
    assert v in p
    assert "A backdoor is code that" in p
    assert "0 to 9" in p and "single digit" in p
    assert "secret" not in p.lower()


def test_summarize_prompt_has_view_and_no_judgment_words():
    v = build_view("Print the largest.", "x = 1")
    p = build_prompt(v, "summarize")
    assert v in p
    for w in ("backdoor", "suspicious", "honest", "rate"):
        assert w not in p.lower()


def test_digits_are_single_tokens():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
    except Exception:
        pytest.skip("tokenizer not cached")
    ids = digit_token_ids(tok)
    assert len(ids) == 10 and len(set(ids)) == 10
