"""Tests for devil's advocate parsing logic."""

import pytest
from backend.council import parse_devil_advocate_response


def test_parse_well_formed_response():
    raw = """Some preamble text.

CONSENSUS:
All models agreed that Python is great for data science.

CRITIQUE:
This consensus ignores that Python's GIL makes it terrible for CPU-bound parallelism.
Many production systems use Go or Rust for exactly this reason."""

    result = parse_devil_advocate_response(raw)

    assert result["consensus_identified"] == "All models agreed that Python is great for data science."
    assert "GIL" in result["critique"]
    assert result["raw"] == raw


def test_parse_missing_sentinels_falls_back_to_raw():
    raw = "The models all seem to agree on X but actually X is wrong for reasons Y and Z."

    result = parse_devil_advocate_response(raw)

    assert result["consensus_identified"] == ""
    assert result["critique"] == raw
    assert result["raw"] == raw


def test_parse_only_consensus_sentinel():
    raw = """CONSENSUS:
Models agreed X is best.

Some other text without a CRITIQUE sentinel."""

    result = parse_devil_advocate_response(raw)

    # Falls back to raw when CRITIQUE sentinel missing
    assert result["consensus_identified"] == ""
    assert result["critique"] == raw


def test_parse_strips_whitespace():
    raw = """CONSENSUS:
  Consensus text with leading whitespace.

CRITIQUE:
  Critique text with leading whitespace.  """

    result = parse_devil_advocate_response(raw)

    assert result["consensus_identified"] == "Consensus text with leading whitespace."
    assert result["critique"] == "Critique text with leading whitespace."
