# tests/test_token_tracking.py
from backend.council import aggregate_tokens

def test_aggregate_tokens_sums_correctly():
    usages = [
        {"prompt_tokens": 100, "completion_tokens": 50},
        {"prompt_tokens": 200, "completion_tokens": 75},
    ]
    result = aggregate_tokens(usages)
    assert result == {"prompt_tokens": 300, "completion_tokens": 125, "total": 425}

def test_aggregate_tokens_skips_none():
    usages = [{"prompt_tokens": 100, "completion_tokens": 50}, None]
    result = aggregate_tokens(usages)
    assert result == {"prompt_tokens": 100, "completion_tokens": 50, "total": 150}

def test_aggregate_tokens_empty():
    assert aggregate_tokens([]) == {"prompt_tokens": 0, "completion_tokens": 0, "total": 0}

def test_aggregate_tokens_all_none():
    assert aggregate_tokens([None, None]) == {"prompt_tokens": 0, "completion_tokens": 0, "total": 0}
