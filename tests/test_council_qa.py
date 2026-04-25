# tests/test_council_qa.py
from backend.council import parse_questions_from_stage1, parse_consolidated_questions


# --- parse_questions_from_stage1 ---

def test_parse_questions_extracts_numbered_list():
    text = """Here is my answer.

CLARIFYING QUESTIONS:
1. Who is your target audience?
2. What is the expected scale?"""
    response, questions = parse_questions_from_stage1(text)
    assert response == "Here is my answer."
    assert questions == ["Who is your target audience?", "What is the expected scale?"]


def test_parse_questions_none_returns_empty_list():
    text = """Here is my answer.

CLARIFYING QUESTIONS:
NONE"""
    response, questions = parse_questions_from_stage1(text)
    assert response == "Here is my answer."
    assert questions == []


def test_parse_questions_missing_sentinel_returns_full_text():
    text = "Here is my answer with no questions."
    response, questions = parse_questions_from_stage1(text)
    assert response == text
    assert questions == []


def test_parse_questions_capped_at_three():
    text = """Answer.

CLARIFYING QUESTIONS:
1. Q1?
2. Q2?
3. Q3?
4. Q4?"""
    response, questions = parse_questions_from_stage1(text)
    assert len(questions) == 3
    assert questions[0] == "Q1?"


def test_parse_questions_strips_whitespace():
    text = "  My response.  \n\nCLARIFYING QUESTIONS:\n1.   What now?  "
    response, questions = parse_questions_from_stage1(text)
    assert response == "My response."
    assert questions == ["What now?"]


# --- parse_consolidated_questions ---

def test_parse_consolidated_questions_multiple():
    text = """CONSOLIDATED QUESTIONS:
1. What is the target audience?
2. What is the expected scale?"""
    assert parse_consolidated_questions(text) == [
        "What is the target audience?",
        "What is the expected scale?",
    ]


def test_parse_consolidated_questions_none():
    text = "CONSOLIDATED QUESTIONS:\nNONE"
    assert parse_consolidated_questions(text) == []


def test_parse_consolidated_questions_missing_sentinel():
    assert parse_consolidated_questions("Some other text.") == []


def test_parse_consolidated_questions_single():
    text = "CONSOLIDATED QUESTIONS:\n1. What is this for?"
    assert parse_consolidated_questions(text) == ["What is this for?"]
