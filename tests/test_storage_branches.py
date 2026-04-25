# tests/test_storage_branches.py
import pytest
from backend import storage


@pytest.fixture
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.storage.DATA_DIR", str(tmp_path))


def test_user_message_has_id(tmp_storage):
    storage.create_conversation("c1")
    storage.add_user_message("c1", "hello")
    conv = storage.get_conversation("c1")
    msg = conv["messages"][0]
    assert "id" in msg
    assert len(msg["id"]) == 36  # UUID4 length


def test_assistant_message_has_id(tmp_storage):
    storage.create_conversation("c1")
    storage.add_user_message("c1", "hello")
    storage.add_assistant_message("c1", [], [], None, {"model": "x", "response": "y"})
    conv = storage.get_conversation("c1")
    assert "id" in conv["messages"][1]


def test_get_message_by_id_returns_message(tmp_storage):
    storage.create_conversation("c1")
    storage.add_user_message("c1", "hello")
    conv = storage.get_conversation("c1")
    msg_id = conv["messages"][0]["id"]
    result = storage.get_message_by_id("c1", msg_id)
    assert result is not None
    assert result["content"] == "hello"


def test_get_message_by_id_unknown_returns_none(tmp_storage):
    storage.create_conversation("c1")
    assert storage.get_message_by_id("c1", "nonexistent-id") is None


def test_append_user_alternative_updates_content(tmp_storage):
    storage.create_conversation("c1")
    storage.add_user_message("c1", "original")
    conv = storage.get_conversation("c1")
    msg_id = conv["messages"][0]["id"]

    storage.append_user_alternative("c1", msg_id, "edited")

    conv2 = storage.get_conversation("c1")
    msg = conv2["messages"][0]
    assert msg["content"] == "edited"
    assert len(msg["alternatives"]) == 2
    assert msg["alternatives"][0]["content"] == "original"
    assert msg["alternatives"][1]["content"] == "edited"
    assert msg["active_alternative"] == 1


def test_add_branch_to_message(tmp_storage):
    storage.create_conversation("c1")
    storage.add_user_message("c1", "hello")
    storage.add_assistant_message("c1", [], [], None, {"model": "x", "response": "y"})
    conv = storage.get_conversation("c1")
    asst_id = conv["messages"][1]["id"]

    branch = {
        "stage1": [{"model": "a", "response": "b"}],
        "stage2": [],
        "stage2_5": None,
        "stage3": {"model": "x", "response": "new"},
        "clarification": None,
    }
    storage.add_branch_to_message("c1", asst_id, branch)

    conv2 = storage.get_conversation("c1")
    msg = conv2["messages"][1]
    assert len(msg["branches"]) == 2
    assert msg["active_branch"] == 1
    assert msg["stage3"]["response"] == "new"  # top-level mirrors active branch
