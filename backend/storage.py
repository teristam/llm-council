"""JSON-based storage for conversations."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from .config import DATA_DIR


def ensure_data_dir():
    """Ensure the data directory exists."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_conversation_path(conversation_id: str) -> str:
    """Get the file path for a conversation."""
    return os.path.join(DATA_DIR, f"{conversation_id}.json")


def create_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    Create a new conversation.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        New conversation dict
    """
    ensure_data_dir()

    conversation = {
        "id": conversation_id,
        "created_at": datetime.utcnow().isoformat(),
        "title": "New Conversation",
        "messages": []
    }

    # Save to file
    path = get_conversation_path(conversation_id)
    with open(path, 'w') as f:
        json.dump(conversation, f, indent=2)

    return conversation


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a conversation from storage.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        Conversation dict or None if not found
    """
    path = get_conversation_path(conversation_id)

    if not os.path.exists(path):
        return None

    with open(path, 'r') as f:
        return json.load(f)


def save_conversation(conversation: Dict[str, Any]):
    """
    Save a conversation to storage.

    Args:
        conversation: Conversation dict to save
    """
    ensure_data_dir()

    path = get_conversation_path(conversation['id'])
    with open(path, 'w') as f:
        json.dump(conversation, f, indent=2)


def list_conversations() -> List[Dict[str, Any]]:
    """
    List all conversations (metadata only).

    Returns:
        List of conversation metadata dicts
    """
    ensure_data_dir()

    conversations = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            path = os.path.join(DATA_DIR, filename)
            with open(path, 'r') as f:
                data = json.load(f)
                # Return metadata only
                conversations.append({
                    "id": data["id"],
                    "created_at": data["created_at"],
                    "title": data.get("title", "New Conversation"),
                    "message_count": len(data["messages"])
                })

    # Sort by creation time, newest first
    conversations.sort(key=lambda x: x["created_at"], reverse=True)

    return conversations


def add_user_message(conversation_id: str, content: str):
    """
    Add a user message to a conversation.
    """
    import uuid
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    now = datetime.utcnow().isoformat()
    conversation["messages"].append({
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": content,
        "alternatives": [{"content": content, "timestamp": now}],
        "active_alternative": 0,
    })
    save_conversation(conversation)


def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage2_5: Dict[str, Any],
    stage3: Dict[str, Any],
    clarification: Dict[str, Any] = None,
):
    """
    Add an assistant message with all stages to a conversation.
    """
    import uuid
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    branch_data = {
        "stage1": stage1,
        "stage2": stage2,
        "stage2_5": stage2_5,
        "stage3": stage3,
        "clarification": clarification,
    }
    conversation["messages"].append({
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "stage1": stage1,
        "stage2": stage2,
        "stage2_5": stage2_5,
        "stage3": stage3,
        "clarification": clarification,
        "branches": [branch_data],
        "active_branch": 0,
    })
    save_conversation(conversation)


def update_conversation_title(conversation_id: str, title: str):
    """
    Update the title of a conversation.

    Args:
        conversation_id: Conversation identifier
        title: New title for the conversation
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["title"] = title
    save_conversation(conversation)


def get_message_by_id(conversation_id: str, message_id: str) -> Optional[Dict[str, Any]]:
    """Return the message with the given id, or None if not found."""
    conversation = get_conversation(conversation_id)
    if conversation is None:
        return None
    for msg in conversation["messages"]:
        if msg.get("id") == message_id:
            return msg
    return None


def append_user_alternative(conversation_id: str, message_id: str, new_content: str):
    """
    Append a new alternative to a user message and set it as active.
    Updates the top-level content field to the new content.
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    now = datetime.utcnow().isoformat()
    for msg in conversation["messages"]:
        if msg.get("id") == message_id and msg["role"] == "user":
            if "alternatives" not in msg:
                msg["alternatives"] = [{"content": msg["content"], "timestamp": now}]
            msg["alternatives"].append({"content": new_content, "timestamp": now})
            msg["active_alternative"] = len(msg["alternatives"]) - 1
            msg["content"] = new_content
            break

    save_conversation(conversation)


def add_branch_to_message(conversation_id: str, message_id: str, branch_data: Dict[str, Any]):
    """
    Append a new branch to an assistant message and set it as active.
    Updates the top-level stage fields to mirror the new active branch.
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    for msg in conversation["messages"]:
        if msg.get("id") == message_id and msg["role"] == "assistant":
            if "branches" not in msg:
                original = {k: msg.get(k) for k in ("stage1", "stage2", "stage2_5", "stage3", "clarification")}
                msg["branches"] = [original]
            msg["branches"].append(branch_data)
            msg["active_branch"] = len(msg["branches"]) - 1
            for key in ("stage1", "stage2", "stage2_5", "stage3", "clarification"):
                if key in branch_data:
                    msg[key] = branch_data[key]
            break

    save_conversation(conversation)
