# Interactivity Improvements Design

**Date:** 2026-03-30
**Status:** Approved

## Overview

Three features to improve the interactivity of LLM Council:

1. **Council Q&A** — council members can ask the user clarifying questions before finalising their responses
2. **Token Tracking** — running token usage displayed in a footer bar, updated live as stages stream in
3. **Edit & Branch** — users can edit a past message and re-run the council, creating a parallel branch they can switch between

All three are built on top of the existing SSE streaming pipeline with additive changes. No breaking changes to existing storage or API contracts.

---

## Architecture Overview

- **Council Q&A** splits the current single `/message/stream` call into a two-pass flow. Pass 1 runs Stage 1a + 1b and waits for user input. Pass 2 runs Stage 1c → 2 → 2.5 → 3 via a new `/message/clarify` endpoint.
- **Token Tracking** threads token counts through existing SSE events. The frontend accumulates a running total displayed in a slim fixed footer strip below the input form.
- **Edit & Branch** adds message IDs to storage and a new `/messages/{id}/edit/stream` endpoint that streams a new council run and appends it as a parallel branch. Branch selection is client-side state.

---

## Feature 1: Council Q&A

### Stage 1a — Responses + Questions

The Stage 1 prompt is updated to ask each council member to optionally append up to 3 clarifying questions using a strict sentinel format:

```
CLARIFYING QUESTIONS:
1. <first question>
2. <second question>
3. <third question>
(or NONE if no questions)
```

`stage1_collect_responses()` parses this sentinel from each response, splitting into:
- `response` — the answer (everything before the sentinel)
- `questions` — a list of up to 3 question strings, or `[]` if `NONE`

The stage1 result shape gains a `questions` field (list).

### Stage 1b — Chairman Question Consolidation

New function `stage1b_consolidate_questions()`:
- Collects all non-empty question lists from Stage 1a (flattening up to 3 questions per model)
- Sends them to the chairman with a prompt to deduplicate and combine into one consolidated question
- Uses strict sentinel format:
  ```
  CONSOLIDATED QUESTION: <combined question, or NONE>
  ```
- Returns the consolidated question string, or `null` if `NONE`

Pass 1 ends here. The SSE stream emits:
- `clarification_needed` event with the consolidated question — UI shows question + Answer/Skip UI
- `clarification_skipped` event if no question — frontend immediately triggers Pass 2 automatically

### Pass 2 — `/message/clarify` Endpoint

`POST /api/conversations/{id}/message/clarify/stream`

Request body:
```json
{
  "user_answer": "developers"   // null if user skipped
}
```

Runs Stage 1c → Stage 2 → Stage 2.5 → Stage 3, streaming SSE events identical to the normal flow. The `user_answer` (or a note that the user skipped) is injected into the context passed to Stage 1c and onwards.

### Stage 1c — Revised Responses

- If the user **skipped** clarification (`user_answer` is `null`), Stage 1c is skipped entirely — all Stage 1a responses are used as-is
- If the user **answered**, council members who submitted a non-empty questions list are re-queried with their original Stage 1a response + the Q&A appended to their message context
- Members who had `NONE` (empty list) keep their Stage 1a response unchanged in both cases
- Only 0–N models are re-queried when an answer is provided (not all), saving tokens when few models asked questions

### Storage

The assistant message gains a `clarification` field stored alongside stage results:

```json
{
  "clarification": {
    "question": "What is your target audience?",
    "answer": "developers",
    "questions_by_model": {
      "openai/gpt-4.1": ["Who is this for?", "What is the expected scale?"],
      "google/gemini-2.5-pro": []
    }
  }
}
```

- `answer` is `null` if the user skipped
- The stage1 results stored are the **final** post-1c responses (not 1a drafts)
- `questions_by_model` is stored for transparency display in the UI

### UI

- When `clarification_needed` arrives, a question card appears in the chat below the stage 1 loading indicator
- Card shows the chairman's consolidated question with a textarea for the answer and a "Skip" button
- On submit or skip, Pass 2 is triggered and the card collapses
- The clarification Q&A is displayed inline in the message history for transparency

---

## Feature 2: Token Tracking

### `openrouter.py`

`query_model()` extracts and returns `usage` from the OpenRouter response:

```python
return {
    'content': ...,
    'reasoning_details': ...,
    'usage': {
        'prompt_tokens': data.get('usage', {}).get('prompt_tokens', 0),
        'completion_tokens': data.get('usage', {}).get('completion_tokens', 0),
    }
}
```

### `council.py`

Each stage aggregates token usage across all its model calls and returns a `tokens_used` summary:

```python
{"prompt_tokens": 4200, "completion_tokens": 1800, "total": 6000}
```

`run_full_council()` also computes and returns a grand total across all stages in the metadata dict.

### SSE Events

Each `stage_complete` event gains a `tokens` field:

```json
{"type": "stage1_complete", "data": [...], "tokens": {"prompt_tokens": 3000, "completion_tokens": 1200, "total": 4200}}
```

The `complete` event includes the grand total:

```json
{"type": "complete", "tokens": {"grand_total": 24500}}
```

### Frontend — Token Footer

A `TokenBar` component renders as a slim fixed strip at the very bottom of the page, below the input form.

- Displays: `Tokens used this request: 24,500`
- Updates on every `stage_complete` event as stages stream in
- Resets to `—` when a new message is sent (before first stage completes)
- Not persisted between page loads — per-request display only, not a billing tracker

---

## Feature 3: Edit & Branch

### Storage Schema Changes

All changes are additive; existing conversations load without modification.

**Every new message** gets an `id` field (UUID) on creation in `storage.py`.

**User messages** gain:
```json
{
  "id": "uuid",
  "role": "user",
  "content": "current active content",
  "alternatives": [
    {"content": "original question", "timestamp": "2026-03-30T10:00:00Z"},
    {"content": "edited question",   "timestamp": "2026-03-30T10:05:00Z"}
  ],
  "active_alternative": 1
}
```

**Assistant messages** gain:
```json
{
  "id": "uuid",
  "role": "assistant",
  "stage1": [...],
  "stage2": [...],
  "stage2_5": {...},
  "stage3": {...},
  "clarification": {...},
  "branches": [
    {"stage1": [...], "stage2": [...], "stage2_5": {...}, "stage3": {...}, "clarification": {...}},
    {"stage1": [...], "stage2": [...], "stage2_5": {...}, "stage3": {...}, "clarification": {...}}
  ],
  "active_branch": 1
}
```

The top-level `stage1/stage2/stage3` fields mirror the active branch's data for backward compatibility with existing conversations that have no `branches` field.

### New Endpoint

`POST /api/conversations/{id}/messages/{msg_id}/edit/stream`

Request body:
```json
{"content": "edited question text"}
```

- Appends new alternative to the user message (by `msg_id`)
- Runs full two-pass council flow (same as normal send, including Q&A pass 1/2)
- Streams identical SSE events to the normal flow
- On `complete`, appends new branch to the following assistant message in storage
- Returns `404` if `msg_id` not found; `400` if message is not a user message

### Frontend — Edit UI

- Each user message shows a pencil icon (visible on hover)
- Clicking opens an inline textarea pre-filled with the current content, replacing the message bubble
- Pressing Enter submits; Escape cancels
- On submit, the edit endpoint stream is opened; loading states follow the same stage-by-stage pattern as normal sends
- **Branch navigator** `← 1/2 →` appears on the assistant message when `branches.length > 1`
- Switching branches updates both the assistant message display and the user message above (showing the corresponding alternative)
- Active branch is tracked in React state; it is not persisted to the backend (refreshing returns to `active_branch` from storage, which is always the latest branch)

---

## Affected Files

### Backend
- `backend/openrouter.py` — return `usage` from `query_model()`
- `backend/council.py` — add `stage1b_consolidate_questions()`, update `stage1_collect_responses()` for question parsing, add `stage1c_revise_responses()`, thread token aggregation through all stages
- `backend/storage.py` — add `id` to messages on creation, add `add_branch_to_message()` helper, add `append_alternative_to_message()` helper
- `backend/main.py` — add `/message/clarify/stream` endpoint, add `/messages/{msg_id}/edit/stream` endpoint, include tokens in SSE events

### Frontend
- `frontend/src/api.js` — add `clarifyMessage()` and `editMessage()` stream methods
- `frontend/src/App.jsx` — handle `clarification_needed`/`clarification_skipped` events, handle edit flow
- `frontend/src/components/ChatInterface.jsx` — render clarification card, render edit UI, render `TokenBar`
- `frontend/src/components/TokenBar.jsx` — new component
- `frontend/src/components/TokenBar.css` — new file
- `frontend/src/components/ClarificationCard.jsx` — new component
- `frontend/src/components/ClarificationCard.css` — new file
