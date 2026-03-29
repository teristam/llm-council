# Devil's Advocate — Design Spec

**Date:** 2026-03-29
**Status:** Approved

## Overview

Add a Devil's Advocate (DA) step at Stage 2.5 of the LLM Council pipeline. After all council models have ranked each other's responses, a dedicated high-capability model identifies where the council reached consensus and argues forcefully against that shared position. The chairman must then directly address this challenge in the final synthesis.

## Goals

- Stress-test the council's consensus before it reaches the chairman
- Surface blind spots and overlooked counterarguments
- Force the final answer to be more robust by requiring it to rebut or concede the DA's challenge

## Data Flow

```
Stage 1: All council models answer in parallel
    ↓
Stage 2: All council models rank anonymized responses
    ↓
Stage 2.5: Devil's Advocate identifies consensus → attacks it   ← NEW
    ↓
Stage 3: Chairman synthesizes, MUST address DA challenge
```

## Backend Changes

### `backend/config.py`

Add one constant:

```python
DEVIL_ADVOCATE_MODEL = "google/gemini-3.1-pro-preview"
```

### `backend/council.py` — new function `stage2_5_devil_advocate()`

**Signature:**
```python
async def stage2_5_devil_advocate(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str],
    conversation_history: List[Dict[str, str]] = []
) -> Dict[str, Any]:
```

**Prompt strategy (single call, Option B):**
- Give the DA all Stage 1 responses and Stage 2 rankings
- Instruct it to produce output with two clearly labelled sections:
  - `CONSENSUS:` — what the majority of models agreed on
  - `CRITIQUE:` — a forceful argument against that consensus
- Parse response by splitting on these sentinel headers

**Return value:**
```python
{
    "model": DEVIL_ADVOCATE_MODEL,
    "consensus_identified": "<parsed consensus text>",
    "critique": "<parsed critique text>",
    "raw": "<full DA response>"
}
```

On parse failure (sentinels not found), the full response is stored in `critique` and `consensus_identified` is set to `""`.

### `backend/council.py` — update `stage3_synthesize_final()`

Accept an optional `devil_advocate_result: Dict[str, Any] = None` parameter. When present, append to the chairman prompt:

```
DEVIL'S ADVOCATE CHALLENGE:
Consensus identified: <consensus_identified>
Challenge: <critique>

You MUST directly address this challenge in your final answer — either rebut it with evidence,
concede the point, or explain why it doesn't change your conclusion.
```

### `backend/council.py` — update `run_full_council()`

Call `stage2_5_devil_advocate()` after Stage 2, pass result to Stage 3. Return DA result as part of the tuple:

```python
return stage1_results, stage2_results, devil_advocate_result, stage3_result, metadata
```

### `backend/main.py`

- Update all call sites of `run_full_council()` to unpack the new 5-tuple
- Include `stage2_5` in the non-streaming API response body
- Add `stage2_5_start` and `stage2_5_complete` SSE events in the streaming endpoint, between the existing `stage2_complete` and `stage3_start` events

## Frontend Changes

### New `frontend/src/components/Stage2_5.jsx`

Displays the DA output with:
- Red-tinted background (`#fff0f0`) to visually distinguish it from other stages
- **"Devil's Advocate"** as the section header
- Two labelled subsections: **"Consensus Identified"** and **"Critique"**
- Explanatory note: *"This model was instructed to find where the other models agreed — and argue against it."*
- ReactMarkdown rendering inside `.markdown-content` wrapper (consistent with other stage components)

### `frontend/src/App.jsx`

- Handle `stage2_5_complete` SSE event: store the DA result in message state
- Include `stage2_5` in the assistant message object passed to `ChatInterface`

### `frontend/src/components/ChatInterface.jsx`

Render `<Stage2_5 />` between `<Stage2 />` and `<Stage3 />`, conditionally (only when `stage2_5` data is present).

## Error Handling

- If the DA call fails, `devil_advocate_result` is `None`; Stage 3 proceeds normally without the challenge section
- If sentinel parsing fails, the full raw response is shown as the critique with an empty consensus field
- Consistent with the existing graceful-degradation philosophy: DA failure never blocks the main council flow

## Storage

The DA result **is persisted** to JSON storage, alongside stage1/stage2/stage3, consistent with how other council member responses are stored.

`storage.add_assistant_message()` gains a `stage2_5` parameter:

```python
def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage2_5: Dict[str, Any],   # ← new
    stage3: Dict[str, Any]
):
```

Stored assistant message shape:
```json
{
  "role": "assistant",
  "stage1": [...],
  "stage2": [...],
  "stage2_5": { "model": "...", "consensus_identified": "...", "critique": "...", "raw": "..." },
  "stage3": { "model": "...", "response": "..." }
}
```

When loading a past conversation, `ChatInterface` reads `stage2_5` from the message and renders `<Stage2_5 />` if present. Existing stored conversations without `stage2_5` are unaffected (the component renders nothing when the field is absent).

Note: `metadata` (label_to_model, aggregate_rankings) remains ephemeral, as before.

## Trade-offs

| Concern | Decision |
|---|---|
| Extra latency | One additional sequential API call after Stage 2; estimated +5–15s |
| Token cost | One extra call to a top-tier model per query |
| DA quality | `gemini-3.1-pro-preview` chosen for reasoning depth; configurable via `DEVIL_ADVOCATE_MODEL` |
| Parse robustness | Sentinel-based parsing with raw fallback; no brittle regex |
