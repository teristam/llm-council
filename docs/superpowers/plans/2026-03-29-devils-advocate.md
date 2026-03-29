# Devil's Advocate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Stage 2.5 Devil's Advocate step that identifies council consensus and argues against it, forcing the chairman to address the challenge in the final synthesis.

**Architecture:** A new `stage2_5_devil_advocate()` function runs sequentially between Stage 2 and Stage 3 in `run_full_council()`. It queries a single designated model (`DEVIL_ADVOCATE_MODEL`) using a prompt that asks it to identify consensus across Stage 1 responses and argue against it. The result is persisted to storage alongside the other stages and rendered in a new red-tinted `Stage2_5` frontend component.

**Tech Stack:** Python/FastAPI (backend), React/Vite (frontend), OpenRouter API (LLM calls), pytest (tests)

---

## File Map

| File | Change |
|---|---|
| `backend/config.py` | Add `DEVIL_ADVOCATE_MODEL` constant |
| `backend/council.py` | Add `stage2_5_devil_advocate()`, update `stage3_synthesize_final()` and `run_full_council()` |
| `backend/storage.py` | Add `stage2_5` param to `add_assistant_message()` |
| `backend/main.py` | Update both endpoints to handle 5-tuple and new SSE events |
| `frontend/src/components/Stage2_5.jsx` | New component — red-tinted DA display |
| `frontend/src/components/Stage2_5.css` | New stylesheet for DA component |
| `frontend/src/App.jsx` | Handle `stage2_5_complete` SSE event, add `stage2_5` to loading state |
| `frontend/src/components/ChatInterface.jsx` | Render `<Stage2_5 />` between Stage 2 and Stage 3 |
| `tests/test_devil_advocate.py` | New — unit tests for DA parsing and integration |

---

## Task 1: Add `DEVIL_ADVOCATE_MODEL` to config

**Files:**
- Modify: `backend/config.py`

- [ ] **Step 1: Add the constant**

Open `backend/config.py` and add after `CHAIRMAN_MODEL`:

```python
# Devil's Advocate model - identifies consensus and argues against it
DEVIL_ADVOCATE_MODEL = "google/gemini-3.1-pro-preview"
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
cd /path/to/llm-council
python -c "from backend.config import DEVIL_ADVOCATE_MODEL; print(DEVIL_ADVOCATE_MODEL)"
```

Expected output:
```
google/gemini-3.1-pro-preview
```

- [ ] **Step 3: Commit**

```bash
git add backend/config.py
git commit -m "feat: add DEVIL_ADVOCATE_MODEL to config"
```

---

## Task 2: Write tests for DA parsing logic

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_devil_advocate.py`

- [ ] **Step 1: Create the tests directory and init file**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: Write failing tests for `parse_devil_advocate_response()`**

Create `tests/test_devil_advocate.py`:

```python
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
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
python -m pytest tests/test_devil_advocate.py -v
```

Expected: 4 failures with `ImportError: cannot import name 'parse_devil_advocate_response'`

- [ ] **Step 4: Commit the tests**

```bash
git add tests/__init__.py tests/test_devil_advocate.py
git commit -m "test: add failing tests for DA parsing"
```

---

## Task 3: Implement `parse_devil_advocate_response()` and `stage2_5_devil_advocate()`

**Files:**
- Modify: `backend/council.py`

- [ ] **Step 1: Add `parse_devil_advocate_response()` to `council.py`**

Add this function after `parse_ranking_from_text()` (around line 290):

```python
def parse_devil_advocate_response(raw: str) -> Dict[str, Any]:
    """
    Parse CONSENSUS: and CRITIQUE: sections from devil's advocate response.

    Args:
        raw: Full text response from the devil's advocate model

    Returns:
        Dict with consensus_identified, critique, and raw keys
    """
    if "CONSENSUS:" in raw and "CRITIQUE:" in raw:
        consensus_start = raw.index("CONSENSUS:") + len("CONSENSUS:")
        critique_start = raw.index("CRITIQUE:")
        critique_content_start = critique_start + len("CRITIQUE:")

        consensus_text = raw[consensus_start:critique_start].strip()
        critique_text = raw[critique_content_start:].strip()

        return {
            "consensus_identified": consensus_text,
            "critique": critique_text,
            "raw": raw
        }

    # Fallback: sentinels not found
    return {
        "consensus_identified": "",
        "critique": raw,
        "raw": raw
    }
```

- [ ] **Step 2: Run the parsing tests — verify they pass**

```bash
python -m pytest tests/test_devil_advocate.py -v
```

Expected: 4 tests pass.

- [ ] **Step 3: Add `stage2_5_devil_advocate()` to `council.py`**

Add this function after `parse_devil_advocate_response()`. Also add `DEVIL_ADVOCATE_MODEL` to the import from `.config` at the top of the file:

```python
# Update the import at the top of council.py:
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL, DEVIL_ADVOCATE_MODEL
```

Then add the function:

```python
async def stage2_5_devil_advocate(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str],
    conversation_history: List[Dict[str, str]] = []
) -> Dict[str, Any]:
    """
    Stage 2.5: Devil's Advocate identifies council consensus and argues against it.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2
        label_to_model: Mapping from anonymous labels to model names
        conversation_history: Previous messages in OpenAI format (optional)

    Returns:
        Dict with model, consensus_identified, critique, and raw keys.
        Returns None if the model call fails.
    """
    # Build a summary of Stage 1 responses (anonymized, consistent with Stage 2)
    labels = [chr(65 + i) for i in range(len(stage1_results))]
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    # Build a summary of Stage 2 rankings
    rankings_text = "\n".join([
        f"- {r['model'].split('/')[-1]}: {', '.join(r['parsed_ranking']) if r['parsed_ranking'] else 'no parsed ranking'}"
        for r in stage2_results
    ])

    history_section = ""
    if conversation_history:
        history_section = f"""Conversation History:
{format_history_for_display(conversation_history)}

"""

    da_prompt = f"""You are a Devil's Advocate. Your job is NOT to be helpful — it is to challenge assumptions and expose weaknesses.

{history_section}Question being discussed: {user_query}

RESPONSES FROM THE COUNCIL:
{responses_text}

PEER RANKINGS:
{rankings_text}

Your task:
1. Read all the responses carefully and identify the KEY POINTS where the majority of models agreed — the consensus view.
2. Then argue FORCEFULLY and specifically against that consensus. Find the strongest possible counterargument, the overlooked evidence, the hidden assumption, or the failure mode that everyone missed.

You MUST format your response EXACTLY as follows:

CONSENSUS:
[State clearly what the majority of models agreed on — be specific, not vague]

CRITIQUE:
[Your forceful argument against that consensus — be specific, cite concrete reasons, don't hedge]"""

    messages = [{"role": "user", "content": da_prompt}]

    response = await query_model(DEVIL_ADVOCATE_MODEL, messages)

    if response is None:
        return None

    raw = response.get('content', '')
    parsed = parse_devil_advocate_response(raw)

    return {
        "model": DEVIL_ADVOCATE_MODEL,
        "consensus_identified": parsed["consensus_identified"],
        "critique": parsed["critique"],
        "raw": raw
    }
```

- [ ] **Step 4: Verify the module imports cleanly**

```bash
python -c "from backend.council import stage2_5_devil_advocate; print('OK')"
```

Expected output: `OK`

- [ ] **Step 5: Commit**

```bash
git add backend/council.py
git commit -m "feat: add stage2_5_devil_advocate() and parse_devil_advocate_response()"
```

---

## Task 4: Update `stage3_synthesize_final()` to accept DA result

**Files:**
- Modify: `backend/council.py`

- [ ] **Step 1: Update the function signature and prompt**

Find `stage3_synthesize_final()` in `council.py` (around line 177). Update its signature and the chairman prompt construction:

```python
async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]] = [],
    devil_advocate_result: Dict[str, Any] = None   # ← new parameter
) -> Dict[str, Any]:
```

Then, inside the function, just before the final `chairman_prompt = f"""..."""` string, add a helper variable:

```python
    # Build devil's advocate section if available
    da_section = ""
    if devil_advocate_result:
        da_section = f"""
DEVIL'S ADVOCATE CHALLENGE:
Consensus identified: {devil_advocate_result.get('consensus_identified', '')}
Challenge: {devil_advocate_result.get('critique', '')}

You MUST directly address this challenge in your final answer — either rebut it with evidence,
concede the point, or explain why it doesn't change your conclusion.
"""
```

Then add `{da_section}` to the chairman prompt, inserting it just before the final `"Provide a clear, well-reasoned final answer..."` line. The end of the prompt should read:

```python
    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

{question_section}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}
{da_section}
Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's {question_type}. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement
{context_instructions}
Provide a clear, well-reasoned final answer {final_answer_instruction}:"""
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
python -c "from backend.council import stage3_synthesize_final; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/council.py
git commit -m "feat: pass devil's advocate challenge to chairman prompt"
```

---

## Task 5: Update `run_full_council()` to call Stage 2.5

**Files:**
- Modify: `backend/council.py`

- [ ] **Step 1: Update `run_full_council()`**

Find `run_full_council()` (around line 378). After the `stage2_collect_rankings` call and `calculate_aggregate_rankings` call, add the Stage 2.5 call. Also update the Stage 3 call and return tuple:

```python
async def run_full_council(
    user_query: str,
    conversation_history: List[Dict[str, str]] = []
) -> Tuple[List, List, Dict, Dict, Dict]:
    """
    Run the complete council process.

    Returns:
        Tuple of (stage1_results, stage2_results, devil_advocate_result, stage3_result, metadata)
    """
    # Stage 1: Collect individual responses WITH HISTORY
    stage1_results = await stage1_collect_responses(user_query, conversation_history)

    if not stage1_results:
        return [], [], None, {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {}

    # Stage 2: Collect rankings WITH HISTORY
    stage2_results, label_to_model = await stage2_collect_rankings(
        user_query,
        stage1_results,
        conversation_history
    )

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 2.5: Devil's Advocate — identify consensus and argue against it
    devil_advocate_result = await stage2_5_devil_advocate(
        user_query,
        stage1_results,
        stage2_results,
        label_to_model,
        conversation_history
    )

    # Stage 3: Synthesize final answer WITH HISTORY and DA challenge
    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results,
        conversation_history,
        devil_advocate_result
    )

    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings
    }

    return stage1_results, stage2_results, devil_advocate_result, stage3_result, metadata
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
python -c "from backend.council import run_full_council; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/council.py
git commit -m "feat: wire stage2_5_devil_advocate into run_full_council"
```

---

## Task 6: Update `storage.add_assistant_message()` to persist DA result

**Files:**
- Modify: `backend/storage.py`

- [ ] **Step 1: Update `add_assistant_message()`**

Find `add_assistant_message()` in `storage.py` (around line 130). Update its signature and the stored message dict:

```python
def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage2_5: Dict[str, Any],
    stage3: Dict[str, Any]
):
    """
    Add an assistant message with all stages to a conversation.

    Args:
        conversation_id: Conversation identifier
        stage1: List of individual model responses
        stage2: List of model rankings
        stage2_5: Devil's advocate result (may be None if DA call failed)
        stage3: Final synthesized response
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["messages"].append({
        "role": "assistant",
        "stage1": stage1,
        "stage2": stage2,
        "stage2_5": stage2_5,
        "stage3": stage3
    })

    save_conversation(conversation)
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
python -c "from backend.storage import add_assistant_message; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/storage.py
git commit -m "feat: persist stage2_5 devil's advocate result to storage"
```

---

## Task 7: Update `main.py` — non-streaming endpoint and streaming endpoint

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Update the non-streaming `send_message` endpoint**

Find the `send_message` function (around line 93). Update the tuple unpack and the call to `add_assistant_message`, and add `stage2_5` to the return body:

```python
    # Run the 3-stage council process WITH HISTORY
    stage1_results, stage2_results, devil_advocate_result, stage3_result, metadata = await run_full_council(
        request.content,
        conversation_history
    )

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        devil_advocate_result,
        stage3_result
    )

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage2_5": devil_advocate_result,
        "stage3": stage3_result,
        "metadata": metadata
    }
```

- [ ] **Step 2: Update the streaming `send_message_stream` endpoint**

Find the `event_generator()` function inside `send_message_stream`. After the `stage2_complete` yield and before `stage3_start`, add the Stage 2.5 block. Also update the `add_assistant_message` call at the end:

```python
            # After stage2_complete yield, add:
            # Stage 2.5: Devil's Advocate
            yield f"data: {json.dumps({'type': 'stage2_5_start'})}\n\n"

            stage2_5_task = asyncio.create_task(stage2_5_devil_advocate(
                request.content,
                stage1_results,
                stage2_results,
                label_to_model,
                conversation_history
            ))
            while not stage2_5_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(stage2_5_task), timeout=15.0)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
            devil_advocate_result = stage2_5_task.result()
            yield f"data: {json.dumps({'type': 'stage2_5_complete', 'data': devil_advocate_result})}\n\n"

            # Stage 3 (update the existing stage3 call to pass devil_advocate_result):
            # ...existing stage3_start yield...
            stage3_task = asyncio.create_task(stage3_synthesize_final(
                request.content,
                stage1_results,
                stage2_results,
                conversation_history,
                devil_advocate_result      # ← add this argument
            ))
```

Also update the `add_assistant_message` call near the end of `event_generator`:

```python
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                devil_advocate_result,
                stage3_result
            )
```

And add `stage2_5_devil_advocate` to the import from `.council` at the top of `main.py`:

```python
from .council import (
    run_full_council,
    generate_conversation_title,
    stage1_collect_responses,
    stage2_collect_rankings,
    stage2_5_devil_advocate,
    stage3_synthesize_final,
    calculate_aggregate_rankings,
    build_conversation_context
)
```

- [ ] **Step 3: Verify the module imports cleanly**

```bash
python -c "from backend.main import app; print('OK')"
```

Expected output: `OK`

- [ ] **Step 4: Start the backend and verify it starts without errors**

```bash
python -m backend.main
```

Expected: Server starts on port 8001 with no import or startup errors. Kill with Ctrl+C.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py
git commit -m "feat: add stage2_5 events to streaming endpoint and non-streaming response"
```

---

## Task 8: Create `Stage2_5.jsx` and `Stage2_5.css`

**Files:**
- Create: `frontend/src/components/Stage2_5.jsx`
- Create: `frontend/src/components/Stage2_5.css`

- [ ] **Step 1: Create `Stage2_5.css`**

```css
.stage2_5 {
  background: #fff0f0;
  border-color: #e6c8c8;
}

.da-content {
  background: #ffffff;
  padding: 20px;
  border-radius: 6px;
  border: 1px solid #e6c8c8;
}

.da-model-label {
  color: #8a2d2d;
  font-size: 12px;
  font-family: monospace;
  margin-bottom: 12px;
  font-weight: 600;
}

.da-section {
  margin-bottom: 20px;
}

.da-section:last-child {
  margin-bottom: 0;
}

.da-section-title {
  font-size: 13px;
  font-weight: 700;
  color: #8a2d2d;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 8px;
}

.da-section-text {
  color: #333;
  line-height: 1.7;
  font-size: 15px;
}
```

- [ ] **Step 2: Create `Stage2_5.jsx`**

```jsx
import ReactMarkdown from 'react-markdown';
import './Stage2_5.css';

export default function Stage2_5({ devilAdvocate }) {
  if (!devilAdvocate) {
    return null;
  }

  const modelShortName = devilAdvocate.model.split('/')[1] || devilAdvocate.model;

  return (
    <div className="stage stage2_5">
      <h3 className="stage-title">Stage 2.5: Devil's Advocate</h3>
      <p className="stage-description">
        This model was instructed to find where the other models agreed — and argue against it.
      </p>

      <div className="da-content">
        <div className="da-model-label">Devil's Advocate: {modelShortName}</div>

        {devilAdvocate.consensus_identified && (
          <div className="da-section">
            <div className="da-section-title">Consensus Identified</div>
            <div className="da-section-text markdown-content">
              <ReactMarkdown>{devilAdvocate.consensus_identified}</ReactMarkdown>
            </div>
          </div>
        )}

        <div className="da-section">
          <div className="da-section-title">Critique</div>
          <div className="da-section-text markdown-content">
            <ReactMarkdown>{devilAdvocate.critique || devilAdvocate.raw}</ReactMarkdown>
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/Stage2_5.jsx frontend/src/components/Stage2_5.css
git commit -m "feat: add Stage2_5 devil's advocate component"
```

---

## Task 9: Update `App.jsx` to handle `stage2_5` events

**Files:**
- Modify: `frontend/src/App.jsx`

- [ ] **Step 1: Add `stage2_5` to the initial assistant message state**

Find the `assistantMessage` object inside `handleSendMessage` (around line 73). Add `stage2_5: null` and `stage2_5: false` to loading:

```javascript
      const assistantMessage = {
        role: 'assistant',
        stage1: null,
        stage2: null,
        stage2_5: null,      // ← add
        stage3: null,
        metadata: null,
        loading: {
          stage1: false,
          stage2: false,
          stage2_5: false,   // ← add
          stage3: false,
        },
      };
```

- [ ] **Step 2: Add `stage2_5_start` and `stage2_5_complete` cases to the switch statement**

Find the switch statement inside `api.sendMessageStream(...)`. Add two new cases after `stage2_complete`:

```javascript
          case 'stage2_5_start':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage2_5 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage2_5_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage2_5 = event.data;
              lastMsg.loading.stage2_5 = false;
              return { ...prev, messages };
            });
            break;
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.jsx
git commit -m "feat: handle stage2_5 SSE events in App.jsx"
```

---

## Task 10: Update `ChatInterface.jsx` to render `<Stage2_5 />`

**Files:**
- Modify: `frontend/src/components/ChatInterface.jsx`

- [ ] **Step 1: Import `Stage2_5`**

Add to the imports at the top of `ChatInterface.jsx`:

```javascript
import Stage2_5 from './Stage2_5';
```

- [ ] **Step 2: Add loading state and component rendering between Stage 2 and Stage 3**

Find the Stage 2 and Stage 3 blocks in the JSX (around lines 84–107). Insert the Stage 2.5 block between them:

```jsx
                  {/* Stage 2.5 - Devil's Advocate */}
                  {msg.loading?.stage2_5 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 2.5: Devil's Advocate...</span>
                    </div>
                  )}
                  {msg.stage2_5 && <Stage2_5 devilAdvocate={msg.stage2_5} />}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChatInterface.jsx
git commit -m "feat: render Stage2_5 devil's advocate between Stage 2 and Stage 3"
```

---

## Task 11: End-to-end smoke test

- [ ] **Step 1: Start the backend**

```bash
python -m backend.main
```

Expected: Server starts on port 8001.

- [ ] **Step 2: Start the frontend**

In a second terminal:

```bash
cd frontend && npm run dev
```

Expected: Vite dev server starts on port 5173.

- [ ] **Step 3: Manual smoke test**

1. Open `http://localhost:5173`
2. Create a new conversation
3. Send a question (e.g. "What is the best programming language for beginners?")
4. Watch the stages load progressively:
   - Stage 1 spinner → Stage 1 tabs appear
   - Stage 2 spinner → Stage 2 rankings appear
   - **Stage 2.5 spinner → Devil's Advocate section appears (red-tinted)**
   - Stage 3 spinner → Final answer appears
5. Verify Stage 2.5 shows:
   - "Consensus Identified" subsection
   - "Critique" subsection
   - Red-tinted background
   - Model name shown
6. Verify Stage 3 response addresses the DA challenge somewhere

- [ ] **Step 4: Test past conversation persistence**

1. Reload the page
2. Select the conversation from the sidebar
3. Verify Stage 2.5 content loads correctly from storage

- [ ] **Step 5: Run unit tests**

```bash
python -m pytest tests/test_devil_advocate.py -v
```

Expected: 4 tests pass.

- [ ] **Step 6: Final commit**

```bash
git add .
git commit -m "feat: devil's advocate stage 2.5 complete

- Adds Stage 2.5 between peer rankings and chairman synthesis
- DA identifies consensus across council responses and argues against it
- Chairman prompt requires direct response to DA challenge
- Result persisted to storage and displayed in red-tinted UI component"
```
