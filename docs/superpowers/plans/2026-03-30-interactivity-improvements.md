# Interactivity Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add token usage tracking, council Q&A clarification flow, and edit-and-branch to LLM Council.

**Architecture:** Token tracking threads usage from OpenRouter responses through SSE events to a live footer bar. Council Q&A splits the send flow into two passes — Pass 1 collects stage1a responses + chairman-consolidated questions, Pass 2 (after user answers or skips) runs stage1c through stage3. Edit & Branch adds message IDs and a `branches` array to storage, with a new edit endpoint that streams a new council run as a parallel branch.

**Tech Stack:** Python 3.10+, FastAPI, asyncio, httpx, pytest, React, Vite, SSE streaming, JSON file storage

---

## File Map

### Modified
- `backend/openrouter.py` — return `usage` dict from `query_model()`
- `backend/council.py` — add `aggregate_tokens()`, `parse_questions_from_stage1()`, `parse_consolidated_questions()`, `stage1b_consolidate_questions()`, `stage1c_revise_responses()`; update all stage functions to return token counts alongside results
- `backend/storage.py` — add UUID `id` to all messages; add `get_message_by_id()`, `append_user_alternative()`, `add_branch_to_message()`; update `add_assistant_message()` to accept `clarification` param
- `backend/main.py` — include `tokens` in SSE stage events; update Pass 1 stream to emit clarification events; add `/message/clarify/stream` and `/messages/{msg_id}/edit/stream` endpoints
- `frontend/src/api.js` — add `clarifyMessageStream()`, `editMessageStream()`
- `frontend/src/App.jsx` — accumulate token totals; handle clarification events; handle edit flow
- `frontend/src/components/ChatInterface.jsx` — render `ClarificationCard`; render edit button + inline textarea; render `TokenBar`; render branch navigator

### Created
- `tests/test_token_tracking.py`
- `tests/test_council_qa.py`
- `tests/test_storage_branches.py`
- `frontend/src/components/TokenBar.jsx`
- `frontend/src/components/TokenBar.css`
- `frontend/src/components/ClarificationCard.jsx`
- `frontend/src/components/ClarificationCard.css`

---

## FEATURE 1: TOKEN TRACKING

### Task 1: Add usage to `query_model` and `aggregate_tokens` helper

**Files:**
- Modify: `backend/openrouter.py`
- Modify: `backend/council.py`
- Create: `tests/test_token_tracking.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_token_tracking.py -v
```
Expected: `ImportError` or `FAILED` — `aggregate_tokens` not defined yet.

- [ ] **Step 3: Add `aggregate_tokens` to `backend/council.py`**

Add after the imports at the top of `backend/council.py`:

```python
def aggregate_tokens(usages: List[Optional[Dict[str, int]]]) -> Dict[str, int]:
    """Sum prompt and completion tokens across a list of usage dicts, skipping None."""
    prompt = sum(u.get('prompt_tokens', 0) for u in usages if u is not None)
    completion = sum(u.get('completion_tokens', 0) for u in usages if u is not None)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total": prompt + completion}
```

Also update the `from typing import` line to ensure `Optional` is imported (it already is).

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_token_tracking.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Update `query_model` in `backend/openrouter.py` to return usage**

Replace the return statement inside `query_model`:

```python
            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details'),
                'usage': {
                    'prompt_tokens': data.get('usage', {}).get('prompt_tokens', 0),
                    'completion_tokens': data.get('usage', {}).get('completion_tokens', 0),
                }
            }
```

- [ ] **Step 6: Commit**

```bash
git add backend/openrouter.py backend/council.py tests/test_token_tracking.py
git commit -m "feat: add usage to query_model return and aggregate_tokens helper"
```

---

### Task 2: Thread tokens through stage functions

**Files:**
- Modify: `backend/council.py` — update return types of all four stage functions
- Modify: `backend/main.py` — unpack new return tuples in SSE handler and run_full_council call

Each stage function gains a second return value: a `tokens_used` dict. The callers unpack the tuple.

- [ ] **Step 1: Update `stage1_collect_responses` to return `(results, tokens)`**

Replace the end of `stage1_collect_responses`:

```python
    stage1_results = []
    usage_list = []
    for model, response in responses.items():
        if response is not None:
            stage1_results.append({
                "model": model,
                "response": response.get('content', '')
            })
            if response.get('usage'):
                usage_list.append(response['usage'])

    return stage1_results, aggregate_tokens(usage_list)
```

- [ ] **Step 2: Update `stage2_collect_rankings` to return `(results, label_to_model, tokens)`**

Replace the end of `stage2_collect_rankings`:

```python
    stage2_results = []
    usage_list = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed
            })
            if response.get('usage'):
                usage_list.append(response['usage'])

    return stage2_results, label_to_model, aggregate_tokens(usage_list)
```

- [ ] **Step 3: Update `stage2_5_devil_advocate` to return `(result, tokens)`**

Replace the end of `stage2_5_devil_advocate` (after `response = await query_model(...)`):

```python
    if response is None:
        return None, aggregate_tokens([])

    raw = response.get('content', '')
    parsed = parse_devil_advocate_response(raw)
    tokens = aggregate_tokens([response.get('usage')])

    return {
        "model": DEVIL_ADVOCATE_MODEL,
        "consensus_identified": parsed["consensus_identified"],
        "critique": parsed["critique"],
        "raw": raw
    }, tokens
```

- [ ] **Step 4: Update `stage3_synthesize_final` to return `(result, tokens)`**

Replace the end of `stage3_synthesize_final` (after `response = await query_model(...)`):

```python
    if response is None:
        return {
            "model": CHAIRMAN_MODEL,
            "response": "Error: Unable to generate final synthesis."
        }, aggregate_tokens([])

    tokens = aggregate_tokens([response.get('usage')])
    return {
        "model": CHAIRMAN_MODEL,
        "response": response.get('content', '')
    }, tokens
```

- [ ] **Step 5: Update `run_full_council` to unpack tuples and aggregate grand total**

Replace the body of `run_full_council` with:

```python
    stage1_results, stage1_tokens = await stage1_collect_responses(user_query, conversation_history)

    if not stage1_results:
        return [], [], None, {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {}

    stage2_results, label_to_model, stage2_tokens = await stage2_collect_rankings(
        user_query, stage1_results, conversation_history
    )

    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    devil_advocate_result, stage2_5_tokens = await stage2_5_devil_advocate(
        user_query, stage1_results, stage2_results, conversation_history
    )

    stage3_result, stage3_tokens = await stage3_synthesize_final(
        user_query, stage1_results, stage2_results, conversation_history, devil_advocate_result
    )

    grand_total = aggregate_tokens([stage1_tokens, stage2_tokens, stage2_5_tokens, stage3_tokens])

    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "tokens": {
            "stage1": stage1_tokens,
            "stage2": stage2_tokens,
            "stage2_5": stage2_5_tokens,
            "stage3": stage3_tokens,
            "grand_total": grand_total
        }
    }

    return stage1_results, stage2_results, devil_advocate_result, stage3_result, metadata
```

- [ ] **Step 6: Update `main.py` SSE handler to unpack tuples**

In `send_message_stream`, replace all stage result assignments:

```python
            # Stage 1
            stage1_results, stage1_tokens = stage1_task.result()
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results, 'tokens': stage1_tokens})}\n\n"

            # Stage 2
            stage2_results, label_to_model, stage2_tokens = stage2_task.result()
            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}, 'tokens': stage2_tokens})}\n\n"

            # Stage 2.5
            devil_advocate_result, stage2_5_tokens = stage2_5_task.result()
            yield f"data: {json.dumps({'type': 'stage2_5_complete', 'data': devil_advocate_result, 'tokens': stage2_5_tokens})}\n\n"

            # Stage 3
            stage3_result, stage3_tokens = stage3_task.result()
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result, 'tokens': stage3_tokens})}\n\n"

            # Complete event with grand total
            grand_total = aggregate_tokens([stage1_tokens, stage2_tokens, stage2_5_tokens, stage3_tokens])
            yield f"data: {json.dumps({'type': 'complete', 'tokens': {'grand_total': grand_total}})}\n\n"
```

Also add `aggregate_tokens` to the imports from `.council` at the top of `main.py`:

```python
from .council import (
    run_full_council,
    generate_conversation_title,
    stage1_collect_responses,
    stage2_collect_rankings,
    stage2_5_devil_advocate,
    stage3_synthesize_final,
    calculate_aggregate_rankings,
    build_conversation_context,
    aggregate_tokens,
)
```

- [ ] **Step 7: Smoke-test the backend starts without errors**

```bash
uv run python -m backend.main &
sleep 2
curl -s http://localhost:8001/ && kill %1
```
Expected: `{"status":"ok","service":"LLM Council API"}`

- [ ] **Step 8: Commit**

```bash
git add backend/council.py backend/main.py
git commit -m "feat: thread token counts through all stage functions and SSE events"
```

---

### Task 3: TokenBar frontend component

**Files:**
- Create: `frontend/src/components/TokenBar.jsx`
- Create: `frontend/src/components/TokenBar.css`
- Modify: `frontend/src/components/ChatInterface.jsx`
- Modify: `frontend/src/App.jsx`

- [ ] **Step 1: Create `frontend/src/components/TokenBar.css`**

```css
.token-bar {
  width: 100%;
  padding: 6px 16px;
  background: #f5f5f5;
  border-top: 1px solid #e0e0e0;
  font-size: 12px;
  color: #888;
  text-align: right;
  flex-shrink: 0;
}

.token-bar .token-count {
  font-variant-numeric: tabular-nums;
}
```

- [ ] **Step 2: Create `frontend/src/components/TokenBar.jsx`**

```jsx
import './TokenBar.css';

export default function TokenBar({ totalTokens }) {
  const display = totalTokens == null
    ? '—'
    : totalTokens.toLocaleString();

  return (
    <div className="token-bar">
      Tokens used this request: <span className="token-count">{display}</span>
    </div>
  );
}
```

- [ ] **Step 3: Update `App.jsx` to track accumulated tokens**

Add `tokenTotal` state after the existing state declarations:

```jsx
const [tokenTotal, setTokenTotal] = useState(null);
```

Reset on send — inside `handleSendMessage`, right after `setIsLoading(true)`:

```jsx
setTokenTotal(null);
```

Accumulate on each stage event — add cases to the `switch` in the SSE callback:

```jsx
case 'stage1_complete':
  setCurrentConversation((prev) => { /* existing */ });
  setTokenTotal((prev) => (prev ?? 0) + (event.tokens?.total ?? 0));
  break;

case 'stage2_complete':
  setCurrentConversation((prev) => { /* existing */ });
  setTokenTotal((prev) => (prev ?? 0) + (event.tokens?.total ?? 0));
  break;

case 'stage2_5_complete':
  setCurrentConversation((prev) => { /* existing */ });
  setTokenTotal((prev) => (prev ?? 0) + (event.tokens?.total ?? 0));
  break;

case 'stage3_complete':
  setCurrentConversation((prev) => { /* existing */ });
  setTokenTotal((prev) => (prev ?? 0) + (event.tokens?.total ?? 0));
  break;
```

Pass `tokenTotal` to `ChatInterface`:

```jsx
<ChatInterface
  conversation={currentConversation}
  onSendMessage={handleSendMessage}
  isLoading={isLoading}
  tokenTotal={tokenTotal}
/>
```

- [ ] **Step 4: Update `ChatInterface.jsx` to render `TokenBar`**

Add import at top:

```jsx
import TokenBar from './TokenBar';
```

Update the component signature to accept `tokenTotal`:

```jsx
export default function ChatInterface({ conversation, onSendMessage, isLoading, tokenTotal }) {
```

Render `TokenBar` after the `</form>` closing tag, before the final `</div>`:

```jsx
      <form className="input-form" onSubmit={handleSubmit}>
        {/* existing form content unchanged */}
      </form>
      <TokenBar totalTokens={tokenTotal} />
    </div>
```

- [ ] **Step 5: Verify in browser**

Start dev servers and send a message. The footer should show `—` while loading then update to a number like `Tokens used this request: 24,500` as each stage completes.

```bash
cd frontend && npm run dev
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/TokenBar.jsx frontend/src/components/TokenBar.css frontend/src/components/ChatInterface.jsx frontend/src/App.jsx
git commit -m "feat: add token usage footer bar with live updates per stage"
```

---

## FEATURE 2: COUNCIL Q&A

### Task 4: Q&A parsing functions

**Files:**
- Modify: `backend/council.py`
- Create: `tests/test_council_qa.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_council_qa.py -v
```
Expected: `ImportError` — functions not defined yet.

- [ ] **Step 3: Implement `parse_questions_from_stage1` in `backend/council.py`**

Add after `aggregate_tokens`:

```python
def parse_questions_from_stage1(text: str) -> Tuple[str, List[str]]:
    """
    Split a stage1 model response into (response_text, questions_list).
    Extracts the CLARIFYING QUESTIONS sentinel section, caps at 3 questions.
    """
    import re
    sentinel = "CLARIFYING QUESTIONS:"
    if sentinel not in text:
        return text.strip(), []

    parts = text.split(sentinel, 1)
    response_text = parts[0].strip()
    questions_section = parts[1].strip()

    if not questions_section or questions_section.upper().startswith("NONE"):
        return response_text, []

    matches = re.findall(r'^\d+\.\s*(.+?)\s*$', questions_section, re.MULTILINE)
    return response_text, matches[:3]
```

Also add `Tuple` to the typing imports at the top of `council.py`:

```python
from typing import List, Dict, Any, Tuple, Optional
```

(It's already there — verify it is.)

- [ ] **Step 4: Implement `parse_consolidated_questions` in `backend/council.py`**

Add after `parse_questions_from_stage1`:

```python
def parse_consolidated_questions(text: str) -> List[str]:
    """
    Extract consolidated questions from chairman response.
    Returns a list of question strings, or [] if NONE.
    """
    import re
    sentinel = "CONSOLIDATED QUESTIONS:"
    if sentinel not in text:
        return []

    section = text.split(sentinel, 1)[1].strip()
    if not section or section.upper().startswith("NONE"):
        return []

    return re.findall(r'^\d+\.\s*(.+?)\s*$', section, re.MULTILINE)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_council_qa.py -v
```
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/council.py tests/test_council_qa.py
git commit -m "feat: add parse_questions_from_stage1 and parse_consolidated_questions"
```

---

### Task 5: Stage 1a — update `stage1_collect_responses` to parse questions

**Files:**
- Modify: `backend/council.py`

- [ ] **Step 1: Update `stage1_collect_responses` to append question instructions and parse questions**

Replace the full function body:

```python
async def stage1_collect_responses(
    user_query: str,
    conversation_history: List[Dict[str, str]] = []
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Stage 1a: Collect individual responses from all council models.
    Each response includes an optional list of clarifying questions.

    Returns:
        (stage1_results, tokens_used)
        Each result has keys: model, response, questions (list of up to 3 strings)
    """
    question_instructions = (
        "\n\n---\n"
        "After your response, list up to 3 clarifying questions that would help you give a better answer. "
        "Use this exact format:\n\n"
        "CLARIFYING QUESTIONS:\n"
        "1. First question\n"
        "2. Second question\n\n"
        "Or if you have no questions:\n\n"
        "CLARIFYING QUESTIONS:\n"
        "NONE"
    )

    user_msg = {"role": "user", "content": user_query + question_instructions}
    messages = conversation_history + [user_msg]

    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    stage1_results = []
    usage_list = []
    for model, response in responses.items():
        if response is not None:
            raw_content = response.get('content', '')
            response_text, questions = parse_questions_from_stage1(raw_content)
            stage1_results.append({
                "model": model,
                "response": response_text,
                "questions": questions,
            })
            if response.get('usage'):
                usage_list.append(response['usage'])

    return stage1_results, aggregate_tokens(usage_list)
```

- [ ] **Step 2: Verify existing tests still pass**

```bash
uv run pytest tests/ -v
```
Expected: all existing tests pass (the new `questions` field is additive).

- [ ] **Step 3: Commit**

```bash
git add backend/council.py
git commit -m "feat: stage1a — parse clarifying questions from council member responses"
```

---

### Task 6: Stage 1b — `stage1b_consolidate_questions`

**Files:**
- Modify: `backend/council.py`

- [ ] **Step 1: Add `stage1b_consolidate_questions` to `backend/council.py`**

Add after `stage1_collect_responses`:

```python
async def stage1b_consolidate_questions(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, int]]:
    """
    Stage 1b: Chairman reviews all questions from council members,
    deduplicates and consolidates into a list of distinct questions.

    Returns:
        (consolidated_questions, questions_by_model, tokens_used)
        consolidated_questions: list of question strings (may be empty)
        questions_by_model: {model_name: [questions]} for all models
    """
    questions_by_model = {r['model']: r.get('questions', []) for r in stage1_results}
    all_questions = [q for qs in questions_by_model.values() for q in qs]

    if not all_questions:
        return [], questions_by_model, aggregate_tokens([])

    questions_text = "\n".join(
        f"- {model.split('/')[-1]}: {'; '.join(qs)}"
        for model, qs in questions_by_model.items()
        if qs
    )

    consolidation_prompt = f"""Multiple AI council members independently answered a user question and each submitted clarifying questions.

User question: {user_query}

Questions submitted by council members:
{questions_text}

Your task: consolidate these into the most important distinct questions. Remove duplicates and combine related questions. You may ask up to 5 questions.

Format your response EXACTLY as:
CONSOLIDATED QUESTIONS:
1. First question
2. Second question

Or if none are worth asking:
CONSOLIDATED QUESTIONS:
NONE"""

    messages = [{"role": "user", "content": consolidation_prompt}]
    response = await query_model(CHAIRMAN_MODEL, messages)

    if response is None:
        return [], questions_by_model, aggregate_tokens([])

    consolidated = parse_consolidated_questions(response.get('content', ''))
    tokens = aggregate_tokens([response.get('usage')])

    return consolidated, questions_by_model, tokens
```

- [ ] **Step 2: Verify all tests still pass**

```bash
uv run pytest tests/ -v
```

- [ ] **Step 3: Commit**

```bash
git add backend/council.py
git commit -m "feat: stage1b — chairman consolidates council questions into final list"
```

---

### Task 7: Stage 1c — `stage1c_revise_responses`

**Files:**
- Modify: `backend/council.py`

- [ ] **Step 1: Add `stage1c_revise_responses` to `backend/council.py`**

Add after `stage1b_consolidate_questions`:

```python
async def stage1c_revise_responses(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    consolidated_questions: List[str],
    user_answer: str,
    conversation_history: List[Dict[str, str]] = []
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Stage 1c: Re-query models that asked questions, appending the Q&A context.
    Models that asked no questions keep their stage1a response unchanged.

    Only called when user_answer is not None (user did not skip).

    Returns:
        (revised_stage1_results, tokens_used)
    """
    questions_text = "\n".join(
        f"{i+1}. {q}" for i, q in enumerate(consolidated_questions)
    )
    qa_context = (
        f"The council asked some clarifying questions:\n{questions_text}\n\n"
        f"User's answer: {user_answer}\n\n"
        "Please revise your response based on this additional context."
    )

    question_instructions = (
        "\n\n---\n"
        "After your response, list up to 3 clarifying questions that would help you give a better answer. "
        "Use this exact format:\n\nCLARIFYING QUESTIONS:\n1. First question\n\nOr:\n\nCLARIFYING QUESTIONS:\nNONE"
    )

    # Build tasks only for models that asked questions
    models_to_revise = [r for r in stage1_results if r.get('questions')]

    if not models_to_revise:
        return stage1_results, aggregate_tokens([])

    async def revise_one(result: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict]]:
        messages = (
            conversation_history
            + [{"role": "user", "content": user_query + question_instructions}]
            + [{"role": "assistant", "content": result['response']}]
            + [{"role": "user", "content": qa_context}]
        )
        response = await query_model(result['model'], messages)
        if response is None:
            return result, None
        revised_text, _ = parse_questions_from_stage1(response.get('content', ''))
        revised = {**result, "response": revised_text, "questions": []}
        return revised, response.get('usage')

    tasks = [revise_one(r) for r in models_to_revise]
    revisions = await asyncio.gather(*tasks)

    revised_by_model = {r['model']: (revised, usage) for r, (revised, usage) in zip(models_to_revise, revisions)}

    final_results = []
    usage_list = []
    for result in stage1_results:
        if result['model'] in revised_by_model:
            revised, usage = revised_by_model[result['model']]
            final_results.append(revised)
            if usage:
                usage_list.append(usage)
        else:
            final_results.append(result)

    return final_results, aggregate_tokens(usage_list)
```

Also add `import asyncio` to the top of council.py if not already present — check the existing imports first. It is not imported at module level currently; it's imported inside functions. Add it at module level:

```python
import asyncio
```

- [ ] **Step 2: Run all tests**

```bash
uv run pytest tests/ -v
```

- [ ] **Step 3: Commit**

```bash
git add backend/council.py
git commit -m "feat: stage1c — revise responses for models that asked clarifying questions"
```

---

### Task 8: Pass 1 SSE — emit clarification events, stop early

**Files:**
- Modify: `backend/main.py`

The `/message/stream` endpoint currently runs all four stages. We split it into Pass 1 (stage1a + stage1b) and let the frontend call `/message/clarify/stream` for Pass 2.

- [ ] **Step 1: Update imports in `main.py`**

Add to the import from `.council`:

```python
from .council import (
    run_full_council,
    generate_conversation_title,
    stage1_collect_responses,
    stage1b_consolidate_questions,
    stage1c_revise_responses,
    stage2_collect_rankings,
    stage2_5_devil_advocate,
    stage3_synthesize_final,
    calculate_aggregate_rankings,
    build_conversation_context,
    aggregate_tokens,
)
```

- [ ] **Step 2: Replace `send_message_stream` with Pass 1 only**

Replace the entire `event_generator` function inside `send_message_stream` with:

```python
    async def event_generator():
        async def run_with_keepalive(coro):
            task = asyncio.create_task(coro)
            while not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=15.0)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
            return task.result()

        try:
            storage.add_user_message(conversation_id, request.content)

            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(request.content))

            # Stage 1a: collect responses with questions
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_task = asyncio.create_task(
                stage1_collect_responses(request.content, conversation_history)
            )
            while not stage1_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(stage1_task), timeout=15.0)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
            stage1_results, stage1_tokens = stage1_task.result()
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results, 'tokens': stage1_tokens})}\n\n"

            # Stage 1b: chairman consolidates questions
            yield f"data: {json.dumps({'type': 'stage1b_start'})}\n\n"
            stage1b_task = asyncio.create_task(
                stage1b_consolidate_questions(request.content, stage1_results)
            )
            while not stage1b_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(stage1b_task), timeout=15.0)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
            consolidated_questions, questions_by_model, stage1b_tokens = stage1b_task.result()

            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            if consolidated_questions:
                yield f"data: {json.dumps({'type': 'clarification_needed', 'questions': consolidated_questions, 'stage1_results': stage1_results, 'questions_by_model': questions_by_model})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'clarification_skipped', 'stage1_results': stage1_results, 'questions_by_model': questions_by_model})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
```

- [ ] **Step 3: Verify backend starts without errors**

```bash
uv run python -m backend.main &
sleep 2
curl -s http://localhost:8001/ && kill %1
```

- [ ] **Step 4: Commit**

```bash
git add backend/main.py
git commit -m "feat: split message/stream into Pass 1 (stage1a+1b) emitting clarification events"
```

---

### Task 9: Pass 2 — `/message/clarify/stream` endpoint

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/storage.py`

- [ ] **Step 1: Update `add_assistant_message` in `storage.py` to accept `clarification`**

Replace the function signature and body:

```python
def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage2_5: Dict[str, Any],
    stage3: Dict[str, Any],
    clarification: Dict[str, Any] = None,
):
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["messages"].append({
        "role": "assistant",
        "stage1": stage1,
        "stage2": stage2,
        "stage2_5": stage2_5,
        "stage3": stage3,
        "clarification": clarification,
    })

    save_conversation(conversation)
```

- [ ] **Step 2: Add `ClarifyMessageRequest` model to `main.py`**

Add after `SendMessageRequest`:

```python
class ClarifyMessageRequest(BaseModel):
    """Pass 2 request: user's answer (or None to skip) plus stage1 context from Pass 1."""
    user_answer: Optional[str]
    user_query: str  # the original question text (needed when editing historical messages)
    stage1_results: List[Dict[str, Any]]
    questions_by_model: Dict[str, List[str]]
    consolidated_questions: List[str]
```

Also ensure `Optional` is imported: `from typing import List, Dict, Any, Optional`.

- [ ] **Step 3: Add `/message/clarify/stream` endpoint to `main.py`**

Add after `send_message_stream`:

```python
@app.post("/api/conversations/{conversation_id}/message/clarify/stream")
async def clarify_message_stream(conversation_id: str, request: ClarifyMessageRequest):
    """
    Pass 2: continue council pipeline after clarification.
    Accepts stage1 results from Pass 1 and the user's answer (or None if skipped).
    """
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation_history = build_conversation_context(conversation["messages"])

    async def event_generator():
        try:
            # Stage 1c: revise responses (only if user answered)
            if request.user_answer is not None:
                yield f"data: {json.dumps({'type': 'stage1c_start'})}\n\n"
                stage1c_task = asyncio.create_task(
                    stage1c_revise_responses(
                        conversation["messages"][-1]["content"],  # last user message
                        request.stage1_results,
                        request.consolidated_questions,
                        request.user_answer,
                        conversation_history,
                    )
                )
                while not stage1c_task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(stage1c_task), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                stage1_results, stage1c_tokens = stage1c_task.result()
                yield f"data: {json.dumps({'type': 'stage1c_complete', 'data': stage1_results, 'tokens': stage1c_tokens})}\n\n"
            else:
                stage1_results = request.stage1_results
                stage1c_tokens = aggregate_tokens([])

            user_query = request.user_query

            # Stage 2
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_task = asyncio.create_task(
                stage2_collect_rankings(user_query, stage1_results, conversation_history)
            )
            while not stage2_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(stage2_task), timeout=15.0)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
            stage2_results, label_to_model, stage2_tokens = stage2_task.result()
            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}, 'tokens': stage2_tokens})}\n\n"

            # Stage 2.5
            yield f"data: {json.dumps({'type': 'stage2_5_start'})}\n\n"
            stage2_5_task = asyncio.create_task(
                stage2_5_devil_advocate(user_query, stage1_results, stage2_results, conversation_history)
            )
            while not stage2_5_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(stage2_5_task), timeout=15.0)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
            devil_advocate_result, stage2_5_tokens = stage2_5_task.result()
            yield f"data: {json.dumps({'type': 'stage2_5_complete', 'data': devil_advocate_result, 'tokens': stage2_5_tokens})}\n\n"

            # Stage 3
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_task = asyncio.create_task(
                stage3_synthesize_final(user_query, stage1_results, stage2_results, conversation_history, devil_advocate_result)
            )
            while not stage3_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(stage3_task), timeout=15.0)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
            stage3_result, stage3_tokens = stage3_task.result()
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result, 'tokens': stage3_tokens})}\n\n"

            # Save
            clarification = {
                "questions": request.consolidated_questions,
                "answer": request.user_answer,
                "questions_by_model": request.questions_by_model,
            }
            storage.add_assistant_message(
                conversation_id, stage1_results, stage2_results, devil_advocate_result, stage3_result, clarification
            )

            grand_total = aggregate_tokens([stage1c_tokens, stage2_tokens, stage2_5_tokens, stage3_tokens])
            yield f"data: {json.dumps({'type': 'complete', 'tokens': {'grand_total': grand_total}})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        }
    )
```

- [ ] **Step 4: Verify backend starts without errors**

```bash
uv run python -m backend.main &
sleep 2
curl -s http://localhost:8001/ && kill %1
```

- [ ] **Step 5: Commit**

```bash
git add backend/main.py backend/storage.py
git commit -m "feat: add /message/clarify/stream Pass 2 endpoint and clarification storage"
```

---

### Task 10: Frontend — Q&A flow (api.js, ClarificationCard, App.jsx)

**Files:**
- Modify: `frontend/src/api.js`
- Create: `frontend/src/components/ClarificationCard.jsx`
- Create: `frontend/src/components/ClarificationCard.css`
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/components/ChatInterface.jsx`

- [ ] **Step 1: Add `clarifyMessageStream` to `frontend/src/api.js`**

Add inside the `api` object, after `sendMessageStream`:

```js
  /**
   * Pass 2: send clarification answer (or null to skip) and stream the rest of the council.
   */
  async clarifyMessageStream(conversationId, payload, onEvent) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message/clarify/stream`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }
    );
    if (!response.ok) throw new Error('Failed to clarify message');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6));
              onEvent(event.type, event);
            } catch (e) {
              console.error('Failed to parse SSE event:', e);
            }
          }
        }
      }
      if (buffer.startsWith('data: ')) {
        try {
          const event = JSON.parse(buffer.slice(6));
          onEvent(event.type, event);
        } catch (e) { /* ignore */ }
      }
    } catch (err) {
      onEvent('error', { message: `Stream interrupted: ${err.message}` });
    } finally {
      onEvent('stream_end', {});
    }
  },
```

- [ ] **Step 2: Create `frontend/src/components/ClarificationCard.css`**

```css
.clarification-card {
  background: #fff8e1;
  border: 1px solid #ffe082;
  border-radius: 8px;
  padding: 16px;
  margin: 12px 0;
}

.clarification-card h4 {
  margin: 0 0 10px;
  font-size: 14px;
  color: #7c6200;
}

.clarification-card ol {
  margin: 0 0 12px;
  padding-left: 20px;
}

.clarification-card li {
  font-size: 14px;
  margin-bottom: 4px;
  color: #333;
}

.clarification-card textarea {
  width: 100%;
  box-sizing: border-box;
  min-height: 72px;
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
  resize: vertical;
  margin-bottom: 8px;
}

.clarification-card .card-actions {
  display: flex;
  gap: 8px;
}

.clarification-card .btn-answer {
  background: #4a90e2;
  color: white;
  border: none;
  padding: 6px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.clarification-card .btn-answer:disabled {
  opacity: 0.5;
  cursor: default;
}

.clarification-card .btn-skip {
  background: none;
  border: 1px solid #ccc;
  padding: 6px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  color: #666;
}
```

- [ ] **Step 3: Create `frontend/src/components/ClarificationCard.jsx`**

```jsx
import { useState } from 'react';
import './ClarificationCard.css';

export default function ClarificationCard({ questions, onAnswer, onSkip }) {
  const [answer, setAnswer] = useState('');

  return (
    <div className="clarification-card">
      <h4>The council has some clarifying questions:</h4>
      <ol>
        {questions.map((q, i) => (
          <li key={i}>{q}</li>
        ))}
      </ol>
      <textarea
        placeholder="Your answer (optional)…"
        value={answer}
        onChange={(e) => setAnswer(e.target.value)}
      />
      <div className="card-actions">
        <button
          className="btn-answer"
          disabled={!answer.trim()}
          onClick={() => onAnswer(answer.trim())}
        >
          Answer
        </button>
        <button className="btn-skip" onClick={onSkip}>
          Skip
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Update `App.jsx` to handle Q&A flow**

Add state for pending clarification after existing state declarations:

```jsx
const [pendingClarification, setPendingClarification] = useState(null);
// Shape: { conversationId, stage1Results, questionsByModel, consolidatedQuestions }
```

In `handleSendMessage`, update the SSE event switch to handle Pass 1 events and trigger Pass 2:

```jsx
case 'clarification_needed':
  setPendingClarification({
    conversationId: currentConversationId,
    userQuery: content,  // `content` is the message text from handleSendMessage's closure
    stage1Results: event.stage1_results,
    questionsByModel: event.questions_by_model,
    consolidatedQuestions: event.questions,
    asstMessageId: null,
  });
  setIsLoading(false);
  break;

case 'clarification_skipped':
  // No questions — auto-trigger Pass 2 immediately
  triggerPass2({
    conversationId: currentConversationId,
    userQuery: content,
    stage1Results: event.stage1_results,
    questionsByModel: event.questions_by_model,
    consolidatedQuestions: [],
    userAnswer: null,
    asstMessageId: null,
  });
  break;
```

Add the `triggerPass2` function (defined before `handleSendMessage`):

```jsx
const triggerPass2 = async ({ conversationId, userQuery, stage1Results, questionsByModel, consolidatedQuestions, userAnswer, asstMessageId = null }) => {
  setPendingClarification(null);
  setIsLoading(true);

  const payload = {
    user_answer: userAnswer,
    user_query: userQuery,
    stage1_results: stage1Results,
    questions_by_model: questionsByModel,
    consolidated_questions: consolidatedQuestions,
    asst_message_id: asstMessageId,
  };

  try {
    await api.clarifyMessageStream(conversationId, payload, (eventType, event) => {
      switch (eventType) {
        case 'stage1c_start':
        case 'stage2_start':
        case 'stage2_5_start':
        case 'stage3_start':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            const stageKey = eventType.replace('_start', '').replace('stage1c', 'stage1');
            lastMsg.loading = { ...lastMsg.loading, [stageKey]: true };
            return { ...prev, messages };
          });
          break;

        case 'stage1c_complete':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.stage1 = event.data;
            lastMsg.loading = { ...lastMsg.loading, stage1: false };
            return { ...prev, messages };
          });
          setTokenTotal((prev) => (prev ?? 0) + (event.tokens?.total ?? 0));
          break;

        case 'stage2_complete':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.stage2 = event.data;
            lastMsg.metadata = event.metadata;
            lastMsg.loading = { ...lastMsg.loading, stage2: false };
            return { ...prev, messages };
          });
          setTokenTotal((prev) => (prev ?? 0) + (event.tokens?.total ?? 0));
          break;

        case 'stage2_5_complete':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.stage2_5 = event.data;
            lastMsg.loading = { ...lastMsg.loading, stage2_5: false };
            return { ...prev, messages };
          });
          setTokenTotal((prev) => (prev ?? 0) + (event.tokens?.total ?? 0));
          break;

        case 'stage3_complete':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.stage3 = event.data;
            lastMsg.loading = { ...lastMsg.loading, stage3: false };
            return { ...prev, messages };
          });
          setTokenTotal((prev) => (prev ?? 0) + (event.tokens?.total ?? 0));
          break;

        case 'complete':
          loadConversations();
          setIsLoading(false);
          break;

        case 'error':
          console.error('Clarify stream error:', event.message);
          setIsLoading(false);
          break;

        case 'stream_end':
          setIsLoading(false);
          break;

        case 'keepalive':
          break;
      }
    });
  } catch (error) {
    console.error('Failed to clarify:', error);
    setIsLoading(false);
  }
};
```

Pass `pendingClarification` and `triggerPass2` to `ChatInterface`:

```jsx
<ChatInterface
  conversation={currentConversation}
  onSendMessage={handleSendMessage}
  isLoading={isLoading}
  tokenTotal={tokenTotal}
  pendingClarification={pendingClarification}
  onClarificationAnswer={(answer) => triggerPass2({ ...pendingClarification, userAnswer: answer })}
  onClarificationSkip={() => triggerPass2({ ...pendingClarification, userAnswer: null })}
/>
```

- [ ] **Step 5: Update `ChatInterface.jsx` to render `ClarificationCard`**

Add import:

```jsx
import ClarificationCard from './ClarificationCard';
```

Update component signature:

```jsx
export default function ChatInterface({
  conversation,
  onSendMessage,
  isLoading,
  tokenTotal,
  pendingClarification,
  onClarificationAnswer,
  onClarificationSkip,
}) {
```

Render the card below the last assistant message loading spinners and above the input form. Add inside `messages-container`, after the `conversation.messages.map(...)` block:

```jsx
        {pendingClarification && (
          <ClarificationCard
            questions={pendingClarification.consolidatedQuestions}
            onAnswer={onClarificationAnswer}
            onSkip={onClarificationSkip}
          />
        )}
```

- [ ] **Step 6: Verify end-to-end in browser**

Start backend and frontend dev servers. Send a question. The council should:
1. Show stage1 responses loading, then complete
2. Show stage1b loading
3. Show the ClarificationCard with the chairman's consolidated questions (or skip straight to stage2 if no questions)
4. On Answer or Skip, continue through stages 2 → 3

- [ ] **Step 7: Commit**

```bash
git add frontend/src/api.js frontend/src/components/ClarificationCard.jsx frontend/src/components/ClarificationCard.css frontend/src/App.jsx frontend/src/components/ChatInterface.jsx
git commit -m "feat: frontend Q&A flow — ClarificationCard, two-pass SSE handling"
```

---

## FEATURE 3: EDIT & BRANCH

### Task 11: Storage — message IDs and branch helpers

**Files:**
- Modify: `backend/storage.py`
- Create: `tests/test_storage_branches.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_storage_branches.py
import pytest
from backend import storage


@pytest.fixture
def conv(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.storage.DATA_DIR", str(tmp_path))
    storage.create_conversation("c1")
    storage.add_user_message("c1", "hello")
    storage.add_assistant_message("c1", [], [], None, {"model": "x", "response": "y"})
    return storage.get_conversation("c1")


def test_user_message_has_id(conv):
    assert "id" in conv["messages"][0]
    assert len(conv["messages"][0]["id"]) == 36  # UUID


def test_assistant_message_has_id(conv):
    assert "id" in conv["messages"][1]


def test_get_message_by_id_returns_message(conv):
    msg_id = conv["messages"][0]["id"]
    result = storage.get_message_by_id("c1", msg_id)
    assert result is not None
    assert result["content"] == "hello"


def test_get_message_by_id_unknown_returns_none(conv):
    assert storage.get_message_by_id("c1", "nonexistent-id") is None


def test_append_user_alternative_updates_content(conv, tmp_path, monkeypatch):
    monkeypatch.setattr("backend.storage.DATA_DIR", str(tmp_path))
    storage.create_conversation("c2")
    storage.add_user_message("c2", "original")
    c = storage.get_conversation("c2")
    msg_id = c["messages"][0]["id"]

    storage.append_user_alternative("c2", msg_id, "edited")

    c2 = storage.get_conversation("c2")
    msg = c2["messages"][0]
    assert msg["content"] == "edited"
    assert len(msg["alternatives"]) == 2
    assert msg["alternatives"][0]["content"] == "original"
    assert msg["alternatives"][1]["content"] == "edited"
    assert msg["active_alternative"] == 1


def test_add_branch_to_message(conv, tmp_path, monkeypatch):
    monkeypatch.setattr("backend.storage.DATA_DIR", str(tmp_path))
    storage.create_conversation("c3")
    storage.add_user_message("c3", "hello")
    storage.add_assistant_message("c3", [], [], None, {"model": "x", "response": "y"})
    c = storage.get_conversation("c3")
    asst_id = c["messages"][1]["id"]

    branch = {"stage1": [{"model": "a", "response": "b"}], "stage2": [], "stage2_5": None, "stage3": {"model": "x", "response": "new"}, "clarification": None}
    storage.add_branch_to_message("c3", asst_id, branch)

    c3 = storage.get_conversation("c3")
    msg = c3["messages"][1]
    assert len(msg["branches"]) == 2
    assert msg["active_branch"] == 1
    assert msg["stage3"]["response"] == "new"  # top-level mirrors active branch
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_storage_branches.py -v
```
Expected: failures — functions not defined, `id` not present on messages.

- [ ] **Step 3: Update `add_user_message` to include UUID and `alternatives`**

Replace the function:

```python
def add_user_message(conversation_id: str, content: str):
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
```

- [ ] **Step 4: Update `add_assistant_message` to include UUID and `branches`**

Replace the function:

```python
def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage2_5: Dict[str, Any],
    stage3: Dict[str, Any],
    clarification: Dict[str, Any] = None,
):
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
```

- [ ] **Step 5: Add `get_message_by_id`, `append_user_alternative`, `add_branch_to_message`**

Add at the end of `storage.py`:

```python
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
    Updates the top-level `content` field to the new content.
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    now = datetime.utcnow().isoformat()
    for msg in conversation["messages"]:
        if msg.get("id") == message_id and msg["role"] == "user":
            msg.setdefault("alternatives", [{"content": msg["content"], "timestamp": now}])
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
                # Migrate old message that has no branches array
                original = {k: msg[k] for k in ("stage1", "stage2", "stage2_5", "stage3", "clarification") if k in msg}
                msg["branches"] = [original]
            msg["branches"].append(branch_data)
            msg["active_branch"] = len(msg["branches"]) - 1
            # Mirror active branch to top-level fields
            for key in ("stage1", "stage2", "stage2_5", "stage3", "clarification"):
                if key in branch_data:
                    msg[key] = branch_data[key]
            break

    save_conversation(conversation)
```

- [ ] **Step 6: Run all tests**

```bash
uv run pytest tests/ -v
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add backend/storage.py tests/test_storage_branches.py
git commit -m "feat: add message IDs and branch/alternative helpers to storage"
```

---

### Task 12: Edit endpoint — `/messages/{msg_id}/edit/stream`

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Add `EditMessageRequest` model to `main.py`**

Add after `ClarifyMessageRequest`:

```python
class EditMessageRequest(BaseModel):
    """Request to edit a user message and re-run the council as a new branch."""
    content: str
```

- [ ] **Step 2: Add the edit endpoint**

Add after the clarify endpoint:

```python
@app.post("/api/conversations/{conversation_id}/messages/{message_id}/edit/stream")
async def edit_message_stream(conversation_id: str, message_id: str, request: EditMessageRequest):
    """
    Edit a past user message and re-run the full council pipeline as a new branch.
    Streams identical SSE events to the normal send flow (two passes).
    """
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Find user message and the following assistant message
    messages = conversation["messages"]
    user_msg = storage.get_message_by_id(conversation_id, message_id)
    if user_msg is None or user_msg["role"] != "user":
        raise HTTPException(status_code=400, detail="Message not found or not a user message")

    # Find the assistant message immediately following the edited user message
    user_idx = next(i for i, m in enumerate(messages) if m.get("id") == message_id)
    asst_msg = messages[user_idx + 1] if user_idx + 1 < len(messages) else None

    # Build history up to (but not including) the edited message
    prior_history = build_conversation_context(messages[:user_idx])

    async def event_generator():
        try:
            # Update user message alternative
            storage.append_user_alternative(conversation_id, message_id, request.content)

            # Pass 1: stage1a + stage1b
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_task = asyncio.create_task(
                stage1_collect_responses(request.content, prior_history)
            )
            while not stage1_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(stage1_task), timeout=15.0)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
            stage1_results, stage1_tokens = stage1_task.result()
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results, 'tokens': stage1_tokens})}\n\n"

            yield f"data: {json.dumps({'type': 'stage1b_start'})}\n\n"
            stage1b_task = asyncio.create_task(
                stage1b_consolidate_questions(request.content, stage1_results)
            )
            while not stage1b_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(stage1b_task), timeout=15.0)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
            consolidated_questions, questions_by_model, _ = stage1b_task.result()

            if consolidated_questions:
                yield f"data: {json.dumps({'type': 'clarification_needed', 'questions': consolidated_questions, 'stage1_results': stage1_results, 'questions_by_model': questions_by_model, 'edit_message_id': message_id, 'asst_message_id': asst_msg['id'] if asst_msg else None})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'clarification_skipped', 'stage1_results': stage1_results, 'questions_by_model': questions_by_model, 'edit_message_id': message_id, 'asst_message_id': asst_msg['id'] if asst_msg else None})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        }
    )
```

Note: Pass 2 for edit uses the same `/message/clarify/stream` endpoint — the frontend sends `asst_message_id` alongside the normal clarify payload, and after completion calls `storage.add_branch_to_message`. Update `ClarifyMessageRequest` to accept optional `asst_message_id`:

```python
class ClarifyMessageRequest(BaseModel):
    user_answer: Optional[str]
    stage1_results: List[Dict[str, Any]]
    questions_by_model: Dict[str, List[str]]
    consolidated_questions: List[str]
    asst_message_id: Optional[str] = None  # set when this is an edit branch
```

In the clarify endpoint's save block, replace `storage.add_assistant_message(...)` with:

```python
            clarification = {
                "questions": request.consolidated_questions,
                "answer": request.user_answer,
                "questions_by_model": request.questions_by_model,
            }
            if request.asst_message_id:
                # Edit branch: append to existing assistant message
                branch_data = {
                    "stage1": stage1_results,
                    "stage2": stage2_results,
                    "stage2_5": devil_advocate_result,
                    "stage3": stage3_result,
                    "clarification": clarification,
                }
                storage.add_branch_to_message(conversation_id, request.asst_message_id, branch_data)
            else:
                # New message: create fresh assistant message
                storage.add_assistant_message(
                    conversation_id, stage1_results, stage2_results, devil_advocate_result, stage3_result, clarification
                )
```

- [ ] **Step 3: Verify backend starts without errors**

```bash
uv run python -m backend.main &
sleep 2
curl -s http://localhost:8001/ && kill %1
```

- [ ] **Step 4: Commit**

```bash
git add backend/main.py
git commit -m "feat: add /messages/{id}/edit/stream endpoint for edit-and-branch"
```

---

### Task 13: Frontend — edit UI and branch navigator

**Files:**
- Modify: `frontend/src/api.js`
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/components/ChatInterface.jsx`
- Modify: `frontend/src/components/ChatInterface.css`

- [ ] **Step 1: Add `editMessageStream` to `frontend/src/api.js`**

Add inside the `api` object:

```js
  async editMessageStream(conversationId, messageId, content, onEvent) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/messages/${messageId}/edit/stream`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      }
    );
    if (!response.ok) throw new Error('Failed to edit message');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6));
              onEvent(event.type, event);
            } catch (e) {
              console.error('Failed to parse SSE event:', e);
            }
          }
        }
      }
    } catch (err) {
      onEvent('error', { message: `Stream interrupted: ${err.message}` });
    } finally {
      onEvent('stream_end', {});
    }
  },
```

- [ ] **Step 2: Add `handleEditMessage` to `App.jsx`**

Add state for tracking which message is being edited:

```jsx
const [editingMessageId, setEditingMessageId] = useState(null);
```

Add `handleEditMessage` before `handleSendMessage`:

```jsx
const handleEditMessage = async (messageId, newContent) => {
  setEditingMessageId(null);
  setIsLoading(true);
  setTokenTotal(null);

  // Optimistically update the user message in the UI
  setCurrentConversation((prev) => {
    const messages = prev.messages.map((m) => {
      if (m.id === messageId) return { ...m, content: newContent };
      return m;
    });
    return { ...prev, messages };
  });

  try {
    await api.editMessageStream(currentConversationId, messageId, newContent, (eventType, event) => {
      switch (eventType) {
        case 'clarification_needed':
          setPendingClarification({
            conversationId: currentConversationId,
            stage1Results: event.stage1_results,
            questionsByModel: event.questions_by_model,
            consolidatedQuestions: event.questions,
            asstMessageId: event.asst_message_id,
          });
          setIsLoading(false);
          break;

        case 'clarification_skipped':
          triggerPass2({
            conversationId: currentConversationId,
            stage1Results: event.stage1_results,
            questionsByModel: event.questions_by_model,
            consolidatedQuestions: [],
            userAnswer: null,
            asstMessageId: event.asst_message_id,
          });
          break;

        case 'stage1_complete':
          setTokenTotal((prev) => (prev ?? 0) + (event.tokens?.total ?? 0));
          break;

        case 'error':
          console.error('Edit stream error:', event.message);
          setIsLoading(false);
          break;

        case 'stream_end':
          setIsLoading(false);
          break;

        case 'keepalive':
          break;
      }
    });
  } catch (error) {
    console.error('Failed to edit message:', error);
    setIsLoading(false);
  }
};
```

Update `triggerPass2` to accept and forward `asstMessageId`:

```jsx
const triggerPass2 = async ({ conversationId, stage1Results, questionsByModel, consolidatedQuestions, userAnswer, asstMessageId = null }) => {
  // ... existing code ...
  const payload = {
    user_answer: userAnswer,
    stage1_results: stage1Results,
    questions_by_model: questionsByModel,
    consolidated_questions: consolidatedQuestions,
    asst_message_id: asstMessageId,  // add this line
  };
  // ... rest unchanged ...
};
```

Also update the clarification answer/skip handlers to forward `asstMessageId`:

```jsx
onClarificationAnswer={(answer) => triggerPass2({ ...pendingClarification, userAnswer: answer, asstMessageId: pendingClarification.asstMessageId })}
onClarificationSkip={() => triggerPass2({ ...pendingClarification, userAnswer: null, asstMessageId: pendingClarification.asstMessageId })}
```

Pass `editingMessageId`, `setEditingMessageId`, `handleEditMessage` to `ChatInterface`:

```jsx
<ChatInterface
  conversation={currentConversation}
  onSendMessage={handleSendMessage}
  isLoading={isLoading}
  tokenTotal={tokenTotal}
  pendingClarification={pendingClarification}
  onClarificationAnswer={(answer) => triggerPass2({ ...pendingClarification, userAnswer: answer, asstMessageId: pendingClarification?.asstMessageId })}
  onClarificationSkip={() => triggerPass2({ ...pendingClarification, userAnswer: null, asstMessageId: pendingClarification?.asstMessageId })}
  editingMessageId={editingMessageId}
  onEditStart={(id) => setEditingMessageId(id)}
  onEditCancel={() => setEditingMessageId(null)}
  onEditSubmit={handleEditMessage}
/>
```

Also update the `triggerPass2` call inside `handleSendMessage`'s `clarification_skipped` case to include `asstMessageId: null`.

- [ ] **Step 3: Update `ChatInterface.jsx` — edit button, inline textarea, branch navigator**

Update component signature:

```jsx
export default function ChatInterface({
  conversation,
  onSendMessage,
  isLoading,
  tokenTotal,
  pendingClarification,
  onClarificationAnswer,
  onClarificationSkip,
  editingMessageId,
  onEditStart,
  onEditCancel,
  onEditSubmit,
}) {
```

Add state for the edit textarea value:

```jsx
const [editValue, setEditValue] = useState('');
```

Replace the user message render block with edit-aware version:

```jsx
              {msg.role === 'user' ? (
                <div className="user-message">
                  <div className="message-label">
                    You
                    {!isLoading && (
                      <button
                        className="edit-btn"
                        title="Edit"
                        onClick={() => { setEditValue(msg.content); onEditStart(msg.id); }}
                      >
                        ✏️
                      </button>
                    )}
                  </div>
                  {editingMessageId === msg.id ? (
                    <div className="edit-area">
                      <textarea
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        rows={3}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onEditSubmit(msg.id, editValue); }
                          if (e.key === 'Escape') onEditCancel();
                        }}
                      />
                      <div className="edit-actions">
                        <button onClick={() => onEditSubmit(msg.id, editValue)} disabled={!editValue.trim()}>Save</button>
                        <button onClick={onEditCancel}>Cancel</button>
                      </div>
                    </div>
                  ) : (
                    <div className="message-content">
                      {msg.alternatives && msg.alternatives.length > 1 && (
                        <div className="alt-nav">
                          ← {(msg.active_alternative ?? 0) + 1}/{msg.alternatives.length} →
                        </div>
                      )}
                      <div className="markdown-content">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    </div>
                  )}
                </div>
```

Add branch navigator to the assistant message render block, just before `{msg.stage1 && <Stage1 .../>}`:

```jsx
                  {msg.branches && msg.branches.length > 1 && (
                    <div className="branch-nav">
                      Branch {(msg.active_branch ?? 0) + 1}/{msg.branches.length}
                    </div>
                  )}
```

- [ ] **Step 4: Add CSS for edit UI to `ChatInterface.css`**

```css
.edit-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 12px;
  padding: 0 4px;
  opacity: 0;
  transition: opacity 0.15s;
}

.user-message:hover .edit-btn {
  opacity: 1;
}

.edit-area textarea {
  width: 100%;
  box-sizing: border-box;
  padding: 8px;
  border: 1px solid #4a90e2;
  border-radius: 4px;
  font-size: 14px;
  resize: vertical;
}

.edit-actions {
  display: flex;
  gap: 8px;
  margin-top: 6px;
}

.edit-actions button {
  padding: 4px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}

.edit-actions button:first-child {
  background: #4a90e2;
  color: white;
  border: none;
}

.edit-actions button:last-child {
  background: none;
  border: 1px solid #ccc;
  color: #666;
}

.alt-nav, .branch-nav {
  font-size: 11px;
  color: #999;
  margin-bottom: 4px;
}
```

- [ ] **Step 5: Verify edit flow in browser**

Send a question, wait for full response. Hover over the user message — pencil icon appears. Click it, edit the text, press Save. The council should re-run and a branch navigator should appear showing `Branch 2/2`.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/api.js frontend/src/App.jsx frontend/src/components/ChatInterface.jsx frontend/src/components/ChatInterface.css
git commit -m "feat: edit-and-branch — edit button, inline textarea, branch navigator"
```

---

## Self-Review Checklist

After completing all tasks, run:

```bash
uv run pytest tests/ -v
cd frontend && npm run build
```

Both should pass without errors before considering the implementation complete.
