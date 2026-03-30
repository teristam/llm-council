"""3-stage LLM Council orchestration."""

import asyncio
from typing import List, Dict, Any, Tuple, Optional
from .openrouter import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL, DEVIL_ADVOCATE_MODEL


def aggregate_tokens(usages: List[Optional[Dict[str, int]]]) -> Dict[str, int]:
    """Sum prompt and completion tokens across a list of usage dicts, skipping None."""
    prompt = sum(u.get('prompt_tokens', 0) for u in usages if u is not None)
    completion = sum(u.get('completion_tokens', 0) for u in usages if u is not None)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total": prompt + completion}


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


def build_conversation_context(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert stored conversation messages to LLM-compatible format.
    Extracts Stage 3 chairman synthesis as assistant content.

    Args:
        messages: List of messages from storage (user + assistant with stages)

    Returns:
        List of {"role": "user/assistant", "content": "text"} dicts
    """
    context = []
    for msg in messages:
        if msg["role"] == "user":
            context.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            # Extract chairman synthesis as the assistant's response
            content = msg.get("stage3", {}).get("response", "")
            if content:  # Only include complete messages
                context.append({"role": "assistant", "content": content})
    return context


def format_history_for_display(conversation_history: List[Dict[str, str]]) -> str:
    """
    Format conversation history for inclusion in prompts.

    Args:
        conversation_history: List of role/content message dicts

    Returns:
        Formatted string for prompt inclusion
    """
    if not conversation_history:
        return ""

    formatted = []
    for msg in conversation_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        # Truncate very long messages to keep prompts manageable
        if len(content) > 500:
            content = content[:497] + "..."
        formatted.append(f"{role}: {content}")

    return "\n\n".join(formatted)


async def stage1_collect_responses(
    user_query: str,
    conversation_history: List[Dict[str, str]] = []
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Stage 1a: Collect individual responses from all council models.
    Each response includes an optional list of clarifying questions (up to 3).

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


async def stage1b_consolidate_questions(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, int]]:
    """
    Stage 1b: Chairman reviews all questions from council members,
    deduplicates and consolidates into a list of distinct questions.

    Returns:
        (consolidated_questions, questions_by_model, tokens_used)
        consolidated_questions: list of question strings (may be empty if NONE)
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

    models_to_revise = [r for r in stage1_results if r.get('questions')]

    if not models_to_revise:
        return stage1_results, aggregate_tokens([])

    async def revise_one(result: Dict[str, Any]):
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

    revisions = await asyncio.gather(*[revise_one(r) for r in models_to_revise])

    revised_by_model = {
        original['model']: (revised, usage)
        for original, (revised, usage) in zip(models_to_revise, revisions)
    }

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


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]] = []
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, int]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1
        conversation_history: Previous messages in OpenAI format (optional)

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    # Build the ranking prompt with optional conversation history
    history_section = ""
    if conversation_history:
        history_section = f"""Conversation History:
{format_history_for_display(conversation_history)}

"""

    ranking_prompt = f"""You are evaluating different responses in an ongoing conversation.

{history_section}Current Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually{', considering whether it appropriately references the conversation history' if conversation_history else ''}. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

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


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]] = [],
    devil_advocate_result: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2
        conversation_history: Previous messages in OpenAI format (optional)
        devil_advocate_result: Devil's advocate challenge result (optional).
            If provided, chairman prompt includes a section requiring direct response.

    Returns:
        Dict with 'model' and 'response' keys
    """
    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    # Build the chairman prompt with optional conversation history
    if conversation_history:
        question_section = f"""Conversation History:
{format_history_for_display(conversation_history)}

Current Question: {user_query}"""
        context_instructions = """- The conversation history and how this question relates to previous discussion
- Ensure your answer builds upon prior answers when relevant
- Maintain consistency with the council's previous positions
"""
        question_type = "current question"
        final_answer_instruction = "that naturally continues the conversation"
    else:
        question_section = f"Original Question: {user_query}"
        context_instructions = ""
        question_type = "original question"
        final_answer_instruction = "that represents the council's collective wisdom"

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

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model with medium reasoning effort
    response = await query_model(CHAIRMAN_MODEL, messages, reasoning={"effort": "medium"})

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


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def parse_devil_advocate_response(raw: str) -> Dict[str, Any]:
    """
    Parse CONSENSUS: and CRITIQUE: sections from devil's advocate response.

    Args:
        raw: Full text response from the devil's advocate model

    Returns:
        Dict with consensus_identified, critique, and raw keys
    """
    consensus_pos = raw.find("CONSENSUS:")
    critique_pos = raw.find("CRITIQUE:")
    if consensus_pos != -1 and critique_pos != -1 and consensus_pos < critique_pos:
        consensus_start = consensus_pos + len("CONSENSUS:")
        critique_content_start = critique_pos + len("CRITIQUE:")

        consensus_text = raw[consensus_start:critique_pos].strip()
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


async def stage2_5_devil_advocate(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]] = []
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    """
    Stage 2.5: Devil's Advocate identifies council consensus and argues against it.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2
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

    da_prompt = f"""You are a Devil's Advocate. Your role is to provide calm, reasoned scrutiny of the council's consensus — not to attack, but to surface overlooked considerations with intellectual honesty.

{history_section}Question being discussed: {user_query}

RESPONSES FROM THE COUNCIL:
{responses_text}

PEER RANKINGS:
{rankings_text}

Your task:
1. Read all the responses carefully and identify the KEY POINTS where the majority of models agreed — the consensus view.
2. Calmly and precisely argue against that consensus. Identify the strongest counterargument, the overlooked evidence, the unexamined assumption, or the potential failure mode — and explain it with clear, measured reasoning.

You MUST format your response EXACTLY as follows:

CONSENSUS:
[State clearly what the majority of models agreed on — be specific, not vague]

CRITIQUE:
[Your reasoned argument against that consensus — be specific and concrete, but measured in tone]"""

    messages = [{"role": "user", "content": da_prompt}]

    response = await query_model(DEVIL_ADVOCATE_MODEL, messages)

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


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use gemini-2.5-flash for title generation (fast and cheap)
    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(
    user_query: str,
    conversation_history: List[Dict[str, str]] = []
) -> Tuple[List, List, Optional[Dict], Dict, Dict]:
    """
    Run the complete council process.

    Args:
        user_query: The user's question
        conversation_history: Previous messages in OpenAI format (optional)

    Returns:
        Tuple of (stage1_results, stage2_results, devil_advocate_result, stage3_result, metadata)
    """
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
