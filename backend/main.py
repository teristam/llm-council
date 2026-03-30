"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import json
import asyncio
import os

from . import storage
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

app = FastAPI(title="LLM Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str


class ClarifyMessageRequest(BaseModel):
    """Pass 2 request: user's answer plus stage1 context from Pass 1."""
    user_answer: Optional[str]
    user_query: str
    stage1_results: List[Dict[str, Any]]
    questions_by_model: Dict[str, List[str]]
    consolidated_questions: List[str]
    asst_message_id: Optional[str] = None


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Add user message
    storage.add_user_message(conversation_id, request.content)

    # If this is the first message, generate a title
    if is_first_message:
        title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    # Build conversation context from stored messages (before the current user message)
    conversation_history = build_conversation_context(conversation["messages"])

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


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Build conversation context from stored messages (before adding current user message)
    conversation_history = build_conversation_context(conversation["messages"])

    async def event_generator():
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
            consolidated_questions, questions_by_model, _ = stage1b_task.result()

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

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx/proxy buffering
            "Transfer-Encoding": "chunked",
        }
    )


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
            user_query = request.user_query

            # Stage 1c: revise responses (only if user answered)
            if request.user_answer is not None:
                yield f"data: {json.dumps({'type': 'stage1c_start'})}\n\n"
                stage1c_task = asyncio.create_task(
                    stage1c_revise_responses(
                        user_query,
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

            # Save assistant message
            clarification = {
                "questions": request.consolidated_questions,
                "answer": request.user_answer,
                "questions_by_model": request.questions_by_model,
            }
            if request.asst_message_id:
                # Edit branch: handled by edit endpoint — storage will be updated there
                pass
            else:
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


# Serve static frontend files in production mode
# Check if frontend/dist exists (production build)
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if FRONTEND_DIST.exists():
    # Serve static assets
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    # Catch-all route for SPA - must be last
    @app.get("/{path:path}")
    async def serve_spa(path: str):
        """Serve the SPA for any non-API route."""
        # Try to serve the exact file first
        file_path = FRONTEND_DIST / path
        if file_path.is_file():
            return FileResponse(file_path)
        # Otherwise serve index.html for SPA routing
        return FileResponse(FRONTEND_DIST / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        timeout_keep_alive=600,  # 10 min keepalive for long SSE connections
    )
