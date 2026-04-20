"""
api/server.py

FastAPI server — the nervous system that connects:
  STT → Brain (LangGraph) → TTS → Client

Endpoints:
  POST /transcribe        — Upload audio, get text back
  POST /chat/stream       — Text in, streaming text out (SSE)
  POST /speak/stream      — Text in, streaming audio out
  POST /pipeline/stream   — Audio in → Text → LLM → Audio out (full pipeline)
  WS   /ws/voice          — WebSocket full-duplex voice session
  GET  /health            — Liveness check
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from brain.agent_graph import AgentState, get_graph
from brain.llm_router import TaskPriority
from config import get_settings
from core.logger import get_logger, setup_logging
from stt.whisper_engine import transcribe_audio, transcribe_via_deepgram
from tts.voice_engine import synthesize_stream

setup_logging()
logger = get_logger("api.server")
settings = get_settings()

app = FastAPI(
    title="JARVIS API",
    description="World's smartest personal AI — Phase 1",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: pre-warm models ───────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("JARVIS starting up...")
    # Pre-warm Whisper in background (loads model into RAM)
    asyncio.create_task(_prewarm_whisper())
    # Pre-compile agent graph
    get_graph()
    logger.info("JARVIS ready", host=settings.app_host, port=settings.app_port)


async def _prewarm_whisper():
    """Load Whisper model at startup so first request is instant."""
    from stt.whisper_engine import get_whisper
    await asyncio.get_event_loop().run_in_executor(None, get_whisper)
    logger.info("Whisper pre-warmed")


# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []        # [{"role": "user"|"assistant", "content": str}]
    language: str = "en"
    priority: str = "standard"      # standard | critical | offline


class TranscribeResponse(BaseModel):
    text: str
    language: str
    latency_ms: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(...),
    x_stt_backend: str = Header(default="deepgram"),   # whisper | deepgram
):
    """
    Upload raw audio (WAV/MP3), get transcription back.
    Mobile app sends audio here when away from home.
    Header X-STT-Backend: deepgram  → use Deepgram (better on slow mobile internet)
    """
    audio_bytes = await audio.read()
    logger.info("Transcribe request", backend=x_stt_backend,
                bytes=len(audio_bytes), content_type=audio.content_type)

    if x_stt_backend == "deepgram":
        result = await transcribe_via_deepgram(audio_bytes, audio.content_type or "audio/wav")
    else:
        result = await transcribe_audio(audio_bytes)

    return TranscribeResponse(**result)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Text in → SSE streaming text out.
    Frontend can start rendering tokens immediately.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    logger.info("Chat request", language=req.language, priority=req.priority,
                msg_preview=req.message[:60])

    # Build message history
    history_msgs = []
    for m in req.history[-10:]:         # last 10 turns max (context window control)
        if m["role"] == "user":
            history_msgs.append(HumanMessage(content=m["content"]))
        else:
            history_msgs.append(AIMessage(content=m["content"]))
    history_msgs.append(HumanMessage(content=req.message))

    initial_state: AgentState = {
        "messages": history_msgs,
        "intent": "unknown",
        "priority": req.priority,
        "language": req.language,
        "response_tokens": [],
    }

    async def token_generator() -> AsyncIterator[str]:
        graph = get_graph()
        async for event in graph.astream_events(initial_state, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.post("/speak/stream")
async def speak_stream(req: ChatRequest):
    """
    Text in → streaming MP3 audio out.
    Client starts playing before full audio is generated.
    """
    logger.info("Speak request", chars=len(req.message))

    async def audio_generator():
        async for chunk in synthesize_stream(req.message):
            yield chunk

    return StreamingResponse(audio_generator(), media_type="audio/mpeg")


@app.post("/pipeline/stream")
async def pipeline_stream(
    audio: UploadFile = File(...),
    language: str = Header(default="en"),
    priority: str = Header(default="standard"),
    x_stt_backend: str = Header(default="whisper"),
    x_output: str = Header(default="audio"),    # audio | text
):
    """
    Full pipeline: Audio → STT → LangGraph → TTS → Streaming Audio
    This is the main endpoint for voice sessions.

    Returns:
      - If X-Output: audio → streaming MP3
      - If X-Output: text  → SSE text tokens
    """
    from langchain_core.messages import HumanMessage

    # Step 1: STT
    audio_bytes = await audio.read()
    if x_stt_backend == "deepgram":
        stt_result = await transcribe_via_deepgram(audio_bytes)
    else:
        stt_result = await transcribe_audio(audio_bytes)

    user_text = stt_result["text"]
    detected_lang = stt_result["language"]
    logger.info("Pipeline STT done", text=user_text[:60], lang=detected_lang,
                stt_ms=stt_result["latency_ms"])

    if not user_text.strip():
        raise HTTPException(status_code=422, detail="No speech detected")

    # Step 2: Brain (collect full response — needed before TTS)
    graph = get_graph()
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_text)],
        "intent": "unknown",
        "priority": priority,
        "language": detected_lang or language,
        "response_tokens": [],
    }

    final_state = await graph.ainvoke(initial_state)
    ai_messages = [m for m in final_state["messages"] if isinstance(m, type(final_state["messages"][-1]))]
    response_text = final_state["messages"][-1].content
    logger.info("Pipeline LLM done", response_preview=response_text[:60])

    if x_output == "text":
        async def text_gen():
            for token in response_text.split():
                yield f"data: {token} \n\n"
                await asyncio.sleep(0)
            yield "data: [DONE]\n\n"
        return StreamingResponse(text_gen(), media_type="text/event-stream")

    # Step 3: TTS stream
    async def audio_gen():
        async for chunk in synthesize_stream(response_text):
            yield chunk

    return StreamingResponse(audio_gen(), media_type="audio/mpeg")


# ── WebSocket: real-time voice session ────────────────────────────────────────

@app.websocket("/ws/voice")
async def voice_websocket(ws: WebSocket):
    """
    WebSocket for continuous voice session.
    Protocol:
      Client sends: {"type": "audio", "data": "<base64 PCM>"}
      Client sends: {"type": "text",  "data": "message text"}
      Server sends: {"type": "token", "data": "word "}
      Server sends: {"type": "done",  "data": ""}
      Server sends: {"type": "error", "data": "message"}
    """
    import base64
    from langchain_core.messages import HumanMessage

    await ws.accept()
    logger.info("WebSocket voice session opened")
    conversation_history = []

    try:
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            if msg_type == "audio":
                # Decode base64 PCM audio
                pcm = base64.b64decode(msg["data"])
                stt = await transcribe_audio(pcm)
                user_text = stt["text"]
                if not user_text.strip():
                    continue
                await ws.send_json({"type": "stt", "data": user_text})

            elif msg_type == "text":
                user_text = msg["data"]
            else:
                continue

            # Run agent graph
            conversation_history.append(HumanMessage(content=user_text))
            initial_state: AgentState = {
                "messages": conversation_history[-20:],  # rolling 20-turn window
                "intent": "unknown",
                "priority": msg.get("priority", "standard"),
                "language": msg.get("language", "en"),
                "response_tokens": [],
            }

            graph = get_graph()
            async for event in graph.astream_events(initial_state, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        await ws.send_json({"type": "token", "data": chunk.content})

            await ws.send_json({"type": "done", "data": ""})

    except WebSocketDisconnect:
        logger.info("WebSocket voice session closed")
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        await ws.send_json({"type": "error", "data": str(e)})
