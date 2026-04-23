"""
api/server.py

FastAPI server — the nervous system that connects:
  STT → Brain (LangGraph) → TTS → Client

Endpoints:
  POST /transcribe        — Upload audio, get text back
  POST /chat/stream       — Text in, streaming text out (SSE)
  POST /speak/stream      — Text in, streaming audio out
  POST /pipeline/stream   — Audio in → Text → LLM → Audio out (full pipeline)
  WS   /ws/voice          — WebSocket full-duplex voice session (legacy)
  WS   /ws/voice2         — WebSocket full-duplex voice session (v2, with TTS)
  GET  /health            — Liveness check
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from typing import AsyncIterator

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from brain.agent_graph import AgentState, get_graph
from config import get_settings
from core.logger import get_logger, setup_logging
from stt.whisper_engine import transcribe_audio
from tts.voice_engine import synthesize_stream

setup_logging()
logger   = get_logger("api.server")
settings = get_settings()

app = FastAPI(
    title="JARVIS API",
    description="World's smartest personal AI — Phase 1",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: pre-warm models ───────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("JARVIS starting up...")
    asyncio.create_task(_prewarm_whisper())
    get_graph()
    logger.info("JARVIS ready", host=settings.app_host, port=settings.app_port)


async def _prewarm_whisper():
    from stt.whisper_engine import get_whisper
    await asyncio.get_event_loop().run_in_executor(None, get_whisper)
    logger.info("Whisper pre-warmed")


# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    language: str = "en"
    priority: str = "standard"      # standard | critical | offline


class TranscribeResponse(BaseModel):
    text: str
    language: str
    latency_ms: int


# ── Helpers ────────────────────────────────────────────────────────────────────

def _split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split text on sentence boundaries. Returns (complete_sentences, remainder)."""
    parts = re.split(r'(?<=[.!?])\s+', buffer)
    if len(parts) <= 1:
        return [], buffer
    return parts[:-1], parts[-1]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(...),
    x_stt_backend: str = Header(default="whisper"),
):
    audio_bytes = await audio.read()
    logger.info("Transcribe request", bytes=len(audio_bytes), content_type=audio.content_type)
    result = await transcribe_audio(audio_bytes)
    return TranscribeResponse(**result)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    from langchain_core.messages import AIMessage, HumanMessage

    logger.info("Chat request", language=req.language, priority=req.priority,
                msg_preview=req.message[:60])

    history_msgs = []
    for m in req.history[-10:]:
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
    x_output: str = Header(default="audio"),
):
    from langchain_core.messages import HumanMessage

    audio_bytes = await audio.read()
    stt_result  = await transcribe_audio(audio_bytes)
    user_text   = stt_result["text"]

    logger.info("Pipeline STT done", text=user_text[:60], stt_ms=stt_result["latency_ms"])

    if not user_text.strip():
        raise HTTPException(status_code=422, detail="No speech detected")

    graph = get_graph()
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_text)],
        "intent": "unknown",
        "priority": priority,
        "language": language,
        "response_tokens": [],
    }

    final_state   = await graph.ainvoke(initial_state)
    response_text = final_state["messages"][-1].content
    logger.info("Pipeline LLM done", response_preview=response_text[:60])

    if x_output == "text":
        async def text_gen():
            for token in response_text.split():
                yield f"data: {token} \n\n"
                await asyncio.sleep(0)
            yield "data: [DONE]\n\n"
        return StreamingResponse(text_gen(), media_type="text/event-stream")

    async def audio_gen():
        async for chunk in synthesize_stream(response_text):
            yield chunk

    return StreamingResponse(audio_gen(), media_type="audio/mpeg")


# ── WebSocket v1: legacy (text only, no TTS) ──────────────────────────────────

@app.websocket("/ws/voice")
async def voice_websocket(ws: WebSocket):
    from langchain_core.messages import HumanMessage

    await ws.accept()
    logger.info("WebSocket voice session opened")
    conversation_history = []

    try:
        while True:
            msg      = await ws.receive_json()
            msg_type = msg.get("type")

            if msg_type == "audio":
                pcm       = base64.b64decode(msg["data"])
                stt       = await transcribe_audio(pcm)
                user_text = stt["text"]
                if not user_text.strip():
                    continue
                await ws.send_json({"type": "stt", "data": user_text})

            elif msg_type == "text":
                user_text = msg["data"]
            else:
                continue

            conversation_history.append(HumanMessage(content=user_text))
            initial_state: AgentState = {
                "messages":        conversation_history[-20:],
                "intent":          "unknown",
                "priority":        msg.get("priority", "standard"),
                "language":        msg.get("language", "en"),
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
        try:
            await ws.send_json({"type": "error", "data": str(e)})
        except Exception:
            pass


# ── WebSocket v2: full pipeline with TTS streaming ────────────────────────────

@app.websocket("/ws/voice2")
async def voice_websocket_v2(ws: WebSocket):
    """
    Full-duplex voice pipeline.

    Client → Server:
      binary bytes               raw 16-bit mono PCM at 16kHz (audio chunks)
      {"type": "end_of_speech"}  VAD detected pause — process buffered audio
      {"type": "ping"}           keepalive

    Server → Client (all JSON):
      {"type": "stt",         "data": "transcribed text"}
      {"type": "token",       "data": "word "}           LLM token stream
      {"type": "audio_chunk", "data": "<base64 MP3>"}    TTS audio stream
      {"type": "done"}                                   turn complete
      {"type": "error",       "data": "message"}
      {"type": "pong"}                                   keepalive reply
    """
    from langchain_core.messages import AIMessage, HumanMessage

    await ws.accept()
    logger.info("Voice v2 session opened")

    audio_buffer:        bytearray  = bytearray()
    conversation_history: list      = []
    is_processing:        bool      = False

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def safe_send(data: dict) -> None:
        """Send JSON frame — silently swallow errors if socket is already closed."""
        try:
            logger.debug("safe_send attempt", type=data.get("type"), client_state=str(ws.client_state))
            await ws.send_json(data)
            logger.debug("safe_send success", type=data.get("type"))
        except Exception:
            logger.warning("safe_send failed", type=data.get("type"))
            pass

    async def stream_tts_sentence(text: str) -> None:
        """
        Synthesize a single sentence via edge_tts and forward each MP3 chunk
        to the client as {"type": "audio_chunk", "data": "<base64>"}.

        Always JSON — never raw binary — so the client onmessage handler
        can JSON.parse every frame without a try/catch split.
        """
        logger.info("TTS sentence", chars=len(text), preview=text[:60])
        chunk_count = 0
        try:
            await safe_send({"type": "tts_sentence_start", "data": {"text": text}})
            async for chunk in synthesize_stream(text):
                if not chunk:
                    continue
                encoded = base64.b64encode(chunk).decode("utf-8")
                await safe_send({"type": "audio_chunk", "data": encoded})
                chunk_count += 1
        except Exception as e:
            logger.warning("TTS chunk failed", error=str(e))
            await safe_send({"type": "tts_sentence_error", "data": str(e)})
        finally:
            await safe_send({"type": "tts_sentence_end", "data": {"text": text, "chunks": chunk_count}})
        logger.info("TTS sentence done", chunks=chunk_count)

    def is_disconnected() -> bool:
        """Check WebSocket client state without raising."""
        try:
            return ws.client_state.value == 3   # 3 = DISCONNECTED
        except Exception:
            return False

    # ── Main loop ─────────────────────────────────────────────────────────────

    try:
        while True:

            # Receive next frame — exits loop on any socket error
            try:
                logger.debug("Waiting for next message from client", client_state=str(ws.client_state))
                message = await ws.receive()
                logger.debug("Received message from client", msg_type=message.get("type"))
            except Exception:
                logger.warning("Error receiving from client, exiting loop")
                break

            # Graceful client disconnect
            if message.get("type") == "websocket.disconnect":
                logger.info("Client disconnect message received")
                break

            # ── Binary frame: PCM audio chunk ──────────────────────────
            if "bytes" in message and message["bytes"]:
                if not is_processing:
                    audio_buffer.extend(message["bytes"])
                    logger.debug("Audio buffer", bytes=len(audio_buffer))
                else:
                    logger.debug("Dropping audio chunk — still processing")
                continue

            # ── Text frame: control message ────────────────────────────
            raw = message.get("text")
            if not raw:
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Unparseable text frame", raw=raw[:40])
                continue

            msg_type = msg.get("type")

            if msg_type == "ping":
                await safe_send({"type": "pong"})
                continue

            if msg_type != "end_of_speech":
                logger.debug("Unknown message type", msg_type=msg_type)
                continue

            # ── end_of_speech received — process buffered audio ────────

            buffer_size = len(audio_buffer)
            logger.info("end_of_speech received", buffer_bytes=buffer_size)

            if buffer_size < 3200:      # < 100ms of 16kHz 16-bit mono
                logger.info("Buffer too short, skipping", bytes=buffer_size)
                audio_buffer.clear()
                continue

            if is_processing:
                logger.info("Already processing, dropping utterance")
                audio_buffer.clear()
                continue

            is_processing = True
            pcm_bytes = bytes(audio_buffer)
            audio_buffer.clear()

            try:

                # ── Step 1: STT ────────────────────────────────────────
                logger.info("STT start", pcm_bytes=len(pcm_bytes))
                stt_result = await transcribe_audio(pcm_bytes)
                user_text  = stt_result["text"].strip()
                logger.info("STT done",
                            text=user_text[:80],
                            latency_ms=stt_result.get("latency_ms"))

                if not user_text:
                    logger.info("Empty transcript — silent audio, skipping")
                    await safe_send({"type": "stt", "data": ""})
                    continue

                await safe_send({"type": "stt", "data": user_text})

                # ── Step 2: LLM streaming ──────────────────────────────
                conversation_history.append(HumanMessage(content=user_text))

                initial_state: AgentState = {
                    "messages":        conversation_history[-20:],
                    "intent":          "unknown",
                    "priority":        msg.get("priority", "standard"),
                    "language":        msg.get("language", "en"),
                    "response_tokens": [],
                }

                graph        = get_graph()
                token_buffer = ""
                full_response = ""
                token_count  = 0

                logger.info("LLM streaming start")

                async for event in graph.astream_events(initial_state, version="v2"):

                    if is_disconnected():
                        logger.info("Client disconnected mid-LLM, aborting")
                        return

                    if event["event"] != "on_chat_model_stream":
                        continue

                    chunk = event["data"]["chunk"]
                    if not chunk.content:
                        continue

                    token         = chunk.content
                    full_response += token
                    token_buffer  += token
                    token_count   += 1
                    await safe_send({"type": "token", "data": token})

                    # ── Step 3: TTS — flush on sentence boundary ───────
                    sentences, token_buffer = _split_sentences(token_buffer)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence:
                            if is_disconnected():
                                logger.info("Client disconnected mid-TTS, aborting")
                                return
                            await stream_tts_sentence(sentence)

                # Flush any remaining text that didn't end with punctuation
                token_buffer = token_buffer.strip()
                if token_buffer:
                    logger.info("Flushing remainder", text=token_buffer[:60])
                    await stream_tts_sentence(token_buffer)

                conversation_history.append(AIMessage(content=full_response))

                logger.info("Turn complete",
                            tokens=token_count,
                            response_chars=len(full_response))

                await safe_send({"type": "done"})

            except Exception as e:
                logger.error("Processing error", error=str(e))
                await safe_send({"type": "error", "data": str(e)})

            finally:
                is_processing = False

    except Exception as e:
        logger.error("Voice v2 fatal error", error=str(e))

    finally:
        logger.info("Voice v2 session closed")