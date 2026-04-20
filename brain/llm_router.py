"""
brain/llm_router.py

Smart LLM routing — the decision layer for which model to call.

Priority chain:
  1. CRITICAL tasks     → OpenAI GPT-4o          (tool-calling, payments, email)
  2. Standard tasks     → NVIDIA NIM Nemotron-70B (free, very smart)
  3. Offline / timeout  → Ollama Llama-3.2 on CPU (zero latency, LAN-only)

Design goal: every call goes through this router.
Agents never import LLM clients directly.
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import AsyncIterator

from langchain_core.messages import BaseMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from config import get_settings
from core.logger import get_logger

logger = get_logger("brain.router")
settings = get_settings()


# ── Task priority enum ─────────────────────────────────────────────────────────

class TaskPriority(str, Enum):
    CRITICAL = "critical"       # Always OpenAI — financial, medical, complex tool-use
    STANDARD = "standard"       # NVIDIA NIM — 90% of conversations
    OFFLINE  = "offline"        # Force Ollama — user requested offline / NIM down


# ── LLM singletons (lazy init, created once) ───────────────────────────────────

_nvidia_llm: ChatNVIDIA | None = None
_openai_llm: ChatOpenAI | None = None
_ollama_llm: ChatOllama | None = None


def _get_nvidia() -> ChatNVIDIA:
    global _nvidia_llm
    if _nvidia_llm is None:
        _nvidia_llm = ChatNVIDIA(
            model=settings.nvidia_model,
            api_key=settings.nvidia_api_key,
            temperature=0.7,
            max_tokens=1024,
            streaming=True,
        )
        logger.info("NVIDIA NIM client ready", model=settings.nvidia_model)
    return _nvidia_llm


def _get_openai() -> ChatOpenAI:
    global _openai_llm
    if _openai_llm is None:
        _openai_llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.3,          # lower temp for critical/precise tasks
            max_tokens=2048,
            streaming=True,
        )
        logger.info("OpenAI client ready", model=settings.openai_model)
    return _openai_llm


def _get_ollama() -> ChatOllama:
    global _ollama_llm
    if _ollama_llm is None:
        _ollama_llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.7,
            num_predict=512,          # limit tokens on local CPU
        )
        logger.info("Ollama client ready", model=settings.ollama_model)
    return _ollama_llm


# ── Public router interface ────────────────────────────────────────────────────

async def stream_response(
    messages: list[BaseMessage],
    priority: TaskPriority = TaskPriority.STANDARD,
    timeout: float | None = None,
) -> AsyncIterator[str]:
    """
    Stream tokens from the best available LLM.

    Yields:  str chunks (tokens) as they arrive
    Raises:  RuntimeError if all backends fail
    """
    if priority == TaskPriority.CRITICAL:
        async for token in _stream_openai(messages):
            yield token
        return

    if priority == TaskPriority.OFFLINE:
        async for token in _stream_ollama(messages):
            yield token
        return

    # STANDARD → try NVIDIA first, fall back to Ollama
    try:
        nvidia_timeout = timeout or settings.nvidia_timeout_sec
        async for token in _stream_with_timeout(_stream_nvidia(messages), nvidia_timeout):
            yield token
    except (TimeoutError, Exception) as e:
        logger.warning("NVIDIA NIM failed, falling back to Ollama", error=str(e))
        async for token in _stream_ollama(messages):
            yield token


async def get_response(
    messages: list[BaseMessage],
    priority: TaskPriority = TaskPriority.STANDARD,
) -> str:
    """Collect full response (non-streaming). Used by tool-calling agents."""
    tokens: list[str] = []
    async for chunk in stream_response(messages, priority):
        tokens.append(chunk)
    return "".join(tokens)


# ── Private streaming helpers ──────────────────────────────────────────────────

async def _stream_nvidia(messages: list[BaseMessage]) -> AsyncIterator[str]:
    llm = _get_nvidia()
    t0 = time.perf_counter()
    first_token = True
    async for chunk in llm.astream(messages):
        if first_token:
            logger.debug("NVIDIA first token", ttft_ms=int((time.perf_counter()-t0)*1000))
            first_token = False
        if chunk.content:
            yield chunk.content


async def _stream_openai(messages: list[BaseMessage]) -> AsyncIterator[str]:
    llm = _get_openai()
    t0 = time.perf_counter()
    first_token = True
    async for chunk in llm.astream(messages):
        if first_token:
            logger.debug("OpenAI first token", ttft_ms=int((time.perf_counter()-t0)*1000))
            first_token = False
        if chunk.content:
            yield chunk.content


async def _stream_ollama(messages: list[BaseMessage]) -> AsyncIterator[str]:
    llm = _get_ollama()
    t0 = time.perf_counter()
    first_token = True
    async for chunk in llm.astream(messages):
        if first_token:
            logger.debug("Ollama first token", ttft_ms=int((time.perf_counter()-t0)*1000))
            first_token = False
        if chunk.content:
            yield chunk.content


async def _stream_with_timeout(gen: AsyncIterator[str], timeout: float) -> AsyncIterator[str]:
    """Wrap an async generator with a hard timeout on the first token."""
    iterator = gen.__aiter__()
    # Wait for the first token with a timeout
    try:
        first = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"LLM first-token timeout after {timeout}s")
    yield first
    # Stream the rest without timeout (already connected)
    async for token in iterator:
        yield token
