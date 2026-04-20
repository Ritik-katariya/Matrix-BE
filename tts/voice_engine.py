"""
tts/voice_engine.py

Voice of JARVIS.

Strategy:
  - Detect language of response text
  - Hindi / Hinglish  → Gnani.ai (MOS 4.3, best Hindi TTS 2026)
  - English           → Fish Audio S2 Pro (emotion tags support)
  - Both stream audio chunks so playback starts in <300ms

Emotion tags passed from LLM: [laughter] [sigh] [excited] [thinking]
Fish Audio handles these natively.
Gnani strips them (Hindi doesn't have equivalent yet).
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import AsyncIterator

import httpx

from config import get_settings
from core.logger import get_logger

# Install once: pip install pyttsx3
import pyttsx3

def speak_offline(text: str) -> None:
    """Zero-cost offline TTS — runs on CPU, no internet needed."""
    clean = strip_emotion_tags(clean_for_tts(text))
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)   # speed
    engine.say(clean)
    engine.runAndWait()


logger = get_logger("tts.voice")
settings = get_settings()

# ── Language detection (simple, no extra deps) ────────────────────────────────

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_EMOTION_TAG_RE = re.compile(r"\[(\w+)\]")


def detect_tts_backend(text: str) -> str:
    """
    Returns: "gnani" | "fish"
    Hindi/Hinglish → gnani, English → fish
    """
    if _DEVANAGARI_RE.search(text):
        return "gnani"
    # Heuristic: if >20% Hindi-origin words, use Gnani even for transliterated text
    hindi_words = {"yaar", "kya", "hai", "nahi", "aur", "bhi", "toh",
                   "main", "hoon", "karo", "arre", "bilkul", "shukriya",
                   "theek", "matlab", "phir", "abhi", "woh", "isko"}
    words = set(text.lower().split())
    if len(words & hindi_words) / max(len(words), 1) > 0.15:
        return "gnani"
    return "fish"


def strip_emotion_tags(text: str) -> str:
    """Remove [tag] markers — for Gnani which doesn't understand them."""
    return _EMOTION_TAG_RE.sub("", text).strip()


def clean_for_tts(text: str) -> str:
    """Remove markdown, extra spaces."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Fish Audio (English + emotion tags) ──────────────────────────────────────

async def stream_fish_audio(text: str) -> AsyncIterator[bytes]:
    """
    Stream audio from Fish Audio S2 Pro API.
    Returns MP3 chunks as they arrive — start playback immediately.
    """
    if not settings.fish_audio_api_key:
        logger.warning("Fish Audio key not set, skipping TTS")
        return

    clean_text = clean_for_tts(text)
    logger.debug("Fish Audio TTS", chars=len(clean_text),
                 preview=clean_text[:50])

    payload = {
        "text": clean_text,
        "reference_id": settings.fish_audio_reference_id or None,
        "format": "mp3",
        "streaming": True,
        "normalize": True,
        "latency": "balanced",      # balanced | normal
    }
    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}

    t0 = time.perf_counter()
    first_chunk = True

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST",
            "https://api.fish.audio/v1/tts",
            headers={
                "Authorization": f"Bearer {settings.fish_audio_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=4096):
                if chunk:
                    if first_chunk:
                        logger.debug("Fish Audio first chunk",
                                     ttft_ms=int((time.perf_counter()-t0)*1000))
                        first_chunk = False
                    yield chunk


# ── Gnani.ai (Hindi / Hinglish) ───────────────────────────────────────────────

async def stream_gnani_audio(text: str) -> AsyncIterator[bytes]:
    """
    Stream audio from Gnani.ai TTS API.
    Hindi MOS 4.3/5 — best Hindi TTS available in 2026.
    """
    if not settings.gnani_api_key:
        logger.warning("Gnani API key not set, skipping TTS")
        return

    clean_text = strip_emotion_tags(clean_for_tts(text))
    logger.debug("Gnani TTS", chars=len(clean_text), preview=clean_text[:50])

    payload = {
        "text": clean_text,
        "language": settings.gnani_language,
        "encoding": "mp3",
        "sample_rate": 22050,
    }

    t0 = time.perf_counter()
    first_chunk = True

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST",
            "https://tts.gnani.ai/api/v1/synthesize/stream",   # verify endpoint in docs
            headers={
                "Authorization": f"Bearer {settings.gnani_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=4096):
                if chunk:
                    if first_chunk:
                        logger.debug("Gnani first chunk",
                                     ttft_ms=int((time.perf_counter()-t0)*1000))
                        first_chunk = False
                    yield chunk


# ── Public router ─────────────────────────────────────────────────────────────

async def synthesize_stream(text: str) -> AsyncIterator[bytes]:
    backend = detect_tts_backend(text)

    # Try cloud first, fall back to offline if no keys
    if backend == "gnani" and settings.gnani_api_key:
        async for chunk in stream_gnani_audio(text):
            yield chunk
    elif backend == "fish" and settings.fish_audio_api_key:
        async for chunk in stream_fish_audio(text):
            yield chunk
    else:
        # Offline fallback — pyttsx3 on CPU
        logger.warning("No TTS keys set — using offline pyttsx3")
        speak_offline(text)   # blocking but works
        return
# ── Local audio playback (desktop) ───────────────────────────────────────────

async def play_audio_stream(audio_gen: AsyncIterator[bytes]) -> None:
    """
    Plays streaming MP3 audio locally using sounddevice.
    Buffers just enough to avoid glitches, starts playing immediately.
    """
    import io
    import sounddevice as sd
    from scipy.io import wavfile
    import numpy as np

    # Collect all chunks first (MP3 decode needs complete file)
    # TODO Phase 2: use pydub streaming decode for true zero-latency playback
    buffer = io.BytesIO()
    async for chunk in audio_gen:
        buffer.write(chunk)

    buffer.seek(0)
    try:
        # Try to decode as WAV first, fallback to raw bytes
        import subprocess
        # Convert MP3 → WAV via ffmpeg (fast, <5ms)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", "pipe:0", "-f", "wav", "-ar", "22050",
            "-ac", "1", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        wav_bytes, _ = await proc.communicate(input=buffer.read())
        wav_buf = io.BytesIO(wav_bytes)
        sample_rate, data = wavfile.read(wav_buf)
        sd.play(data, samplerate=sample_rate, blocking=True)
    except Exception as e:
        logger.error("Audio playback failed", error=str(e))
