"""
stt/whisper_engine.py

Ears of JARVIS.
- Primary:  Faster-Whisper Large-v3-Turbo (CPU, int8) — best Hinglish accuracy
- Fallback: Deepgram Nova-2 API         — when mobile on slow internet
- VAD:      webrtcvad                   — skip silence, save CPU
"""
from __future__ import annotations

import asyncio
import io
import time
import wave
from typing import AsyncGenerator

import numpy as np
import webrtcvad
from faster_whisper import WhisperModel

from config import get_settings
from core.logger import get_logger

logger = get_logger("stt.whisper")
settings = get_settings()

# ── Singleton model (loaded once at startup) ───────────────────────────────────
_whisper: WhisperModel | None = None


def get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        logger.info("Loading Faster-Whisper", model=settings.whisper_model,
                    device=settings.whisper_device, compute=settings.whisper_compute_type)
        t0 = time.perf_counter()
        _whisper = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        logger.info("Whisper ready", load_sec=round(time.perf_counter() - t0, 2))
    return _whisper


# ── VAD helper ─────────────────────────────────────────────────────────────────

class VADFilter:
    """
    Wraps webrtcvad to strip silent frames before sending to Whisper.
    Saves ~40% CPU on quiet gaps in speech.
    """

    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000,
                 frame_ms: int = 30):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit

    def has_speech(self, pcm_bytes: bytes) -> bool:
        if len(pcm_bytes) != self.frame_bytes:
            return True  # non-standard chunk → pass through
        try:
            return self.vad.is_speech(pcm_bytes, self.sample_rate)
        except Exception:
            return True


# ── Core transcription ─────────────────────────────────────────────────────────

def _build_wav_bytes(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw PCM in WAV container (in-memory)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


async def transcribe_audio(pcm_bytes: bytes) -> dict:
    """
    Transcribe raw 16kHz mono PCM.
    Returns: {"text": str, "language": str, "latency_ms": int}

    Runs Whisper in a thread-pool to not block the FastAPI event loop.
    """
    t0 = time.perf_counter()
    wav_bytes = _build_wav_bytes(pcm_bytes, settings.sample_rate)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_whisper, wav_bytes)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    result["latency_ms"] = latency_ms
    logger.debug("STT done", text_preview=result["text"][:60],
                 lang=result["language"], latency_ms=latency_ms)
    return result


def _run_whisper(wav_bytes: bytes) -> dict:
    """Blocking call — runs inside executor."""
    model = get_whisper()
    audio_buf = io.BytesIO(wav_bytes)

    segments, info = model.transcribe(
        audio_buf,
        language=None,              # auto-detect (handles Hinglish switch)
        task="transcribe",
        beam_size=5,
        vad_filter=True,            # Whisper's own VAD on top of webrtcvad
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=False,      # skip word stamps — saves 30ms
    )
    text = " ".join(seg.text.strip() for seg in segments)
    return {"text": text.strip(), "language": info.language}


# ── Streaming microphone capture ───────────────────────────────────────────────

async def stream_microphone() -> AsyncGenerator[bytes, None]:
    """
    Yield VAD-filtered PCM chunks from the local microphone.
    Used by the desktop app; mobile sends audio via HTTP endpoint.
    """
    import pyaudio

    vad = VADFilter(
        aggressiveness=settings.vad_aggressiveness,
        sample_rate=settings.sample_rate,
        frame_ms=settings.chunk_duration_ms,
    )
    frame_size = int(settings.sample_rate * settings.chunk_duration_ms / 1000)

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=settings.sample_rate,
        input=True,
        frames_per_buffer=frame_size,
    )

    logger.info("Microphone stream started", sample_rate=settings.sample_rate)
    speech_buffer: list[bytes] = []
    silence_count = 0
    SILENCE_LIMIT = 20  # ~600ms of silence → flush

    try:
        while True:
            chunk = await asyncio.get_event_loop().run_in_executor(
                None, stream.read, frame_size, False
            )
            if vad.has_speech(chunk):
                speech_buffer.append(chunk)
                silence_count = 0
            else:
                if speech_buffer:
                    silence_count += 1
                    if silence_count >= SILENCE_LIMIT:
                        yield b"".join(speech_buffer)
                        speech_buffer.clear()
                        silence_count = 0
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ── Deepgram fallback (mobile slow internet) ───────────────────────────────────

async def transcribe_via_deepgram(audio_bytes: bytes, mime: str = "audio/wav") -> dict:
    """
    Used when: mobile client sets header X-STT-Backend: deepgram
    Requires DEEPGRAM_API_KEY in .env
    """
    if not settings.deepgram_api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not set")

    from deepgram import DeepgramClient, PrerecordedOptions

    t0 = time.perf_counter()
    dg = DeepgramClient(settings.deepgram_api_key)
    options = PrerecordedOptions(
        model="nova-2",
        language="hi-en",           # Hinglish model
        smart_format=True,
        punctuate=True,
    )
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: dg.listen.prerecorded.v("1").transcribe_file(
            {"buffer": audio_bytes, "mimetype": mime}, options
        ),
    )
    text = response.results.channels[0].alternatives[0].transcript
    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.debug("Deepgram STT", latency_ms=latency_ms)
    return {"text": text, "language": "hi-en", "latency_ms": latency_ms}
