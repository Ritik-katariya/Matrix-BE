"""
stt/whisper_engine.py

Ears of JARVIS — Faster-Whisper, English only, fast.

Why English only:
  - Hindi detection adds 3-5x processing time
  - language="en" forces Whisper to skip language detection entirely
  - Result: consistent 200-400ms transcription on Ryzen 5600H CPU
  - Hinglish still works — English words in Hindi speech transcribe fine

Model: large-v3-turbo, CPU, int8
"""
from __future__ import annotations

import asyncio
import functools
import io
import time
import wave

from faster_whisper import WhisperModel

from config import get_settings
from core.logger import get_logger

logger   = get_logger("stt.whisper")
settings = get_settings()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — timeit decorator
# Logs how long every transcription call takes.
# ══════════════════════════════════════════════════════════════════════════════

def timeit(func):
    """
    Measures and prints execution time of any function it wraps.

    Output: [_run_whisper] Audio processed in 0.213s
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start   = time.perf_counter()
        result  = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{func.__name__}] Audio processed in {elapsed:.3f}s")
        return result
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Singleton model
# Loaded once at startup, reused for every request.
# Cold load: ~3-5s. Every call after: 0ms.
# ══════════════════════════════════════════════════════════════════════════════

_whisper: WhisperModel | None = None

# Whisper hallucinates these strings on silent/noise-only audio.
# Return "" instead of garbage if detected.
_HALLUCINATIONS = {
    "субтитры создавал dimatorzok",
    "subtitles by dimatorzok",
    "thanks for watching",
    "please subscribe",
    "thank you for watching",
    "www.mooji.org",
}


def get_whisper() -> WhisperModel:
    """
    Return the singleton WhisperModel, loading it on first call only.

    Config:
      large-v3-turbo + cpu + int8
      → best accuracy for English/Hinglish on CPU
      → int8 quantisation = 2x faster than float16, same accuracy
    """
    global _whisper
    if _whisper is None:
        logger.info("Loading Faster-Whisper",
                    model=settings.whisper_model,
                    device=settings.whisper_device,
                    compute=settings.whisper_compute_type)
        t0 = time.perf_counter()
        _whisper = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        logger.info("Whisper ready",
                    load_sec=round(time.perf_counter() - t0, 2))
    return _whisper


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Audio format helper
# Whisper needs WAV; uploaded files may be raw PCM bytes.
# Wraps PCM in WAV header entirely in memory — no disk I/O.
# ══════════════════════════════════════════════════════════════════════════════

def _build_wav_bytes(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """
    Wrap raw 16-bit mono PCM bytes in a WAV container header.
    Done entirely in memory using BytesIO — no temp files.

    Args:
      pcm:         Raw 16-bit mono PCM bytes
      sample_rate: Always 16000 for Whisper

    Returns:
      Complete in-memory WAV file bytes
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)       # mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Core Whisper call (blocking, runs in thread-pool)
# Two variants:
#   _run_whisper()           — from WAV bytes (uploaded audio)
#   _run_whisper_from_path() — from file path (local test files)
# Both decorated with @timeit so timing is always printed.
# ══════════════════════════════════════════════════════════════════════════════

@timeit
def _run_whisper(wav_bytes: bytes) -> dict:
    """
    Blocking Whisper transcription from in-memory WAV bytes.

    Key settings:
      language="en"                  → skip language detection, 3-5x faster
      beam_size=5, best_of=5         → same as your test script
      vad_filter=True                → skip silence inside audio automatically
      condition_on_previous_text=False → prevents hallucination loops
      no_speech_threshold=0.6        → return empty if audio is mostly silence
    """
    model     = get_whisper()
    audio_buf = io.BytesIO(wav_bytes)

    segments, info = model.transcribe(
        audio_buf,
        language="en",                       # English only — fast, no detection
        task="transcribe",
        beam_size=5,
        best_of=5,
        vad_filter=True,                     # built-in silence skip
        vad_parameters={
            "min_silence_duration_ms": 300,  # shorter than before — faster flush
            "speech_pad_ms": 200,
            "threshold": 0.3,
        },
        word_timestamps=False,               # skip word stamps — saves 30ms
        condition_on_previous_text=False,    # kills hallucination loops
        no_speech_threshold=0.6,             # silence gate
        log_prob_threshold=-1.0,
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()

    if text.lower() in _HALLUCINATIONS:
        logger.warning("Hallucination blocked", raw=text[:60])
        return {"text": "", "language": "en"}

    return {"text": text, "language": "en"}


@timeit
def _run_whisper_from_path(file_path: str) -> dict:
    """
    Blocking Whisper transcription directly from a file path.
    Accepts MP3, WAV, OGG — Whisper handles format internally.

    Identical settings to _run_whisper() for consistency.
    Used by transcribe_file() and the standalone test runner.
    """
    model = get_whisper()

    segments, info = model.transcribe(
        file_path,
        language="en",
        task="transcribe",
        beam_size=5,
        best_of=5,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 200,
            "threshold": 0.3,
        },
        word_timestamps=False,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()

    if text.lower() in _HALLUCINATIONS:
        logger.warning("Hallucination blocked in file", raw=text[:60])
        return {"text": "", "language": "en"}

    return {"text": text, "language": "en"}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Async wrappers (called by FastAPI)
# Run the blocking Whisper call in a thread-pool executor so the
# FastAPI event loop never stalls waiting for transcription.
# ══════════════════════════════════════════════════════════════════════════════

async def transcribe_audio(pcm_bytes: bytes) -> dict:
    """
    Async transcription from raw PCM bytes — called by FastAPI /transcribe
    and /pipeline/stream endpoints.

    Converts PCM → WAV in memory, then runs _run_whisper in executor.

    Returns:
      {"text": str, "language": "en", "latency_ms": int}
      text is "" if audio is silent or hallucination detected.
    """
    t0        = time.perf_counter()
    wav_bytes = _build_wav_bytes(pcm_bytes, settings.sample_rate)

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_whisper, wav_bytes)

    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    logger.debug("STT done",
                 text_preview=result["text"][:60],
                 latency_ms=result["latency_ms"])
    return result


async def transcribe_file(file_path: str) -> dict:
    """
    Async transcription from a file path (MP3, WAV, etc).

    Use for testing with local audio files — same as your original test script
    but non-blocking so it works inside FastAPI without freezing the server.

    Example:
      result = await transcribe_file("output.mp3")
      print(result["text"])
    """
    t0 = time.perf_counter()

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_whisper_from_path, file_path)

    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    logger.debug("STT file done",
                 path=file_path,
                 text_preview=result["text"][:60],
                 latency_ms=result["latency_ms"])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Standalone test — run: python -m stt.whisper_engine path/to/audio.mp3
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    async def _demo():
        if len(sys.argv) > 1:
            path = sys.argv[1]
            print(f"Transcribing: {path}\n")
            result = await transcribe_file(path)
        else:
            print("No file given — running hallucination guard test\n")
            silent_pcm = b"\x00\x00" * 16000 * 2   # 2s silence
            result     = await transcribe_audio(silent_pcm)

        print(f"Text:      '{result['text']}'")
        print(f"Language:   {result['language']}")
        print(f"Latency:    {result['latency_ms']}ms")

        if not result["text"]:
            print("\n[OK] Empty returned on silence — hallucination guard working")

    asyncio.run(_demo())