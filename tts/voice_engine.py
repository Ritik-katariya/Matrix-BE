"""
tts/voice_engine.py

Voice of JARVIS — powered entirely by edge_tts (Microsoft Edge TTS).

Why edge_tts:
  - Completely FREE, no API key needed
  - Supports Hinglish naturally via en-IN-PrabhatNeural
  - Streams audio chunks so playback starts before synthesis finishes
  - 400+ voices built in

Voice strategy (auto-switched by language detection):
  Hinglish  → en-IN-PrabhatNeural  (best for Hinglish rhythm)
  Hindi     → hi-IN-MadhurNeural   (pure Hindi, natural)
  English   → en-US-GuyNeural      (serious JARVIS feel)

Install: pip install edge-tts
"""
from __future__ import annotations

import asyncio
import io
import re
import time
from typing import AsyncIterator

import edge_tts

# from config import get_settings
from core.logger import get_logger

logger = get_logger("tts.voice")
# settings = get_settings()
# 

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Voice profiles
# All available voices in one dict. Change DEFAULT_VOICE to switch globally.
# ══════════════════════════════════════════════════════════════════════════════

VOICES = {
    "hinglish_male":   "en-IN-PrabhatNeural",   # Best for Hinglish mixing
    "hinglish_female": "en-IN-NeerjaNeural",     # Female Hinglish
    "hindi_male":      "hi-IN-MadhurNeural",     # Pure Hindi male
    "hindi_female":    "hi-IN-SwaraNeural",      # Pure Hindi female
    "english_jarvis":  "en-US-GuyNeural",        # Serious JARVIS English
    "english_natural": "en-GB-RyanNeural",       # Warm natural English
}

DEFAULT_VOICE = VOICES["hinglish_male"]

# emotion tag from LLM → (rate adjustment, pitch adjustment) for edge_tts
EMOTION_PROSODY: dict[str, tuple[str, str]] = {
    "excited":  ("+20%", "+25Hz"),   # faster + higher
    "laughter": ("+15%", "+20Hz"),   # cheerful
    "sigh":     ("-15%", "-10Hz"),   # slower + lower
    "thinking": ("-10%", "-5Hz"),    # slow, thoughtful
    "sad":      ("-20%", "-15Hz"),   # slow + low
    "angry":    ("+25%", "+10Hz"),   # fast
}

DEFAULT_RATE  = "+1%"
DEFAULT_PITCH = "+18Hz"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Text preprocessing
# Clean and analyse text BEFORE sending to edge_tts.
# ══════════════════════════════════════════════════════════════════════════════

_DEVANAGARI_RE  = re.compile(r"[\u0900-\u097F]")
_EMOTION_TAG_RE = re.compile(r"\[(\w+)\]")

_HINDI_LATIN_WORDS = {
    "yaar", "kya", "hai", "nahi", "aur", "bhi", "toh", "main", "hoon",
    "karo", "arre", "bilkul", "shukriya", "theek", "matlab", "phir",
    "abhi", "woh", "isko", "usko", "mera", "tera", "apna", "bhai",
    "dost", "accha", "haan", "nah", "bas", "chal", "ruk", "sun",
}


def detect_language(text: str) -> str:
    """
    Detect the dominant language of the response text.
    Returns: "hindi" | "hinglish" | "english"

    How it works:
      1. Devanagari script detected      → "hindi"
      2. More than 15% Hindi-origin words → "hinglish"
      3. Otherwise                        → "english"

    This drives select_voice() to automatically pick the right voice
    without any manual configuration per request.
    """
    if _DEVANAGARI_RE.search(text):
        return "hindi"
    words = set(text.lower().split())
    ratio = len(words & _HINDI_LATIN_WORDS) / max(len(words), 1)
    return "hinglish" if ratio > 0.15 else "english"


def extract_dominant_emotion(text: str) -> str | None:
    """
    Find the first [emotion_tag] in the LLM's response text.
    Returns the tag name if it's a recognised emotion, else None.

    The LLM places emotion tags at sentence starts:
      "[excited] Arre yaar, that is brilliant!"  → returns "excited"
      "Hello, how are you?"                      → returns None

    The returned emotion is used by select_prosody() to adjust
    edge_tts rate and pitch for that sentence's delivery.
    Tags are stripped from text before TTS sees it (in strip_emotion_tags).
    """
    match = _EMOTION_TAG_RE.search(text)
    if match:
        tag = match.group(1).lower()
        if tag in EMOTION_PROSODY:
            return tag
    return None


def strip_emotion_tags(text: str) -> str:
    """
    Remove all [tag] markers from text before passing to edge_tts.
    These are JARVIS-internal prosody signals — edge_tts must never see them
    or it will literally speak the word "excited" aloud.

    "[excited] Hello there!" → "Hello there!"
    "[sigh] yeh mushkil hai" → "yeh mushkil hai"
    """
    return _EMOTION_TAG_RE.sub("", text).strip()


def clean_for_tts(text: str) -> str:
    """
    Strip markdown formatting that sounds wrong when spoken aloud.
    Removes: **bold**, *italic*, `code blocks`, # headers, extra spaces.

    "Use **pip install** to setup" → "Use pip install to setup"
    "`print('hello')`"            → "print hello"
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*",     r"\1", text)
    text = re.sub(r"`(.+?)`",       r"\1", text)
    text = re.sub(r"#{1,6}\s+",     "",    text)
    text = re.sub(r"\s+",           " ",   text)
    return text.strip()


def prepare_text(text: str) -> str:
    """
    Full text preparation pipeline — always call this before TTS synthesis.
    Runs: clean_for_tts → strip_emotion_tags in the correct order.

    Order matters: clean markdown first (may reveal hidden chars),
    then strip emotion tags so nothing JARVIS-internal reaches edge_tts.
    """
    return strip_emotion_tags(clean_for_tts(text))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Voice and prosody selection
# Decide which voice and speed/pitch to use for each response.
# ══════════════════════════════════════════════════════════════════════════════

def select_voice(text: str, voice_override: str | None = None) -> str:
    """
    Pick the best edge_tts voice for the response text.

    Priority order:
      1. voice_override  → caller explicitly requested a specific voice
      2. detect_language → auto-map detected language to best voice
      3. DEFAULT_VOICE   → final fallback (en-IN-PrabhatNeural)

    Examples:
      "Hello how are you today"    → en-US-GuyNeural     (english)
      "arre yaar kya haal hai"     → en-IN-PrabhatNeural (hinglish)
      "नमस्ते कैसे हो"              → hi-IN-MadhurNeural  (hindi)
    """
    if voice_override:
        return voice_override

    lang = detect_language(text)
    voice = {
        "hindi":    VOICES["hindi_male"],
        "hinglish": VOICES["hinglish_male"],
        "english":  VOICES["english_jarvis"],
    }.get(lang, DEFAULT_VOICE)

    logger.debug("Voice selected", lang=lang, voice=voice)
    return voice


def select_prosody(emotion: str | None) -> tuple[str, str]:
    """
    Map an emotion tag to (rate, pitch) adjustments for edge_tts.
    Returns defaults if emotion is None or unrecognised.

    "excited" → ("+20%", "+25Hz")  — speaks faster and higher pitched
    "sigh"    → ("-15%", "-10Hz")  — slower and deeper
    None      → ("+1%",  "+18Hz")  — normal JARVIS tone (slight depth)

    These strings go directly into:
      edge_tts.Communicate(rate=rate, pitch=pitch)
    """
    if emotion and emotion in EMOTION_PROSODY:
        rate, pitch = EMOTION_PROSODY[emotion]
        logger.debug("Emotion prosody", emotion=emotion, rate=rate, pitch=pitch)
        return rate, pitch
    return DEFAULT_RATE, DEFAULT_PITCH


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Core synthesis
# The actual edge_tts calls. Two modes: streaming (for API) and file (for test).
# ══════════════════════════════════════════════════════════════════════════════

async def synthesize_stream(
    text: str,
    voice_override: str | None = None,
) -> AsyncIterator[bytes]:
    """
    PRIMARY TTS FUNCTION — called by FastAPI /speak/stream and /pipeline/stream.

    Streams raw MP3 audio bytes from edge_tts as they are generated.
    FastAPI forwards these chunks to the client immediately, so the user
    hears the first words in ~300ms, before synthesis is even finished.

    Args:
      text:           Raw LLM output. Emotion tags and markdown are OK —
                      this function cleans everything internally.
      voice_override: Force a specific voice e.g. "hi-IN-SwaraNeural".
                      Pass None to let language detection decide.

    Yields:
      Raw MP3 bytes in ~4KB chunks as edge_tts generates them.

    Internal flow:
      1. extract_dominant_emotion  → detect [tag] for prosody
      2. select_prosody            → map tag to rate + pitch strings
      3. prepare_text              → strip tags + markdown
      4. select_voice              → pick voice from detected language
      5. edge_tts.Communicate      → start synthesis
      6. stream() loop             → yield only "audio" type chunks
    """
    t0 = time.perf_counter()

    # Step 1+2 — emotion must be extracted BEFORE prepare_text strips the tags
    emotion = extract_dominant_emotion(text)
    rate, pitch = select_prosody(emotion)

    # Step 3 — clean text (strips tags + markdown)
    clean_text = prepare_text(text)
    if not clean_text:
        logger.warning("Empty text after cleaning, skipping TTS")
        return

    # Step 4 — pick voice
    voice = select_voice(clean_text, voice_override)
    logger.info("TTS synthesizing", voice=voice, emotion=emotion,
                rate=rate, pitch=pitch, chars=len(clean_text),
                preview=clean_text[:60])

    # Step 5+6 — stream from edge_tts
    communicate = edge_tts.Communicate(
        text=clean_text,
        voice=voice,
        rate=rate,
        pitch=pitch,
    )

    first_chunk = True
    chunk_count = 0

    async for chunk in communicate.stream():
        # edge_tts yields two types:
        #   {"type": "audio",         "data": bytes}  ← we want this
        #   {"type": "WordBoundary",  ...}             ← timing info, skip
        if chunk["type"] == "audio" and chunk["data"]:
            if first_chunk:
                logger.debug("First audio chunk",
                             ttft_ms=int((time.perf_counter() - t0) * 1000))
                first_chunk = False
            chunk_count += 1
            yield chunk["data"]

    logger.debug("TTS stream done", chunks=chunk_count,
                 total_ms=int((time.perf_counter() - t0) * 1000))


async def synthesize_to_file(
    text: str,
    output_path: str = "output.mp3",
    voice_override: str | None = None,
) -> str:
    """
    Synthesize speech and save the result to an MP3 file on disk.
    Does NOT stream — waits for full synthesis then writes file.

    Use this for:
      - Testing and auditioning voices quickly
      - Pre-generating audio for known fixed responses
      - Debugging prosody / rate / pitch settings

    Args:
      text:           Text to speak (raw LLM output OK, cleaned here)
      output_path:    Where to write the MP3 (default: "output.mp3")
      voice_override: Force a specific edge_tts voice name

    Returns:
      output_path string (so calls can be chained)

    Quick test from terminal:
      python -c "
      import asyncio
      from tts.voice_engine import synthesize_to_file
      asyncio.run(synthesize_to_file('arre yaar kya haal hai!', 'test.mp3'))
      "
    """
    emotion = extract_dominant_emotion(text)
    rate, pitch = select_prosody(emotion)
    clean_text = prepare_text(text)
    voice = select_voice(clean_text, voice_override)

    logger.info("TTS saving to file", path=output_path, voice=voice)
    communicate = edge_tts.Communicate(text=clean_text, voice=voice,
                                       rate=rate, pitch=pitch)
    await communicate.save(output_path)
    logger.info("TTS file saved", path=output_path)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Local desktop playback
# Play audio directly on PC speakers. Used when running JARVIS on desktop.
# ══════════════════════════════════════════════════════════════════════════════

async def play_audio_stream(audio_gen: AsyncIterator[bytes]) -> None:
    """
    Collect streamed MP3 bytes and play them through local PC speakers.

    Called when JARVIS runs on desktop and should speak out loud instead
    of returning audio bytes to a client over HTTP.

    Flow:
      1. Collect all MP3 chunks from the async generator into memory
      2. Pipe them through ffmpeg to decode MP3 → WAV (no temp file)
      3. Play the WAV using sounddevice (blocking until audio finishes)

    Requires:
      pip install sounddevice scipy
      ffmpeg installed and in PATH (winget install ffmpeg on Windows)
    """
    import sounddevice as sd
    from scipy.io import wavfile

    # Collect full MP3 into memory first
    buffer = io.BytesIO()
    async for chunk in audio_gen:
        buffer.write(chunk)
    buffer.seek(0)

    try:
        # ffmpeg: MP3 in via stdin, WAV out via stdout — no disk I/O
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", "pipe:0",
            "-f", "wav", "-ar", "22050", "-ac", "1", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        wav_bytes, _ = await proc.communicate(input=buffer.read())
        sample_rate, data = wavfile.read(io.BytesIO(wav_bytes))
        sd.play(data, samplerate=sample_rate, blocking=True)
        logger.debug("Local playback complete")

    except Exception as e:
        logger.error("Local playback failed", error=str(e))


async def speak_local(text: str, voice_override: str | None = None) -> None:
    """
    Convenience one-liner: synthesize + play on local speakers.

    await speak_local("arre yaar, kya haal hai!")

    This is shorthand for:
      stream = synthesize_stream(text, voice_override)
      await play_audio_stream(stream)
    """
    await play_audio_stream(synthesize_stream(text, voice_override))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Utilities / dev tools
# ══════════════════════════════════════════════════════════════════════════════

async def list_available_voices(locale_filter: str | None = None) -> list[dict]:
    """
    Return all voices available in edge_tts, optionally filtered by locale.

    locale_filter examples: "hi-IN", "en-IN", "en-US", "te-IN", "ta-IN"
    Pass None to list all 400+ voices.

    Usage:
      voices = await list_available_voices("hi-IN")
      for v in voices:
          print(v["ShortName"], v["Gender"])
    """
    voices = await edge_tts.list_voices()
    if locale_filter:
        voices = [v for v in voices if v["Locale"].startswith(locale_filter)]
    return voices


async def test_all_jarvis_voices() -> None:
    """
    Audition all JARVIS voice profiles by generating one MP3 per profile.
    Run this once to choose which voice you like best, then set DEFAULT_VOICE.

    Output files: test_hinglish_male.mp3, test_hindi_male.mp3, etc.
    Play them with: ffplay test_hinglish_male.mp3
    """
    samples = {
        "hinglish_male":   "arre yaar, I am JARVIS, your personal AI assistant!",
        "hinglish_female": "Hello! Main aapki kaise help kar sakti hoon?",
        "hindi_male":      "नमस्ते! मैं जार्विस हूं, आपका AI सहायक।",
        "hindi_female":    "नमस्ते! आज आपकी क्या मदद करूं?",
        "english_jarvis":  "Good morning. All systems are operational.",
        "english_natural": "[excited] That is a brilliant idea! Let me help.",
    }
    for profile, text in samples.items():
        path = f"test_{profile}.mp3"
        await synthesize_to_file(text, path, voice_override=VOICES[profile])
        print(f"Saved: {path}")


# ── Quick standalone demo ─────────────────────────────────────────────────────
# Run from project root: python -m tts.voice_engine

if __name__ == "__main__":
    async def _demo():
        print("Testing edge_tts voice engine...\n")
        demos = [
            "arre yaar kya chal raha hai bhai, [excited] aaj ka plan kya hai?",
            "Good morning. I am JARVIS. All systems are operational.",
            "नमस्ते! आज मैं आपकी क्या मदद कर सकता हूं?",
            "[sigh] Main samajh sakta hoon, yeh thoda mushkil hai.",
        ]
        for i, text in enumerate(demos, 1):
            path = f"demo_{i}.mp3"
            print(f"[{i}] {text[:70]}")
            await synthesize_to_file(text, path)
            print(f"    Saved → {path}\n")
        print("Done! Play demo_*.mp3 to hear JARVIS.")

    asyncio.run(_demo())