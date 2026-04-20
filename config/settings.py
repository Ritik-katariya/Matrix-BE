"""
config/settings.py
Central settings — read once at startup, used everywhere.
All values come from .env (never hardcoded).
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    primary_language: str = "hinglish"       # english | hindi | hinglish
    mobile_stream_mode: str = "lan"          # lan | ngrok | disabled

    # ── LLMs ─────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    nvidia_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Model names
    nvidia_model: str = "nvidia/llama-3.1-nemotron-70b-instruct"
    openai_model: str = "gpt-4o"
    ollama_model: str = "llama3.2:latest"

    # ── STT ──────────────────────────────────────────────────────────────────
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cpu"             # cpu (5600H is fast enough)
    whisper_compute_type: str = "int8"      # int8 → fastest on CPU
    deepgram_api_key: str = ""

    # ── TTS ──────────────────────────────────────────────────────────────────
    fish_audio_api_key: str = ""
    fish_audio_reference_id: str = ""
    gnani_api_key: str = ""
    gnani_language: str = "hi-IN"

    # ── Audio ─────────────────────────────────────────────────────────────────
    vad_aggressiveness: int = 2             # 0-3, 2 = balanced
    sample_rate: int = 16000
    chunk_duration_ms: int = 30             # VAD frame size

    # ── LLM Routing Thresholds ────────────────────────────────────────────────
    # Tasks tagged "critical" always go to OpenAI.
    # Everything else → NVIDIA NIM → Ollama fallback.
    nvidia_timeout_sec: float = 8.0
    ollama_timeout_sec: float = 30.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — parse .env once."""
    return Settings()
