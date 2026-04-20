# tts/__init__.py
from .voice_engine import synthesize_stream, detect_tts_backend

__all__ = ["synthesize_stream", "detect_tts_backend"]
