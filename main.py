"""
main.py
Entry point — run with: python main.py
"""
import uvicorn
from config import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,               # False in prod — True only for dev
        log_level=settings.log_level.lower(),
        workers=1,                  # Single worker — Whisper model is shared singleton
        loop="asyncio",
    )
