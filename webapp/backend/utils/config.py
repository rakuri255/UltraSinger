"""Configuration settings for the UltraSinger web application"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Server configuration
    backend_port: int = 8000
    backend_host: str = "0.0.0.0"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # File storage
    upload_dir: Path = Path("/tmp/ultrasinger/uploads")
    output_dir: Path = Path("/tmp/ultrasinger/outputs")
    max_file_size: int = 100 * 1024 * 1024  # 100MB

    # Job management
    job_retention_hours: int = 24
    max_concurrent_jobs: int = 2

    # UltraSinger configuration
    ultrasinger_src_path: Path = Path(__file__).parent.parent.parent.parent / "src"
    whisper_model: str = "medium"  # tiny|small|medium|large-v2|large-v3
    crepe_model: str = "full"  # tiny|small|medium|large|full
    force_cpu: bool = False

    # Supported languages
    supported_languages: dict[str, str] = {
        "it": "Italian",
        "en": "English",
        "pl": "Polish"
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
