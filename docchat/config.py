"""Settings, read from environment variables / the .env file.

All tunable knobs live here so the rest of the code never calls os.getenv directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Every setting DocChat uses. `frozen=True` means it can't be changed by accident."""

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_score: float
    storage_dir: Path

    def __post_init__(self) -> None:
        # Catch bad values early with a clear message instead of a confusing crash later.
        if self.chunk_size <= 0:
            raise ValueError(f"CHUNK_SIZE must be positive, got {self.chunk_size}.")
        if not 0 <= self.chunk_overlap <= self.chunk_size // 2:
            raise ValueError(
                f"CHUNK_OVERLAP must be between 0 and half of CHUNK_SIZE "
                f"(got overlap={self.chunk_overlap}, size={self.chunk_size})."
            )
        if self.top_k < 1:
            raise ValueError(f"TOP_K must be at least 1, got {self.top_k}.")
        if not -1.0 <= self.min_score <= 1.0:
            raise ValueError(f"MIN_SCORE must be between -1 and 1, got {self.min_score}.")


def get_settings() -> Settings:
    """Load `.env` (if present) and build a Settings object with sensible defaults."""
    load_dotenv()
    return Settings(
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        top_k=int(os.getenv("TOP_K", "4")),
        min_score=float(os.getenv("MIN_SCORE", "0.25")),
        storage_dir=Path(os.getenv("STORAGE_DIR", "storage")),
    )
