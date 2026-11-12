"""Shared test helpers: fake models so tests run instantly, offline, with no API key."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

# Make `import docchat` work when running `pytest` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docchat.config import Settings  # noqa: E402


class FakeEmbedder:
    """Turns text into a vector by hashing each word into one of `dim` buckets.

    Texts that share words get similar vectors, which is enough to test
    retrieval logic. Same interface as the real Embedder.
    """

    model_name = "fake-embedder"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def embed_one(self, text: str) -> np.ndarray:
        return self._vector(text)


class FakeLLM:
    """Records every prompt it receives and returns a fixed answer."""

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return "FAKE ANSWER [1]"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_base_url="http://unused",
        llm_api_key="unused",
        llm_model="unused",
        embedding_model="fake-embedder",
        chunk_size=200,
        chunk_overlap=40,
        top_k=3,
        min_score=0.1,
        storage_dir=tmp_path / "storage",
    )


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
