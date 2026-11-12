from pathlib import Path

import pytest

from docchat.config import Settings


def make_settings(chunk_size: int = 800, chunk_overlap: int = 150) -> Settings:
    return Settings(
        llm_base_url="unused",
        llm_api_key="",
        llm_model="unused",
        embedding_model="unused",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=4,
        min_score=0.25,
        storage_dir=Path("storage"),
    )


def test_default_overlap_is_allowed():
    assert make_settings().chunk_overlap == 150


def test_overlap_of_exactly_half_is_allowed():
    assert make_settings(chunk_size=800, chunk_overlap=400).chunk_overlap == 400


def test_overlap_above_half_is_rejected():
    with pytest.raises(ValueError, match="half of CHUNK_SIZE"):
        make_settings(chunk_size=800, chunk_overlap=401)


def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError):
        make_settings(chunk_overlap=-1)
