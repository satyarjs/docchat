import numpy as np
import pytest

from docchat.schema import Chunk
from docchat.vector_store import VectorStore


def make_chunk(i: int) -> Chunk:
    return Chunk(id=f"f.txt:p1:c{i}", text=f"chunk {i}", source="f.txt", page_number=1)


@pytest.fixture
def store() -> VectorStore:
    # Three unit vectors in 2D, like the worked example in the README.
    s = VectorStore(dim=2)
    vectors = np.array([[0.8, 0.6], [-0.6, 0.8], [1.0, 0.0]], dtype=np.float32)
    s.add([make_chunk(0), make_chunk(1), make_chunk(2)], vectors)
    return s


def test_closest_ranks_first_and_scores_descend(store):
    results = store.search(np.array([0.6, 0.8]), k=3)
    assert results[0].chunk.id == "f.txt:p1:c0"  # 0.96, the README example
    assert results[0].score == pytest.approx(0.96, abs=1e-5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_k_larger_than_store_returns_all(store):
    assert len(store.search(np.array([1.0, 0.0]), k=10)) == 3


def test_empty_store_returns_nothing():
    assert VectorStore(dim=2).search(np.array([1.0, 0.0]), k=3) == []


def test_wrong_dimension_raises():
    s = VectorStore(dim=3)
    with pytest.raises(ValueError):
        s.add([make_chunk(0)], np.ones((1, 2)))


def test_mismatched_counts_raise():
    s = VectorStore(dim=2)
    with pytest.raises(ValueError):
        s.add([make_chunk(0), make_chunk(1)], np.ones((1, 2)))


def test_save_load_round_trip(store, tmp_path):
    store.save(tmp_path, embedding_model="model-a")
    assert VectorStore.exists(tmp_path)
    loaded = VectorStore.load(tmp_path, expected_model="model-a")
    assert len(loaded) == 3
    np.testing.assert_allclose(loaded.vectors, store.vectors)
    assert [c.id for c in loaded.chunks] == [c.id for c in store.chunks]


def test_load_with_different_model_raises(store, tmp_path):
    store.save(tmp_path, embedding_model="model-a")
    with pytest.raises(ValueError, match="different embedding model"):
        VectorStore.load(tmp_path, expected_model="model-b")


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "nothing", expected_model="model-a")
