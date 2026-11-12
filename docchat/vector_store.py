"""Step 4 of indexing / step 2 of querying: store vectors and find the closest ones.

The whole "vector database" is:
  * self.vectors -> a NumPy matrix, shape (num_chunks, dim)
  * self.chunks  -> a list where chunks[i] describes row i of the matrix

Cosine similarity: cos(a, b) = (a . b) / (|a| * |b|)
Our vectors all have length 1, so cosine similarity is just the dot product a . b.
  1.0 -> same meaning, 0.0 -> unrelated.

Searching = one matrix-vector multiply that scores every chunk at once
("brute-force" search). 10,000 chunks x 384 dims takes a few milliseconds.
Real vector databases (FAISS, Chroma, pgvector) only pay off at much larger scale.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .schema import Chunk, SearchResult

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"


class VectorStore:
    """An in-memory list of chunks plus a matching matrix of their vectors."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        # Start with an empty matrix that has 0 rows and `dim` columns.
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Append chunks and their vectors (row i of `vectors` belongs to chunks[i])."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors of shape (n, {self.dim}), got {vectors.shape}."
            )
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"Got {len(chunks)} chunks but {vectors.shape[0]} vectors; "
                "they must match one-to-one."
            )
        # vstack glues the new rows under the existing ones.
        self.vectors = np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        """Return the k chunks most similar to `query_vector`, best first."""
        if len(self) == 0:
            return []
        query_vector = np.asarray(query_vector, dtype=np.float32)

        # (num_chunks, dim) @ (dim,) -> (num_chunks,): one similarity score per chunk.
        scores = self.vectors @ query_vector

        k = min(k, len(self))
        # argsort sorts ascending, so we sort the NEGATED scores to get highest first,
        # then keep the first k indices.
        top_indices = np.argsort(-scores)[:k]

        return [
            SearchResult(chunk=self.chunks[i], score=float(scores[i]))
            for i in top_indices
        ]

    def save(self, directory: Path, embedding_model: str) -> None:
        """Write the index to disk as three files (see README section 5.4)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        np.save(directory / VECTORS_FILE, self.vectors)
        (directory / CHUNKS_FILE).write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        meta = {
            "embedding_model": embedding_model,
            "dim": self.dim,
            "count": len(self),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def exists(directory: Path) -> bool:
        """True if all three index files are present."""
        directory = Path(directory)
        return all((directory / name).is_file() for name in (VECTORS_FILE, CHUNKS_FILE, META_FILE))

    @classmethod
    def load(cls, directory: Path, expected_model: str) -> "VectorStore":
        """Load an index saved by `save`, refusing one built with another model."""
        directory = Path(directory)
        if not cls.exists(directory):
            raise FileNotFoundError(
                f"No index found in '{directory}'. Run ingest first."
            )

        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        if meta["embedding_model"] != expected_model:
            raise ValueError(
                f"Index was built with a different embedding model "
                f"('{meta['embedding_model']}', but EMBEDDING_MODEL is "
                f"'{expected_model}'). Re-run ingest."
            )

        vectors = np.load(directory / VECTORS_FILE)
        raw_chunks = json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(c) for c in raw_chunks]
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                "Index files are out of sync (vector rows != chunks). Re-run ingest."
            )

        store = cls(dim=int(meta["dim"]))
        store.add(chunks, vectors)
        return store
