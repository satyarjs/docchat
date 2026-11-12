"""Step 3 of indexing (and step 1 of querying): turn text into vectors.

An embedding model maps text to a fixed-length list of numbers such that texts
with similar MEANING point in similar directions. We use all-MiniLM-L6-v2:
384 numbers per text, ~90 MB, runs fine on a laptop CPU, free and local.

Golden rule: questions and chunks must be embedded with the SAME model,
otherwise their vectors live in different "spaces" and can't be compared.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    """Wraps a sentence-transformers model.

    The model is loaded lazily (on first use), so importing this module and
    creating an Embedder is instant. The first embed() call downloads the model
    (~90 MB, once) into your Hugging Face cache.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None  # loaded on first use

    def _load(self):
        if self._model is None:
            # Imported here so that tests (which use a fake embedder) never need torch.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        """Length of each vector (384 for all-MiniLM-L6-v2)."""
        model = self._load()
        # The method was renamed in newer sentence-transformers versions.
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        return int(getter())

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed many texts at once. Returns shape (len(texts), dim), dtype float32.

        normalize_embeddings=True scales every vector to length 1. That makes
        cosine similarity equal to a plain dot product (see vector_store.py).
        Note: the model reads at most 256 tokens per text; longer text is cut
        off silently, which is why chunks are kept to ~800 characters.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._load().encode(
            texts,
            batch_size=32,  # how many texts go through the network at once
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns shape (dim,)."""
        return self.embed([text])[0]
