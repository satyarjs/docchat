"""The glue: DocChat ties loading, chunking, embedding, search and the LLM together.

Both front-ends (cli.py and app.py) only talk to this class.

`embedder` and `llm` can be passed in from outside ("dependency injection").
The real app passes nothing and gets the real ones; the tests pass fakes, so
they run instantly, offline, with no API key.
"""

from __future__ import annotations

from pathlib import Path

from .chunker import chunk_pages
from .config import Settings
from .embedder import Embedder
from .llm import LLMClient
from .loader import load_paths
from .prompts import NO_RESULTS_MESSAGE, build_messages
from .schema import Answer, IngestReport, SearchResult
from .vector_store import VectorStore


class DocChat:
    """Ask questions about your documents."""

    def __init__(self, settings: Settings, embedder=None, llm=None) -> None:
        self.settings = settings
        self.embedder = embedder or Embedder(settings.embedding_model)
        self.llm = llm or LLMClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
        self.store: VectorStore | None = None  # filled by ingest() or load_index()

    # ---------- Phase 1: indexing ----------

    def ingest(self, paths: list[Path]) -> IngestReport:
        """Load -> chunk -> embed -> store -> save. Rebuilds the index from scratch."""
        pages = load_paths(paths)
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError(
                "No text could be extracted from the given files. "
                "Are they scanned PDFs or empty?"
            )

        vectors = self.embedder.embed([c.text for c in chunks])  # shape (n, dim)

        store = VectorStore(dim=vectors.shape[1])
        store.add(chunks, vectors)
        store.save(self.settings.storage_dir, self.embedder.model_name)
        self.store = store

        return IngestReport(
            files=len({p.source for p in pages}),
            pages=len(pages),
            chunks=len(chunks),
        )

    def load_index(self) -> bool:
        """Load a saved index from disk. Returns False if there isn't one yet."""
        if not VectorStore.exists(self.settings.storage_dir):
            return False
        self.store = VectorStore.load(self.settings.storage_dir, self.embedder.model_name)
        return True

    @property
    def is_ready(self) -> bool:
        """True once there is at least one chunk to search."""
        return self.store is not None and len(self.store) > 0

    # ---------- Phase 2: querying ----------

    def retrieve(self, question: str) -> list[SearchResult]:
        """Find the top-k chunks for a question, dropping ones below MIN_SCORE."""
        if self.store is None:
            return []
        query_vector = self.embedder.embed_one(question)  # shape (dim,)
        results = self.store.search(query_vector, self.settings.top_k)
        return [r for r in results if r.score >= self.settings.min_score]

    def ask(self, question: str) -> Answer:
        """Answer a question using only the indexed documents.

        In plain English:
          1. Make sure there is an index to search.
          2. Turn the question into a vector with the same embedding model.
          3. Score it against every chunk and keep the best few above MIN_SCORE.
          4. If nothing is relevant, say so and DON'T call the LLM
             (prevents made-up answers and saves an API call).
          5. Otherwise paste the chunks into a prompt and let the LLM write the
             answer, citing [1], [2]... which map to answer.sources.
        """
        if not self.is_ready:
            raise RuntimeError("No documents indexed yet. Run ingest first.")

        question = question.strip()
        if not question:
            raise ValueError("Please type a question.")

        results = self.retrieve(question)
        if not results:
            return Answer(text=NO_RESULTS_MESSAGE, sources=[], used_llm=False)

        messages = build_messages(question, results)
        text = self.llm.chat(messages)
        return Answer(text=text, sources=results, used_llm=True)
