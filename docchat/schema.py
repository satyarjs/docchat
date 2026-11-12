"""Plain data objects that flow through the RAG pipeline.

Keeping these as simple dataclasses (instead of dicts) gives you autocomplete,
type checking and one obvious place to see what each piece of data contains.

Flow:  files -> Page -> Chunk -> (vector) -> SearchResult -> Answer
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Page:
    """One page of text from a source file.

    We keep the page number because it travels all the way to the final answer,
    which is what makes citations like "page 3" possible.
    """

    source: str  # file name, e.g. "handbook.pdf"
    page_number: int  # 1-based, the way a human counts pages
    text: str


@dataclass
class Chunk:
    """A small piece of a page that gets its own embedding vector.

    `id` looks like "handbook.pdf:p3:c0" -> file, page 3, first chunk on that page.
    """

    id: str
    text: str
    source: str
    page_number: int

    def to_dict(self) -> dict:
        """Convert to a plain dict so it can be saved as JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Rebuild a Chunk from the dict produced by `to_dict`."""
        return cls(
            id=data["id"],
            text=data["text"],
            source=data["source"],
            page_number=int(data["page_number"]),
        )


@dataclass
class SearchResult:
    """A chunk plus how similar it is to the question (cosine similarity, -1..1)."""

    chunk: Chunk
    score: float


@dataclass
class Answer:
    """What `DocChat.ask` returns.

    `sources[i]` is the chunk the LLM refers to as `[i + 1]` in its text.
    `used_llm` is False when no chunk was relevant enough and we skipped the LLM.
    """

    text: str
    sources: list[SearchResult] = field(default_factory=list)
    used_llm: bool = False


@dataclass
class IngestReport:
    """Summary of what `DocChat.ingest` indexed."""

    files: int
    pages: int
    chunks: int
