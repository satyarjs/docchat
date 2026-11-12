"""Step 2 of indexing: split page text into small, overlapping chunks.

Why chunk?
  * The embedding model reads at most ~256 tokens and silently ignores the rest.
    800 characters of English is roughly 150-200 tokens, which fits.
  * One vector for a whole document is a blurry average; one per paragraph is sharp.
  * We only want to send the LLM a few relevant paragraphs, not the whole book.

Why overlap?
  If an answer sits across a cut, overlap makes sure it appears whole in at
  least one chunk.

Picture (size=800, overlap=150, so each chunk starts 650 chars after the last):
  chunk 0: |---------- 800 ----------|
  chunk 1:                 |---------- 800 ----------|
                           |<- 150 ->| shared text
"""

from __future__ import annotations

from .schema import Chunk, Page

SENTENCE_ENDS = (". ", "? ", "! ")


def _best_cut(text: str, start: int, end: int, min_length: int) -> int:
    """Move `end` back to a sentence end or a space so we don't cut words in half.

    Only moves the cut if the chunk stays at least `min_length` characters long.
    """
    window = text[start:end]

    # Prefer the last sentence boundary inside the window.
    sentence_cut = max(window.rfind(mark) for mark in SENTENCE_ENDS)
    if sentence_cut != -1 and sentence_cut + 1 >= min_length:
        return start + sentence_cut + 1  # +1 keeps the "." inside this chunk

    # Otherwise fall back to the last space.
    space_cut = window.rfind(" ")
    if space_cut != -1 and space_cut >= min_length:
        return start + space_cut

    # No good boundary: cut exactly at `end`.
    return end


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of up to `chunk_size` characters with `overlap` shared."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0 <= overlap < chunk_size:
        raise ValueError(
            f"overlap must be >= 0 and smaller than chunk_size "
            f"(got overlap={overlap}, chunk_size={chunk_size})."
        )

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _best_cut(text, start, end, min_length=chunk_size // 2)

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break  # reached the end of the text

        # Next chunk starts `overlap` characters before this one ended...
        next_start = end - overlap
        # ...but always move forward, otherwise we could loop forever.
        start = max(next_start, start + 1)

    return chunks


def chunk_pages(pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk each page separately so every chunk belongs to exactly one page.

    This keeps citations accurate. The trade-off: a sentence spanning a page
    break gets split.
    """
    chunks: list[Chunk] = []
    for page in pages:
        for i, piece in enumerate(chunk_text(page.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{page.source}:p{page.page_number}:c{i}",
                    text=piece,
                    source=page.source,
                    page_number=page.page_number,
                )
            )
    return chunks
