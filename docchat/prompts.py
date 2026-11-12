"""Step 3 of querying: build the exact text the LLM will read.

Anatomy of the prompt:
  * system message  -> the rules (answer only from context, cite, admit "not found")
  * numbered context -> short labels [1], [2] the model can cite and we can map back
  * question last    -> models pay strong attention to the end of the prompt
"""

from __future__ import annotations

from .schema import SearchResult

SYSTEM_PROMPT = """You are DocChat, an assistant that answers questions using ONLY the provided context from the user's documents.
Rules:
1. Use only the information in the context. Do not use outside knowledge.
2. Cite every factual claim with the matching source number in square brackets, like [1] or [2].
3. If the context does not contain the answer, say: "I couldn't find that in your documents."
4. Be concise. Prefer short paragraphs or bullet points."""

NO_RESULTS_MESSAGE = "I couldn't find anything about that in your documents."


def build_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks as numbered blocks: "[1] (file, page N)\\n<text>"."""
    blocks = []
    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        blocks.append(f"[{i}] ({chunk.source}, page {chunk.page_number})\n{chunk.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, results: list[SearchResult]) -> list[dict]:
    """Build the chat messages list in the OpenAI format."""
    user_content = f"Context:\n{build_context(results)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
