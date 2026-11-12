"""Terminal interface for DocChat.

Usage:
  python cli.py ingest data/
  python cli.py ask "What time is the curfew on Saturday?"
  python cli.py ask "Can I bring a friend over?" --show-context
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docchat import DocChat, get_settings
from docchat.llm import LLMError


def cmd_ingest(app: DocChat, paths: list[str]) -> None:
    report = app.ingest([Path(p) for p in paths])
    print(
        f"Indexed {report.files} file(s), {report.pages} page(s), "
        f"{report.chunks} chunk(s) -> {app.settings.storage_dir}/"
    )


def cmd_ask(app: DocChat, question: str, show_context: bool) -> None:
    if not app.load_index():
        raise RuntimeError("No documents indexed yet. Run: python cli.py ingest data/")

    if show_context:
        # Show what retrieval found BEFORE the LLM sees it. Best way to learn/tune.
        results = app.retrieve(question)
        print(f"--- Retrieved {len(results)} chunk(s) (MIN_SCORE={app.settings.min_score}) ---")
        for i, r in enumerate(results, start=1):
            preview = r.chunk.text[:300] + ("..." if len(r.chunk.text) > 300 else "")
            print(f"[{i}] score={r.score:.3f}  {r.chunk.source} p{r.chunk.page_number}")
            print(f"    {preview}\n")
        print("--- Answer ---")

    answer = app.ask(question)
    print(answer.text)

    if answer.sources:
        print("\nSources:")
        for i, r in enumerate(answer.sources, start=1):
            print(f"  [{i}] {r.chunk.source}, page {r.chunk.page_number} (score {r.score:.2f})")
    elif not answer.used_llm:
        print("\n(No relevant passages found, so the LLM was not called.)")


def main() -> int:
    parser = argparse.ArgumentParser(description="DocChat: chat with your PDFs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index .pdf/.txt files or folders")
    p_ingest.add_argument("paths", nargs="+", help="Files and/or folders to index")

    p_ask = sub.add_parser("ask", help="Ask a question about your indexed documents")
    p_ask.add_argument("question", help="Your question, in quotes")
    p_ask.add_argument(
        "--show-context", action="store_true", help="Print retrieved chunks and scores"
    )

    args = parser.parse_args()

    try:
        app = DocChat(get_settings())
        if args.command == "ingest":
            cmd_ingest(app, args.paths)
        else:
            cmd_ask(app, args.question, args.show_context)
    except (LLMError, RuntimeError, ValueError, FileNotFoundError) as exc:
        # Friendly one-line errors instead of a scary traceback.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
