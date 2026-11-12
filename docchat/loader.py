"""Step 1 of indexing: read files and turn them into Page objects.

PDFs don't store paragraphs; they store "draw these glyphs at these coordinates".
pypdf reconstructs the reading order, which works well for normal text but can
jumble tables and multi-column layouts. Scanned PDFs are images and contain no
text at all, so they need OCR (not included in this project).
"""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from .schema import Page

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def _clean(text: str) -> str:
    """Collapse runs of spaces/newlines/tabs into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def load_file(path: Path) -> list[Page]:
    """Read one .pdf or .txt file into a list of non-empty Pages."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        text = _clean(path.read_text(encoding="utf-8", errors="replace"))
        # A text file has no pages, so we treat the whole file as page 1.
        return [Page(source=path.name, page_number=1, text=text)] if text else []

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages: list[Page] = []
        # enumerate(..., start=1) gives human-style page numbers: 1, 2, 3...
        for number, pdf_page in enumerate(reader.pages, start=1):
            text = _clean(pdf_page.extract_text() or "")
            if text:
                pages.append(Page(source=path.name, page_number=number, text=text))
        if not pages:
            print(
                f"Warning: no text extracted from {path.name}. "
                "It is probably a scanned PDF and would need OCR."
            )
        return pages

    raise ValueError(
        f"Unsupported file type '{suffix}' for {path.name}. "
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def load_directory(path: Path) -> list[Page]:
    """Load every .pdf and .txt file directly inside a folder (not sub-folders)."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Folder not found: {path}")
    pages: list[Page] = []
    for file in sorted(path.iterdir()):
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            pages.extend(load_file(file))
    return pages


def load_paths(paths: list[Path]) -> list[Page]:
    """Load a mix of files and folders, e.g. ["data/", "notes.pdf"]."""
    pages: list[Page] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            pages.extend(load_directory(p))
        elif p.is_file():
            pages.extend(load_file(p))
        else:
            raise FileNotFoundError(f"Path not found: {p}")
    return pages
