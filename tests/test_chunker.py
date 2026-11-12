import pytest

from docchat.chunker import chunk_pages, chunk_text
from docchat.schema import Page

LONG_TEXT = " ".join(
    f"This is sentence number {i} about hostel rules and mess timings." for i in range(60)
)


def test_short_text_gives_one_chunk():
    assert chunk_text("Hello world.", chunk_size=100, overlap=10) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ", chunk_size=100, overlap=10) == []


def test_long_text_gives_multiple_chunks_within_size():
    chunks = chunk_text(LONG_TEXT, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(0 < len(c) <= 200 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk_text(LONG_TEXT, chunk_size=200, overlap=40)
    for first, second in zip(chunks, chunks[1:]):
        # The start of the next chunk should already appear at the end of the previous one.
        assert second[:15] in first


def test_all_text_is_covered():
    chunks = chunk_text(LONG_TEXT, chunk_size=200, overlap=40)
    assert chunks[0].startswith("This is sentence number 0")
    assert chunks[-1].endswith("sentence number 59 about hostel rules and mess timings.")


def test_text_without_spaces_still_terminates():
    chunks = chunk_text("x" * 1000, chunk_size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) >= 10


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text(LONG_TEXT, chunk_size=100, overlap=100)


def test_chunk_pages_ids_and_page_numbers():
    pages = [
        Page(source="a.pdf", page_number=1, text="Short page one."),
        Page(source="a.pdf", page_number=2, text=LONG_TEXT),
    ]
    chunks = chunk_pages(pages, chunk_size=200, overlap=40)
    assert chunks[0].id == "a.pdf:p1:c0"
    assert chunks[0].page_number == 1
    assert chunks[1].id == "a.pdf:p2:c0"
    assert chunks[2].id == "a.pdf:p2:c1"
    assert all(c.source == "a.pdf" for c in chunks)
