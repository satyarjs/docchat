from pathlib import Path

import pytest

from docchat.pipeline import DocChat
from docchat.prompts import NO_RESULTS_MESSAGE


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "hostel.txt").write_text(
        "The curfew is 10:30 PM on weekdays and 11:30 PM on Saturday. "
        "Breakfast is served from 7:30 AM to 9:00 AM in the mess.",
        encoding="utf-8",
    )
    (folder / "library.txt").write_text(
        "The library lends books for fourteen days. Late returns cost five rupees per day.",
        encoding="utf-8",
    )
    return folder


@pytest.fixture
def app(settings, fake_embedder, fake_llm) -> DocChat:
    return DocChat(settings, embedder=fake_embedder, llm=fake_llm)


def test_ingest_reports_counts_and_writes_files(app, docs, settings):
    report = app.ingest([docs])
    assert report.files == 2
    assert report.pages == 2
    assert report.chunks >= 2
    for name in ("vectors.npy", "chunks.json", "meta.json"):
        assert (settings.storage_dir / name).is_file()


def test_matching_question_reaches_llm_with_context(app, docs, fake_llm):
    app.ingest([docs])
    answer = app.ask("What time is the curfew on Saturday?")

    assert answer.used_llm is True
    assert answer.text == "FAKE ANSWER [1]"
    assert len(fake_llm.calls) == 1

    user_message = fake_llm.calls[0][1]["content"]
    assert "[1] (hostel.txt, page 1)" in user_message
    assert "11:30 PM on Saturday" in user_message
    assert user_message.endswith("Question: What time is the curfew on Saturday?")
    assert answer.sources[0].chunk.source == "hostel.txt"


def test_unrelated_question_skips_llm(app, docs, fake_llm):
    app.ingest([docs])
    # No words in common with the documents, so FakeEmbedder scores it near 0.
    answer = app.ask("Which football club signed Ronaldo?")
    assert answer.text == NO_RESULTS_MESSAGE
    assert answer.used_llm is False
    assert answer.sources == []
    assert fake_llm.calls == []  # the LLM was never called


def test_ask_before_ingest_raises(app):
    with pytest.raises(RuntimeError, match="No documents indexed"):
        app.ask("anything")


def test_empty_question_raises(app, docs):
    app.ingest([docs])
    with pytest.raises(ValueError):
        app.ask("   ")


def test_fresh_instance_can_load_saved_index(app, docs, settings, fake_embedder, fake_llm):
    app.ingest([docs])
    fresh = DocChat(settings, embedder=fake_embedder, llm=fake_llm)
    assert fresh.is_ready is False
    assert fresh.load_index() is True
    assert fresh.is_ready is True
    assert fresh.ask("When is breakfast served?").used_llm is True


def test_ingest_with_no_text_raises(app, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "blank.txt").write_text("   ", encoding="utf-8")
    with pytest.raises(ValueError, match="No text"):
        app.ingest([empty])
