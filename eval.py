"""Evaluation script: proves DocChat works with a pass/fail scorecard.

Run:
  python eval.py              # retrieval only: no API key needed
  python eval.py --with-llm   # also checks the LLM's final answers (needs .env)

What it measures:
  * Retrieval hit rate: for each question with a known answer, is the chunk
    containing that answer among the top-k retrieved chunks?
  * Refusals: for questions the documents can't answer, does DocChat refuse
    (no chunk above MIN_SCORE, so the LLM is never called)?
  * Answer check (--with-llm): does the final answer contain the expected
    fact and a [n] citation?

It indexes data/sample.txt into a temporary folder, so your real index in
storage/ is never touched.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

from docchat import DocChat, get_settings
from docchat.llm import LLMError

SAMPLE = Path("data/sample.txt")

# (question, a fact that must appear in the retrieved text / answer)
# Several questions deliberately use different words from the handbook,
# e.g. "friend" vs "guests", to prove the search works on meaning, not keywords.
ANSWERABLE = [
    ("What time is the curfew on Saturday?", "11:30 PM"),
    ("Can I bring a friend over?", "4:00 PM"),
    ("When is breakfast served?", "7:30 AM"),
    ("How many gadgets can I connect to the internet?", "2 devices"),
    ("Where do I go if I forget my Wi-Fi password?", "G-12"),
    ("Which days can I get my clothes washed?", "Tuesdays and Fridays"),
    ("What is the penalty for paying hostel fees late?", "₹50"),
    ("How early must I apply before going home?", "24 hours"),
]

# Questions the handbook cannot answer: DocChat should refuse.
UNANSWERABLE = [
    "Who won the 2011 Cricket World Cup?",
    "What is the capital of France?",
    "Explain quantum entanglement.",
]


def _norm(text: str) -> str:
    """Lowercase and drop spaces/dots so '11:30 PM' matches '11:30 p.m.'."""
    return text.lower().replace(" ", "").replace(".", "")


def run_eval(app: DocChat, with_llm: bool) -> bool:
    """Print the scorecard. Returns True if every check passed."""
    width = 50
    print(f"\n{'#':<3}{'Question':<{width}}{'Expected':<22}{'Top':>6}  Retrieval" + ("  Answer" if with_llm else ""))
    print("-" * (width + 45))

    hits = 0
    answers_ok = 0
    for i, (question, expected) in enumerate(ANSWERABLE, start=1):
        results = app.retrieve(question)
        top = results[0].score if results else 0.0
        hit = any(_norm(expected) in _norm(r.chunk.text) for r in results)
        hits += hit
        line = f"{i:<3}{question[:width - 2]:<{width}}{expected:<22}{top:>6.2f}  {'PASS' if hit else 'FAIL':<9}"

        if with_llm:
            answer = app.ask(question)
            ok = _norm(expected) in _norm(answer.text) and "[" in answer.text
            answers_ok += ok
            line += f"  {'PASS' if ok else 'FAIL'}"
        print(line)

    print("\nOut-of-scope questions (should be refused without calling the LLM):")
    refusals = 0
    for question in UNANSWERABLE:
        results = app.retrieve(question)
        top = max((r.score for r in results), default=0.0)
        refused = not results
        refusals += refused
        print(f"   {question:<{width}}{'refused' if refused else f'NOT refused (score {top:.2f})'}")

    n, m = len(ANSWERABLE), len(UNANSWERABLE)
    print("\n" + "=" * 40)
    print(f"Retrieval hit rate @k={app.settings.top_k}: {hits}/{n} ({hits / n:.0%})")
    if with_llm:
        print(f"Answers with correct fact + citation: {answers_ok}/{n} ({answers_ok / n:.0%})")
    print(f"Correct refusals: {refusals}/{m}")
    print("=" * 40)

    all_passed = hits == n and refusals == m and (not with_llm or answers_ok == n)
    if not all_passed:
        print("Tip: failures here are useful. Try `python cli.py ask \"...\" --show-context`")
        print("on a failing question, then tune MIN_SCORE / CHUNK_SIZE / TOP_K in .env.")
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DocChat on the sample handbook.")
    parser.add_argument("--with-llm", action="store_true", help="Also check LLM answers (needs .env)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        # Same settings as the app, but the index goes to a throwaway folder.
        settings = dataclasses.replace(get_settings(), storage_dir=Path(tmp))
        app = DocChat(settings)
        report = app.ingest([SAMPLE])
        print(f"Indexed {SAMPLE} -> {report.chunks} chunks "
              f"(chunk_size={settings.chunk_size}, top_k={settings.top_k}, min_score={settings.min_score})")
        try:
            passed = run_eval(app, args.with_llm)
        except LLMError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
