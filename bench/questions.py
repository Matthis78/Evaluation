"""Build the labelled question set.

This is the part of the benchmark that decides whether the numbers mean
anything. Questions are generated *from* a passage, which biases retrieval
towards that passage — the three difficulty levels exist to control how much
of that bias survives:

    easy    the user borrows the page's own vocabulary (high lexical overlap)
    medium  same need, paraphrased, distinctive terms swapped for synonyms
    hard    the user describes a symptom or a goal and never names the feature

A model that only does lexical matching looks fine on `easy` and collapses on
`hard`. That spread is the signal; the headline number alone hides it.

Every generated question then goes through a separate validation call that
throws out anything not answerable from its own passage, or that refers back
to "the text". Roughly 10-20% gets dropped, and keeping them would quietly
put a ceiling on every pipeline in the matrix.
"""

from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from .corpus import Chunk
from .openrouter import OpenRouter

DIFFICULTIES = ("easy", "medium", "hard")

GEN_SYSTEM = """You write evaluation questions for a document retrieval benchmark.

You are given one passage from a help centre. Write exactly three questions \
that this passage answers, one per difficulty level.

easy    - What a user types when they have skimmed this page. Reuses the \
passage's own distinctive words and product terms.
medium  - The same information need, paraphrased. Keep the meaning, swap the \
distinctive nouns and verbs for natural synonyms. No verbatim phrases of 4+ \
words from the passage.
hard    - A user who does not know the feature's name. They describe a \
situation, a symptom or a goal in their own words. The passage still answers \
it, but almost no vocabulary is shared.

Hard rules for all three:
- Self-contained. Never write "this document", "the passage", "the above", \
"according to the text".
- Answerable from this passage alone, with no outside knowledge.
- Specific enough that this passage is the right answer, not a generic \
question that fifty other pages would also answer.
- One sentence each. No preamble, no numbering.

Reply with JSON only: {"easy": "...", "medium": "...", "hard": "..."}"""

VALIDATE_SYSTEM = """You audit questions for a retrieval benchmark.

Given a PASSAGE and a QUESTION, decide:
- answerable: is the question fully answered by this passage alone?
- self_contained: does the question stand on its own, without referring to \
"the text"/"the document"/"the above"?
- specific: would this passage be a correct answer, rather than the question \
being so generic that any page would do?

Reply with JSON only: {"answerable": true/false, "self_contained": true/false, \
"specific": true/false, "reason": "<10 words>"}"""


@dataclass
class Question:
    id: str
    text: str
    difficulty: str
    gold_chunk_id: str
    gold_doc_id: str


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if not brace:
            return {}
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            return {}


def sample_chunks(
    chunks: list[Chunk], n: int, *, seed: int = 20261027, min_chars: int = 500
) -> list[Chunk]:
    """One chunk per document, so no document is over-represented."""
    rng = random.Random(seed)
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        if c.n_chars >= min_chars:
            by_doc.setdefault(c.doc_id, []).append(c)
    picked = [rng.choice(v) for v in by_doc.values()]
    rng.shuffle(picked)
    return picked[:n]


def generate(
    client: OpenRouter, chunks: list[Chunk], *, model: str, concurrency: int = 8
) -> list[Question]:
    def one(chunk: Chunk) -> list[Question]:
        raw = client.chat(
            model,
            [
                {"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": f"PASSAGE:\n{chunk.embed_text()}"},
            ],
            temperature=0.3,
            json_mode=True,
            cache_salt=chunk.id,
        )
        data = _parse_json(raw)
        out = []
        for level in DIFFICULTIES:
            text = (data.get(level) or "").strip()
            if len(text) < 12:
                continue
            out.append(
                Question(
                    id=f"{chunk.id}:{level}",
                    text=text,
                    difficulty=level,
                    gold_chunk_id=chunk.id,
                    gold_doc_id=chunk.doc_id,
                )
            )
        return out

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        batches = list(pool.map(one, chunks))
    return [q for batch in batches for q in batch]


def validate(
    client: OpenRouter,
    questions: list[Question],
    chunks: list[Chunk],
    *,
    model: str,
    concurrency: int = 8,
) -> tuple[list[Question], list[dict]]:
    by_id = {c.id: c for c in chunks}

    def one(q: Question) -> tuple[Question, dict]:
        chunk = by_id[q.gold_chunk_id]
        raw = client.chat(
            model,
            [
                {"role": "system", "content": VALIDATE_SYSTEM},
                {
                    "role": "user",
                    "content": f"PASSAGE:\n{chunk.embed_text()}\n\nQUESTION:\n{q.text}",
                },
            ],
            temperature=0.0,
            json_mode=True,
            cache_salt=q.id,
        )
        return q, _parse_json(raw)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        judged = list(pool.map(one, questions))

    kept, rejected = [], []
    for q, verdict in judged:
        ok = (
            bool(verdict.get("answerable"))
            and bool(verdict.get("self_contained"))
            and bool(verdict.get("specific"))
        )
        if ok:
            kept.append(q)
        else:
            rejected.append({**asdict(q), "verdict": verdict})
    return kept, rejected


def save(questions: list[Question], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")


def load(path: str) -> list[Question]:
    with open(path, encoding="utf-8") as fh:
        return [Question(**json.loads(line)) for line in fh if line.strip()]
