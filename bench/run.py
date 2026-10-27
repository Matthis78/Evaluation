"""Run the pipeline matrix and score it.

Shape of a run, for N questions and a corpus of M chunks:

    embed the corpus once per *embedder*        (M vectors, cached)
    embed the questions once per *embedder*     (N vectors, cached)
    cosine top-`candidate_k` per question       (one matrix product)
    optional rerank of those candidates         (N calls per reranker, cached)
    collapse chunks to documents, keeping order
    rank of the gold document -> recall@k

Embedding the corpus is the expensive step and it is shared by every pipeline
that uses the same embedder, so adding a reranker to the matrix costs almost
nothing.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import metrics
from .corpus import Chunk
from .openrouter import OpenRouter
from .questions import Question


@dataclass
class Pipeline:
    id: str
    label: str
    embedder: str
    family: str
    reranker: str | None = None
    dimensions: int | None = None
    input_type: bool = False          # send search_query / search_document
    production: bool = False

    @property
    def embed_key(self) -> tuple:
        return (self.embedder, self.dimensions, self.input_type)


def _progress(label: str, i: int, total: int) -> None:
    if not sys.stderr.isatty():          # keep logs readable when piped to a file
        if i == total:
            print(f"  {label:<34} {total}/{total}")
        return
    bar = int(28 * i / max(total, 1))
    sys.stderr.write(f"\r  {label:<34} [{'#' * bar}{'.' * (28 - bar)}] {i}/{total}")
    sys.stderr.flush()
    if i == total:
        sys.stderr.write("\n")


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def embed_everything(
    client: OpenRouter, pipelines: list[Pipeline], chunks: list[Chunk], questions: list[Question]
) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    """One (chunk_matrix, question_matrix) per distinct embedder setting."""
    out: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    keys = {p.embed_key: p for p in pipelines}
    chunk_texts = [c.embed_text() for c in chunks]
    question_texts = [q.text for q in questions]

    for key, p in keys.items():
        model, dims, typed = key
        print(f"\n  embedding with {model}" + (f" @ {dims}d" if dims else ""))
        t0 = time.time()
        cvecs = client.embed(
            model, chunk_texts, dimensions=dims,
            input_type="search_document" if typed else None,
        )
        qvecs = client.embed(
            model, question_texts, dimensions=dims,
            input_type="search_query" if typed else None,
        )
        out[key] = (_normalise(np.asarray(cvecs, dtype=np.float32)),
                    _normalise(np.asarray(qvecs, dtype=np.float32)))
        print(f"    {len(chunk_texts):,} chunks + {len(question_texts):,} questions "
              f"in {time.time() - t0:.0f}s ({client.summary()})")
    return out


def vector_candidates(
    qmat: np.ndarray, cmat: np.ndarray, candidate_k: int, block: int = 256
) -> np.ndarray:
    """Top-`candidate_k` chunk indices per question, best first."""
    tops = []
    for start in range(0, qmat.shape[0], block):
        sims = qmat[start : start + block] @ cmat.T
        k = min(candidate_k, sims.shape[1])
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(part, np.argsort(-np.take_along_axis(sims, part, 1), axis=1), 1)
        tops.append(ordered)
    return np.vstack(tops)


def docs_in_order(chunk_indices: list[int], chunks: list[Chunk]) -> list[str]:
    """Collapse a chunk ranking to a document ranking, first occurrence wins.

    The metric is "right *document* in the top 3", so two chunks of the same
    page must not eat two of the three slots.
    """
    seen: list[str] = []
    for idx in chunk_indices:
        doc = chunks[idx].doc_id
        if doc not in seen:
            seen.append(doc)
    return seen


def run_pipeline(
    client: OpenRouter,
    pipeline: Pipeline,
    candidates: np.ndarray,
    chunks: list[Chunk],
    questions: list[Question],
    *,
    final_k: int = 10,
    concurrency: int = 8,
) -> list[int | None]:
    """Return the rank of the gold document for each question (None if absent)."""
    ranks: list[int | None] = [None] * len(questions)
    done = 0

    def one(i: int) -> tuple[int, int | None]:
        cand = [int(x) for x in candidates[i]]
        if pipeline.reranker:
            docs = [chunks[j].embed_text() for j in cand]
            ranked = client.rerank(pipeline.reranker, questions[i].text, docs, top_n=len(docs))
            cand = [cand[pos] for pos, _ in ranked]
        doc_rank = metrics.rank_of(docs_in_order(cand, chunks)[:final_k], questions[i].gold_doc_id)
        return i, doc_rank

    with ThreadPoolExecutor(max_workers=concurrency if pipeline.reranker else 1) as pool:
        for i, rank in pool.map(one, range(len(questions))):
            ranks[i] = rank
            done += 1
            if done % 25 == 0 or done == len(questions):
                _progress(pipeline.label, done, len(questions))
    return ranks


def evaluate(
    client: OpenRouter,
    pipelines: list[Pipeline],
    chunks: list[Chunk],
    questions: list[Question],
    *,
    candidate_k: int = 20,
    concurrency: int = 8,
) -> dict:
    embeddings = embed_everything(client, pipelines, chunks, questions)

    cand_cache: dict[tuple, np.ndarray] = {}
    for key, (cmat, qmat) in embeddings.items():
        cand_cache[key] = vector_candidates(qmat, cmat, candidate_k)

    print("\n  scoring pipelines")
    all_ranks: dict[str, list[int | None]] = {}
    for p in pipelines:
        all_ranks[p.id] = run_pipeline(
            client, p, cand_cache[p.embed_key], chunks, questions, concurrency=concurrency
        )

    groups = {"all": list(range(len(questions)))}
    for level in ("easy", "medium", "hard"):
        idx = [i for i, q in enumerate(questions) if q.difficulty == level]
        if idx:
            groups[level] = idx

    baseline = next((p.id for p in pipelines if p.production), pipelines[0].id)
    results: dict[str, dict] = {}
    for group, idx in groups.items():
        results[group] = {}
        for p in pipelines:
            sub = [all_ranks[p.id][i] for i in idx]
            s = metrics.score(sub).as_dict()
            delta, lo, hi = metrics.paired_delta_ci(
                [all_ranks[baseline][i] for i in idx], sub
            )
            s["delta_vs_production"] = {"delta": delta, "ci": [lo, hi]}
            results[group][p.id] = s

    return {
        "candidate_k": candidate_k,
        "n_chunks": len(chunks),
        "n_documents": len({c.doc_id for c in chunks}),
        "n_questions": len(questions),
        "baseline": baseline,
        "pipelines": [p.__dict__ for p in pipelines],
        "results": results,
        "ranks": {pid: r for pid, r in all_ranks.items()},
        "spend": {"cost_usd": round(client.cost, 4), "api_calls": client.api_calls},
    }
