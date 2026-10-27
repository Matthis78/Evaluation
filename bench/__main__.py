"""CLI: python -m bench <command>

    models     list the embedding and rerank slugs your key can reach
    fetch      clone the corpus
    chunk      corpus -> chunks.jsonl
    questions  chunks -> questions.jsonl (generate, then validate)
    run        the pipeline matrix -> results/run.json
    report     results -> chart + console summary
    all        fetch, chunk, questions, run, report

Each step is resumable: everything the API returns is cached, so re-running
costs nothing and a crash halfway through loses no work.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import yaml

from . import corpus as corpus_mod
from . import questions as q_mod
from . import report as report_mod
from .openrouter import OpenRouter
from .run import Pipeline, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS = {
    "config": os.path.join(ROOT, "config.yaml"),
    "corpus_dir": os.path.join(ROOT, "data", "corpus"),
    "chunks": os.path.join(ROOT, "data", "chunks.jsonl"),
    "questions": os.path.join(ROOT, "data", "questions.jsonl"),
    "rejected": os.path.join(ROOT, "data", "questions.rejected.jsonl"),
    "run": os.path.join(ROOT, "results", "run.json"),
    "chart_data": os.path.join(ROOT, "results", "retrieval_benchmark.json"),
    "chart": os.path.join(ROOT, "chart", "index.html"),
}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def client_from(cfg: dict) -> OpenRouter:
    return OpenRouter(
        cache_path=cfg.get("cache_path", os.path.join(ROOT, ".cache", "openrouter.sqlite")),
        concurrency=cfg.get("concurrency", 8),
    )


# -- commands --------------------------------------------------------------

def cmd_models(args, cfg) -> None:
    import requests

    c = client_from(cfg)
    r = requests.get(
        f"{c.base_url}/embeddings/models", headers=c._headers(), timeout=60
    )
    r.raise_for_status()
    rows = r.json().get("data", [])
    print(f"\nembedding models ({len(rows)}):")
    for m in sorted(rows, key=lambda m: m.get("id", "")):
        print(f"  {m.get('id')}")

    # /models with no params returns only text/image/audio models — rerankers
    # (and the :batch embedding variants) need output_modalities=all.
    r = requests.get(
        f"{c.base_url}/models", headers=c._headers(),
        params={"output_modalities": "all"}, timeout=60,
    )
    r.raise_for_status()
    rr = [m for m in r.json().get("data", []) if "rerank" in (m.get("id") or "").lower()]
    print(f"\nrerank models ({len(rr)}):")
    for m in sorted(rr, key=lambda m: m.get("id", "")):
        print(f"  {m.get('id')}")
    print("\nPut the slugs you want into config.yaml.\n")


def cmd_fetch(args, cfg) -> None:
    src = cfg["corpus"]
    dest = DEFAULTS["corpus_dir"]
    if os.path.isdir(os.path.join(dest, ".git")):
        print(f"corpus already present at {dest} (delete it to re-fetch)")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"cloning {src['repo']} (sparse: {src['subdir']})")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", src["repo"], dest],
        check=True,
    )
    subprocess.run(["git", "sparse-checkout", "set", src["subdir"]], cwd=dest, check=True)
    print(f"corpus at {dest}/{src['subdir']}")


def cmd_chunk(args, cfg) -> None:
    src = os.path.join(DEFAULTS["corpus_dir"], cfg["corpus"]["subdir"])
    chunks = corpus_mod.build(
        src,
        target_chars=cfg["chunking"]["target_chars"],
        hard_max=cfg["chunking"]["hard_max"],
        min_chars=cfg["chunking"]["min_chars"],
    )
    corpus_mod.save(chunks, DEFAULTS["chunks"])
    docs = len({c.doc_id for c in chunks})
    print(f"{len(chunks):,} chunks from {docs:,} documents -> {DEFAULTS['chunks']}")


def cmd_questions(args, cfg) -> None:
    chunks = corpus_mod.load(DEFAULTS["chunks"])
    picked = q_mod.sample_chunks(
        chunks, cfg["questions"]["n_chunks"], seed=cfg["questions"]["seed"]
    )
    client = client_from(cfg)
    model = cfg["questions"]["model"]
    print(f"generating 3 questions for each of {len(picked):,} passages with {model}")
    generated = q_mod.generate(client, picked, model=model, concurrency=cfg.get("concurrency", 8))
    print(f"  {len(generated):,} generated ({client.summary()})")

    print("validating (answerable / self-contained / specific)")
    kept, rejected = q_mod.validate(
        client, generated, chunks, model=cfg["questions"]["validator_model"],
        concurrency=cfg.get("concurrency", 8),
    )
    q_mod.save(kept, DEFAULTS["questions"])
    with open(DEFAULTS["rejected"], "w", encoding="utf-8") as fh:
        for r in rejected:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_level = {d: sum(1 for q in kept if q.difficulty == d) for d in q_mod.DIFFICULTIES}
    drop = 100 * len(rejected) / max(len(generated), 1)
    print(f"  kept {len(kept):,} ({drop:.0f}% dropped) -> {DEFAULTS['questions']}")
    print(f"  by difficulty: {by_level}")


def cmd_run(args, cfg) -> None:
    chunks = corpus_mod.load(DEFAULTS["chunks"])
    questions = q_mod.load(DEFAULTS["questions"])
    pipelines = [Pipeline(**p) for p in cfg["pipelines"]]
    client = client_from(cfg)
    print(f"{len(pipelines)} pipelines x {len(questions):,} questions over {len(chunks):,} chunks")
    out = evaluate(
        client, pipelines, chunks, questions,
        candidate_k=cfg["retrieval"]["candidate_k"], concurrency=cfg.get("concurrency", 8),
    )
    os.makedirs(os.path.dirname(DEFAULTS["run"]), exist_ok=True)
    with open(DEFAULTS["run"], "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n-> {DEFAULTS['run']}")
    print(report_mod.console_table(out))


def cmd_report(args, cfg) -> None:
    with open(DEFAULTS["run"], encoding="utf-8") as fh:
        run = json.load(fh)
    chart_data = report_mod.to_chart_data(run, corpus_name=cfg["corpus"]["name"])
    with open(DEFAULTS["chart_data"], "w", encoding="utf-8") as fh:
        json.dump(chart_data, fh, indent=2, ensure_ascii=False)
    report_mod.splice(DEFAULTS["chart"], chart_data)
    print(report_mod.console_table(run))
    print(f"chart data -> {DEFAULTS['chart_data']}")
    print(f"chart page -> {DEFAULTS['chart']}")
    print(
        "\nNote: the headline and the deck paragraph in chart/index.html are editorial.\n"
        "Re-read them against these numbers before you publish."
    )


def cmd_all(args, cfg) -> None:
    for step in (cmd_fetch, cmd_chunk, cmd_questions, cmd_run, cmd_report):
        print(f"\n=== {step.__name__.removeprefix('cmd_')} ===")
        step(args, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=[
        "models", "fetch", "chunk", "questions", "run", "report", "all"])
    parser.add_argument("--config", default=DEFAULTS["config"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    globals()[f"cmd_{args.command}"](args, cfg)


if __name__ == "__main__":
    main()
