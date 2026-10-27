"""Turn a run into the published chart and a console summary.

The chart page owns its own design; this module only splices the data block
between the BENCH-DATA markers, so re-running the benchmark refreshes the
figures without touching the layout.
"""

from __future__ import annotations

import json
import os
import re

MARK_START = "/* BENCH-DATA:START */"
MARK_END = "/* BENCH-DATA:END */"
PANEL_LABELS = {
    "all": "All questions",
    "easy": "Easy questions",
    "medium": "Medium questions",
    "hard": "Hard questions",
}


def _pct(x: float) -> int:
    return int(round(100 * x))


def _lead(pipelines: list[dict], family: str) -> dict | None:
    """The no-reranker pipeline of a family — the embedder on its own."""
    return next((p for p in pipelines if p["family"] == family and not p.get("reranker")), None)


def _best(results: dict, pipelines: list[dict], group: str, family: str | None = None) -> dict:
    pool = [p for p in pipelines if family is None or p["family"] == family]
    return max(pool, key=lambda p: results[group][p["id"]]["recall_at_3"])


def build_tiles(run: dict) -> list[dict]:
    """The four headline deltas. Each is a paired comparison on the same questions."""
    pipelines, results = run["pipelines"], run["results"]
    families = list(dict.fromkeys(p["family"] for p in pipelines))
    tiles: list[dict] = []
    if len(families) < 2:
        return tiles

    base_lead, chall_lead = _lead(pipelines, families[0]), _lead(pipelines, families[-1])
    prod = next((p for p in pipelines if p.get("production")), pipelines[0])

    def r3(pid: str, group: str = "all") -> float:
        return results[group][pid]["recall_at_3"]

    if base_lead and chall_lead:
        tiles.append({
            "delta": _pct(r3(chall_lead["id"]) - r3(base_lead["id"])),
            "what": "Swapping the embedder, no reranker on either side",
            "from": f"{_pct(r3(base_lead['id']))}% → {_pct(r3(chall_lead['id']))}%",
            "tone": "b",
        })
        best_rr = _best(results, pipelines, "all", families[0])
        tiles.append({
            "delta": _pct(r3(best_rr["id"]) - r3(base_lead["id"])),
            "what": "Best reranker bolted onto the old embedder",
            "from": f"{_pct(r3(base_lead['id']))}% → {_pct(r3(best_rr['id']))}%",
            "tone": "a",
        })
        if "hard" in results:
            tiles.append({
                "delta": _pct(r3(chall_lead["id"], "hard") - r3(base_lead["id"], "hard")),
                "what": "Same embedder swap, hard questions only",
                "from": f"{_pct(r3(base_lead['id'], 'hard'))}% → "
                        f"{_pct(r3(chall_lead['id'], 'hard'))}%",
                "tone": "b",
            })
    best_overall = _best(results, pipelines, "all")
    tiles.append({
        "delta": _pct(r3(best_overall["id"]) - r3(prod["id"])),
        "what": "Today's production pipeline → best pipeline tested",
        "from": f"{_pct(r3(prod['id']))}% → {_pct(r3(best_overall['id']))}%",
        "tone": "b",
    })
    return tiles


def to_chart_data(run: dict, *, corpus_name: str) -> dict:
    pipelines, results = run["pipelines"], run["results"]
    families = list(dict.fromkeys(p["family"] for p in pipelines))
    return {
        "configs": [
            {
                "id": p["id"],
                "label": p["label"],
                "family": "a" if p["family"] == families[0] else "b",
                **({"production": True} if p.get("production") else {}),
            }
            for p in pipelines
        ],
        "panels": [
            {"id": g, "label": PANEL_LABELS.get(g, g.title()),
             "n": f"n = {results[g][pipelines[0]['id']]['n']:,}" if g == "all" else ""}
            for g in ("all", "easy", "medium", "hard") if g in results
        ],
        "data": {
            g: {p["id"]: _pct(results[g][p["id"]]["recall_at_3"]) for p in pipelines}
            for g in results
        },
        "meta": {
            "corpus": corpus_name,
            "n_questions": run["n_questions"],
            "n_documents": run["n_documents"],
            "n_chunks": run["n_chunks"],
            "candidate_k": run["candidate_k"],
            "tiles": build_tiles(run),
            "families": [
                {"key": "a", "label": next(p["embedder"] for p in pipelines
                                               if p["family"] == families[0])},
                {"key": "b", "label": next(p["embedder"] for p in pipelines
                                                if p["family"] == families[-1])},
            ],
        },
    }


def _js(obj) -> str:
    """Compact JSON for the page's inline data block."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def splice(chart_path: str, chart_data: dict) -> None:
    with open(chart_path, encoding="utf-8") as fh:
        html = fh.read()
    if MARK_START not in html or MARK_END not in html:
        raise RuntimeError(f"{chart_path} has no BENCH-DATA markers")
    block = (
        f"{MARK_START}\n"
        f"const CONFIGS={_js(chart_data['configs'])};\n"
        f"const PANELS={_js(chart_data['panels'])};\n"
        f"const DATA={_js(chart_data['data'])};\n"
        f"const META={_js(chart_data['meta'])};\n"
        f"{MARK_END}"
    )
    html = re.sub(
        re.escape(MARK_START) + r".*?" + re.escape(MARK_END), lambda _: block, html, flags=re.DOTALL
    )
    with open(chart_path, "w", encoding="utf-8") as fh:
        fh.write(html)


def console_table(run: dict) -> str:
    pipelines, results = run["pipelines"], run["results"]
    groups = [g for g in ("all", "easy", "medium", "hard") if g in results]
    width = max(len(p["label"]) for p in pipelines) + 2
    lines = [
        "",
        f"recall@3 · {run['n_questions']:,} questions · "
        f"{run['n_documents']:,} documents · candidate_k={run['candidate_k']}",
        "",
        f"{'pipeline':<{width}}" + "".join(f"{g:>9}" for g in groups)
        + "   95% CI (all)      Δ vs prod (paired)",
        "-" * (width + 9 * len(groups) + 42),
    ]
    for p in pipelines:
        row = f"{p['label']:<{width}}"
        for g in groups:
            row += f"{_pct(results[g][p['id']]['recall_at_3']):>8}%"
        lo, hi = results["all"][p["id"]]["recall_at_3_ci"]
        d = results["all"][p["id"]]["delta_vs_production"]
        sig = "" if d["ci"][0] <= 0 <= d["ci"][1] else " *"
        row += (
            f"   [{_pct(lo):>2}%, {_pct(hi):>2}%]"
            f"   {_pct(d['delta']):+4} pts [{_pct(d['ci'][0]):+3}, {_pct(d['ci'][1]):+3}]{sig}"
        )
        if p.get("production"):
            row += "  <- prod"
        lines.append(row)
    lines += [
        "",
        "* paired 95% interval excludes zero: the difference is real, not sampling noise.",
        f"spend this run: ${run['spend']['cost_usd']:.4f} over {run['spend']['api_calls']:,} API calls.",
        "",
    ]
    return "\n".join(lines)
