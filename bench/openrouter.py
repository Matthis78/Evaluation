"""OpenRouter client for the three calls this benchmark makes.

One API key covers embeddings, reranking and the question-generation LLM.
Every response is cached in SQLite, so re-running a matrix is free and a run
that dies halfway resumes where it stopped.

Endpoints (confirmed against the official SDK, openrouter==1.3.84):
    POST /embeddings  {model, input, dimensions?, input_type?} -> {data:[{embedding,index}], usage}
    POST /rerank      {model, query, documents, top_n?}        -> {results:[{index,relevance_score}], usage}
    POST /chat/completions                                      (OpenAI-compatible)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import requests

BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_CACHE = os.environ.get("BENCH_CACHE", ".cache/openrouter.sqlite")


def _key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    """SQLite key/value store. Safe to share across threads."""

    def __init__(self, path: str = DEFAULT_CACHE):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._db.commit()

    def get(self, key: str):
        with self._lock:
            row = self._db.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (key, payload))
            self._db.commit()

    def stats(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM kv").fetchone()[0]


class RateLimited(RuntimeError):
    pass


class OpenRouter:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_path: str = DEFAULT_CACHE,
        base_url: str = BASE_URL,
        timeout: int = 180,
        max_retries: int = 6,
        concurrency: int = 8,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "No API key. Set OPENROUTER_API_KEY in the environment:\n"
                "    export OPENROUTER_API_KEY=sk-or-...\n"
                "Create one at https://openrouter.ai/keys"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.cache = Cache(cache_path)
        self._session = requests.Session()
        self._cost_lock = threading.Lock()
        self.cost = 0.0           # credits spent on calls that actually hit the API
        self.api_calls = 0
        self.cache_hits = 0

    # -- transport ---------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/",
            "X-OpenRouter-Title": "rag-retrieval-benchmark",
        }

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        last = None
        for attempt in range(self.max_retries):
            try:
                r = self._session.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
            except requests.RequestException as exc:          # network blip
                last = exc
                time.sleep(min(2 ** attempt, 30) + random.random())
                continue

            if r.status_code == 200:
                with self._cost_lock:
                    self.api_calls += 1
                body = r.json()
                usage = body.get("usage") or {}
                if usage.get("cost"):
                    with self._cost_lock:
                        self.cost += float(usage["cost"])
                return body

            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After") or 0) or min(2 ** attempt, 30)
                last = RateLimited(f"{r.status_code} {r.text[:200]}")
                time.sleep(wait + random.random())
                continue

            raise RuntimeError(f"OpenRouter {path} -> {r.status_code}: {r.text[:500]}")

        raise RuntimeError(f"OpenRouter {path} failed after {self.max_retries} attempts: {last}")

    # -- embeddings --------------------------------------------------------

    def embed(
        self,
        model: str,
        texts: Sequence[str],
        *,
        input_type: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 64,
    ) -> list[list[float]]:
        """Embed texts, reusing anything already cached. Order is preserved."""
        keys = [_key("embed", model, input_type, dimensions, t) for t in texts]
        out: list[list[float] | None] = [self.cache.get(k) for k in keys]
        self.cache_hits += sum(1 for v in out if v is not None)

        missing = [i for i, v in enumerate(out) if v is None]
        if not missing:
            return out  # type: ignore[return-value]

        batches = [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]

        def run(batch: list[int]) -> None:
            payload: dict[str, Any] = {"model": model, "input": [texts[i] for i in batch]}
            if dimensions:
                payload["dimensions"] = dimensions
            if input_type:
                payload["input_type"] = input_type
            body = self._post("/embeddings", payload)
            data = sorted(body["data"], key=lambda d: d.get("index", 0))
            if len(data) != len(batch):
                raise RuntimeError(f"{model}: asked {len(batch)} embeddings, got {len(data)}")
            for pos, item in zip(batch, data):
                vec = item["embedding"]
                out[pos] = vec
                self.cache.put(keys[pos], vec)

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            list(pool.map(run, batches))

        if any(v is None for v in out):
            raise RuntimeError(f"{model}: some embeddings are still missing")
        return out  # type: ignore[return-value]

    # -- rerank ------------------------------------------------------------

    def rerank(
        self, model: str, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[tuple[int, float]]:
        """Return [(original_index, relevance_score)] sorted most relevant first."""
        k = _key("rerank", model, query, list(documents), top_n)
        hit = self.cache.get(k)
        if hit is not None:
            self.cache_hits += 1
            return [(int(i), float(s)) for i, s in hit]

        payload: dict[str, Any] = {"model": model, "query": query, "documents": list(documents)}
        if top_n:
            payload["top_n"] = top_n
        body = self._post("/rerank", payload)
        ranked = [(int(r["index"]), float(r["relevance_score"])) for r in body["results"]]
        ranked.sort(key=lambda p: -p[1])
        self.cache.put(k, ranked)
        return ranked

    # -- chat (question generation / validation) ---------------------------

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        cache_salt: str = "",
    ) -> str:
        k = _key("chat", model, messages, temperature, max_tokens, json_mode, cache_salt)
        hit = self.cache.get(k)
        if hit is not None:
            self.cache_hits += 1
            return hit

        payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = self._post("/chat/completions", payload)
        text = body["choices"][0]["message"]["content"] or ""
        self.cache.put(k, text)
        return text

    def summary(self) -> str:
        return (
            f"{self.api_calls} API calls, {self.cache_hits} cache hits, "
            f"${self.cost:.4f} spent this run"
        )
