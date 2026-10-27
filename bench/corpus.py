"""Turn a folder of documents into retrievable chunks.

The benchmark is corpus-agnostic: point it at any directory of .md / .mdx /
.txt files. The default recipe uses github/docs, a real public help centre
(3,700+ articles, CC BY 4.0) — the closest public analogue to an internal
customer portal.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# Markdown noise that carries no retrievable meaning.
RE_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
RE_LIQUID = re.compile(r"\{%.*?%\}|\{\{.*?\}\}", re.DOTALL)   # github/docs templating
RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RE_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
RE_HTML_TAG = re.compile(r"</?[A-Za-z][^>\n]*>")
RE_BLANKS = re.compile(r"\n{3,}")
RE_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$", re.MULTILINE)


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    heading: str
    text: str
    n_chars: int

    def embed_text(self) -> str:
        """What actually gets embedded: the breadcrumb plus the body.

        Headings carry a lot of the signal in documentation, and dropping them
        quietly costs every model a few points.
        """
        crumb = " > ".join(p for p in (self.title, self.heading) if p)
        return f"{crumb}\n\n{self.text}" if crumb else self.text


def _front_matter(raw: str) -> tuple[dict, str]:
    m = RE_FRONTMATTER.match(raw)
    if not m:
        return {}, raw
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "-", "\t")):
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip("'\"")
    return meta, raw[m.end():]


def clean(text: str) -> str:
    text = RE_HTML_COMMENT.sub("", text)
    text = RE_LIQUID.sub("", text)
    text = RE_MD_IMAGE.sub("", text)
    text = RE_MD_LINK.sub(r"\1", text)
    text = RE_HTML_TAG.sub("", text)
    text = RE_BLANKS.sub("\n\n", text)
    return text.strip()


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split on markdown headings, keeping (heading, body) pairs in order."""
    marks = list(RE_HEADING.finditer(body))
    if not marks:
        return [("", body)]
    out: list[tuple[str, str]] = []
    lead = body[: marks[0].start()].strip()
    if lead:
        out.append(("", lead))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        section = body[m.end():end].strip()
        if section:
            out.append((m.group(2).strip(), section))
    return out


def pack(sections: list[tuple[str, str]], target: int, hard_max: int) -> list[tuple[str, str]]:
    """Merge sections up to `target` chars; split anything past `hard_max`."""
    packed: list[tuple[str, str]] = []
    buf_head, buf_text = "", ""

    def flush():
        nonlocal buf_head, buf_text
        if buf_text.strip():
            packed.append((buf_head, buf_text.strip()))
        buf_head, buf_text = "", ""

    for head, text in sections:
        if len(text) > hard_max:
            flush()
            paras, cur = text.split("\n\n"), ""
            for p in paras:
                if len(cur) + len(p) > target and cur:
                    packed.append((head, cur.strip()))
                    cur = ""
                cur += p + "\n\n"
            if cur.strip():
                packed.append((head, cur.strip()))
            continue
        if len(text) > target and not buf_text:
            # A single oversized block (big table, long code fence): hard-cut it
            # rather than letting one chunk dominate the index.
            for j in range(0, len(text), target):
                packed.append((head, text[j : j + target].strip()))
            continue
        if len(buf_text) + len(text) > target and buf_text:
            flush()
        buf_head = buf_head or head
        buf_text += (("\n\n" if buf_text else "") + text)
    flush()
    return packed


def build(
    root: str,
    *,
    target_chars: int = 1500,
    hard_max: int = 4000,
    min_chars: int = 220,
    extensions: tuple[str, ...] = (".md", ".mdx", ".txt"),
    skip_dirs: tuple[str, ...] = (".git", "node_modules", "assets", "images"),
    skip_stems: tuple[str, ...] = ("readme", "contributing", "code_of_conduct", "license"),
) -> list[Chunk]:
    chunks: list[Chunk] = []
    root_path = Path(root)
    files = sorted(
        p for p in root_path.rglob("*")
        if p.suffix.lower() in extensions
        and not any(part in skip_dirs for part in p.parts)
        and p.stem.lower() not in skip_stems
    )
    for path in files:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        meta, body = _front_matter(raw)
        body = clean(body)
        if len(body) < min_chars:
            continue
        rel = str(path.relative_to(root_path))
        title = meta.get("title") or meta.get("shortTitle") or path.stem.replace("-", " ").title()
        doc_id = hashlib.sha1(rel.encode()).hexdigest()[:12]
        for i, (head, text) in enumerate(pack(split_sections(body), target_chars, hard_max)):
            if len(text) < min_chars:
                continue
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{i}",
                    doc_id=rel,
                    title=title,
                    heading=head,
                    text=text,
                    n_chars=len(text),
                )
            )
    return chunks


def save(chunks: list[Chunk], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load(path: str) -> list[Chunk]:
    with open(path, encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]
