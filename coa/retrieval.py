"""BM25 over one patient's note sections (one section = one passage)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

TOP_K = 3
STOP = {"the", "a", "an", "of", "is", "in", "what", "which", "and", "at", "to", "for", "by", "on", "this", "that",
        "patient", "patient's", "tumor", "with", "are", "was", "it", "be", "or", "as", "from"}

# Field hints added to the question to form the BM25 query. Shown on the Data page.
QUERY_HINTS: dict[str, str] = {
    "laterality": "left right breast side biopsy specimen",
    "histology": "carcinoma ductal lobular diagnosis",
    "grade": "nottingham grade diagnosis",
    "er": "estrogen receptor er",
    "her2": "her2 ihc",
    "btherapy": "currently taking started mg daily treatment",
    "lobe": "lobe upper middle lower mass biopsy",
    "lhistology": "adenocarcinoma squamous carcinoma diagnosis",
    "egfr": "egfr mutation exon",
    "alk": "alk rearrangement fish",
    "pdl1": "pd-l1 tumor proportion score",
    "ltherapy": "currently receiving started mg treatment",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


@dataclass(frozen=True)
class Hit:
    passage_id: str
    score: float


def query_for(question: str, field: str) -> str:
    return f"{question} {QUERY_HINTS.get(field, '')}".strip()


def retrieve(query: str, passages: list[tuple[str, str]], k: int = TOP_K) -> list[Hit]:
    """passages: (id, text) for one patient. Ties keep note order, so results are deterministic."""
    if not passages:
        return []
    bm25 = BM25Okapi([tokenize(t) for _, t in passages])
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(passages)), key=lambda i: (-scores[i], i))[:k]
    return [Hit(passages[i][0], float(scores[i])) for i in order]
