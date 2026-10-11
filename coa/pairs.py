"""Build DPO preference pairs from the base model's own answers on train questions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from . import db
from .checker import FAILURES, Label
from .config import ABSTAIN
from .dataset import QuestionView, question_views
from .fields import FIELDS

# Which of the model's own failures we prefer as the rejected side, most informative first.
FAILURE_ORDER = [Label.INVENTED, Label.WRONG_VALUE, Label.FALSE_REFUSAL, Label.NON_ANSWER]


@dataclass(frozen=True)
class Sample:
    mode: str
    text: str
    label: Label
    citation_valid: bool | None


@dataclass(frozen=True)
class Pair:
    question_id: str
    chosen: str
    chosen_source: str  # "model" (its own correct answer) or "reference"
    rejected: str
    rejected_source: str  # "model" or "perturbation"
    rejected_label: str


def reference_answer(q: QuestionView) -> str:
    if not q.gold.answerable:
        return ABSTAIN
    first = min(q.gold.evidence)
    return f"{q.gold_display} [P{first}]"


def _h(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16)


def perturb(q: QuestionView) -> tuple[str, str]:
    """A labelled negative for questions the model already gets right on every sample."""
    f = FIELDS[q.field]
    if not q.gold.answerable:
        v = f.values[_h(q.id, "inv") % len(f.values)]
        return f"{v.display} [P1]", Label.INVENTED.value
    if _h(q.id, "kind") % 2 == 0:
        return ABSTAIN, Label.FALSE_REFUSAL.value
    others = [v for v in f.values if v.canonical != q.gold.value]
    v = others[_h(q.id, "wrong") % len(others)]
    return f"{v.display} [P{min(q.gold.evidence)}]", Label.WRONG_VALUE.value


def build_pair(q: QuestionView, samples: list[Sample]) -> Pair:
    good = [s for s in samples if s.label == Label.CORRECT_ABSTAIN or (s.label == Label.CORRECT and s.citation_valid)]
    good.sort(key=lambda s: s.mode != "greedy")
    chosen, chosen_src = (good[0].text, "model") if good else (reference_answer(q), "reference")
    for wanted in FAILURE_ORDER:
        bad = [s for s in samples if s.label == wanted and s.text.strip() != chosen.strip()]
        if bad:
            bad.sort(key=lambda s: s.mode != "greedy")
            return Pair(q.id, chosen, chosen_src, bad[0].text, "model", wanted.value)
    rejected, label = perturb(q)
    return Pair(q.id, chosen, chosen_src, rejected, "perturbation", label)


def content_hash(q: QuestionView, chosen: str, rejected: str) -> str:
    blob = json.dumps([q.text, [p[2] for p in q.passages], chosen, rejected])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_all(conn: sqlite3.Connection) -> dict:
    qs = question_views(conn, "train")
    by_q: dict[str, list[Sample]] = {}
    for r in db.rows(conn, "SELECT question_id, mode, text, label, citation_valid FROM samples ORDER BY id"):
        by_q.setdefault(r["question_id"], []).append(
            Sample(r["mode"], r["text"], Label(r["label"]),
                   None if r["citation_valid"] is None else bool(r["citation_valid"])))
    with db.tx(conn):
        conn.execute("DELETE FROM reviews")
        conn.execute("DELETE FROM audits")
        conn.execute("DELETE FROM pairs")
        for q in qs:
            p = build_pair(q, by_q.get(q.id, []))
            conn.execute(
                "INSERT INTO pairs(question_id, chosen, chosen_source, rejected, rejected_source, rejected_label, "
                "content_hash) VALUES (?,?,?,?,?,?,?)",
                (p.question_id, p.chosen, p.chosen_source, p.rejected, p.rejected_source, p.rejected_label,
                 content_hash(q, p.chosen, p.rejected)),
            )
    return mix(conn)


def mix(conn: sqlite3.Connection) -> dict:
    out = {"total": 0, "by_label": {}, "by_source": {}}
    for r in db.rows(conn, "SELECT rejected_label, rejected_source, COUNT(*) AS n FROM pairs GROUP BY 1, 2"):
        out["total"] += r["n"]
        out["by_label"][r["rejected_label"]] = out["by_label"].get(r["rejected_label"], 0) + r["n"]
        out["by_source"][r["rejected_source"]] = out["by_source"].get(r["rejected_source"], 0) + r["n"]
    return out


def is_failure(label: Label) -> bool:
    return label in FAILURES
