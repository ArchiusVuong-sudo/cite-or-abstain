"""Audit a sample of pairs with Claude through one tool call per pair; disputed pairs go to the review queue."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import db
from .checker import canonical_value
from .dataset import question_views
from .fields import FIELDS, THERAPY_FIELDS
from .llm import AuditProvider, BatchResult, record_replay

VALUE_WORDS = re.compile(r"\b(none|no|not|negative|positive|detected|observed|wild|mutat\w*|deletion|grade|"
                         r"carcinoma|lobe)\b|lobe|\d+\s*mg", re.I)
N_BATCHES = 3
BATCH_SIZE = 8


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def _unknown_value_phrase(field: str, text: str) -> bool:
    """A value-like phrase that names no value of this field and no value of a sibling field."""
    if canonical_value(field, text) is not None or not VALUE_WORDS.search(text):
        return False
    cancer = FIELDS[field].cancer
    return not any(canonical_value(f, text) for f, d in FIELDS.items() if d.cancer == cancer and f != field)


def quoted_from_passage(answer: str, passages: list[str]) -> bool:
    """The answer's words (2+ words, citation stripped) appear verbatim in a retrieved passage."""
    core = re.sub(r"\[[^\]]*\]|\(P\d\)|\bP\d\b", " ", answer).strip(" .,;:\n").lower()
    core = re.sub(r"\s+", " ", core)
    return len(core.split()) >= 2 and any(core in re.sub(r"\s+", " ", t.lower()) for t in passages)


def uncertainty(p: dict, passages: list[str] | None = None) -> int:
    """How likely the rule-based checker got this pair wrong; the audit looks there first."""
    score = 0
    if p["rejected_source"] == "model":
        score += 1
        if p["rejected_label"] == "invented" and passages and quoted_from_passage(p["rejected"], passages):
            score += 4  # "invented" words that a passage states: maybe a coarser true value, a label-policy call
        if p["rejected_label"] == "wrong_value" and _unknown_value_phrase(p["field"], p["rejected"]):
            score += 3  # a value-like phrase the checker does not know: "wrong" may be a phrasing it lacks
        if p["field"] in THERAPY_FIELDS and canonical_value(p["field"], p["rejected"]) is not None:
            score += 3  # right drug, missing or different dose
        if p["rejected_label"] == "non_answer":
            score += 1
    if p["chosen_source"] == "model" and len(p["chosen"]) > 40:
        score += 2  # a long chosen answer may carry unsupported extra claims
    if p["trap"]:
        score += 1
    return score


def select_pairs(conn: sqlite3.Connection, n: int = N_BATCHES * BATCH_SIZE) -> list[dict]:
    """Deterministic sample of the pairs where the rules are least reliable, plus a few perturbations."""
    pairs = db.rows(conn, "SELECT p.*, q.trap, q.field FROM pairs p JOIN questions q ON q.id = p.question_id "
                          "WHERE p.status = 'active'")
    texts = {q.id: [t for _, t in q.prompt_passages()] for q in question_views(conn, "train")}
    pairs.sort(key=lambda p: (-uncertainty(p, texts.get(p["question_id"])), _h(p["content_hash"])))
    n_perturbed = n // 6
    model = [p for p in pairs if p["rejected_source"] == "model"][: n - n_perturbed]
    perturbed = [p for p in pairs if p["rejected_source"] == "perturbation"][: n - len(model)]
    return sorted(model + perturbed, key=lambda p: p["id"])


def batch_prompt(conn: sqlite3.Connection, batch: list[dict]) -> str:
    qs = {q.id: q for q in question_views(conn, "train", ids=[p["question_id"] for p in batch])}
    items = []
    for p in batch:
        q = qs[p["question_id"]]
        items.append({
            "pair_id": p["id"],
            "question": q.text,
            "passages": {f"P{i + 1}": f"({label}) {text}" for i, (label, text) in enumerate(q.prompt_passages())},
            "chosen": p["chosen"],
            "rejected": p["rejected"],
            "checker_label_of_rejected": p["rejected_label"],
            "rejected_source": p["rejected_source"],
        })
    return ("Audit these preference pairs. Call record_audit once for each pair_id.\n\n"
            + json.dumps(items, indent=1))


def run(conn: sqlite3.Connection, provider: AuditProvider, fixtures_dir: Path, record: bool,
        progress: Callable[[int, int, str], None]) -> dict:
    picked = select_pairs(conn)
    batches = [picked[i:i + BATCH_SIZE] for i in range(0, len(picked), BATCH_SIZE)]
    with db.tx(conn):
        conn.execute("DELETE FROM audits")
        conn.execute("DELETE FROM llm_traces")
    totals = {"pairs": 0, "disputed": 0, "errors": 0, "cost_usd": 0.0, "latency_ms": 0}
    for b, batch in enumerate(batches, start=1):
        progress(b - 1, len(batches), f"Batch {b} of {len(batches)}: {len(batch)} pairs")
        prompt = batch_prompt(conn, batch)
        keys = {p["id"]: p["content_hash"] for p in batch}
        res: BatchResult = provider.audit_batch(b, [p["id"] for p in batch], prompt, keys)
        with db.tx(conn):
            cur = conn.execute(
                "INSERT INTO llm_traces(batch, model, source, started_at, latency_ms, turns, input_tokens, "
                "output_tokens, cost_usd, attempts, prompt, tool_calls, result_text, error) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (b, res.model, res.source, datetime.now(timezone.utc).isoformat(timespec="seconds"), res.latency_ms,
                 res.turns, res.input_tokens, res.output_tokens, res.cost_usd, res.attempts, prompt,
                 json.dumps(res.tool_calls), res.result_text, res.error),
            )
            trace_id = cur.lastrowid
            for rec in res.records:
                conn.execute(
                    "INSERT INTO audits(pair_id, batch, verdict, issue, suggested_action, reason, source, trace_id) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (rec.pair_id, b, rec.verdict, rec.issue, rec.suggested_action, rec.reason, res.source, trace_id),
                )
        totals["pairs"] += len(res.records)
        totals["disputed"] += sum(r.verdict == "dispute" for r in res.records)
        totals["errors"] += int(bool(res.error))
        totals["cost_usd"] += res.cost_usd or 0.0
        totals["latency_ms"] += res.latency_ms or 0
        if record and res.source == "live" and res.records:
            record_replay(fixtures_dir, [(keys[r.pair_id], r) for r in res.records], list(keys.values()), res)
    progress(len(batches), len(batches), "Done")
    totals["cost_usd"] = round(totals["cost_usd"], 4)
    totals["source"] = provider.mode
    return totals


def queue(conn: sqlite3.Connection) -> list[dict]:
    """Disputed pairs with their latest review, oldest first."""
    return db.rows(
        conn,
        "SELECT a.*, p.question_id, p.chosen, p.rejected, p.rejected_label, p.rejected_source, p.status, p.flipped, "
        "(SELECT action FROM reviews r WHERE r.pair_id = a.pair_id ORDER BY r.id DESC LIMIT 1) AS decision, "
        "(SELECT at FROM reviews r WHERE r.pair_id = a.pair_id ORDER BY r.id DESC LIMIT 1) AS decided_at "
        "FROM audits a JOIN pairs p ON p.id = a.pair_id WHERE a.verdict = 'dispute' ORDER BY a.pair_id",
    )
