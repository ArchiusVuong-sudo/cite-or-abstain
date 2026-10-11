"""Sample the base model on every train question (greedy plus two samples at T=0.8) and label each answer."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Callable

from . import db
from .checker import check
from .dataset import QuestionView, question_views
from .prompts import messages

FIXTURE = "base_samples.jsonl"
N_SAMPLED = 2
TEMPERATURE = 0.8


def prompt_hash(q: QuestionView) -> str:
    return hashlib.sha256(json.dumps(messages(q.text, q.prompt_passages())).encode()).hexdigest()[:16]


def _store(conn: sqlite3.Connection, q: QuestionView, model: str, answers: list[tuple[str, str]], source: str) -> None:
    for mode, text in answers:
        c = check(text, q.gold)
        conn.execute(
            "INSERT INTO samples(question_id, model, mode, text, label, value, cited, citation_valid, source) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (q.id, model, mode, text, c.label.value, c.value, json.dumps(list(c.cited)),
             None if c.citation_valid is None else int(c.citation_valid), source),
        )


def run_live(conn: sqlite3.Connection, model_id: str, device: str, seed: int,
             progress: Callable[[int, int, str], None]) -> dict:
    from . import modeling

    qs = question_views(conn, "train")
    progress(0, len(qs), f"Loading {model_id}")
    lm = modeling.load(model_id, device)
    convs = [messages(q.text, q.prompt_passages()) for q in qs]
    greedy = modeling.generate(lm, convs, on_batch=lambda d, t: progress(d // 2, t, "Greedy answers"))
    sampled = modeling.generate(lm, convs, n=N_SAMPLED, sample=True, temperature=TEMPERATURE, seed=seed,
                                on_batch=lambda d, t: progress(t // 2 + d // 2, t, "Sampled answers at T=0.8"))
    with db.tx(conn):
        conn.execute("DELETE FROM samples")
        for q, g, s in zip(qs, greedy, sampled):
            _store(conn, q, model_id, [("greedy", g[0])] + [(f"t{TEMPERATURE}", x) for x in s], "live")
    progress(len(qs), len(qs), "Done")
    return {"questions": len(qs), "answers": len(qs) * (1 + N_SAMPLED), "source": "live", "model": model_id}


def run_recorded(conn: sqlite3.Connection, fixtures_dir: Path, progress: Callable[[int, int, str], None]) -> dict:
    path = fixtures_dir / FIXTURE
    if not path.exists():
        raise FileNotFoundError(f"No recorded samples at {path.name}; run the live sampler once to record them")
    recorded: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recorded[r["prompt_hash"]] = r
    qs = question_views(conn, "train")
    missing = 0
    with db.tx(conn):
        conn.execute("DELETE FROM samples")
        for q in qs:
            r = recorded.get(prompt_hash(q))
            if r is None:
                missing += 1
                continue
            _store(conn, q, r["model"], [(a["mode"], a["text"]) for a in r["answers"]], "recorded")
    progress(len(qs), len(qs), "Loaded recorded answers")
    if missing == len(qs):
        raise ValueError("Recorded samples do not match this dataset (different seed or prompt)")
    return {"questions": len(qs) - missing, "missing": missing, "source": "recorded",
            "model": next(iter(recorded.values()))["model"] if recorded else None}


def record_fixture(conn: sqlite3.Connection, fixtures_dir: Path) -> Path:
    """Write the current live samples to fixtures/ so replay mode can load them."""
    qs = {q.id: q for q in question_views(conn, "train")}
    by_q: dict[str, list[dict]] = {}
    model = None
    for r in db.rows(conn, "SELECT question_id, model, mode, text FROM samples WHERE source='live' ORDER BY id"):
        by_q.setdefault(r["question_id"], []).append({"mode": r["mode"], "text": r["text"]})
        model = r["model"]
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    path = fixtures_dir / FIXTURE
    with path.open("w") as fh:
        for qid in sorted(by_q):
            fh.write(json.dumps({"question_id": qid, "prompt_hash": prompt_hash(qs[qid]), "model": model,
                                 "answers": by_q[qid]}) + "\n")
    return path
