"""Seed step: generate the cohort, scrub PHI, cut passages, retrieve, and write gold labels."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass

from . import db
from .checker import Gold
from .fields import FIELDS, fields_for
from .phi import Scrubber, date_shift, leak_scan
from .retrieval import TOP_K, query_for, retrieve
from .synth import STAMP, Patient, generate
from .typos import add_typos


@dataclass(frozen=True)
class QuestionView:
    id: str
    patient_id: str
    split: str
    field: str
    text: str
    gold: Gold
    gold_display: str | None
    documented: bool
    trap: str | None
    passages: list[tuple[str, str, str]]  # (passage_id, label, text) in retrieval order

    def prompt_passages(self, typo: bool = False) -> list[tuple[str, str]]:
        return [(label, add_typos(text, f"{self.id}:{pid}") if typo else text) for pid, label, text in self.passages]


def originals(p: Patient) -> list[str]:
    i = p.identity
    raw = "\n".join(n.render() for n in p.notes)
    dates = re.findall(r"\b\d{2}/\d{2}/\d{4}\b", raw)
    return [i.first, i.last, f"{i.first} {i.last}", i.mrn, i.phone, i.dob, i.clinician, *dates]


def seed(conn: sqlite3.Connection, seed_value: int) -> dict:
    """Rebuild every data table from the seed. Returns the summary shown on the Data page."""
    patients = generate(seed_value)
    with db.tx(conn):
        for t in ("notes", "phi", "passages", "questions", "samples", "pairs", "audits", "reviews", "llm_traces",
                  "patients"):
            conn.execute(f"DELETE FROM {t}")
        for p in patients:
            raw_texts = [n.render() for n in p.notes]
            shift = date_shift(seed_value, p.id, raw_texts)
            scrubber = Scrubber(f"{seed_value}:{p.id}", shift)
            conn.execute("INSERT INTO patients VALUES (?,?,?,?,?)", (p.id, p.split, p.cancer, p.record_hash(), shift))
            plist: list[tuple[str, str, str, dict]] = []
            for k, note in enumerate(p.notes):
                nid = f"{p.id}-N{k + 1}"
                scrubbed = scrubber.scrub(note.render()).text
                conn.execute("INSERT INTO notes VALUES (?,?,?,?,?,?)",
                             (nid, p.id, note.kind, note.title, note.render(), scrubbed))
                for j, sec in enumerate(note.sections):
                    pid = f"{nid}-S{j + 1}"
                    text = scrubber.scrub(sec.text).text
                    plist.append((pid, f"{note.title} - {sec.name}", text, sec.supports))
                    conn.execute("INSERT INTO passages VALUES (?,?,?,?,?,?)",
                                 (pid, p.id, nid, sec.name, text, json.dumps(sec.supports)))
            for (kind, orig), sur in scrubber.mapping.items():
                conn.execute("INSERT INTO phi VALUES (?,?,?,?)", (p.id, kind, orig, sur))
            for f in fields_for(p.cancer):
                fdef = FIELDS[f]
                hits = retrieve(query_for(fdef.question, f), [(x[0], f"{x[1]} {x[2]}") for x in plist], TOP_K)
                retrieved = [h.passage_id for h in hits]
                by_id = {x[0]: x for x in plist}
                documented = p.documented[f]
                gold_value = p.values.get(f) if documented else None
                evidence = [i + 1 for i, pid in enumerate(retrieved) if gold_value and
                            by_id[pid][3].get(f) == gold_value]
                answerable = bool(evidence)
                conn.execute(
                    "INSERT INTO questions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f"{p.id}-{f}", p.id, p.split, f, fdef.question, int(documented), gold_value if answerable else None,
                     fdef.value(gold_value).display if answerable and gold_value else None, int(answerable),
                     p.traps.get(f), json.dumps(retrieved), json.dumps(evidence)),
                )
        db.set_meta(conn, "seed", seed_value)
    return summary(conn)


def question_views(conn: sqlite3.Connection, split: str | None = None, ids: list[str] | None = None
                   ) -> list[QuestionView]:
    sql, args = "SELECT * FROM questions", []
    if split:
        sql, args = sql + " WHERE split=?", [split]
    qs = db.rows(conn, sql + " ORDER BY id", *args)
    if ids is not None:
        keep = set(ids)
        qs = [q for q in qs if q["id"] in keep]
    pas = {r["id"]: r for r in db.rows(conn, "SELECT p.id, p.text, p.section, n.title FROM passages p "
                                             "JOIN notes n ON n.id = p.note_id")}
    out = []
    for q in qs:
        retrieved = json.loads(q["retrieved"])
        out.append(QuestionView(
            id=q["id"], patient_id=q["patient_id"], split=q["split"], field=q["field"], text=q["text"],
            gold=Gold(q["field"], bool(q["answerable"]), q["gold_value"], frozenset(json.loads(q["evidence"]))),
            gold_display=q["gold_display"], documented=bool(q["documented"]), trap=q["trap"],
            passages=[(pid, f"{pas[pid]['title']} - {pas[pid]['section']}", pas[pid]["text"]) for pid in retrieved],
        ))
    return out


def recall_at_k(conn: sqlite3.Connection) -> dict:
    """Share of documented fields whose stating passage is in the top-k."""
    r = db.one(conn, "SELECT SUM(documented) AS documented, SUM(answerable) AS hit FROM questions")
    documented, hit = (r or {}).get("documented") or 0, (r or {}).get("hit") or 0
    return {"k": TOP_K, "documented": documented, "hit": hit, "recall": (hit / documented) if documented else None}


def leak_gate(conn: sqlite3.Connection, items: list[tuple[str, str]]) -> list[str]:
    """items: (patient_id, text). Each text is scanned for that patient's original identifiers."""
    seed_value = db.get_meta(conn, "seed")
    if seed_value is None:
        return []
    truth = {p.id: originals(p) for p in generate(seed_value)}
    texts: dict[str, list[str]] = {}
    for pid, text in items:
        texts.setdefault(pid, []).append(text)
    found: set[str] = set()
    for pid, ts in texts.items():
        found.update(f"{pid}:{o}" for o in leak_scan(ts, truth.get(pid, [])))
    return sorted(found)


def summary(conn: sqlite3.Connection) -> dict:
    splits = {r["split"]: r for r in db.rows(
        conn, "SELECT split, COUNT(*) AS patients, SUM(cancer='breast') AS breast, SUM(cancer='lung') AS lung "
              "FROM patients GROUP BY split")}
    q = db.one(conn, "SELECT COUNT(*) AS n, SUM(documented) AS documented, SUM(answerable) AS answerable, "
                     "SUM(trap IS NOT NULL AND documented=0) AS traps FROM questions") or {}
    return {
        "stamp": STAMP,
        "patients": sum(s["patients"] for s in splits.values()),
        "notes": (db.one(conn, "SELECT COUNT(*) AS n FROM notes") or {}).get("n", 0),
        "passages": (db.one(conn, "SELECT COUNT(*) AS n FROM passages") or {}).get("n", 0),
        "splits": splits,
        "questions": q.get("n") or 0,
        "documented": q.get("documented") or 0,
        "answerable": q.get("answerable") or 0,
        "traps": q.get("traps") or 0,
        "recall": recall_at_k(conn),
        "phi_replacements": (db.one(conn, "SELECT COUNT(*) AS n FROM phi") or {}).get("n", 0),
    }
