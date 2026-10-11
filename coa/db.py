"""SQLite storage: one file, WAL mode so job workers write while the web app reads."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS patients (
  id TEXT PRIMARY KEY, split TEXT NOT NULL, cancer TEXT NOT NULL,
  record_hash TEXT NOT NULL, date_shift_days INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
  id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id),
  kind TEXT NOT NULL, title TEXT NOT NULL, raw_text TEXT NOT NULL, scrubbed_text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS phi (
  patient_id TEXT NOT NULL, kind TEXT NOT NULL, original TEXT NOT NULL, surrogate TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS passages (
  id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, note_id TEXT NOT NULL,
  section TEXT NOT NULL, text TEXT NOT NULL, supports TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS questions (
  id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, split TEXT NOT NULL, field TEXT NOT NULL,
  text TEXT NOT NULL, documented INTEGER NOT NULL, gold_value TEXT, gold_display TEXT,
  answerable INTEGER NOT NULL, trap TEXT, retrieved TEXT NOT NULL, evidence TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
  id INTEGER PRIMARY KEY AUTOINCREMENT, question_id TEXT NOT NULL, model TEXT NOT NULL,
  mode TEXT NOT NULL, text TEXT NOT NULL, label TEXT NOT NULL, value TEXT, cited TEXT,
  citation_valid INTEGER, source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pairs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, question_id TEXT NOT NULL UNIQUE,
  chosen TEXT NOT NULL, chosen_source TEXT NOT NULL,
  rejected TEXT NOT NULL, rejected_source TEXT NOT NULL, rejected_label TEXT NOT NULL,
  content_hash TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', flipped INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS audits (
  id INTEGER PRIMARY KEY AUTOINCREMENT, pair_id INTEGER NOT NULL, batch INTEGER NOT NULL,
  verdict TEXT NOT NULL, issue TEXT NOT NULL, suggested_action TEXT NOT NULL, reason TEXT NOT NULL,
  source TEXT NOT NULL, trace_id INTEGER
);
CREATE TABLE IF NOT EXISTS llm_traces (
  id INTEGER PRIMARY KEY AUTOINCREMENT, batch INTEGER NOT NULL, model TEXT NOT NULL,
  source TEXT NOT NULL, started_at TEXT NOT NULL, latency_ms INTEGER, turns INTEGER,
  input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, attempts INTEGER NOT NULL DEFAULT 1,
  prompt TEXT NOT NULL, tool_calls TEXT NOT NULL, result_text TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT, pair_id INTEGER NOT NULL, action TEXT NOT NULL, at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, state TEXT NOT NULL,
  progress INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0, message TEXT NOT NULL DEFAULT '',
  params TEXT NOT NULL DEFAULT '{}', result TEXT, error TEXT, pid INTEGER,
  created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS axolotl_checks (
  dataset_hash TEXT PRIMARY KEY, passed INTEGER NOT NULL, log TEXT NOT NULL, at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS train_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER, dataset_hash TEXT NOT NULL, config TEXT NOT NULL,
  adapter_dir TEXT, status TEXT NOT NULL, seconds REAL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS train_metrics (
  run_id INTEGER NOT NULL, step INTEGER NOT NULL, loss REAL, margin REAL, accuracy REAL,
  learning_rate REAL, epoch REAL
);
CREATE TABLE IF NOT EXISTS evals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, split TEXT NOT NULL, created_at TEXT NOT NULL,
  summary TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_answers (
  eval_id INTEGER NOT NULL, question_id TEXT NOT NULL, system TEXT NOT NULL, slice TEXT NOT NULL,
  text TEXT NOT NULL, label TEXT NOT NULL, citation_valid INTEGER
);
CREATE TABLE IF NOT EXISTS exports (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, kind TEXT NOT NULL, status TEXT NOT NULL,
  path TEXT, size_bytes INTEGER, note TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_samples_q ON samples(question_id);
CREATE INDEX IF NOT EXISTS ix_eval_answers ON eval_answers(eval_id, system, slice);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def rows(conn: sqlite3.Connection, sql: str, *args: Any) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def one(conn: sqlite3.Connection, sql: str, *args: Any) -> dict[str, Any] | None:
    r = conn.execute(sql, args).fetchone()
    return dict(r) if r else None


def get_meta(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    r = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(r[0]) if r else default


def set_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value)),
    )
