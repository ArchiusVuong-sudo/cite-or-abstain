"""Job runner: one long job at a time, each in its own worker process, state kept in SQLite.

Model steps never overlap (one lock), a crash cannot take the web app down, and a worker that
dies leaves a job the app marks as failed instead of one that spins forever.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import db
from .config import ROOT, Settings
from .llm import ENV_ALLOW

KINDS = {"sample", "audit", "axolotl", "train", "eval", "export"}
TERMINAL = {"done", "error"}
LLM_KINDS = {"audit"}


class Busy(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A finished child stays a zombie until reaped; reap it so it reads as gone.
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def reap_stale(conn: sqlite3.Connection) -> None:
    for j in db.rows(conn, "SELECT id, pid, state FROM jobs WHERE state IN ('queued', 'running')"):
        if j["pid"] and not _alive(j["pid"]):
            conn.execute("UPDATE jobs SET state='error', error=?, finished_at=? "
                         "WHERE id=? AND state IN ('queued', 'running')",
                         ("The worker process exited without finishing", now(), j["id"]))


def active(conn: sqlite3.Connection) -> dict | None:
    reap_stale(conn)
    return db.one(conn, "SELECT * FROM jobs WHERE state IN ('queued', 'running') ORDER BY id LIMIT 1")


def get(conn: sqlite3.Connection, job_id: int) -> dict | None:
    reap_stale(conn)
    j = db.one(conn, "SELECT * FROM jobs WHERE id=?", job_id)
    if j:
        j["params"] = json.loads(j["params"])
        j["result"] = json.loads(j["result"]) if j["result"] else None
    return j


def latest(conn: sqlite3.Connection, kind: str) -> dict | None:
    j = db.one(conn, "SELECT id FROM jobs WHERE kind=? ORDER BY id DESC LIMIT 1", kind)
    return get(conn, j["id"]) if j else None


def child_env(kind: str) -> dict[str, str]:
    if kind in LLM_KINDS:
        return {k: os.environ[k] for k in ENV_ALLOW if os.environ.get(k)}
    return {k: v for k, v in os.environ.items() if k != "CLAUDECODE" and not k.startswith("CLAUDE_CODE_")}


def start(conn: sqlite3.Connection, s: Settings, kind: str, params: dict | None = None) -> int:
    if kind not in KINDS:
        raise ValueError(f"unknown job kind {kind}")
    with db.tx(conn):
        busy = db.one(conn, "SELECT id, kind FROM jobs WHERE state IN ('queued', 'running')")
        if busy:
            raise Busy(f"Job {busy['id']} ({busy['kind']}) is still running")
        cur = conn.execute("INSERT INTO jobs(kind, state, params, created_at) VALUES (?, 'queued', ?, ?)",
                           (kind, json.dumps(params or {}), now()))
        job_id = int(cur.lastrowid)
    log = s.path("logs", f"job-{job_id}-{kind}.log")
    settings_arg = json.dumps({"data_dir": str(s.data_dir), "fixtures_dir": str(s.fixtures_dir),
                               "base_model": s.base_model, "audit_model": s.audit_model, "device": s.device,
                               "seed": s.seed, "eval_limit": s.eval_limit})
    with open(log, "w") as fh:
        proc = subprocess.Popen([sys.executable, "-m", "coa.worker", str(job_id), settings_arg], cwd=str(ROOT),
                                env=child_env(kind), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    conn.execute("UPDATE jobs SET pid=? WHERE id=?", (proc.pid, job_id))
    return job_id


def log_path(s: Settings, job: dict) -> Path:
    return s.data_dir / "logs" / f"job-{job['id']}-{job['kind']}.log"
