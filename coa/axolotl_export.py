"""Export the reviewed pairs as Axolotl DPO data plus config, and check them with `axolotl preprocess`."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import db
from .dataset import leak_gate, question_views
from .prompts import messages
from .runconfig import RunConfig


@dataclass(frozen=True)
class ExportedSet:
    rows: list[dict]
    dataset_hash: str
    leaks: list[str]


def export_rows(conn: sqlite3.Connection) -> ExportedSet:
    """Active pairs after review, with flips applied, in Axolotl chat_template.default form."""
    qs = {q.id: q for q in question_views(conn, "train")}
    rows: list[dict] = []
    items: list[tuple[str, str]] = []
    for p in db.rows(conn, "SELECT * FROM pairs WHERE status='active' ORDER BY id"):
        q = qs[p["question_id"]]
        chosen, rejected = (p["rejected"], p["chosen"]) if p["flipped"] else (p["chosen"], p["rejected"])
        row = {
            "messages": messages(q.text, q.prompt_passages()),
            "chosen": {"role": "assistant", "content": chosen},
            "rejected": {"role": "assistant", "content": rejected},
        }
        rows.append(row)
        items.append((q.patient_id, json.dumps(row)))
    blob = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
    return ExportedSet(rows, hashlib.sha256(blob.encode()).hexdigest()[:12], leak_gate(conn, items))


def write(conn: sqlite3.Connection, cfg: RunConfig, data_dir: Path) -> dict:
    ex = export_rows(conn)
    if ex.leaks:
        raise ValueError(f"Leak gate failed: {len(ex.leaks)} original identifiers in the export")
    out = data_dir / "export" / ex.dataset_hash
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "dpo_train.jsonl"
    data_path.write_text("".join(json.dumps(r) + "\n" for r in ex.rows))
    conf = cfg.axolotl(str(data_path), str(out / "axolotl-out"), str(out / "prepared"))
    conf_path = out / "axolotl_dpo.yml"
    conf_path.write_text(yaml.safe_dump(conf, sort_keys=False))
    return {"dataset_hash": ex.dataset_hash, "rows": len(ex.rows), "data_path": str(data_path),
            "config_path": str(conf_path), "leaks": 0, "yaml": conf_path.read_text()}


def cached_check(conn: sqlite3.Connection, dataset_hash: str) -> dict | None:
    return db.one(conn, "SELECT * FROM axolotl_checks WHERE dataset_hash=?", dataset_hash)


def axolotl_cli() -> Path | None:
    exe = Path(sys.executable).parent / "axolotl"
    return exe if exe.exists() else None


def preprocess(conn: sqlite3.Connection, dataset_hash: str, config_path: Path, timeout: int = 900) -> dict:
    """Run `axolotl preprocess` once per dataset hash; the result is cached."""
    hit = cached_check(conn, dataset_hash)
    if hit:
        return {**hit, "cached": True}
    cli = axolotl_cli()
    if cli is None:
        result = {"dataset_hash": dataset_hash, "passed": 0, "log": "axolotl is not installed (uv sync --extra "
                  "axolotl)", "at": _now()}
    else:
        env = dict(os.environ)
        # The CLI execs `accelerate` from PATH, so the venv's bin must come first.
        env["PATH"] = f"{cli.parent}{os.pathsep}{env.get('PATH', '')}"
        env.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        env.setdefault("WANDB_MODE", "disabled")
        env.setdefault("DO_NOT_TRACK", "1")
        prepared = config_path.parent / "prepared"
        if prepared.exists():
            shutil.rmtree(prepared)
        proc = subprocess.run([str(cli), "preprocess", str(config_path)], capture_output=True, text=True,
                              timeout=timeout, env=env, cwd=str(config_path.parent))
        log = (proc.stdout + "\n" + proc.stderr)[-6000:]
        result = {"dataset_hash": dataset_hash, "passed": int(proc.returncode == 0), "log": log, "at": _now()}
    conn.execute("INSERT OR REPLACE INTO axolotl_checks VALUES (?,?,?,?)",
                 (result["dataset_hash"], result["passed"], result["log"], result["at"]))
    return {**result, "cached": False}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
