"""Export: merge the LoRA adapter into safetensors (reload check), write a model card, convert to GGUF Q8_0."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import db
from .dataset import question_views
from .evaluate import SYSTEM_LABELS
from .prompts import messages
from .synth import STAMP

RATE_LABELS = {"invented_rate": "Invented-answer rate (undocumented fields)", "accuracy": "Accuracy (documented)",
               "false_refusal_rate": "False-refusal rate (documented)", "citation_validity": "Citation validity"}


def converter_path(data_dir: Path) -> Path:
    return data_dir / "llama.cpp" / "convert_hf_to_gguf.py"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


def _record(conn: sqlite3.Connection, run_id: int, kind: str, status: str, path: Path | None, note: str) -> dict:
    size = dir_size(path) if path and path.exists() else None
    conn.execute("INSERT INTO exports(run_id, kind, status, path, size_bytes, note, created_at) VALUES (?,?,?,?,?,?,?)",
                 (run_id, kind, status, str(path) if path else None, size, note,
                  datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return {"kind": kind, "status": status, "path": str(path) if path else None, "size_bytes": size, "note": note}


def model_card(base_model: str, run: dict, ev: dict | None) -> str:
    lines = [
        "# Registry abstraction adapter (cite or abstain)", "",
        f"Base model: `{base_model}`, LoRA adapter trained with DPO (TRL DPOTrainer), merged into the weights.",
        "Task: answer one cancer registry field from 3 retrieved note passages, citing the passage id, or answer "
        "\"Not documented in the provided notes.\"", "",
        "## Training", "",
        f"- Preference pairs: {run.get('pairs', '?')} (dataset hash `{run.get('dataset_hash', '?')}`)",
        f"- Steps: {run.get('steps', '?')}, wall time {run.get('seconds', '?')} s on {run.get('device', '?')}",
        "",
    ]
    if ev:
        m = ev["metrics"]["clean"]
        systems = [s for s in ("base", "two_shot", "tuned") if s in m]
        lines += [f"## Held-out {ev['split']} split ({ev['questions']} questions), clean slice", "",
                  "| Rate | " + " | ".join(SYSTEM_LABELS[s] for s in systems) + " |",
                  "|---|" + "---|" * len(systems)]
        for key, label in RATE_LABELS.items():
            cells = []
            for s in systems:
                r = m[s][key]
                cells.append("n/a" if r["rate"] is None else
                             f"{r['rate'] * 100:.1f}% ({r['k']}/{r['n']}, 95% CI {r['lo'] * 100:.1f}-{r['hi'] * 100:.1f})")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines += ["", f"Build gate: {'passed' if ev['gate'].get('passed') else 'not passed'}.", ""]
    lines += [
        "## Data card", "",
        f"- All training and evaluation records are synthetic ({STAMP}). Made by a seeded generator, no LLM.",
        "- 80 invented patients (40 breast, 40 lung), 3 notes each, split by patient into train, dev and test.",
        "- Names, MRNs, phone numbers and dates were replaced with consistent surrogates; the leak gate found 0 "
        "original identifiers in the exported training file.",
        "- Not for clinical use. A research and engineering demo only.", "",
    ]
    return "\n".join(lines)


def run(conn: sqlite3.Connection, base_model: str, run_id: int, data_dir: Path, device: str,
        progress: Callable[[int, int, str], None]) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import modeling

    tr = db.one(conn, "SELECT * FROM train_runs WHERE id=?", run_id)
    if not tr or not tr["adapter_dir"]:
        raise ValueError("No finished training run to export")
    out = data_dir / "artifacts" / f"run-{run_id}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    conn.execute("DELETE FROM exports WHERE run_id=?", (run_id,))
    results = []

    progress(0, 4, "Merging the adapter into the base weights")
    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, tr["adapter_dir"]).merge_and_unload()
    merged_dir = out / "merged-safetensors"
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tok.save_pretrained(str(merged_dir))
    del merged, base

    progress(1, 4, "Reloading the merged model and answering one test question")
    q = question_views(conn, "test")[0]
    lm = modeling.load(str(merged_dir), device)
    answer = modeling.generate(lm, [messages(q.text, q.prompt_passages())])[0][0]
    del lm
    note = f"Reloaded; {q.id}: {answer}"
    results.append(_record(conn, run_id, "safetensors", "done", merged_dir, note))

    progress(2, 4, "Writing the model card")
    ev_row = db.one(conn, "SELECT summary FROM evals WHERE run_id=? ORDER BY id DESC LIMIT 1", run_id)
    ev = json.loads(ev_row["summary"]) if ev_row else None
    run_info = json.loads(tr["config"]).get("result", {})
    card = model_card(base_model, run_info, ev)
    (merged_dir / "README.md").write_text(card)
    results.append(_record(conn, run_id, "model_card", "done", merged_dir / "README.md", "Model card with test table"))

    progress(3, 4, "Converting to GGUF Q8_0")
    conv = converter_path(data_dir)
    gguf = out / "model-q8_0.gguf"
    if not conv.exists():
        results.append(_record(conn, run_id, "gguf", "skipped", None,
                               "llama.cpp converter not fetched (run scripts/fetch_converter.sh)"))
    else:
        # Not run with python -I: the converter imports its sibling `conversion` package.
        proc = subprocess.run([sys.executable, str(conv), str(merged_dir), "--outtype", "q8_0", "--outfile",
                               str(gguf)], capture_output=True, text=True, timeout=900, cwd=str(conv.parent))
        if proc.returncode == 0 and gguf.exists():
            results.append(_record(conn, run_id, "gguf", "done", gguf, "Q8_0 via llama.cpp convert_hf_to_gguf.py"))
        else:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
            results.append(_record(conn, run_id, "gguf", "skipped", None, f"converter failed: {tail[0][:300]}"))
    progress(4, 4, "Done")
    return {"artifacts": results, "answer": answer, "question_id": q.id}
