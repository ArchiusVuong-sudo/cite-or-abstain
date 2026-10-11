"""Held-out evaluation: base, base with a 2-shot prompt, and the tuned adapter, on clean and typo slices."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable

from . import db
from .checker import Check, Label, check
from .dataset import QuestionView, question_views
from .prompts import messages, two_shot_messages

SYSTEMS = ["base", "two_shot", "tuned"]
SYSTEM_LABELS = {"base": "Base model", "two_shot": "Base + 2-shot prompt", "tuned": "Tuned (DPO + LoRA)"}
SLICES = ["clean", "typo"]
GATE_INVENTED_DROP = 0.15
GATE_FALSE_REFUSAL_RISE = 0.10


@dataclass(frozen=True)
class Rate:
    k: int
    n: int
    rate: float | None
    lo: float | None
    hi: float | None


def wilson(k: int, n: int, z: float = 1.959964) -> Rate:
    """Wilson score interval; stays inside [0, 1] and behaves at k=0 or k=n."""
    if n == 0:
        return Rate(k, n, None, None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Rate(k, n, p, max(0.0, centre - half), min(1.0, centre + half))


def score(results: list[tuple[QuestionView, Check]]) -> dict[str, Rate]:
    undocumented = [c for q, c in results if not q.gold.answerable]
    documented = [c for q, c in results if q.gold.answerable]
    valued = [c for _, c in results if c.label in {Label.CORRECT, Label.WRONG_VALUE, Label.INVENTED}]
    return {
        "invented_rate": wilson(sum(c.label == Label.INVENTED for c in undocumented), len(undocumented)),
        "accuracy": wilson(sum(c.label == Label.CORRECT for c in documented), len(documented)),
        "false_refusal_rate": wilson(sum(c.label == Label.FALSE_REFUSAL for c in documented), len(documented)),
        "citation_validity": wilson(sum(bool(c.citation_valid) for c in valued), len(valued)),
    }


def gate(metrics: dict[str, dict[str, dict]]) -> dict:
    """Build gate on the clean slice: invented rate down 15+ points vs both baselines, refusals up 10 at most."""
    def r(system: str, key: str) -> float | None:
        return metrics.get(system, {}).get(key, {}).get("rate")

    t_inv, b_inv, s_inv = r("tuned", "invented_rate"), r("base", "invented_rate"), r("two_shot", "invented_rate")
    t_fr, b_fr = r("tuned", "false_refusal_rate"), r("base", "false_refusal_rate")
    if None in (t_inv, b_inv, s_inv, t_fr, b_fr):
        return {"evaluated": False}
    inv_ok = t_inv <= min(b_inv, s_inv) - GATE_INVENTED_DROP
    fr_ok = t_fr <= b_fr + GATE_FALSE_REFUSAL_RISE
    return {"evaluated": True, "passed": inv_ok and fr_ok, "invented_ok": inv_ok, "false_refusal_ok": fr_ok,
            "invented_drop_vs_best_baseline": round(min(b_inv, s_inv) - t_inv, 4),
            "false_refusal_rise_vs_base": round(t_fr - b_fr, 4)}


def run(conn: sqlite3.Connection, model_id: str, adapter_dir: str | None, run_id: int | None, split: str,
        device: str, limit: int, progress: Callable[[int, int, str], None]) -> dict:
    from . import modeling

    qs = question_views(conn, split)
    if limit:
        qs = qs[:limit]
    plan = [(s, sl) for s in SYSTEMS if s != "tuned" or adapter_dir for sl in SLICES]
    total = len(plan) * len(qs)
    done = 0
    answers: dict[tuple[str, str], list[str]] = {}
    progress(0, total, f"Loading {model_id}")
    lm = modeling.load(model_id, device)
    for system, sl in plan:
        if system == "tuned" and not getattr(lm, "_has_adapter", False):
            from peft import PeftModel

            lm.model = PeftModel.from_pretrained(lm.model, adapter_dir).to(lm.device).eval()
            lm._has_adapter = True  # type: ignore[attr-defined]
        build = two_shot_messages if system == "two_shot" else messages
        convs = [build(q.text, q.prompt_passages(typo=sl == "typo")) for q in qs]
        label = f"{SYSTEM_LABELS[system]}, {sl} slice"
        base_done = done
        answers[(system, sl)] = [a[0] for a in modeling.generate(
            lm, convs, on_batch=lambda d, t, b=base_done, lab=label: progress(b + d, total, lab))]
        done += len(qs)

    metrics: dict[str, dict] = {sl: {} for sl in SLICES}
    with db.tx(conn):
        cur = conn.execute("INSERT INTO evals(run_id, split, created_at, summary) VALUES (?,?,?,?)",
                           (run_id, split, datetime.now(timezone.utc).isoformat(timespec="seconds"), "{}"))
        eval_id = cur.lastrowid
        for (system, sl), outs in answers.items():
            results = [(q, check(a, q.gold)) for q, a in zip(qs, outs)]
            metrics[sl][system] = {k: asdict(v) for k, v in score(results).items()}
            for (q, c), a in zip(results, outs):
                conn.execute("INSERT INTO eval_answers VALUES (?,?,?,?,?,?,?)",
                             (eval_id, q.id, system, sl, a, c.label.value,
                              None if c.citation_valid is None else int(c.citation_valid)))
        summary = {"metrics": metrics, "gate": gate(metrics["clean"]), "questions": len(qs), "split": split,
                   "model": model_id, "adapter": adapter_dir,
                   "n_undocumented": sum(not q.gold.answerable for q in qs),
                   "n_documented": sum(q.gold.answerable for q in qs)}
        conn.execute("UPDATE evals SET summary=? WHERE id=?", (json.dumps(summary), eval_id))
    progress(total, total, "Done")
    return {"eval_id": eval_id, **summary}


GOOD = {Label.CORRECT.value, Label.CORRECT_ABSTAIN.value}


def diff(conn: sqlite3.Connection, eval_id: int, sl: str = "clean", baseline: str = "base") -> list[dict]:
    """Questions whose label changed between a baseline and the tuned model."""
    by: dict[str, dict[str, dict]] = {}
    for r in db.rows(conn, "SELECT * FROM eval_answers WHERE eval_id=? AND slice=? AND system IN (?, 'tuned')",
                     eval_id, sl, baseline):
        by.setdefault(r["question_id"], {})[r["system"]] = r
    out = []
    for qid, d in sorted(by.items()):
        if baseline not in d or "tuned" not in d or d[baseline]["label"] == d["tuned"]["label"]:
            continue
        b_good, t_good = d[baseline]["label"] in GOOD, d["tuned"]["label"] in GOOD
        kind = "improved" if t_good and not b_good else "regressed" if b_good and not t_good else "changed"
        out.append({"question_id": qid, "kind": kind, "before": d[baseline], "after": d["tuned"]})
    return out
