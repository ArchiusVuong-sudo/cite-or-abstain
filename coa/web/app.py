"""FastAPI app: four pages (Data & Pairs, Review, Train, Evaluate & Export), HTMX partials and SSE job streams."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from .. import audit, axolotl_export, dataset, db, evaluate, jobs, llm, pairs
from ..checker import value_spans
from ..config import ABSTAIN, settings
from ..fields import FIELDS
from ..retrieval import QUERY_HINTS, TOP_K
from ..runconfig import RunConfig
from ..synth import STAMP

HERE = Path(__file__).resolve().parent
LABEL_NAMES = {"correct": "Correct", "correct_abstain": "Correct abstain", "invented": "Invented",
               "wrong_value": "Wrong value", "false_refusal": "False refusal", "non_answer": "Non-answer"}
TRAP_NAMES = {"family": "Family history", "pending": "Pending result", "nos": "Not subtyped",
              "not_started": "Not started", "not_performed": "Not performed"}


def run_config(s) -> RunConfig:
    return RunConfig(base_model=s.base_model, max_steps=s.train_max_steps, seed=s.seed)


def provider_mode(s) -> str:
    return "replay" if s.replay_forced or not llm.has_login() else "live"


def create_app() -> FastAPI:
    s = settings()
    app = FastAPI(title="Cite or Abstain")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    tpl = Jinja2Templates(directory=HERE / "templates")
    tpl.env.filters["label"] = lambda v: LABEL_NAMES.get(v, v or "")
    tpl.env.filters["trap"] = lambda v: TRAP_NAMES.get(v, v or "")
    tpl.env.filters["pct"] = lambda v: "n/a" if v is None else f"{v * 100:.1f}%"
    tpl.env.filters["size"] = _size
    tpl.env.filters["fromjson"] = json.loads
    app.state.mode = provider_mode(s)

    def conn():
        return db.connect(s.db_path)

    c0 = conn()
    if not db.get_meta(c0, "seed"):
        dataset.seed(c0, s.seed)
    jobs.reap_stale(c0)
    c0.close()

    def page(request: Request, name: str, ctx: dict[str, Any]) -> HTMLResponse:
        c = conn()
        active = jobs.active(c)
        base = {"request": request, "stamp": STAMP, "mode": app.state.mode, "active_job": active,
                "nav": name, "audit_model": s.audit_model, "base_model": s.base_model}
        return tpl.TemplateResponse(request, f"{name}.html", {**base, **ctx})

    # ---------- pages ----------

    @app.get("/", response_class=HTMLResponse)
    def data_page(request: Request, label: str = "all", patient: str | None = None):
        c = conn()
        summ = dataset.summary(c)
        export = axolotl_export.export_rows(c) if _has_pairs(c) else None
        all_items = [(r["patient_id"], r["scrubbed_text"]) for r in db.rows(c, "SELECT patient_id, scrubbed_text "
                                                                                 "FROM notes")]
        leaks_notes = dataset.leak_gate(c, all_items)
        patients = db.rows(c, "SELECT id, split, cancer FROM patients ORDER BY id")
        pick = patient or (patients[0]["id"] if patients else None)
        samp = db.one(c, "SELECT COUNT(*) AS n, MIN(source) AS source, MIN(model) AS model FROM samples")
        return page(request, "data", {
            "summary": summ, "patients": patients, "patient": pick,
            "leak": {"export_rows": len(export.rows) if export else 0, "export_leaks": len(export.leaks) if export
                     else 0, "note_leaks": len(leaks_notes), "notes": len(all_items)},
            "hints": QUERY_HINTS, "top_k": TOP_K, "mix": pairs.mix(c), "label": label, "samples": samp,
            "sample_job": jobs.latest(c, "sample"),
        })

    @app.get("/review", response_class=HTMLResponse)
    def review_page(request: Request):
        c = conn()
        traces = db.rows(c, "SELECT id, batch, model, source, latency_ms, turns, input_tokens, output_tokens, "
                            "cost_usd, attempts, error FROM llm_traces ORDER BY batch")
        stats = db.one(c, "SELECT COUNT(*) AS audited, SUM(verdict='dispute') AS disputed, "
                          "SUM(verdict='agree') AS agreed FROM audits") or {}
        return page(request, "review", {"queue": _queue(c), "traces": traces, "stats": stats,
                                        "counts": _review_counts(c), "has_pairs": _has_pairs(c),
                                        "audit_job": jobs.latest(c, "audit"),
                                        "totals": _trace_totals(traces)})

    @app.get("/train", response_class=HTMLResponse)
    def train_page(request: Request):
        c = conn()
        cfg = run_config(s)
        ax = jobs.latest(c, "axolotl")
        ex = axolotl_export.export_rows(c) if _has_pairs(c) else None
        check = axolotl_export.cached_check(c, ex.dataset_hash) if ex else None
        yaml_text = None
        if ex:
            p = s.data_dir / "export" / ex.dataset_hash / "axolotl_dpo.yml"
            yaml_text = p.read_text() if p.exists() else None
        runs = db.rows(c, "SELECT * FROM train_runs ORDER BY id DESC LIMIT 5")
        for r in runs:
            r["cfg"] = json.loads(r["config"])
        return page(request, "train", {"cfg": cfg, "mapping": cfg.mapping(), "export": ex, "check": check,
                                       "yaml": yaml_text, "ax_job": ax, "runs": runs,
                                       "preview_yaml": yaml.safe_dump(cfg.axolotl("var/export/<hash>/dpo_train.jsonl",
                                                                                  "var/export/<hash>/axolotl-out",
                                                                                  "var/export/<hash>/prepared"),
                                                                      sort_keys=False),
                                       "train_job": jobs.latest(c, "train")})

    @app.get("/evaluate", response_class=HTMLResponse)
    def evaluate_page(request: Request, slice: str = "clean", kind: str = "all"):  # noqa: A002
        c = conn()
        run = db.one(c, "SELECT * FROM train_runs WHERE status='done' ORDER BY id DESC LIMIT 1")
        ev = db.one(c, "SELECT * FROM evals ORDER BY id DESC LIMIT 1")
        summary = json.loads(ev["summary"]) if ev else None
        rows = evaluate.diff(c, ev["id"], slice) if ev and summary and "tuned" in summary["metrics"][slice] else []
        counts = {"improved": sum(r["kind"] == "improved" for r in rows),
                  "regressed": sum(r["kind"] == "regressed" for r in rows),
                  "changed": sum(r["kind"] == "changed" for r in rows)}
        if kind != "all":
            rows = [r for r in rows if r["kind"] == kind]
        qs = {q.id: q for q in dataset.question_views(c, summary["split"])} if summary else {}
        exports = db.rows(c, "SELECT * FROM exports WHERE run_id=? ORDER BY id", run["id"]) if run else []
        card = None
        for e in exports:
            if e["kind"] == "model_card" and e["path"] and Path(e["path"]).exists():
                card = Path(e["path"]).read_text()
        return page(request, "evaluate", {
            "run": run, "ev": ev, "summary": summary, "slice": slice, "kind": kind, "diff": rows[:60],
            "diff_total": len(rows), "counts": counts, "qs": qs, "exports": exports, "card": card,
            "systems": evaluate.SYSTEMS, "system_labels": evaluate.SYSTEM_LABELS,
            "rate_keys": [("invented_rate", "Invented-answer rate", "undocumented fields, lower is better"),
                          ("accuracy", "Accuracy", "documented fields, higher is better"),
                          ("false_refusal_rate", "False-refusal rate", "documented fields, lower is better"),
                          ("citation_validity", "Citation validity", "answers that state a value, higher is better")],
            "eval_job": jobs.latest(c, "eval"), "export_job": jobs.latest(c, "export"),
        })

    # ---------- partials ----------

    @app.get("/partials/note", response_class=HTMLResponse)
    def note_partial(request: Request, patient: str, view: str = "scrubbed"):
        c = conn()
        notes = db.rows(c, "SELECT * FROM notes WHERE patient_id=? ORDER BY id", patient)
        if not notes:
            raise HTTPException(404, "unknown patient")
        phi = db.rows(c, "SELECT kind, original, surrogate FROM phi WHERE patient_id=?", patient)
        p = db.one(c, "SELECT * FROM patients WHERE id=?", patient)
        return tpl.TemplateResponse(request, "partials/note.html", {
            "notes": [{**n, "html": _mark_phi(n["raw_text"] if view == "raw" else n["scrubbed_text"], phi, view)}
                      for n in notes], "view": view, "patient": p, "phi_count": len(phi)})

    @app.get("/partials/pairs", response_class=HTMLResponse)
    def pairs_partial(request: Request, label: str = "all"):
        c = conn()
        sql = ("SELECT p.*, q.field, q.trap, q.text AS question, q.answerable FROM pairs p "
               "JOIN questions q ON q.id = p.question_id")
        args: list[Any] = []
        if label != "all":
            sql += " WHERE p.rejected_label = ?"
            args.append(label)
        rows = db.rows(c, sql + " ORDER BY (q.trap IS NULL), p.id", *args)
        return tpl.TemplateResponse(request, "partials/pairs.html", {"rows": rows, "label": label,
                                                                     "mix": pairs.mix(c)})

    @app.get("/partials/pair/{pair_id}", response_class=HTMLResponse)
    def pair_partial(request: Request, pair_id: int):
        c = conn()
        return tpl.TemplateResponse(request, "partials/pair.html", _pair_ctx(c, pair_id))

    @app.get("/partials/trace/{trace_id}", response_class=HTMLResponse)
    def trace_partial(request: Request, trace_id: int):
        c = conn()
        t = db.one(c, "SELECT * FROM llm_traces WHERE id=?", trace_id)
        if not t:
            raise HTTPException(404, "unknown trace")
        t["calls"] = json.loads(t["tool_calls"])
        return tpl.TemplateResponse(request, "partials/trace.html", {"t": t})

    @app.get("/partials/review-item/{pair_id}", response_class=HTMLResponse)
    def review_item(request: Request, pair_id: int):
        c = conn()
        return tpl.TemplateResponse(request, "partials/review_item.html", {"item": _queue_item(c, pair_id),
                                                                           **_pair_ctx(c, pair_id)})

    # ---------- actions ----------

    @app.post("/api/review/{pair_id}/{action}")
    def review_action(request: Request, pair_id: int, action: str):
        if action not in {"approve", "flip", "drop"}:
            raise HTTPException(400, "action must be approve, flip or drop")
        c = conn()
        p = db.one(c, "SELECT * FROM pairs WHERE id=?", pair_id)
        if not p:
            raise HTTPException(404, "unknown pair")
        with db.tx(c):
            c.execute("INSERT INTO reviews(pair_id, action, at) VALUES (?,?,?)",
                      (pair_id, action, datetime.now(timezone.utc).isoformat(timespec="seconds")))
            c.execute("UPDATE pairs SET status=?, flipped=? WHERE id=?",
                      ("dropped" if action == "drop" else "active", int(action == "flip"), pair_id))
        if request.headers.get("hx-request"):
            return tpl.TemplateResponse(request, "partials/review_item.html", {"item": _queue_item(c, pair_id),
                                                                               **_pair_ctx(c, pair_id),
                                                                               "counts": _review_counts(c),
                                                                               "oob": True})
        return JSONResponse({"pair_id": pair_id, "action": action, "counts": _review_counts(c)})

    @app.post("/api/seed")
    def reseed():
        c = conn()
        if jobs.active(c):
            raise HTTPException(409, "A job is running")
        return dataset.seed(c, s.seed)

    @app.post("/api/jobs/{kind}")
    def start_job(kind: str):
        c = conn()
        params: dict[str, Any] = {}
        if kind == "sample":
            params = {"source": "recorded" if app.state.mode == "replay" else "live",
                      "record": app.state.mode == "live"}
        elif kind == "audit":
            if not _has_pairs(c):
                raise HTTPException(409, "Sample the base model first to build pairs")
            params = {"provider": app.state.mode, "record": True}
        elif kind in {"axolotl", "train"}:
            if not _has_pairs(c):
                raise HTTPException(409, "Sample the base model first to build pairs")
            params = {"config": run_config(s).model_dump()}
        elif kind == "eval":
            run = db.one(c, "SELECT id FROM train_runs WHERE status='done' ORDER BY id DESC LIMIT 1")
            params = {"run_id": run["id"] if run else None, "split": "test"}
        elif kind == "export":
            run = db.one(c, "SELECT id FROM train_runs WHERE status='done' ORDER BY id DESC LIMIT 1")
            if not run:
                raise HTTPException(409, "Train a model first")
            params = {"run_id": run["id"]}
        else:
            raise HTTPException(404, f"unknown job kind {kind}")
        try:
            job_id = jobs.start(c, s, kind, params)
        except jobs.Busy as e:
            raise HTTPException(409, str(e)) from e
        return jobs.get(c, job_id)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: int):
        j = jobs.get(conn(), job_id)
        if not j:
            raise HTTPException(404, "unknown job")
        return j

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: int):
        c = conn()
        j = jobs.get(c, job_id)
        if not j:
            raise HTTPException(404, "unknown job")
        p = jobs.log_path(s, j)
        lines = p.read_text(errors="replace").splitlines()[-40:] if p.exists() else []
        return {"lines": [ln[:300] for ln in lines if "Warning" not in ln]}

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: int):
        async def stream():
            c = conn()
            last, metric_step = None, -1
            while True:
                j = jobs.get(c, job_id)
                if not j:
                    yield _sse("end", {"state": "error", "error": "unknown job"})
                    return
                snap = {k: j[k] for k in ("id", "kind", "state", "progress", "total", "message", "error")}
                if j["kind"] == "train":
                    run = db.one(c, "SELECT id FROM train_runs WHERE job_id=?", job_id)
                    if run:
                        new = db.rows(c, "SELECT step, loss, margin, accuracy FROM train_metrics WHERE run_id=? "
                                         "AND step>? ORDER BY step", run["id"], metric_step)
                        if new:
                            metric_step = new[-1]["step"]
                            snap["metrics"] = new
                if snap != last:
                    yield _sse("job", snap)
                    last = {k: v for k, v in snap.items() if k != "metrics"}
                if j["state"] in jobs.TERMINAL:
                    yield _sse("end", {"state": j["state"], "error": j["error"], "result": j["result"]})
                    return
                await asyncio.sleep(0.7)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/train/{run_id}/metrics")
    def train_metrics(run_id: int):
        return db.rows(conn(), "SELECT step, loss, margin, accuracy FROM train_metrics WHERE run_id=? ORDER BY step",
                       run_id)

    @app.get("/api/summary")
    def api_summary():
        c = conn()
        ev = db.one(c, "SELECT summary FROM evals ORDER BY id DESC LIMIT 1")
        return {"mode": app.state.mode, "data": dataset.summary(c), "pairs": pairs.mix(c),
                "review": _review_counts(c), "eval": json.loads(ev["summary"]) if ev else None}

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "mode": app.state.mode}

    # ---------- helpers bound to templates ----------

    def _queue(c) -> list[dict]:
        return audit.queue(c)

    def _queue_item(c, pair_id: int) -> dict | None:
        return next((q for q in audit.queue(c) if q["pair_id"] == pair_id), None)

    return app


def _has_pairs(c) -> bool:
    return bool(db.one(c, "SELECT 1 AS x FROM pairs LIMIT 1"))


def _review_counts(c) -> dict:
    q = audit.queue(c)
    return {"flagged": len(q), "pending": sum(1 for x in q if not x["decision"]),
            "approve": sum(1 for x in q if x["decision"] == "approve"),
            "flip": sum(1 for x in q if x["decision"] == "flip"),
            "drop": sum(1 for x in q if x["decision"] == "drop"),
            "active_pairs": (db.one(c, "SELECT COUNT(*) AS n FROM pairs WHERE status='active'") or {}).get("n", 0)}


def _trace_totals(traces: list[dict]) -> dict:
    return {"calls": len(traces), "latency_ms": sum(t["latency_ms"] or 0 for t in traces),
            "input_tokens": sum(t["input_tokens"] or 0 for t in traces),
            "output_tokens": sum(t["output_tokens"] or 0 for t in traces),
            "cost_usd": round(sum(t["cost_usd"] or 0 for t in traces), 4),
            "source": traces[0]["source"] if traces else None}


def _pair_ctx(c, pair_id: int) -> dict:
    p = db.one(c, "SELECT * FROM pairs WHERE id=?", pair_id)
    if not p:
        raise HTTPException(404, "unknown pair")
    q = dataset.question_views(c, ids=[p["question_id"]])[0]
    a = db.one(c, "SELECT * FROM audits WHERE pair_id=? ORDER BY id DESC LIMIT 1", pair_id)
    samples = db.rows(c, "SELECT mode, text, label, citation_valid FROM samples WHERE question_id=? ORDER BY id",
                      q.id)
    passages = [{"n": i + 1, "label": lab, "html": _mark(text, value_spans(q.field, text)),
                 "evidence": (i + 1) in q.gold.evidence} for i, (_, lab, text) in enumerate(q.passages)]
    return {"p": p, "q": q, "audit": a, "samples": samples, "passages": passages,
            "field": FIELDS[q.field], "abstain": ABSTAIN,
            "chosen_html": _mark(p["chosen"], value_spans(q.field, p["chosen"])),
            "rejected_html": _mark(p["rejected"], value_spans(q.field, p["rejected"]))}


def _mark(text: str, spans: list[tuple[int, int]]) -> Markup:
    out, last = [], 0
    for a, b in spans:
        out.append(escape(text[last:a]))
        out.append(f"<mark>{escape(text[a:b])}</mark>")
        last = b
    out.append(escape(text[last:]))
    return Markup("".join(out))


def _mark_phi(text: str, phi: list[dict], view: str) -> Markup:
    terms = sorted({(r["original"] if view == "raw" else r["surrogate"]) for r in phi}, key=len, reverse=True)
    html = escape(text)
    if terms:
        pat = re.compile("|".join(rf"(?<![\w]){re.escape(escape(t))}(?![\w])" for t in terms))
        html = pat.sub(lambda m: f'<span class="phi phi-{view}">{m.group(0)}</span>', html)
    return Markup(html)


def _size(n: int | None) -> str:
    if n is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"

