"""Worker process for one job: `python -m coa.worker <job_id> <settings-json>`."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from . import db
from .jobs import now


def main(job_id: int, cfg: dict) -> int:
    data_dir, fixtures_dir = Path(cfg["data_dir"]), Path(cfg["fixtures_dir"])
    conn = db.connect(data_dir / "coa.sqlite3")
    job = db.one(conn, "SELECT * FROM jobs WHERE id=?", job_id)
    if job is None:
        return 2
    params = json.loads(job["params"])
    import os

    conn.execute("UPDATE jobs SET state='running', started_at=?, pid=? WHERE id=?", (now(), os.getpid(), job_id))

    def progress(done: int, total: int, message: str) -> None:
        conn.execute("UPDATE jobs SET progress=?, total=?, message=? WHERE id=?", (done, total, message, job_id))

    try:
        result = run(job["kind"], params, cfg, conn, data_dir, fixtures_dir, job_id, progress)
        conn.execute("UPDATE jobs SET state='done', result=?, finished_at=?, message='Done' WHERE id=?",
                     (json.dumps(result), now(), job_id))
        return 0
    except Exception as e:  # the job records the failure; the web app shows it
        traceback.print_exc()
        conn.execute("UPDATE jobs SET state='error', error=?, finished_at=? WHERE id=?",
                     (f"{type(e).__name__}: {e}"[:2000], now(), job_id))
        return 1


def run(kind: str, params: dict, cfg: dict, conn, data_dir: Path, fixtures_dir: Path, job_id: int, progress):
    if kind == "sample":
        from . import pairs, sampling

        if params.get("source") == "recorded":
            res = sampling.run_recorded(conn, fixtures_dir, progress)
        else:
            res = sampling.run_live(conn, cfg["base_model"], cfg["device"], cfg["seed"], progress)
            if params.get("record"):
                sampling.record_fixture(conn, fixtures_dir)
        res["pairs"] = pairs.build_all(conn)
        return res
    if kind == "audit":
        from . import audit
        from .llm import LiveProvider, ReplayProvider

        provider = ReplayProvider(fixtures_dir) if params.get("provider") == "replay" else \
            LiveProvider(cfg["audit_model"])
        return audit.run(conn, provider, fixtures_dir, bool(params.get("record", True)), progress)
    if kind == "axolotl":
        from . import axolotl_export
        from .runconfig import RunConfig

        progress(0, 2, "Writing the Axolotl DPO config and JSONL")
        out = axolotl_export.write(conn, RunConfig(**params["config"]), data_dir)
        progress(1, 2, "Running axolotl preprocess")
        check = axolotl_export.preprocess(conn, out["dataset_hash"], Path(out["config_path"]))
        progress(2, 2, "Done")
        return {**out, "preprocess": {"passed": bool(check["passed"]), "cached": check["cached"]}}
    if kind == "train":
        from . import train
        from .runconfig import RunConfig

        rc = RunConfig(**params["config"])
        cur = conn.execute("INSERT INTO train_runs(job_id, dataset_hash, config, status, created_at) "
                           "VALUES (?,?,?,?,?)", (job_id, "", json.dumps({"config": rc.model_dump()}), "running",
                                                  now()))
        run_id = int(cur.lastrowid)
        try:
            res = train.run(conn, rc, data_dir / "runs" / f"run-{run_id}", cfg["device"], run_id, progress)
        except Exception:
            conn.execute("UPDATE train_runs SET status='error' WHERE id=?", (run_id,))
            raise
        conn.execute("UPDATE train_runs SET status='done', adapter_dir=?, dataset_hash=?, seconds=?, config=? "
                     "WHERE id=?", (res["adapter_dir"], res["dataset_hash"], res["seconds"],
                                    json.dumps({"config": rc.model_dump(), "result": res}), run_id))
        return {**res, "run_id": run_id}
    if kind == "eval":
        from . import evaluate

        run_id = params.get("run_id")
        tr = db.one(conn, "SELECT adapter_dir FROM train_runs WHERE id=?", run_id) if run_id else None
        return evaluate.run(conn, cfg["base_model"], tr["adapter_dir"] if tr else None, run_id,
                            params.get("split", "test"), cfg["device"], int(cfg.get("eval_limit") or 0), progress)
    if kind == "export":
        from . import export_model

        return export_model.run(conn, cfg["base_model"], int(params["run_id"]), data_dir, cfg["device"], progress)
    raise ValueError(f"unknown job kind {kind}")


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]), json.loads(sys.argv[2])))
