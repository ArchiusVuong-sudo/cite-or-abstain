"""Model tests on the tiny test model (CPU, seconds). Real 0.5B runs are `make test-slow`."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from coa import db
from coa.config import TINY_MODEL
from coa.runconfig import RunConfig

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def trained(with_samples, data_dir):
    from coa import train

    cfg = RunConfig(base_model=TINY_MODEL, max_steps=2, micro_batch_size=2, gradient_accumulation_steps=1,
                    sequence_len=512)
    with_samples.execute("INSERT INTO train_runs(id, dataset_hash, config, status, created_at) "
                         "VALUES (1, '', '{}', 'running', 'x')")
    res = train.run(with_samples, cfg, data_dir / "runs" / "run-1", "cpu", 1, lambda *a: None)
    with_samples.execute("UPDATE train_runs SET status='done', adapter_dir=?, config=? WHERE id=1",
                         (res["adapter_dir"], json.dumps({"result": res})))
    return res


def test_trl_smoke(trained, with_samples):
    assert trained["steps"] == 2 and Path(trained["adapter_dir"], "adapter_config.json").exists()
    m = db.rows(with_samples, "SELECT * FROM train_metrics WHERE run_id=1 ORDER BY step")
    assert [r["step"] for r in m] == [1, 2] and all(r["loss"] is not None and r["margin"] is not None for r in m)


def test_export_roundtrip(trained, with_samples, data_dir, monkeypatch):
    from coa import export_model

    monkeypatch.setenv("LLAMA_CPP_DIR", str(data_dir / "no-converter"))
    monkeypatch.setattr(export_model, "ROOT", data_dir)  # do not pick up a fetched converter in tests
    res = export_model.run(with_samples, TINY_MODEL, 1, data_dir, "cpu", lambda *a: None)
    kinds = {a["kind"]: a for a in res["artifacts"]}
    st = Path(kinds["safetensors"]["path"])
    assert any(st.glob("*.safetensors")) and kinds["safetensors"]["size_bytes"] > 0
    assert res["question_id"] and isinstance(res["answer"], str)
    assert "Synthetic" in (st / "README.md").read_text() or "synthetic" in (st / "README.md").read_text()
    assert kinds["gguf"]["status"] == "skipped" and "not fetched" in kinds["gguf"]["note"]


def test_eval_tiny(trained, with_samples):
    from coa import evaluate

    out = evaluate.run(with_samples, TINY_MODEL, trained["adapter_dir"], 1, "test", "cpu", 6, lambda *a: None)
    assert set(out["metrics"]["clean"]) == {"base", "two_shot", "tuned"} and out["questions"] == 6
    n = db.one(with_samples, "SELECT COUNT(*) AS n FROM eval_answers WHERE eval_id=?", out["eval_id"])["n"]
    assert n == 6 * 3 * 2


@pytest.mark.slow
def test_axolotl_preprocess(with_samples, data_dir):
    from coa import axolotl_export

    if axolotl_export.axolotl_cli() is None:
        pytest.skip("axolotl extra not installed")
    out = axolotl_export.write(with_samples, RunConfig(base_model=TINY_MODEL), data_dir)
    res = axolotl_export.preprocess(with_samples, out["dataset_hash"], Path(out["config_path"]))
    assert res["passed"] == 1, res["log"][-2000:]
    again = axolotl_export.preprocess(with_samples, out["dataset_hash"], Path(out["config_path"]))
    assert again["cached"] is True


@pytest.mark.slow
def test_gguf_conversion(trained, with_samples, data_dir):
    from coa import export_model

    conv = ROOT / "var" / "llama.cpp" / "convert_hf_to_gguf.py"
    if not conv.exists():
        subprocess.run(["bash", str(ROOT / "scripts" / "fetch_converter.sh")], check=True)
    res = export_model.run(with_samples, TINY_MODEL, 1, data_dir, "cpu", lambda *a: None)
    g = [a for a in res["artifacts"] if a["kind"] == "gguf"][0]
    assert g["status"] == "done" and g["size_bytes"] > 0, g["note"]
