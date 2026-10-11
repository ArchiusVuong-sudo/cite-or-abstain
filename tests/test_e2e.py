"""The whole main flow through the HTTP API in replay mode: real worker processes, tiny model, no network."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from coa.config import TINY_MODEL

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def client(data_dir, monkeypatch):
    for k, v in {"COA_DATA_DIR": str(data_dir), "DEMO_REPLAY": "1", "BASE_MODEL": TINY_MODEL,
                 "TRAIN_MAX_STEPS": "2", "EVAL_LIMIT": "6", "COA_DEVICE": "cpu",
                 "LLAMA_CPP_DIR": str(data_dir / "none")}.items():
        monkeypatch.setenv(k, v)
    from coa.web.app import create_app

    return TestClient(create_app())


def run_job(client, kind, timeout=300):
    r = client.post(f"/api/jobs/{kind}")
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["state"] in ("done", "error"):
            assert j["state"] == "done", j["error"]
            return j
        time.sleep(0.5)
    raise AssertionError(f"{kind} timed out")


def test_main_flow_replay(client):
    assert client.get("/healthz").json()["mode"] == "replay"
    s = run_job(client, "sample")["result"]
    assert s["source"] == "recorded" and s["pairs"]["total"] == 240

    a = run_job(client, "audit")["result"]
    assert a["source"] == "replay" and a["pairs"] == 24 and a["disputed"] >= 3
    page = client.get("/review").text
    assert "recorded" in page and "Review queue" in page

    summary = client.get("/api/summary").json()
    assert summary["review"]["flagged"] == a["disputed"]
    first = client.get("/review").text.split('hx-get="/partials/review-item/')[1].split('"')[0]
    assert client.post(f"/api/review/{first}/drop").json()["counts"]["active_pairs"] == 239

    t = run_job(client, "train")["result"]
    assert t["steps"] == 2 and t["pairs"] == 239
    e = run_job(client, "eval")["result"]
    assert e["questions"] == 6 and "tuned" in e["metrics"]["clean"]
    x = run_job(client, "export")["result"]
    assert {a["kind"] for a in x["artifacts"]} == {"safetensors", "model_card", "gguf"}

    for url in ["/", "/review", "/train", "/evaluate", "/evaluate?slice=typo&kind=improved"]:
        assert client.get(url).status_code == 200
    ev = client.get("/evaluate").text
    assert "Tuned (DPO + LoRA)" in ev and "95% CI" in ev


def test_job_stream_ends_with_terminal_event(client):
    jid = run_job(client, "sample")["id"]
    with client.stream("GET", f"/api/jobs/{jid}/events") as r:
        body = "".join(r.iter_text())
    assert "event: job" in body and body.rstrip().split("\n\n")[-1].startswith("event: end")


def test_second_job_is_refused_while_one_runs(client):
    r1 = client.post("/api/jobs/sample")
    r2 = client.post("/api/jobs/sample")
    assert r1.status_code == 200 and r2.status_code == 409
    jid = r1.json()["id"]
    while client.get(f"/api/jobs/{jid}").json()["state"] not in ("done", "error"):
        time.sleep(0.3)
