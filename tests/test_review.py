from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from coa import axolotl_export, db
from coa.llm import ReplayProvider

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def client(with_samples, data_dir, monkeypatch):
    from coa import audit

    audit.run(with_samples, ReplayProvider(ROOT / "fixtures"), data_dir, False, lambda *a: None)
    monkeypatch.setenv("COA_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DEMO_REPLAY", "1")
    from coa.web.app import create_app

    return TestClient(create_app())


def test_review_changes_export(client, with_samples):
    queue = db.rows(with_samples, "SELECT pair_id FROM audits WHERE verdict='dispute' ORDER BY pair_id")
    assert len(queue) >= 3, "the recorded audit disputes at least three pairs"
    a, f, d = (r["pair_id"] for r in queue[:3])
    before = axolotl_export.export_rows(with_samples)
    pf = db.one(with_samples, "SELECT chosen, rejected FROM pairs WHERE id=?", f)

    assert client.post(f"/api/review/{a}/approve").status_code == 200
    assert client.post(f"/api/review/{f}/flip").status_code == 200
    r = client.post(f"/api/review/{d}/drop")
    assert r.json()["counts"]["drop"] == 1 and r.json()["counts"]["flip"] == 1

    after = axolotl_export.export_rows(with_samples)
    assert len(after.rows) == len(before.rows) - 1 and after.dataset_hash != before.dataset_hash
    flipped = [x for x in after.rows if x["chosen"]["content"] == pf["rejected"]
               and x["rejected"]["content"] == pf["chosen"]]
    assert flipped
    rows = db.rows(with_samples, "SELECT pair_id, action, at FROM reviews ORDER BY id")
    assert [(x["pair_id"], x["action"]) for x in rows] == [(a, "approve"), (f, "flip"), (d, "drop")]
    assert all(x["at"] for x in rows)


def test_review_rejects_unknown_action(client):
    assert client.post("/api/review/1/delete").status_code == 400
    assert client.post("/api/review/99999/approve").status_code == 404


def test_htmx_review_returns_item_and_counters(client, with_samples):
    pid = db.one(with_samples, "SELECT pair_id FROM audits WHERE verdict='dispute' LIMIT 1")["pair_id"]
    r = client.post(f"/api/review/{pid}/approve", headers={"HX-Request": "true"})
    assert r.status_code == 200 and f'id="ri-{pid}"' in r.text and 'hx-swap-oob="true"' in r.text


def test_pages_render(client):
    for url in ["/", "/review", "/train", "/evaluate", "/partials/note?patient=B001&view=raw",
                "/partials/pairs?label=invented", "/partials/pair/1", "/partials/trace/1", "/healthz"]:
        r = client.get(url)
        assert r.status_code == 200, url
    assert "Replay mode" in client.get("/").text
    assert "Synthetic data" in client.get("/review").text
