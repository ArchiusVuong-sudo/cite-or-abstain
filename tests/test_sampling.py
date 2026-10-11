import json
from pathlib import Path

import pytest

from coa import db, sampling
from coa.dataset import question_views

ROOT = Path(__file__).resolve().parent.parent


def test_recorded_answers_load_per_question(conn):
    """Patients can share identical passages; each question must still get its own recorded answers."""
    res = sampling.run_recorded(conn, ROOT / "fixtures", lambda *a: None)
    assert res["missing"] == 0 and res["source"] == "recorded"
    fixture = {json.loads(x)["question_id"]: json.loads(x) for x in
               (ROOT / "fixtures" / sampling.FIXTURE).read_text().splitlines() if x.strip()}
    for qid, rec in fixture.items():
        got = [r["text"] for r in db.rows(conn, "SELECT text FROM samples WHERE question_id=? ORDER BY id", qid)]
        assert got == [a["text"] for a in rec["answers"]], qid
    prompts = [sampling.prompt_hash(q) for q in question_views(conn, "train")]
    assert len(prompts) == 240


def test_recorded_answers_reject_a_changed_prompt(conn, tmp_path):
    line = json.loads((ROOT / "fixtures" / sampling.FIXTURE).read_text().splitlines()[0])
    line["prompt_hash"] = "0" * 16
    (tmp_path / sampling.FIXTURE).write_text(json.dumps(line) + "\n")
    with pytest.raises(ValueError, match="do not match"):
        sampling.run_recorded(conn, tmp_path, lambda *a: None)
