import pytest

from coa import db, jobs
from coa.config import Settings


def test_one_job_at_a_time(conn, data_dir, monkeypatch):
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: type("P", (), {"pid": 999999})())
    s = Settings(data_dir=data_dir)
    jid = jobs.start(conn, s, "audit", {})
    conn.execute("UPDATE jobs SET pid=NULL WHERE id=?", (jid,))
    with pytest.raises(jobs.Busy):
        jobs.start(conn, s, "train", {})
    with pytest.raises(ValueError):
        jobs.start(conn, s, "rm -rf", {})


def test_dead_worker_is_marked_failed(conn):
    conn.execute("INSERT INTO jobs(kind, state, params, created_at, pid) VALUES ('train', 'running', '{}', 'x', ?)",
                 (2 ** 22 + 12345,))
    jobs.reap_stale(conn)
    j = db.one(conn, "SELECT state, error FROM jobs")
    assert j["state"] == "error" and "exited" in j["error"]


def test_llm_job_env_is_allowlisted(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "x")
    monkeypatch.setenv("HF_TOKEN_FAKE", "x")
    assert set(jobs.child_env("audit")) <= {"PATH", "HOME", "USER", "LANG", "TMPDIR", "ANTHROPIC_API_KEY"}
    model_env = jobs.child_env("train")
    assert "CLAUDE_CODE_SESSION_ID" not in model_env and "HF_TOKEN_FAKE" in model_env
