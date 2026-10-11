import asyncio
import json
import os

import pytest
from pydantic import ValidationError

from coa import audit, db, llm
from coa.llm import AuditRecord, BatchResult, ReplayProvider, record_replay

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_audit_record_validation():
    ok = AuditRecord(pair_id=1, verdict="dispute", issue="ambiguous_evidence", suggested_action="drop",
                     reason="P1 says pending")
    assert ok.verdict == "dispute"
    for bad in [{"verdict": "maybe"}, {"suggested_action": "delete"}, {"reason": ""}, {"issue": "other"}]:
        with pytest.raises(ValidationError):
            AuditRecord(**{**ok.model_dump(), **bad})


def test_select_pairs_is_deterministic_and_sized(with_samples):
    a = audit.select_pairs(with_samples)
    assert len(a) == 24 and [p["id"] for p in a] == [p["id"] for p in audit.select_pairs(with_samples)]
    assert sum(p["rejected_source"] == "perturbation" for p in a) == 4


def test_uncertainty_flags_invented_words_found_in_a_passage():
    p = {"rejected_source": "model", "rejected_label": "invented", "rejected": "Invasive carcinoma [P1]",
         "field": "histology", "chosen_source": "reference", "chosen": "x", "trap": "pending"}
    assert audit.uncertainty(p, ["Invasive carcinoma; histologic type is pending."]) >= 5
    assert audit.uncertainty(p, ["Grade will follow."]) < 5


def test_audit_replay(with_samples, tmp_path):
    """Recorded verdicts replay by pair content, marked 'recorded'; a missing one is never invented."""
    res = audit.run(with_samples, ReplayProvider(ROOT / "fixtures"), tmp_path, False, lambda *a: None)
    assert res["source"] == "replay" and res["pairs"] == 24 and res["errors"] == 0
    traces = db.rows(with_samples, "SELECT * FROM llm_traces")
    assert len(traces) == 3 and {t["source"] for t in traces} == {"recorded"}
    assert all(json.loads(t["tool_calls"]) for t in traces)
    assert {a["source"] for a in db.rows(with_samples, "SELECT source FROM audits")} == {"recorded"}
    empty = ReplayProvider(tmp_path).audit_batch(1, [1, 2], "x", {1: "h1", 2: "h2"})
    assert empty.records == [] and "no recording for 2 of 2" in empty.error


def test_replay_roundtrip(tmp_path):
    rec = AuditRecord(pair_id=5, verdict="agree", issue="none", suggested_action="keep", reason="P1 states it")
    record_replay(tmp_path, [("hash5", rec)], ["hash5"], BatchResult([rec], "live", "m", latency_ms=10,
                                                                      cost_usd=0.01))
    out = ReplayProvider(tmp_path).audit_batch(1, [9], "x", {9: "hash5"})
    assert out.records[0].pair_id == 9 and out.records[0].reason == "P1 states it"
    assert out.latency_ms == 10 and out.source == "recorded" and out.error is None


def test_child_env_is_an_allowlist(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "s")
    monkeypatch.setenv("SOME_SECRET", "x")
    env = llm.child_env()
    assert set(env) <= set(llm.ENV_ALLOW) and "PATH" in env


def test_live_query_is_isolated(monkeypatch):
    """The live provider's SDK options: no built-ins, one MCP tool, no settings, strict, no persistence."""
    import claude_agent_sdk

    seen = {}

    async def fake_query(prompt, options):
        seen["options"], seen["env"] = options, dict(os.environ)
        server = options.mcp_servers[llm.SERVER]
        seen["server"] = server
        if False:
            yield None

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "leak")
    prov = llm.LiveProvider("claude-haiku-5-5", max_attempts=1)
    res = asyncio.run(prov._query([1], "prompt"))
    o = seen["options"]
    assert o.tools == [] and o.allowed_tools == [llm.ALLOWED_TOOL] and o.strict_mcp_config is True
    assert o.setting_sources == [] and o.permission_mode == "dontAsk" and o.max_turns == 14
    assert o.extra_args == {"no-session-persistence": None} and o.model == "claude-haiku-5-5"
    assert "coa-audit-" in str(o.cwd) and set(o.env) <= set(llm.ENV_ALLOW)
    assert "CLAUDE_CODE_SESSION_ID" not in seen["env"]  # the spawn sees only the allowlist
    assert res.error == "no result message"


def test_retry_classification():
    assert llm.is_retryable("API Error: 529 overloaded_error") and llm.is_retryable("rate limit reached")
    assert not llm.is_retryable("invalid model name")
    assert llm.is_auth_error("Not logged in. Please run /login")


def test_live_provider_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    async def flaky(self, ids, prompt):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("API Error: 529 overloaded")
        return BatchResult([], "live", "m")

    monkeypatch.setattr(llm.LiveProvider, "_query", flaky)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    res = llm.LiveProvider("m", base_delay=0).audit_batch(1, [1], "p", {1: "h"})
    assert calls["n"] == 3 and res.attempts == 3 and res.error is None


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("LIVE") != "1", reason="set LIVE=1 to call the real login")
def test_live_smoke(with_samples):
    picked = audit.select_pairs(with_samples)[:2]
    prompt = audit.batch_prompt(with_samples, picked)
    res = llm.LiveProvider(os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")).audit_batch(
        1, [p["id"] for p in picked], prompt, {p["id"]: p["content_hash"] for p in picked})
    assert res.error is None, res.error
    assert {r.pair_id for r in res.records} == {p["id"] for p in picked}
