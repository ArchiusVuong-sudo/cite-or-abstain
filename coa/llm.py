"""Audit providers: live (Claude Agent SDK on the local login) and replay (responses recorded from live runs).

The live provider runs each query with no built-in tools, one in-process MCP tool (record_audit),
no filesystem settings, strict MCP config, no session persistence, a fresh temp cwd, bounded turns,
a permission mode that denies anything not pre-approved, and a child env built from an allowlist.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol
from unittest import mock

from pydantic import BaseModel, Field, ValidationError

ENV_ALLOW = ("PATH", "HOME", "USER", "LANG", "TMPDIR", "ANTHROPIC_API_KEY")
SERVER = "audit"
TOOL = "record_audit"
ALLOWED_TOOL = f"mcp__{SERVER}__{TOOL}"
RETRYABLE = ("overloaded", "rate limit", "rate_limit", "429", "529", "500", "502", "503", "504", "internal server")
AUTH_ERRORS = ("not logged in", "invalid api key", "authentication", "please run /login", "unauthorized", "401")
REPLAY_FILE = "audit_replay.json"


class AuditRecord(BaseModel):
    """What the model must send for each pair. Validated before it is accepted."""

    pair_id: int
    verdict: Literal["agree", "dispute"]
    issue: Literal["none", "chosen_wrong", "rejected_is_correct", "both_wrong", "ambiguous_evidence",
                   "format_only"]
    suggested_action: Literal["keep", "flip", "drop"]
    reason: str = Field(min_length=3, max_length=400)


TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pair_id": {"type": "integer", "description": "The pair_id being audited"},
        "verdict": {"type": "string", "enum": ["agree", "dispute"],
                    "description": "agree: chosen is grounded and better than rejected. dispute: otherwise"},
        "issue": {"type": "string", "enum": ["none", "chosen_wrong", "rejected_is_correct", "both_wrong",
                                             "ambiguous_evidence", "format_only"]},
        "suggested_action": {"type": "string", "enum": ["keep", "flip", "drop"]},
        "reason": {"type": "string", "description": "One short sentence citing the passage id"},
    },
    "required": ["pair_id", "verdict", "issue", "suggested_action", "reason"],
}


@dataclass
class BatchResult:
    records: list[AuditRecord]
    source: str  # "live" or "recorded"
    model: str
    latency_ms: int | None = None
    turns: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    attempts: int = 1
    tool_calls: list[dict] = field(default_factory=list)
    result_text: str | None = None
    error: str | None = None


class AuditProvider(Protocol):
    mode: str

    def audit_batch(self, batch: int, pair_ids: list[int], prompt: str, keys: dict[int, str]) -> BatchResult: ...


def child_env() -> dict[str, str]:
    """Only allowlisted variables reach the Claude Code child; CLAUDECODE and CLAUDE_CODE_* never do."""
    return {k: os.environ[k] for k in ENV_ALLOW if os.environ.get(k)}


def has_login() -> bool:
    """A cheap check for a key or a local Claude Code login; no network call."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    home = Path(os.environ.get("HOME", "~")).expanduser()
    if (home / ".claude" / ".credentials.json").exists():
        return True
    if shutil.which("security"):
        r = subprocess.run(["security", "find-generic-password", "-s", "Claude Code-credentials"],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    return False


def is_retryable(text: str) -> bool:
    t = text.lower()
    return any(s in t for s in RETRYABLE)


def is_auth_error(text: str) -> bool:
    t = text.lower()
    return any(s in t for s in AUTH_ERRORS)


class LiveProvider:
    mode = "live"

    def __init__(self, model: str, max_turns: int = 14, max_attempts: int = 5, base_delay: float = 3.0):
        self.model = model
        self.max_turns = max_turns
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.env = child_env()

    def audit_batch(self, batch: int, pair_ids: list[int], prompt: str, keys: dict[int, str]) -> BatchResult:
        last_err = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                res = asyncio.run(self._query(pair_ids, prompt))
                res.attempts = attempt
                if res.error and is_retryable(res.error) and attempt < self.max_attempts:
                    last_err = res.error
                else:
                    return res
            except Exception as e:  # SDK raises ProcessError / CLIConnectionError on API failures
                last_err = f"{type(e).__name__}: {e}"
                if is_auth_error(last_err) or not is_retryable(last_err) or attempt == self.max_attempts:
                    return BatchResult([], "live", self.model, attempts=attempt, error=last_err[:2000])
            time.sleep(self.base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1))
        return BatchResult([], "live", self.model, attempts=self.max_attempts, error=last_err[:2000])

    async def _query(self, pair_ids: list[int], prompt: str) -> BatchResult:
        from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock,
                                      create_sdk_mcp_server, query, tool)

        accepted: dict[int, AuditRecord] = {}
        calls: list[dict] = []
        allowed = set(pair_ids)

        @tool(TOOL, "Record your audit of one preference pair. Call once per pair.", TOOL_SCHEMA)
        async def record_audit(args: dict[str, Any]) -> dict[str, Any]:
            try:
                rec = AuditRecord.model_validate(args)
            except ValidationError as e:
                calls.append({"input": args, "ok": False, "error": e.errors()[0]["msg"]})
                return {"content": [{"type": "text", "text": f"Rejected: {e.errors()[0]['msg']}. Fix and retry."}],
                        "is_error": True}
            if rec.pair_id not in allowed:
                calls.append({"input": args, "ok": False, "error": "unknown pair_id"})
                return {"content": [{"type": "text", "text": f"Rejected: pair_id {rec.pair_id} is not in this "
                                                             "batch."}], "is_error": True}
            accepted[rec.pair_id] = rec
            calls.append({"input": args, "ok": True})
            left = sorted(allowed - set(accepted))
            return {"content": [{"type": "text", "text": f"Recorded. Pairs left: {left or 'none'}."}]}

        server = create_sdk_mcp_server(name=SERVER, version="1.0.0", tools=[record_audit])
        with tempfile.TemporaryDirectory(prefix="coa-audit-") as cwd:
            options = ClaudeAgentOptions(
                tools=[],
                allowed_tools=[ALLOWED_TOOL],
                mcp_servers={SERVER: server},
                strict_mcp_config=True,
                setting_sources=[],
                permission_mode="dontAsk",
                max_turns=self.max_turns,
                cwd=cwd,
                env=self.env,
                model=self.model,
                system_prompt=AUDITOR_SYSTEM,
                extra_args={"no-session-persistence": None},
            )
            t0 = time.time()
            text_parts: list[str] = []
            result: Any = None
            # The SDK merges os.environ into the child env; swap in the allowlist for the spawn.
            with mock.patch.dict(os.environ, self.env, clear=True):
                async for msg in query(prompt=prompt, options=options):
                    if isinstance(msg, AssistantMessage):
                        text_parts += [b.text for b in msg.content if isinstance(b, TextBlock)]
                    elif isinstance(msg, ResultMessage):
                        result = msg
            latency = int((time.time() - t0) * 1000)
        usage = (getattr(result, "usage", None) or {}) if result else {}
        error = None
        if result is not None and getattr(result, "is_error", False):
            error = str(getattr(result, "result", None) or getattr(result, "subtype", "error"))
        elif result is None:
            error = "no result message"
        return BatchResult(
            records=[accepted[i] for i in pair_ids if i in accepted], source="live", model=self.model,
            latency_ms=latency, turns=getattr(result, "num_turns", None),
            input_tokens=(usage.get("input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0)
            + (usage.get("cache_creation_input_tokens") or 0) if usage else None,
            output_tokens=usage.get("output_tokens") if usage else None,
            cost_usd=getattr(result, "total_cost_usd", None), tool_calls=calls,
            result_text=(getattr(result, "result", None) or "\n".join(text_parts))[:4000], error=error,
        )


class ReplayProvider:
    """Serves verdicts recorded from live runs, keyed by each pair's content hash. Never invents one."""

    mode = "replay"

    def __init__(self, fixtures_dir: Path):
        path = fixtures_dir / REPLAY_FILE
        self.data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {"pairs": {}, "batches": {}}

    def audit_batch(self, batch: int, pair_ids: list[int], prompt: str, keys: dict[int, str]) -> BatchResult:
        recs, calls = [], []
        for pid in pair_ids:
            r = self.data["pairs"].get(keys[pid])
            if r:
                rec = AuditRecord(**{**r["record"], "pair_id": pid})
                recs.append(rec)
                calls.append({"input": rec.model_dump(), "ok": True})
        bkey = batch_key([keys[p] for p in pair_ids])
        t = self.data["batches"].get(bkey, {})
        missing = len(pair_ids) - len(recs)
        return BatchResult(
            records=recs, source="recorded", model=t.get("model", "recorded"), latency_ms=t.get("latency_ms"),
            turns=t.get("turns"), input_tokens=t.get("input_tokens"), output_tokens=t.get("output_tokens"),
            cost_usd=t.get("cost_usd"), tool_calls=calls, result_text=t.get("result_text"),
            error=f"no recording for {missing} of {len(pair_ids)} pairs" if missing else None,
        )


def batch_key(hashes: list[str]) -> str:
    return ",".join(sorted(hashes))


def record_replay(fixtures_dir: Path, entries: list[tuple[str, AuditRecord]], batch_hashes: list[str],
                  res: BatchResult) -> None:
    """Append a live batch to the replay fixture."""
    path = fixtures_dir / REPLAY_FILE
    data = json.loads(path.read_text()) if path.exists() else {"pairs": {}, "batches": {}}
    for h, rec in entries:
        data["pairs"][h] = {"record": rec.model_dump(exclude={"pair_id"})}
    data["batches"][batch_key(batch_hashes)] = {
        "model": res.model, "latency_ms": res.latency_ms, "turns": res.turns, "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens, "cost_usd": res.cost_usd, "result_text": res.result_text,
    }
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True))


def make_provider(model: str, fixtures_dir: Path, replay_forced: bool) -> AuditProvider:
    if replay_forced or not has_login():
        return ReplayProvider(fixtures_dir)
    return LiveProvider(model)


AUDITOR_SYSTEM = (
    "You audit preference pairs for training a small model that abstracts cancer registry fields from "
    "synthetic clinical notes. For each pair you see the question, the retrieved passages [P1]-[P3], the "
    "chosen answer, the rejected answer and the label a rule-based checker gave the rejected answer. "
    "The chosen answer must state the patient's own value with a citation to a passage that states it, or "
    "say 'Not documented in the provided notes.' when no passage states the patient's value (a relative's "
    "result, a pending test, a plan or a drug that was not started do not count). "
    "Agree when the chosen answer is right and clearly better than the rejected one. Dispute when the chosen "
    "answer is wrong, when the rejected answer is actually acceptable, or when the evidence is ambiguous. "
    "A less specific value that a passage does state (for example 'invasive carcinoma' while the exact type is "
    "pending) is a labelling-policy question: dispute it as ambiguous_evidence so a human reviewer decides. "
    "Suggest keep, flip (swap chosen and rejected) or drop. Call record_audit exactly once per pair, then "
    "stop. Do not write anything else."
)
