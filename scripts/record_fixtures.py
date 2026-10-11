"""Re-record the replay fixtures from real live runs.

1. Seeds a scratch dataset, samples the base model live (greedy + 2 samples at T=0.8 on 240 train
   questions) and writes fixtures/base_samples.jsonl.
2. Builds the pairs and runs the Claude audit live (3 batches of 8), writing fixtures/audit_replay.json.

Needs the base model weights (downloaded on first use) and a Claude Code login or ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coa import audit, dataset, db, llm, pairs, sampling  # noqa: E402
from coa.config import settings  # noqa: E402


def main() -> int:
    s = settings()
    fixtures = s.fixtures_dir
    if not llm.has_login():
        print("No Claude login or ANTHROPIC_API_KEY found; cannot record live audit responses.")
        return 2
    with tempfile.TemporaryDirectory(prefix="coa-record-") as tmp:
        conn = db.connect(Path(tmp) / "coa.sqlite3")
        dataset.seed(conn, s.seed)
        log = lambda d, t, m: print(f"  {m} {d}/{t}", flush=True)  # noqa: E731
        print("Sampling the base model live:", s.base_model)
        print(json.dumps(sampling.run_live(conn, s.base_model, s.device, s.seed, log)))
        print("Wrote", sampling.record_fixture(conn, fixtures))
        print(json.dumps(pairs.build_all(conn)))
        replay = fixtures / llm.REPLAY_FILE
        if replay.exists():
            replay.unlink()
        print("Auditing live with", s.audit_model)
        res = audit.run(conn, llm.LiveProvider(s.audit_model), fixtures, True, log)
        print(json.dumps(res))
        if res["errors"]:
            print("Some audit batches failed; the fixture is incomplete.")
            return 1
        print("Wrote", replay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
