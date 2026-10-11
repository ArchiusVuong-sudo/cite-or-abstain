"""`python -m coa serve` runs the web app; `python -m coa seed` rebuilds the synthetic dataset."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(prog="coa")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run the web app on HOST:PORT (default 127.0.0.1:4104)")
    sub.add_parser("seed", help="rebuild the synthetic dataset from COA_SEED")
    args = ap.parse_args()
    from .config import settings

    s = settings()
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("coa.web.app:create_app", factory=True, host=s.host, port=s.port, log_level="info")
    elif args.cmd == "seed":
        from . import dataset, db

        print(json.dumps(dataset.seed(db.connect(s.db_path), s.seed), indent=1, default=str))


if __name__ == "__main__":
    main()
