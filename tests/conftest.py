from __future__ import annotations

import os
from pathlib import Path

import pytest

from coa import db
from coa.config import TINY_MODEL

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _cached(model_id: str) -> bool:
    from huggingface_hub.constants import HF_HUB_CACHE

    return (Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}").exists()


if _cached(TINY_MODEL):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # no network once the tiny test model is cached


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "var"
    d.mkdir()
    return d


@pytest.fixture()
def conn(data_dir: Path):
    from coa import dataset

    c = db.connect(data_dir / "coa.sqlite3")
    dataset.seed(c, 7)
    yield c
    c.close()


@pytest.fixture()
def with_samples(conn):
    """The seeded DB plus the recorded base-model answers and the pairs built from them."""
    from coa import pairs, sampling

    sampling.run_recorded(conn, ROOT / "fixtures", lambda *a: None)
    pairs.build_all(conn)
    return conn
