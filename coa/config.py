"""Runtime settings, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ABSTAIN = "Not documented in the provided notes."
TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("COA_DATA_DIR", ROOT / "var")))
    fixtures_dir: Path = field(default_factory=lambda: Path(os.environ.get("COA_FIXTURES_DIR", ROOT / "fixtures")))
    base_model: str = field(default_factory=lambda: os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    audit_model: str = field(default_factory=lambda: os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"))
    device: str = field(default_factory=lambda: os.environ.get("COA_DEVICE", "auto"))
    seed: int = field(default_factory=lambda: int(os.environ.get("COA_SEED", "7")))
    eval_limit: int = field(default_factory=lambda: int(os.environ.get("EVAL_LIMIT", "0")))
    train_max_steps: int = field(default_factory=lambda: int(os.environ.get("TRAIN_MAX_STEPS", "60")))
    host: str = field(default_factory=lambda: os.environ.get("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "4104")))

    @property
    def replay_forced(self) -> bool:
        return _flag("DEMO_REPLAY") or _flag("CLAUDE_REPLAY")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "coa.sqlite3"

    def path(self, *parts: str) -> Path:
        p = self.data_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
