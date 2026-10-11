"""Deterministic typo injection for the robustness slice. Value words are never touched."""

from __future__ import annotations

import random
import re

from .fields import value_terms

_PROTECT = value_terms()


def add_typos(text: str, seed: str, rate: float = 0.18) -> str:
    rng = random.Random(seed)

    def swap(m: re.Match) -> str:
        w = m.group(0)
        if len(w) < 5 or w.lower() in _PROTECT or any(c.isdigit() for c in w) or rng.random() > rate:
            return w
        i = rng.randint(1, len(w) - 3)
        op = rng.random()
        if op < 0.5:
            return w[:i] + w[i + 1] + w[i] + w[i + 2:]
        if op < 0.8:
            return w[:i] + w[i + 1:]
        return w[:i] + w[i] + w[i:]

    return re.sub(r"[A-Za-z]+", swap, text)
