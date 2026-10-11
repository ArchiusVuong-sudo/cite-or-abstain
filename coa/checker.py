"""Deterministic grounding checker: labels one answer against the gold label and the retrieved passages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .fields import FIELDS, THERAPY_DOSE, THERAPY_FIELDS


class Label(str, Enum):
    CORRECT = "correct"
    CORRECT_ABSTAIN = "correct_abstain"
    INVENTED = "invented"
    WRONG_VALUE = "wrong_value"
    FALSE_REFUSAL = "false_refusal"
    NON_ANSWER = "non_answer"


FAILURES = {Label.INVENTED, Label.WRONG_VALUE, Label.FALSE_REFUSAL, Label.NON_ANSWER}

ABSTAIN_RE = re.compile(r"\bnot\s+documented\b", re.I)
HEDGE_RE = re.compile(r"\b(pending|unknown|not (stated|specified|mentioned|available|provided|reported|given)|"
                      r"cannot (be )?determine|unable to determine|no information|n/?a)\b", re.I)
CITE_RE = re.compile(r"\[\s*P\s*(\d+)\s*\]", re.I)


@dataclass(frozen=True)
class Gold:
    field: str
    answerable: bool
    value: str | None  # canonical gold value when answerable
    evidence: frozenset[int]  # 1-based positions of retrieved passages that state the gold value


@dataclass(frozen=True)
class Check:
    label: Label
    value: str | None
    cited: tuple[int, ...]
    citation_valid: bool | None  # None when the answer states no value


def canonical_value(field: str, text: str) -> str | None:
    """Earliest phrase in the text that names one of the field's values; None if none does."""
    f = FIELDS[field]
    low = CITE_RE.sub(" ", text).lower().strip()
    best: tuple[int, str] | None = None
    for v in f.values:
        for pat in v.patterns:
            m = re.search(pat, low, re.I)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), v.canonical)
    return best[1] if best else None


def dose_matches(canonical: str, text: str) -> bool:
    pat = THERAPY_DOSE.get(canonical)
    return bool(pat and re.search(pat, text, re.I))


def check(answer: str, gold: Gold) -> Check:
    text = (answer or "").strip()
    cited = tuple(int(n) for n in CITE_RE.findall(text))
    body = CITE_RE.sub(" ", text).strip(" .\n\t")
    if not re.search(r"[A-Za-z0-9]", body):
        return Check(Label.NON_ANSWER, None, cited, None)
    if ABSTAIN_RE.search(text):
        return Check(Label.FALSE_REFUSAL if gold.answerable else Label.CORRECT_ABSTAIN, None, cited, None)
    value = canonical_value(gold.field, body)
    if value is None and HEDGE_RE.search(body):
        return Check(Label.FALSE_REFUSAL if gold.answerable else Label.NON_ANSWER, None, cited, None)
    if not gold.answerable:
        return Check(Label.INVENTED, value, cited, False)
    ok_value = value == gold.value and (gold.field not in THERAPY_FIELDS or dose_matches(value, body))
    citation_valid = any(c in gold.evidence for c in cited)
    return Check(Label.CORRECT if ok_value else Label.WRONG_VALUE, value, cited, citation_valid if ok_value else False)


def value_spans(field: str, text: str) -> list[tuple[int, int]]:
    """Character spans in `text` that name any value of the field (for highlighting in the UI)."""
    spans: list[tuple[int, int]] = []
    for v in FIELDS[field].values:
        for pat in v.patterns:
            if pat.startswith("^"):
                continue
            spans += [(m.start(), m.end()) for m in re.finditer(pat, text, re.I) if m.end() > m.start()]
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    return merged
