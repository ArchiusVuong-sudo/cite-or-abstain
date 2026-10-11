"""PHI scrub with consistent surrogates, plus a leak gate for anything we export.

The scrubber finds identifiers by pattern (header fields, titles plus surnames, clinician names,
MRNs, phone numbers, dates). It never reads the generator's ground truth; the leak gate does,
so a scrubber miss shows up as a failed gate instead of a silent leak.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

SURROGATE_FIRST = ["Alex", "Sam", "Jordan", "Robin", "Casey", "Morgan", "Riley", "Quinn", "Avery", "Jamie"]
CLIN_FIRST = ["Chris", "Dana", "Eli", "Frankie", "Gale", "Harper"]
CLIN_LAST = ["Novak", "Ortiz", "Price", "Quade", "Reid", "Stone"]
SURROGATE_LAST = ["Doe", "Roe", "Hale", "Penn", "Shaw", "Wells", "Ames", "Burke", "Cole", "Dale"]

NAME_RE = re.compile(r"Patient:\s*([A-Z][a-z]+)\s+([A-Z][a-z]+)")
CLINICIAN_RE = re.compile(r"Dr\.\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)")
MRN_RE = re.compile(r"\b(?:MRN:?\s*)?(4\d{6})\b")
PHONE_RE = re.compile(r"\(\d{3}\)\s?\d{3}-\d{4}")
DATE_RE = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")


def _pick(options: list[str], *key: str) -> str:
    h = int(hashlib.sha256("|".join(key).encode()).hexdigest(), 16)
    return options[h % len(options)]


@dataclass
class ScrubResult:
    text: str
    replacements: list[tuple[str, str, str]] = field(default_factory=list)  # (kind, original, surrogate)


@dataclass
class Scrubber:
    """One scrubber per patient, so every note of that patient gets the same surrogates."""

    patient_key: str
    shift_days: int
    mapping: dict[tuple[str, str], str] = field(default_factory=dict)
    names: list[tuple[str, str]] = field(default_factory=list)  # learned from headers, applied to later texts

    def _sur(self, kind: str, original: str, make) -> str:
        k = (kind, original)
        if k not in self.mapping:
            self.mapping[k] = make()
        return self.mapping[k]

    def scrub(self, text: str) -> ScrubResult:
        out: list[tuple[str, str, str]] = []

        def note(kind: str, orig: str, sur: str) -> str:
            out.append((kind, orig, sur))
            return sur

        for m in NAME_RE.finditer(text):
            if (m.group(1), m.group(2)) not in self.names:
                self.names.append((m.group(1), m.group(2)))
        for first, last in self.names:
            sf = self._sur("first", first, lambda: _pick(SURROGATE_FIRST, self.patient_key, first))
            sl = self._sur("last", last, lambda: _pick(SURROGATE_LAST, self.patient_key, last))
            text = re.sub(rf"\b{re.escape(first)}\b", lambda _m: note("name", first, sf), text)
            text = re.sub(rf"\b{re.escape(last)}\b", lambda _m: note("name", last, sl), text)

        def clin(m: re.Match) -> str:
            orig = f"{m.group(1)} {m.group(2)}"
            sur = self._sur("clinician", orig,
                            lambda: f"{_pick(CLIN_FIRST, 'dr', orig)} {_pick(CLIN_LAST, 'dr', orig)}")
            return "Dr. " + note("clinician", orig, sur)

        text = CLINICIAN_RE.sub(clin, text)

        def mrn(m: re.Match) -> str:
            orig = m.group(1)
            sur = self._sur("mrn", orig, lambda: str(9_000_000 + int(hashlib.sha256(
                (self.patient_key + orig).encode()).hexdigest(), 16) % 999_999))
            return m.group(0).replace(orig, note("mrn", orig, sur))

        text = MRN_RE.sub(mrn, text)

        def phone(m: re.Match) -> str:
            orig = m.group(0)
            h = int(hashlib.sha256((self.patient_key + orig).encode()).hexdigest(), 16)
            sur = self._sur("phone", orig, lambda: f"(555) 09{h % 10}-{1000 + h % 9000}")
            return note("phone", orig, sur)

        text = PHONE_RE.sub(phone, text)

        def dt(m: re.Match) -> str:
            orig = m.group(0)
            try:
                d = datetime.strptime(orig, "%m/%d/%Y") + timedelta(days=self.shift_days)
            except ValueError:
                return orig
            return note("date", orig, d.strftime("%m/%d/%Y"))

        text = DATE_RE.sub(dt, text)
        return ScrubResult(text, out)


def date_shift(seed: int, patient_id: str, texts: list[str]) -> int:
    """A per-patient shift in [-60, 60] days (never 0), so intervals inside a record are kept.

    The shift is moved until no shifted date lands on one of the record's original dates;
    otherwise a surrogate could equal a real identifier and the leak gate would rightly fail.
    """
    found = {m.group(0) for t in texts for m in DATE_RE.finditer(t)}
    parsed = []
    for f in found:
        try:
            parsed.append(datetime.strptime(f, "%m/%d/%Y"))
        except ValueError:
            continue
    h = int(hashlib.sha256(f"{seed}:{patient_id}:shift".encode()).hexdigest(), 16)
    s = h % 121 - 60
    for _ in range(200):
        if s != 0 and not any((d + timedelta(days=s)).strftime("%m/%d/%Y") in found for d in parsed):
            return s
        s = s + 1 if s < 60 else -60
    raise ValueError(f"no safe date shift for {patient_id}")


def leak_scan(texts: list[str], originals: list[str]) -> list[str]:
    """Return every original identifier found in any text (whole-word match)."""
    found: set[str] = set()
    for orig in set(o for o in originals if o):
        pat = re.compile(rf"(?<![\w]){re.escape(orig)}(?![\w])")
        if any(pat.search(t) for t in texts):
            found.add(orig)
    return sorted(found)
