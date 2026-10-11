"""Registry fields: the question asked for each, and the canonical values with the phrasings that mean them."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Value:
    canonical: str
    display: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class Field:
    key: str
    cancer: str
    label: str
    question: str
    values: tuple[Value, ...]

    def value(self, canonical: str) -> Value:
        for v in self.values:
            if v.canonical == canonical:
                return v
        raise KeyError(f"{self.key}: unknown value {canonical!r}")


def _v(canonical: str, display: str, *patterns: str) -> Value:
    return Value(canonical, display, patterns or (re.escape(display.lower()),))


POS = (r"\bpositive\b", r"\bpos\b", r"\ber\s?\+", r"\bher2\s?\+")
NEG = (r"\bnegative\b", r"\bneg\b", r"\ber\s?-(?!\w)", r"\bher2\s?-(?!\w)")

FIELDS: dict[str, Field] = {
    f.key: f
    for f in [
        Field("laterality", "breast", "Laterality", "Which breast is affected by the cancer?", (
            _v("left", "Left", r"\bleft\b"),
            _v("right", "Right", r"\bright\b"),
        )),
        Field("histology", "breast", "Histology", "What is the histologic type of the breast tumor?", (
            _v("idc", "Invasive ductal carcinoma", r"\bductal\b", r"\bidc\b"),
            _v("ilc", "Invasive lobular carcinoma", r"\blobular\b", r"\bilc\b"),
        )),
        Field("grade", "breast", "Grade", "What is the histologic grade of the breast tumor?", (
            _v("1", "Grade 1", r"\bgrade[:\s]*(1|i)\b", r"^\s*1\b", r"\bwell[- ]differentiated\b"),
            _v("2", "Grade 2", r"\bgrade[:\s]*(2|ii)\b", r"^\s*2\b", r"\bmoderately[- ]differentiated\b"),
            _v("3", "Grade 3", r"\bgrade[:\s]*(3|iii)\b", r"^\s*3\b", r"\bpoorly[- ]differentiated\b"),
        )),
        Field("er", "breast", "ER status", "What is the estrogen receptor (ER) status of the tumor?", (
            _v("positive", "ER positive", *POS),
            _v("negative", "ER negative", *NEG),
        )),
        Field("her2", "breast", "HER2 status", "What is the HER2 status of the tumor?", (
            _v("positive", "HER2 positive", *POS, r"\b3\+"),
            _v("negative", "HER2 negative", *NEG, r"\b(0|1\+)"),
        )),
        Field("btherapy", "breast", "Current endocrine therapy",
              "What endocrine therapy is the patient currently taking, and at what dose?", (
            _v("letrozole 2.5 mg daily", "Letrozole 2.5 mg daily", r"\bletrozole\b"),
            _v("anastrozole 1 mg daily", "Anastrozole 1 mg daily", r"\banastrozole\b"),
            _v("tamoxifen 20 mg daily", "Tamoxifen 20 mg daily", r"\btamoxifen\b"),
            _v("exemestane 25 mg daily", "Exemestane 25 mg daily", r"\bexemestane\b"),
        )),
        Field("lobe", "lung", "Primary lobe", "Which lobe of the lung is the primary tumor in?", (
            _v("rul", "Right upper lobe", r"\bright upper\b", r"\brul\b"),
            _v("rml", "Right middle lobe", r"\bright middle\b", r"\brml\b"),
            _v("rll", "Right lower lobe", r"\bright lower\b", r"\brll\b"),
            _v("lul", "Left upper lobe", r"\bleft upper\b", r"\blul\b"),
            _v("lll", "Left lower lobe", r"\bleft lower\b", r"\blll\b"),
        )),
        Field("lhistology", "lung", "Histology", "What is the histologic type of the lung tumor?", (
            _v("adenocarcinoma", "Adenocarcinoma", r"\badenocarcinoma\b"),
            _v("squamous", "Squamous cell carcinoma", r"\bsquamous\b"),
        )),
        Field("egfr", "lung", "EGFR status", "What is the EGFR mutation status of the tumor?", (
            _v("exon19", "EGFR exon 19 deletion", r"\bexon\s*19\b", r"\bdel(etion)?\s*19\b"),
            _v("l858r", "EGFR L858R mutation", r"\bl858r\b", r"\bexon\s*21\b"),
            _v("none", "No EGFR mutation detected", r"\bnot detected\b", r"\bno\b.*\bmutation", r"\bwild[- ]?type\b",
               r"\bnegative\b"),
        )),
        Field("alk", "lung", "ALK status", "What is the ALK rearrangement status of the tumor?", (
            _v("rearranged", "ALK rearrangement detected", r"\brearrange(d|ment) (was |is )?detected\b",
               r"\bpositive\b", r"\bfusion detected\b", r"^\s*(alk )?rearranged\b"),
            _v("none", "No ALK rearrangement detected", r"\bnot detected\b", r"\bno\b.*\brearrangement",
               r"\bnegative\b"),
        )),
        Field("pdl1", "lung", "PD-L1 TPS", "What is the PD-L1 tumor proportion score (TPS)?", tuple(
            [_v("<1", "PD-L1 TPS <1%", r"<\s*1\s*%", r"less than 1\s*%")]
            + [_v(str(p), f"PD-L1 TPS {p}%", rf"(?<![\d<.]){p}\s*%") for p in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90)]
        )),
        Field("ltherapy", "lung", "Current systemic therapy",
              "What systemic therapy is the patient currently receiving, and at what dose?", (
            _v("osimertinib 80 mg daily", "Osimertinib 80 mg daily", r"\bosimertinib\b"),
            _v("alectinib 600 mg twice daily", "Alectinib 600 mg twice daily", r"\balectinib\b"),
            _v("pembrolizumab 200 mg every 3 weeks", "Pembrolizumab 200 mg every 3 weeks", r"\bpembrolizumab\b"),
        )),
    ]
}

BREAST = ["laterality", "histology", "grade", "er", "her2", "btherapy"]
LUNG = ["lobe", "lhistology", "egfr", "alk", "pdl1", "ltherapy"]
THERAPY_FIELDS = {"btherapy", "ltherapy"}

# Dose strings the checker requires for a therapy answer to count as correct.
THERAPY_DOSE = {
    "letrozole 2.5 mg daily": r"2\.5\s*mg",
    "anastrozole 1 mg daily": r"(?<![\d.])1\s*mg",
    "tamoxifen 20 mg daily": r"20\s*mg",
    "exemestane 25 mg daily": r"25\s*mg",
    "osimertinib 80 mg daily": r"80\s*mg",
    "alectinib 600 mg twice daily": r"600\s*mg",
    "pembrolizumab 200 mg every 3 weeks": r"200\s*mg",
}


def fields_for(cancer: str) -> list[str]:
    return BREAST if cancer == "breast" else LUNG


def value_terms() -> set[str]:
    """Words that carry a registry value; the typo slice never touches them."""
    words: set[str] = set()
    for f in FIELDS.values():
        for v in f.values:
            words.update(w.lower() for w in re.findall(r"[A-Za-z0-9]+", v.display))
    words.update({"positive", "negative", "detected", "pending", "mother", "sister", "aunt", "family",
                  "not", "no", "declined", "discussed", "started", "performed", "insufficient"})
    return words
