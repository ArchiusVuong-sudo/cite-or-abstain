import pytest

from coa.checker import Gold, Label, canonical_value, check, value_spans
from coa.config import ABSTAIN

ER_POS = Gold("er", True, "positive", frozenset({2}))
ER_NONE = Gold("er", False, None, frozenset())  # e.g. only the mother's ER status is in the passages
HER2_NEG = Gold("her2", True, "negative", frozenset({1}))
HER2_PENDING = Gold("her2", False, None, frozenset())
GRADE2 = Gold("grade", True, "2", frozenset({1}))
LET = Gold("btherapy", True, "letrozole 2.5 mg daily", frozenset({1}))
THERAPY_NONE = Gold("btherapy", False, None, frozenset())
EGFR19 = Gold("egfr", True, "exon19", frozenset({1}))
EGFR_NONE_DOC = Gold("egfr", False, None, frozenset())
ALK_NEG = Gold("alk", True, "none", frozenset({1}))
PDL1 = Gold("pdl1", True, "60", frozenset({1}))
PDL1_LOW = Gold("pdl1", True, "<1", frozenset({1}))
LOBE = Gold("lobe", True, "rul", frozenset({1, 2}))
HIST_NOS = Gold("lhistology", False, None, frozenset())

CASES = [
    # (answer, gold, label, citation_valid)
    ("ER positive [P2]", ER_POS, Label.CORRECT, True),
    ("Positive [P1]", ER_POS, Label.CORRECT, False),  # right value, wrong passage
    ("ER positive", ER_POS, Label.CORRECT, False),  # no citation
    ("ER negative [P2]", ER_POS, Label.WRONG_VALUE, False),
    (ABSTAIN, ER_POS, Label.FALSE_REFUSAL, None),
    ("ER status is not stated [P2]", ER_POS, Label.FALSE_REFUSAL, None),
    ("ER positive [P1]", ER_NONE, Label.INVENTED, False),  # family-history trap: mother's ER read as the patient's
    (ABSTAIN, ER_NONE, Label.CORRECT_ABSTAIN, None),
    ("[Not documented in the provided notes.]", ER_NONE, Label.CORRECT_ABSTAIN, None),
    ("Pending [P1]", HER2_PENDING, Label.NON_ANSWER, None),  # pending-result trap: not a value, not an invention
    ("HER2 negative [P1]", HER2_PENDING, Label.INVENTED, False),
    ("HER2 negative (IHC 1+) [P1]", HER2_NEG, Label.CORRECT, True),
    ("HER2 1+ [P1]", HER2_NEG, Label.CORRECT, True),
    ("Nottingham grade 2 [P1]", GRADE2, Label.CORRECT, True),
    ("2 [P1]", GRADE2, Label.CORRECT, True),
    ("Grade III [P1]", GRADE2, Label.WRONG_VALUE, False),
    ("Letrozole 2.5 mg daily [P1]", LET, Label.CORRECT, True),
    ("Letrozole [P1]", LET, Label.WRONG_VALUE, False),  # dose was asked
    ("Anastrozole 1 mg daily [P1]", LET, Label.WRONG_VALUE, False),
    ("Tamoxifen 20 mg daily [P1]", THERAPY_NONE, Label.INVENTED, False),  # "discussed but not started" trap
    ("EGFR exon 19 deletion [P1]", EGFR19, Label.CORRECT, True),
    ("Exon 19 del detected [P1]", EGFR19, Label.CORRECT, True),
    ("EGFR-mutated [P2]", EGFR_NONE_DOC, Label.INVENTED, False),  # a brother's EGFR status
    ("No ALK rearrangement detected [P1]", ALK_NEG, Label.CORRECT, True),
    ("ALK rearrangement detected [P1]", ALK_NEG, Label.WRONG_VALUE, False),
    ("PD-L1 TPS 60% [P1]", PDL1, Label.CORRECT, True),
    ("PD-L1 TPS 6% [P1]", PDL1, Label.WRONG_VALUE, False),
    ("TPS <1% [P1]", PDL1_LOW, Label.CORRECT, True),
    ("Right upper lobe [P2]", LOBE, Label.CORRECT, True),
    ("RUL [P1]", LOBE, Label.CORRECT, True),
    ("Adenocarcinoma [P3]", HIST_NOS, Label.INVENTED, False),
    ("", ER_POS, Label.NON_ANSWER, None),
    ("[P1]", ER_POS, Label.NON_ANSWER, None),
]


@pytest.mark.parametrize("answer,gold,label,cv", CASES, ids=[f"{i}:{c[0][:24]}" for i, c in enumerate(CASES)])
def test_checker(answer, gold, label, cv):
    r = check(answer, gold)
    assert r.label == label
    assert r.citation_valid == cv


def test_checker_has_25_plus_cases():
    assert len(CASES) >= 25


def test_cited_ids_parsed():
    assert check("ER positive [P2] [ P3 ]", ER_POS).cited == (2, 3)


def test_canonical_value_takes_earliest_phrase():
    assert canonical_value("egfr", "Exon 19 deletion detected; no other mutation") == "exon19"
    assert canonical_value("er", "ER-positive") == "positive"


def test_value_spans_highlight_family_history_value():
    text = "Her mother was treated for ER-positive breast cancer."
    spans = value_spans("er", text)
    assert [text[a:b] for a, b in spans] == ["positive"]
