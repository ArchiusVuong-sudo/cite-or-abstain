import pytest

from coa.checker import Check, Gold, Label
from coa.dataset import QuestionView
from coa.evaluate import gate, score, wilson


def test_wilson():
    r = wilson(8, 10)
    assert r.rate == 0.8 and r.lo == pytest.approx(0.4902, abs=1e-3) and r.hi == pytest.approx(0.9433, abs=1e-3)
    zero = wilson(0, 66)
    assert zero.lo == 0.0 and 0 < zero.hi < 0.06
    assert wilson(5, 5).hi == 1.0
    assert wilson(0, 0).rate is None


def _q(answerable):
    return QuestionView("q", "p", "test", "er", "?", Gold("er", answerable, "positive" if answerable else None,
                                                          frozenset({1})), None, answerable, None, [])


def test_score_rates():
    res = [(_q(False), Check(Label.INVENTED, "positive", (1,), False)),
           (_q(False), Check(Label.CORRECT_ABSTAIN, None, (), None)),
           (_q(True), Check(Label.CORRECT, "positive", (1,), True)),
           (_q(True), Check(Label.FALSE_REFUSAL, None, (), None))]
    s = score(res)
    assert (s["invented_rate"].k, s["invented_rate"].n) == (1, 2)
    assert (s["accuracy"].k, s["accuracy"].n) == (1, 2)
    assert (s["false_refusal_rate"].k, s["false_refusal_rate"].n) == (1, 2)
    assert (s["citation_validity"].k, s["citation_validity"].n) == (1, 2)


def _m(inv, fr):
    return {"invented_rate": {"rate": inv}, "false_refusal_rate": {"rate": fr}}


def test_gate():
    assert gate({"base": _m(0.36, 0.17), "two_shot": _m(0.80, 0.13), "tuned": _m(0.09, 0.0)})["passed"]
    near = gate({"base": _m(0.30, 0.10), "two_shot": _m(0.80, 0.10), "tuned": _m(0.20, 0.10)})
    assert not near["passed"] and not near["invented_ok"]
    refuses = gate({"base": _m(0.40, 0.05), "two_shot": _m(0.80, 0.05), "tuned": _m(0.0, 0.30)})
    assert refuses["invented_ok"] and not refuses["false_refusal_ok"]
    assert gate({"base": _m(0.3, 0.1)}) == {"evaluated": False}
