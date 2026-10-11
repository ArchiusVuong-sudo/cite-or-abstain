from coa import db
from coa.checker import Gold, Label
from coa.config import ABSTAIN
from coa.dataset import QuestionView
from coa.pairs import Sample, build_pair, mix, perturb


def q(answerable=True, field="er", value="positive", display="ER positive"):
    return QuestionView("B001-er", "B001", "train", field, "What is the ER status?",
                        Gold(field, answerable, value if answerable else None, frozenset({2}) if answerable else
                             frozenset()), display if answerable else None, answerable, None,
                        [("x1", "L", "a"), ("x2", "L", "b"), ("x3", "L", "c")])


def test_rejected_prefers_own_invented_answer():
    p = build_pair(q(False), [Sample("greedy", ABSTAIN, Label.CORRECT_ABSTAIN, None),
                              Sample("t0.8", "Pending", Label.NON_ANSWER, None),
                              Sample("t0.8", "ER positive [P1]", Label.INVENTED, False)])
    assert (p.chosen, p.chosen_source) == (ABSTAIN, "model")
    assert (p.rejected, p.rejected_source, p.rejected_label) == ("ER positive [P1]", "model", "invented")


def test_chosen_falls_back_to_reference_when_format_is_messy():
    p = build_pair(q(), [Sample("greedy", "[ER positive (P2)] and more", Label.CORRECT, False),
                         Sample("t0.8", ABSTAIN, Label.FALSE_REFUSAL, None)])
    assert (p.chosen, p.chosen_source) == ("ER positive [P2]", "reference")
    assert p.rejected_label == "false_refusal"


def test_perturbation_when_model_never_fails():
    good = [Sample("greedy", "ER positive [P2]", Label.CORRECT, True)]
    p = build_pair(q(), good)
    assert p.rejected_source == "perturbation" and p.rejected_label in {"false_refusal", "wrong_value"}
    assert p.rejected != p.chosen
    text, label = perturb(q(False))
    assert label == "invented" and text.endswith("[P1]")


def test_pair_mix(with_samples):
    m = mix(with_samples)
    assert m["total"] == 240
    assert m["by_label"].get("invented", 0) > 20 and m["by_label"].get("false_refusal", 0) > 20
    own = db.one(with_samples, "SELECT COUNT(*) AS n FROM pairs WHERE rejected_source='model' "
                               "AND rejected_label IN ('invented', 'false_refusal')")["n"]
    assert own > 0  # the model's own hallucinations and refusals, not only perturbations
