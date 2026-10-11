"""Prompt formats shared by sampling, training data, evaluation and the Axolotl export."""

from __future__ import annotations

from .config import ABSTAIN

SYSTEM = (
    "You abstract cancer registry fields from clinical notes. Use only the passages given. "
    "Answer with the patient's value followed by the passage id in brackets, for example: "
    "Squamous cell carcinoma [P3]. "
    f"If the passages do not state this patient's value, answer exactly: {ABSTAIN}"
)


def user_message(question: str, passages: list[tuple[str, str]]) -> str:
    """passages: (label, text) in retrieval order; ids are P1..Pk by position."""
    lines = [f"[P{i + 1}] ({label}) {text}" for i, (label, text) in enumerate(passages)]
    return "Passages:\n" + "\n".join(lines) + f"\n\nQuestion: {question}"


def messages(question: str, passages: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(question, passages)}]


# Two worked examples for the 2-shot baseline: one answer with a citation, one family-history abstain.
SHOTS: list[tuple[str, list[tuple[str, str]], str]] = [
    (
        "What is the HER2 status of the tumor?",
        [("Pathology report - Biomarkers", "Estrogen receptor (ER): positive (90% of tumor cells). HER2 by IHC: "
                                           "negative (1+)."),
         ("Oncology consult - Plan", "Reviewed the pathology with the patient."),
         ("Treatment summary - Follow-up", "Return visit in three months.")],
        "HER2 negative [P1]",
    ),
    (
        "What is the estrogen receptor (ER) status of the tumor?",
        [("Oncology consult - Family history", "Her mother was treated for ER-positive breast cancer."),
         ("Pathology report - Biomarkers", "HER2 testing has been sent out; result pending."),
         ("Pathology report - Diagnosis", "Invasive ductal carcinoma, Nottingham grade 2.")],
        ABSTAIN,
    ),
]


def two_shot_messages(question: str, passages: list[tuple[str, str]]) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, ps, a in SHOTS:
        msgs.append({"role": "user", "content": user_message(q, ps)})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_message(question, passages)})
    return msgs
