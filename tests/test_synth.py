from collections import Counter

from coa import db
from coa.fields import fields_for
from coa.synth import STAMP, generate


def test_synth_deterministic():
    a, b = generate(7), generate(7)
    assert [p.record_hash() for p in a] == [p.record_hash() for p in b]
    assert [p.split for p in a] == [p.split for p in b]
    assert [p.record_hash() for p in generate(8)] != [p.record_hash() for p in a]


def test_split_by_patient(conn):
    splits = Counter(p.split for p in generate(7))
    assert splits == {"train": 40, "dev": 16, "test": 24}
    rows = db.rows(conn, "SELECT q.patient_id, q.split AS qs, p.split AS ps FROM questions q "
                         "JOIN patients p ON p.id = q.patient_id")
    assert all(r["qs"] == r["ps"] for r in rows)
    per_patient = {}
    for r in rows:
        per_patient.setdefault(r["patient_id"], set()).add(r["qs"])
    assert all(len(v) == 1 for v in per_patient.values())


def test_cohort_shape():
    ps = generate(7)
    assert Counter(p.cancer for p in ps) == {"breast": 40, "lung": 40}
    assert all(len(p.notes) == 3 and all(n.render().startswith(STAMP) for n in p.notes) for p in ps)
    fields = [(p, f) for p in ps for f in fields_for(p.cancer)]
    assert len(fields) == 480
    missing = sum(not p.documented[f] for p, f in fields) / len(fields)
    assert 0.25 < missing < 0.5


def test_traps_only_on_undocumented_fields():
    for p in generate(7):
        for f in p.traps:
            assert not p.documented[f]


def test_sections_state_only_documented_values():
    for p in generate(7):
        for n in p.notes:
            for s in n.sections:
                for f, v in s.supports.items():
                    assert p.documented[f] and p.values[f] == v
