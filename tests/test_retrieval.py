from coa import dataset
from coa.retrieval import retrieve, tokenize


def test_tokenize_splits_hyphens_and_drops_stopwords():
    assert tokenize("The ER-positive tumor") == ["er", "positive"]


def test_retrieve_is_deterministic_and_ranked():
    ps = [("a", "No family history."), ("b", "Estrogen receptor (ER): positive."), ("c", "Grade 2.")]
    hits = retrieve("estrogen receptor er status", ps, 2)
    assert [h.passage_id for h in hits] == ["b", "a"] or hits[0].passage_id == "b"
    assert retrieve("estrogen receptor er status", ps, 2) == hits


def test_recall_at_3(conn):
    r = dataset.recall_at_k(conn)
    assert r["k"] == 3 and r["documented"] > 0 and 0 <= r["recall"] <= 1


def test_family_history_trap_is_retrieved(conn):
    traps = [q for q in dataset.question_views(conn) if q.trap == "family" and q.field == "er"]
    assert traps
    assert any("mother" in text.lower() for q in traps for _, _, text in q.passages)
    assert all(not q.gold.answerable for q in traps)
