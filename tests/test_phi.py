import re
from datetime import datetime

from coa import db, dataset
from coa.phi import DATE_RE, Scrubber, date_shift, leak_scan


def test_leak_gate(conn):
    scrubbed = [(r["patient_id"], r["scrubbed_text"]) for r in db.rows(conn, "SELECT * FROM notes")]
    passages = [(r["patient_id"], r["text"]) for r in db.rows(conn, "SELECT * FROM passages")]
    assert dataset.leak_gate(conn, scrubbed + passages) == []
    raw = [(r["patient_id"], r["raw_text"]) for r in db.rows(conn, "SELECT * FROM notes")]
    assert len(dataset.leak_gate(conn, raw)) > 100  # the gate does catch identifiers when they are there


def test_leak_gate_on_exported_jsonl(with_samples):
    from coa.axolotl_export import export_rows

    ex = export_rows(with_samples)
    assert len(ex.rows) == 240 and ex.leaks == []


def test_surrogates_consistent_across_notes():
    s = Scrubber("k", 10)
    a = s.scrub("Patient: Mara Varnell   MRN: 4123456\nMs. Varnell is here.").text
    b = s.scrub("Ms. Varnell returns. Mara is well. MRN 4123456").text
    assert "Varnell" not in a + b and "Mara" not in a + b and "4123456" not in a + b
    sur_last = re.search(r"Ms\. (\w+)", a).group(1)
    assert f"Ms. {sur_last} returns" in b


def test_dates_shift_keeps_intervals():
    s = Scrubber("k", 30)
    out = s.scrub("Collected: 01/10/2025 then 01/20/2025").text
    d = [datetime.strptime(m.group(0), "%m/%d/%Y") for m in DATE_RE.finditer(out)]
    assert d[0].strftime("%m/%d/%Y") == "02/09/2025" and (d[1] - d[0]).days == 10


def test_clinician_and_phone_scrubbed():
    out = Scrubber("k", 5).scrub("Clinician: Dr. Ilse Varga   Phone: (555) 012-3456").text
    assert "Ilse" not in out and "Varga" not in out and "012-3456" not in out and "Dr. " in out


def test_date_shift_never_lands_on_an_original_date():
    texts = ["01/01/2025 01/11/2025 01/21/2025"]
    shift = date_shift(7, "X", texts)
    out = Scrubber("k", shift).scrub(texts[0]).text
    assert shift != 0 and leak_scan([out], ["01/01/2025", "01/11/2025", "01/21/2025"]) == []
