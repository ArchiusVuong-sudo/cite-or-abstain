"""Seeded generator of synthetic breast and lung cancer patients (no LLM, no real data).

Each patient gets three short notes (pathology report, oncology consult, treatment summary).
Every section records which registry facts it states, so labels never depend on parsing text back.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from .fields import FIELDS, fields_for

STAMP = "SYNTHETIC - not a real patient"
MISSING_RATE = 0.35

FIRST_F = ["Mara", "Ilka", "Tamsin", "Odile", "Rhea", "Juno", "Wren", "Marit", "Liesl", "Noor", "Asha", "Calla",
           "Petra", "Signe", "Yara", "Brisa", "Coralie", "Dagny", "Elin", "Freya", "Ines", "Lorna", "Mirela", "Sabine"]
FIRST_M = ["Tobin", "Anselm", "Bram", "Caspian", "Dorian", "Emrys", "Fintan", "Gideon", "Hollis", "Ivo", "Joris",
           "Kestrel", "Lucan", "Matteo", "Niall", "Orrin", "Pascal", "Rafferty", "Soren", "Teodor"]
SUR_A = ["Var", "Quil", "Hal", "Bren", "Tol", "Oder", "Sael", "Cor", "Mel", "Thal", "Ost", "Prav", "Lund", "Kest"]
SUR_B = ["nell", "lan", "vik", "nick", "var", "in", "ind", "vath", "dor", "sen", "rick", "holm", "by", "wick"]
CLINICIANS = ["Ilse Varga", "Rowan Achterberg", "Selin Dumont", "Arlo Benedek", "Pia Lindqvist", "Taro Okafor",
              "Nadia Ferreira", "Elias Kranz"]


@dataclass
class Section:
    name: str
    text: str
    supports: dict[str, str] = field(default_factory=dict)


@dataclass
class Note:
    kind: str
    title: str
    header: str
    sections: list[Section]

    def render(self) -> str:
        body = "\n".join(f"{s.name}: {s.text}" for s in self.sections)
        return f"{STAMP}\n{self.title.upper()}\n{self.header}\n{body}\n"


@dataclass
class Identity:
    first: str
    last: str
    title: str
    mrn: str
    phone: str
    dob: str
    clinician: str


@dataclass
class Patient:
    id: str
    cancer: str
    split: str
    identity: Identity
    values: dict[str, str]
    documented: dict[str, bool]
    traps: dict[str, str]
    visit_dates: list[str]
    notes: list[Note]

    def record_hash(self) -> str:
        blob = json.dumps({"id": self.id, "notes": [n.render() for n in self.notes]}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _fmt(d: date) -> str:
    return d.strftime("%m/%d/%Y")


def _identity(rng: random.Random, cancer: str, used: set[str]) -> Identity:
    female = cancer == "breast" or rng.random() < 0.5
    while True:
        first = rng.choice(FIRST_F if female else FIRST_M)
        last = rng.choice(SUR_A) + rng.choice(SUR_B)
        if f"{first} {last}" not in used:
            used.add(f"{first} {last}")
            break
    dob = date(rng.randint(1942, 1976), rng.randint(1, 12), rng.randint(1, 28))
    return Identity(
        first=first, last=last, title="Ms." if female else "Mr.",
        mrn=str(rng.randint(4_000_000, 4_999_999)),
        phone=f"(555) 01{rng.randint(0, 9)}-{rng.randint(1000, 9999)}",
        dob=_fmt(dob), clinician=rng.choice(CLINICIANS),
    )


def _breast_values(rng: random.Random) -> dict[str, str]:
    er = "positive" if rng.random() < 0.75 else "negative"
    v = {
        "laterality": rng.choice(["left", "right"]),
        "histology": "idc" if rng.random() < 0.8 else "ilc",
        "grade": rng.choice(["1", "2", "2", "3"]),
        "er": er,
        "her2": "positive" if rng.random() < 0.2 else "negative",
    }
    if er == "positive":
        v["btherapy"] = rng.choice(["letrozole 2.5 mg daily", "anastrozole 1 mg daily", "tamoxifen 20 mg daily",
                                    "exemestane 25 mg daily"])
    return v


def _lung_values(rng: random.Random) -> dict[str, str]:
    hist = "adenocarcinoma" if rng.random() < 0.7 else "squamous"
    egfr, alk = "none", "none"
    if hist == "adenocarcinoma":
        r = rng.random()
        egfr = "exon19" if r < 0.2 else "l858r" if r < 0.35 else "none"
        if egfr == "none" and rng.random() < 0.12:
            alk = "rearranged"
    pdl1 = rng.choice(["<1", "5", "10", "20", "40", "50", "60", "70", "80", "90"])
    v = {"lobe": rng.choice(["rul", "rml", "rll", "lul", "lll"]), "lhistology": hist, "egfr": egfr, "alk": alk,
         "pdl1": pdl1}
    if egfr != "none":
        v["ltherapy"] = "osimertinib 80 mg daily"
    elif alk == "rearranged":
        v["ltherapy"] = "alectinib 600 mg twice daily"
    elif pdl1 not in {"<1", "5", "10", "20", "40"}:
        v["ltherapy"] = "pembrolizumab 200 mg every 3 weeks"
    return v


# Trap kinds used when a field is not documented: a misleading mention that is not the patient's value.
TRAPS: dict[str, list[str]] = {
    "histology": ["pending"], "grade": ["pending"], "er": ["family", "pending"], "her2": ["family", "pending"],
    "btherapy": ["not_started", ""], "laterality": [""],
    "lobe": [""], "lhistology": ["nos"], "egfr": ["not_performed", "family"], "alk": ["pending"],
    "pdl1": ["pending"], "ltherapy": ["not_started", ""],
}


def _breast_notes(p: Patient, rng: random.Random, d: list[date]) -> list[Note]:
    v, doc, tr, idn = p.values, p.documented, p.traps, p.identity
    side = v["laterality"].capitalize() if doc["laterality"] else ""
    path: list[Section] = []
    if doc["laterality"]:
        path.append(Section("Specimen", f"{side} breast, ultrasound-guided core needle biopsy at the "
                                        f"{rng.choice([2, 3, 10, 11])} o'clock position.", {"laterality": v["laterality"]}))
    else:
        path.append(Section("Specimen", "Breast, ultrasound-guided core needle biopsy; side not stated on the "
                                        "requisition."))
    hist_txt = FIELDS["histology"].value(v["histology"]).display
    if doc["histology"] and doc["grade"]:
        path.append(Section("Diagnosis", f"{hist_txt}, Nottingham grade {v['grade']}.",
                            {"histology": v["histology"], "grade": v["grade"]}))
    elif doc["histology"]:
        path.append(Section("Diagnosis", f"{hist_txt}. Grade will be assigned on the excision specimen.",
                            {"histology": v["histology"]}))
    elif doc["grade"]:
        path.append(Section("Diagnosis", f"Invasive carcinoma, Nottingham grade {v['grade']}; histologic type is "
                                         "pending further stains.", {"grade": v["grade"]}))
    else:
        path.append(Section("Diagnosis", "Invasive carcinoma; histologic type and grade are pending further "
                                         "stains."))
    bio: list[str] = []
    sup: dict[str, str] = {}
    if doc["er"]:
        pct = rng.choice([80, 90, 95]) if v["er"] == "positive" else 0
        bio.append(f"Estrogen receptor (ER): {v['er']} ({pct}% of tumor cells).")
        sup["er"] = v["er"]
    elif tr.get("er") == "pending":
        bio.append("Estrogen receptor (ER) studies are pending.")
    if doc["her2"]:
        ihc = "3+" if v["her2"] == "positive" else rng.choice(["0", "1+"])
        bio.append(f"HER2 by IHC: {v['her2']} ({ihc}).")
        sup["her2"] = v["her2"]
    elif tr.get("her2") == "pending":
        bio.append("HER2 testing has been sent to the reference laboratory; result pending.")
    bio.append("Progesterone receptor: reported separately.")
    path.append(Section("Biomarkers", " ".join(bio), sup))

    age = d[0].year - int(idn.dob[-4:])
    consult: list[Section] = [Section(
        "History",
        f"{idn.title} {idn.last} is a {age}-year-old woman seen at the request of Dr. {idn.clinician} for a newly "
        f"diagnosed {side.lower() + ' ' if side else ''}breast cancer found on screening mammogram.",
        {"laterality": v["laterality"]} if side else {},
    )]
    if tr.get("er") == "family":
        consult.append(Section("Family history", "Her mother was treated for ER-positive breast cancer at age 61."))
    elif tr.get("her2") == "family":
        consult.append(Section("Family history", "Her sister had HER2-positive breast cancer at age 49."))
    else:
        consult.append(Section("Family history", rng.choice(["No family history of breast or ovarian cancer.",
                                                             "Father with type 2 diabetes; no known cancers."])))
    summary, ssup = [], {}
    if doc["histology"]:
        summary.append(hist_txt.lower())
        ssup["histology"] = v["histology"]
    if doc["er"]:
        summary.append(f"ER {v['er']}")
        ssup["er"] = v["er"]
    consult.append(Section("Assessment", ("Early-stage breast cancer: " + ", ".join(summary) + ".") if summary
                           else "Early-stage breast cancer; workup in progress.", ssup))
    consult.append(Section("Plan", "Reviewed the pathology with the patient. Follow-up visit after surgery."))

    tx: list[Section] = []
    if doc.get("btherapy") and v.get("btherapy"):
        tx.append(Section("Current treatment", f"She is currently taking {v['btherapy']}, started on {_fmt(d[1])}.",
                          {"btherapy": v["btherapy"]}))
        tx.append(Section("Tolerance", rng.choice(["She reports mild hot flashes and joint stiffness.",
                                                   "She reports no side effects at this time."])))
    elif tr.get("btherapy") == "not_started":
        drug = rng.choice(["Tamoxifen", "Letrozole"])
        tx.append(Section("Current treatment", f"{drug} was discussed at the last visit, but it has not been "
                                               "started."))
    else:
        tx.append(Section("Current treatment", f"She completed radiation therapy on {_fmt(d[1])}."))
    tx.append(Section("Follow-up", f"Return visit on {_fmt(d[2])} with routine labs."))
    return [
        Note("pathology", "Pathology report", _header(idn, d[0], "Collected"), path),
        Note("consult", "Oncology consult", _header(idn, d[0] + timedelta(days=9), "Visit"), consult),
        Note("treatment", "Treatment summary", _header(idn, d[2] - timedelta(days=30), "Visit"), tx),
    ]


def _lung_notes(p: Patient, rng: random.Random, d: list[date]) -> list[Note]:
    v, doc, tr, idn = p.values, p.documented, p.traps, p.identity
    lobe = FIELDS["lobe"].value(v["lobe"]).display.lower() if doc["lobe"] else ""
    size = rng.choice(["2.1", "2.8", "3.4", "4.0"])
    path = [Section("Specimen", f"CT-guided core biopsy of a {size} cm mass in the {lobe}." if lobe else
                    f"CT-guided core biopsy of a {size} cm lung mass.", {"lobe": v["lobe"]} if lobe else {})]
    if doc["lhistology"]:
        path.append(Section("Diagnosis", FIELDS["lhistology"].value(v["lhistology"]).display + ".",
                            {"lhistology": v["lhistology"]}))
    else:
        path.append(Section("Diagnosis", "Non-small cell carcinoma; further subtyping is limited by the sample "
                                         "size."))
    bio, sup = [], {}
    if doc["egfr"]:
        bio.append({"exon19": "EGFR: exon 19 deletion detected.", "l858r": "EGFR: L858R mutation detected.",
                    "none": "EGFR: no mutation detected."}[v["egfr"]])
        sup["egfr"] = v["egfr"]
    elif tr.get("egfr") == "not_performed":
        bio.append("EGFR testing was not performed due to insufficient tissue.")
    if doc["alk"]:
        bio.append("ALK FISH: rearrangement detected." if v["alk"] == "rearranged" else
                   "ALK FISH: no rearrangement detected.")
        sup["alk"] = v["alk"]
    elif tr.get("alk") == "pending":
        bio.append("ALK FISH is pending.")
    if doc["pdl1"]:
        bio.append(f"PD-L1 (22C3) tumor proportion score: {v['pdl1']}%.")
        sup["pdl1"] = v["pdl1"]
    elif tr.get("pdl1") == "pending":
        bio.append("PD-L1 testing has been requested; result pending.")
    if not bio:
        bio.append("Molecular testing has been requested.")
    path.append(Section("Biomarkers", " ".join(bio), sup))

    age = d[0].year - int(idn.dob[-4:])
    pronoun = "woman" if idn.title == "Ms." else "man"
    consult = [Section("History", f"{idn.title} {idn.last} is a {age}-year-old {pronoun}, former smoker, seen at the "
                                  f"request of Dr. {idn.clinician} for a newly diagnosed "
                                  f"{lobe + ' ' if lobe else 'lung '}mass.", {"lobe": v["lobe"]} if lobe else {})]
    if tr.get("egfr") == "family":
        consult.append(Section("Family history", "A brother was treated for EGFR-mutated lung cancer."))
    else:
        consult.append(Section("Family history", rng.choice(["Father had coronary artery disease.",
                                                             "No family history of lung cancer."])))
    consult.append(Section("Plan", "Reviewed the biopsy results and next steps with the patient."))

    tx: list[Section] = []
    if doc.get("ltherapy") and v.get("ltherapy"):
        pr = "She" if idn.title == "Ms." else "He"
        tx.append(Section("Current treatment", f"{pr} is currently receiving {v['ltherapy']}, started on "
                                               f"{_fmt(d[1])}.", {"ltherapy": v["ltherapy"]}))
        tx.append(Section("Tolerance", rng.choice(["Mild fatigue; no dose changes.", "Grade 1 rash, managed "
                                                   "with topical care."])))
    elif tr.get("ltherapy") == "not_started":
        tx.append(Section("Current treatment", "Pembrolizumab was considered; treatment has not yet started."))
    else:
        tx.append(Section("Current treatment", f"Treatment will be decided at the tumor board on {_fmt(d[1])}."))
    tx.append(Section("Follow-up", f"Repeat CT chest before the visit on {_fmt(d[2])}."))
    return [
        Note("pathology", "Pathology report", _header(idn, d[0], "Collected"), path),
        Note("consult", "Oncology consult", _header(idn, d[0] + timedelta(days=8), "Visit"), consult),
        Note("treatment", "Treatment summary", _header(idn, d[2] - timedelta(days=28), "Visit"), tx),
    ]


def _header(idn: Identity, d: date, label: str) -> str:
    return (f"Patient: {idn.first} {idn.last}   MRN: {idn.mrn}   DOB: {idn.dob}\n"
            f"{label}: {_fmt(d)}   Phone: {idn.phone}   Clinician: Dr. {idn.clinician}")


def generate(seed: int = 7, n_breast: int = 40, n_lung: int = 40) -> list[Patient]:
    """Build the cohort. Same seed, same patients, byte for byte."""
    rng = random.Random(seed)
    used: set[str] = set()
    patients: list[Patient] = []
    for cancer, n in (("breast", n_breast), ("lung", n_lung)):
        for i in range(n):
            prng = random.Random(f"{seed}:{cancer}:{i}")
            pid = f"{'B' if cancer == 'breast' else 'L'}{i + 1:03d}"
            values = _breast_values(prng) if cancer == "breast" else _lung_values(prng)
            documented, traps = {}, {}
            for f in fields_for(cancer):
                has_value = f in values
                documented[f] = has_value and prng.random() >= MISSING_RATE
                if not documented[f]:
                    trap = prng.choice(TRAPS[f])
                    if trap:
                        traps[f] = trap
            if traps.get("er") == "family" and traps.get("her2") == "family":
                traps["her2"] = "pending"
            start = date(2025, 1, 6) + timedelta(days=prng.randint(0, 420))
            d = [start, start + timedelta(days=prng.randint(20, 40)), start + timedelta(days=prng.randint(90, 140))]
            p = Patient(pid, cancer, "", _identity(prng, cancer, used), values, documented, traps,
                        [_fmt(x) for x in d], [])
            p.notes = _breast_notes(p, prng, d) if cancer == "breast" else _lung_notes(p, prng, d)
            patients.append(p)
    _assign_splits(patients, rng)
    return patients


def _assign_splits(patients: list[Patient], rng: random.Random) -> None:
    """Split by patient, stratified by cancer type: half train, a fifth dev, the rest test."""
    for cancer in ("breast", "lung"):
        group = [p for p in patients if p.cancer == cancer]
        rng.shuffle(group)
        n = len(group)
        n_train, n_dev = n // 2, n // 5
        for j, p in enumerate(group):
            p.split = "train" if j < n_train else "dev" if j < n_train + n_dev else "test"


def to_json(p: Patient) -> dict:
    return asdict(p)
