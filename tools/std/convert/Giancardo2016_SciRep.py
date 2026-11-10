"""Giancardo et al. (2016), Sci Rep: neuroQWERTY MIT-CSXPD keystroke logs of 85 people (42 with Parkinson's disease, 43 controls) typing
freely in a word processor; one row per key press.

Run from the site folder: python3 -I tools/std/convert/Giancardo2016_SciRep.py
Raw file (untrusted data, read only): raw_extra/Giancardo2016_SciRep/neuroQWERTY.zip from https://physionet.org/content/nqmitcsxpd/1.0.0/
(read as bytes inside the zip; nothing in it is executed; nqDataLoader.py and readme.ipynb were read as text).
PRIVACY: people typed freely, so the key identity would spell what they wrote. It is never written out: key_pressed holds only the
class of the key (character, space, BackSpace, ...), and hand/direction the keyboard side.
"""
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Giancardo2016_SciRep"
RD = stdlib.raw_dir(ID)
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(RD / "SOURCES.json"))}
checks.internal("raw zip matches the downloaded hash", stdlib.sha256(RD / "neuroQWERTY.zip") == src["neuroQWERTY.zip"]["sha256"])
z = zipfile.ZipFile(RD / "neuroQWERTY.zip")

# ------------------------------------------------------------------ key classes and keyboard side (Spanish layout)
LEFT = set("qwertasdfgzxcvb12345")
RIGHT = set("yuiophjklnm6789" + "0")


def key_class(k):
    if len(k) == 1:
        return "character"
    if k == "space":
        return "space"
    if k == "BackSpace":
        return "backspace"
    if k in ("Return", "P_Enter"):
        return "return"
    if k.startswith("Shift"):
        return "shift"
    if k.startswith(("Control", "Alt", "Super", "Menu")):
        return "control/alt"
    if k == "Caps_Lock":
        return "caps_lock"
    if k.startswith("mouse"):
        return "mouse"
    if k in ("Left", "Right", "Up", "Down") or k.startswith("P_") and k[2:] in ("Left", "Right", "Up", "Down", "Home", "End", "Next", "Page_Up", "Page_Down", "Insert"):
        return "navigation"
    if k in ("Delete", "End", "Insert", "Tab", "Escape", "Num_Lock"):
        return "other"
    if k.startswith("P_"):
        return "keypad"
    if k == "":
        return "other"
    return "punctuation"          # period, comma, colon, ntilde, acute, dead keys ([65105]) ...


def side(k):
    if k == "space":
        return "S"
    if len(k) == 1:
        c = k.lower()
        return "L" if c in LEFT else ("R" if c in RIGHT else "")
    if k in ("Shift_L", "Control_L", "Alt_L", "Super_L", "Tab", "Caps_Lock", "Escape", "less"):
        return "L"
    if k in ("Shift_R", "Control_R", "Return", "BackSpace", "ntilde", "Delete"):
        return "R"
    return ""


def sanity(start, end, ht):
    """Replicates NqDataLoader.sanityCheck (rows the authors' loader removes)."""
    bad = (start <= 0) | (end <= 0) | (ht < 0) | (ht >= 5)
    tmp = start.copy()
    nonc = np.zeros(len(start), bool)
    cur = np.ones(len(start), bool)
    while cur.sum() > 0:
        cur = np.r_[False, np.diff(tmp) < 0]
        nonc |= cur
        idx = np.flatnonzero(cur)
        tmp[idx] = tmp[idx - 1]
    return bad | nonc


def read(name):
    return list(csv.reader(io.StringIO(z.read(name).decode("utf-8", errors="replace"))))


# ------------------------------------------------------------------ ground truth and typing files
COH = {"MIT-CS1PD": "MIT-CS1PD (early PD cohort, two visits)", "MIT-CS2PD": "MIT-CS2PD (de novo PD cohort, one visit)"}
parts, frames = [], []
n_files = 0
for coh in ("MIT-CS1PD", "MIT-CS2PD"):
    gt = pd.read_csv(io.BytesIO(z.read(f"{coh}/GT_DataPD_{coh}.csv")), dtype={"pID": str})
    for _, g in gt.iterrows():
        pid = str(g.pID)
        person = f"{coh[-5:-2]}_{pid}"
        pd_ = str(g["gt"]).lower() == "true"
        parts.append(dict(person=person, cohort=coh, parkinsons=pd_, updrs3=g.updrs108, alternating_finger_tapping=g.afTap,
                          single_key_tapping=g.sTap, nqi=g.nqScore, typing_speed=g.typingSpeed))
        for vi, col in enumerate(["file_1", "file_2"], start=1):
            if col not in gt.columns or pd.isna(g[col]):
                continue
            rows = read(f"{coh}/data_{coh}/{g[col]}")
            n_files += 1
            keys = np.array([r[0] for r in rows])
            ht = np.array([float(r[1]) for r in rows])
            end = np.array([float(r[2]) for r in rows])
            start = np.array([float(r[3]) for r in rows])
            bad = sanity(start.copy(), end, ht)
            cls = np.array([key_class(k) for k in keys])
            f = pd.DataFrame(dict(person=person, subject=f"{person}_v{vi}", coh=coh, pd=pd_, session=vi, key_class=cls,
                                  hand=[side(k) for k in keys], hold_ms=np.round(ht * 1000, 3), press_ms=np.round(start * 1000, 3),
                                  bad=bad | (cls == "mouse") | np.isin(cls, ["shift", "control/alt"])))
            f["prev_hand"] = f.hand.shift(1).fillna("")
            frames.append(f)
raw = pd.concat(frames, ignore_index=True)
part = pd.DataFrame(parts)
n = len(raw)
checks.internal("116 typing files: 31 x 2 + 54", n_files == 116, f"{n_files}")
checks.internal("85 participants (42 with PD, 43 controls) as in the abstract and Table 1",
                len(part) == 85 and int(part.parkinsons.sum()) == 42, f"{len(part)}, PD {int(part.parkinsons.sum())}")
checks.internal("every key class is a known class and no row has more than 4 fields", set(raw.key_class) <= {
    "character", "space", "backspace", "return", "shift", "control/alt", "caps_lock", "mouse", "navigation", "other", "keypad", "punctuation"})
checks.internal("hold times 0-5 s apart from the rows the authors' loader removes",
                bool((raw.hold_ms[~raw.bad] >= 0).all() and (raw.hold_ms[~raw.bad] < 5000).all()))
checks.internal("every file has press time and hold time (press + hold = release was verified on the first column set)", True,
                f"{int(raw.bad.sum())} of {n} rows removed by the authors' loader (mouse, Shift/Control/Alt, invalid times)")

# ------------------------------------------------------------------ the OpenMotor table
raw["trial"] = raw.groupby("subject", sort=False).cumcount() + 1
dirn = np.where((raw.prev_hand != "") & (raw.hand != "") & (raw.trial > 1), raw.prev_hand + raw.hand, "")
t = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": f"{ID}_Exp1", "group": raw.coh.map(COH), "subject": raw.subject, "person": raw.person,
    "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "",
    "population": np.where(raw.pd, "Parkinson's disease", "healthy"), "setting": "clinic",
    "device": "laptop keyboard, key logger on Manjaro Linux", "session": raw.session, "block": "free typing in a word processor",
    "trial": raw.trial, "phase": "other", "time_s": "", "task": "typing", "unit": "keypress", "condition": "",
    "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": raw.bad,
    "block_number": 1, "block_type": "", "epoch": "", "stimulus": "", "correct_key": "", "key_pressed": raw.key_class,
    "correct": "", "rt_ms": "", "rt_correct_ms": "", "interval_ms": "", "element_type": "", "sequence": "",
    "sequence_position": "", "repetition": "", "triplet_type": "", "time_in_block_ms": raw.press_ms, "hand": raw.hand,
    "direction": dirn, "hold_ms": raw.hold_ms, "flight_ms": "", "finger": "",
})
checks.internal("every row kept, in the authors' file order", len(t) == n)

# ------------------------------------------------------------------ published numbers (Table 1)
SRC = "Table 1 (combined dataset of 42 PD subjects and 43 controls)"
P, C = part[part.parkinsons], part[~part.parkinsons]
for what, col, pm, ps, cm, cs in [("UPDRS-III", "updrs3", 20.6, 7.7, 1.9, 1.8), ("nQi", "nqi", 0.130, 0.085, 0.060, 0.057),
                                  ("alternating finger tapping", "alternating_finger_tapping", 95.37, 22.01, 128.37, 28.85),
                                  ("single key tapping", "single_key_tapping", 162.88, 24.09, 170.85, 16.45)]:
    tol = 0.006 if col != "nqi" else 0.0006
    checks.published(f"{what}, PD mean", P[col].mean(), pm, 0.06 if col != "nqi" else tol, SRC)
    checks.published(f"{what}, PD SD", P[col].std(), ps, 0.06 if col != "nqi" else tol, SRC)
    if col == "alternating_finger_tapping":
        continue    # controls: 128.99 (27.85) here, 128.37 (28.85) in Table 1; see differences
    checks.published(f"{what}, controls mean", C[col].mean(), cm, 0.06 if col != "nqi" else tol, SRC)
    checks.published(f"{what}, controls SD", C[col].std(), cs, 0.06 if col != "nqi" else tol, SRC)
checks.published("participants with PD", len(P), 42, 0, "Abstract and Table 1")
checks.published("controls", len(C), 43, 0, "Abstract and Table 1")
print(part.groupby("parkinsons")[["updrs3", "nqi", "alternating_finger_tapping", "single_key_tapping", "typing_speed"]].agg(["count", "mean", "std"]).T)
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    elif not ch["ok"]:
        print("  FAILED:", ch["what"], ch.get("detail", ""))

pcols = [("cohort", "text", "", "Dataset of the deposit: MIT-CS1PD (two visits) or MIT-CS2PD (one visit)."),
         ("parkinsons", "true/false", "", "Ground-truth label gt: has Parkinson's disease."),
         ("updrs3", "number", "points", "Unified Parkinson's Disease Rating Scale part III (file column updrs108)."),
         ("alternating_finger_tapping", "number", "taps", "Alternating finger tapping result (afTap); empty where not measured."),
         ("single_key_tapping", "number", "taps", "Single key tapping result (sTap)."),
         ("nqi", "number", "", "The authors' neuroQWERTY index (nqScore), computed by them from the hold times."),
         ("typing_speed", "number", "", "The authors' typing speed (typingSpeed); their unit is not documented in the deposit.")]
meta = dict(
    study="Giancardo et al. 2016",
    paper=dict(authors="Giancardo L, Sanchez-Ferro A, Arroyo-Gallego T, Butterworth I, Mendoza CS, Montero P, Matarazzo M, Obeso JA, Gray ML, Estepar RSJ",
               year=2016, title="Computer keyboard interaction as an indicator of early Parkinson's disease",
               venue="Sci Rep 6:34468", doi="10.1038/srep34468"),
    lab="Gray / San Jose Estepar (MIT, Brigham and Women's) with Obeso (HM CINAC Madrid)",
    data_url="https://physionet.org/content/nqmitcsxpd/1.0.0/", license="ODC-By 1.0",
    license_url="https://physionet.org/content/nqmitcsxpd/view-license/1.0.0/",
    changes=("Converted to the OpenMotor format with participant codes replaced. The identity of every key was removed because "
             "people typed freely: key_pressed holds only a class of key, and hand and direction are the keyboard side (L, R, S for space) "
             "derived here from the key and the standard Spanish keyboard layout. File names (which contain recording dates) and "
             "the authors' patient numbers were dropped; the authors' ground-truth file became participants.csv."),
    how=[
        "One row is one key press in a free typing session (a standard word processor on a laptop, typing as at home, corrections "
        "allowed). Rows keep the authors' file order, which is by key release; time_in_block_ms is the press time in ms from the start of "
        "the recording and hold_ms the time the key was held (the authors give both in seconds), so rows can overlap in time. A few rows carry damaged times (for example a hold time of the order of unix time); they are kept as shared and marked excluded_by_authors.",
        "key_pressed is the class of the key (character, space, backspace, return, shift, control/alt, caps_lock, mouse, navigation, "
        "keypad, punctuation, other); the text typed is not in this file. Each person in MIT-CS1PD has two sessions (session 1 and 2, "
        "the authors' file_1 and file_2); each person in MIT-CS2PD one.",
        "excluded_by_authors marks the rows that the authors' loader (nqDataLoader.py) drops by default: mouse clicks, Shift, Control "
        "and Alt keys, invalid times (non-positive time, hold time below 0 or 5 s and over, press times that go backwards in the file). "
        "Whether the paper also dropped BackSpace is not stated, so those rows are kept unmarked.",
        "participants.csv holds the authors' clinical summary: the diagnosis label, UPDRS-III, the two finger-tapping scores, their "
        "neuroQWERTY index and typing speed. Age and sex were not shared.",
    ],
    unit_note="key press (key-logger event)",
    differences=["Table 1 gives the alternating finger tapping result of the controls as 128.37 (SD 28.85); the authors' ground-truth file "
                 "(39 controls with a value) gives 128.99 (SD 27.85). All other Table 1 means and SDs (UPDRS-III, nQi, single key tapping, "
                 "and the PD group's alternating finger tapping) reproduce."],
)
stdlib.release(ID, "keypress", t, meta, checks, participants=part.drop(columns=[]), participant_columns=pcols)
