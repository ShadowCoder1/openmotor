"""Feit, Weir & Oulasvirta (2016), CHI: How We Type (Typing data v2.0, sentences condition, 30 participants, one row per key press).

Run from the site folder: python3 -I tools/std/convert/Feit2016_CHI.py
Raw files (untrusted data, read only) from https://doi.org/10.5281/zenodo.4047775: typing.zip ("Typing data v2.0.zip") and
Background.xlsx. The motion capture, eye tracking and video archives are not used. The typed responses (column transcribed) are not kept.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Feit2016_CHI"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for f in ("typing.zip", "Background.xlsx"):
    checks.internal(f"{f} matches the downloaded hash", stdlib.sha256(RAW / f) == src[f]["sha256"])

z = zipfile.ZipFile(RAW / "typing.zip")
frames = []
for info in z.infolist():
    if "matched" in info.filename:
        frames.append(pd.read_csv(io.BytesIO(z.read(info)), sep="\t", dtype=str, keep_default_na=False))
a = pd.concat(frames, ignore_index=True)
assert list(a.columns) == ["input_time", "user_id", "input_index", "iki", "key_symbol", "finger", "transcribed", "stimulus"], list(a.columns)
n = len(a)
uid = a.user_id
ii = a.input_index.astype(int)
t = a.input_time.astype(float)
iki = a.iki.astype(float)
first_seen = pd.factorize(uid)[0]
checks.internal("30 participants, rows grouped by participant in the authors' file order", uid.nunique() == 30 and bool((np.diff(first_seen) >= 0).all()))

# sentence number = count of input_index resets; the authors' stimulus changes exactly there
sent = (ii == 0).groupby(uid).cumsum().to_numpy()
chg = ((a.stimulus != a.stimulus.shift()) | (uid != uid.shift())).to_numpy()
checks.internal("the stimulus sentence changes exactly where input_index returns to 0", bool((chg == (ii == 0).to_numpy()).all()))
_step = ii.to_numpy()[1:][~chg[1:]] - ii.to_numpy()[:-1][~chg[1:]]
checks.internal("input_index mostly counts 0,1,2... within a sentence (the authors' own index is kept)", float((_step == 1).mean()) > 0.99,
                f"{(_step != 1).sum()} steps other than +1 of {len(_step)}")
checks.internal("49 or 50 sentences per participant", bool(pd.Series(sent).groupby(uid.to_numpy()).max().between(49, 50).all()))
dt_neg = int((t.groupby(uid).diff() < 0).sum())
checks.internal("timestamps run forward within a participant except in a few places the authors' file has", dt_neg <= 2, f"{dt_neg} backward steps")
finger = a.finger
checks.internal("every key press has an executing finger", bool(finger.isin([f"{h}_{f}" for h in "LR" for f in ("Little", "Ring", "Middle", "Index", "Thumb")]).all()))
hand = finger.str[0]

# background questionnaire: its 'Name' column holds the participant's 6-digit key, linking it to the typing files
ws = openpyxl.load_workbook(RAW / "Background.xlsx").active
rows = list(ws.iter_rows(values_only=True))
B = pd.DataFrame(rows[1:], columns=rows[0])
key, fem, age, hnd, lang = rows[0][0], rows[0][1], rows[0][2], rows[0][5], rows[0][7]
B[key] = B[key].astype(str)
B = B[B[key].isin(set(uid))].set_index(key)
checks.internal("questionnaire rows found for all 30 participants", len(B) == 30)
age_of = B[age].astype(float).to_dict()
sex_of = {k: ("F" if str(v) == "1" else "M") for k, v in B[fem].items()}
hand_of = {k: ("left" if "Left" in str(v) else "right") for k, v in B[hnd].items()}
lang_of = {k: str(v) for k, v in B[lang].items()}

time0 = t.groupby(uid).transform("min")
rowsdf = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": ID + "_Exp1", "group": uid.map(lambda u: {"Fi": "Finnish", "En": "English"}[lang_of[u]]),
    "subject": uid, "person": uid, "subject_excluded_by_authors": False,
    "age": uid.map(age_of), "sex": uid.map(sex_of), "handedness": uid.map(hand_of), "population": "healthy", "setting": "lab",
    "device": "keyboard (Swedish/Finnish layout), 240 fps hand motion capture", "session": 1,
    "block": ("sentence " + pd.Series(sent).astype(str)).to_numpy(),
    "trial": pd.Series(np.ones(n, dtype=int)).groupby(uid.to_numpy()).cumsum().to_numpy(), "phase": "test",
    "time_s": (t - time0).round(3), "task": "typing", "unit": "keypress", "condition": "sentences", "pert_type": "none",
    "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
    "block_number": sent, "block_type": "test", "epoch": "", "stimulus": "", "correct_key": "", "key_pressed": a.key_symbol, "correct": "",
    "rt_ms": "", "rt_correct_ms": "", "interval_ms": iki, "element_type": "", "sequence": a.stimulus,
    "sequence_position": ii + 1, "repetition": "", "triplet_type": "", "time_in_block_ms": "", "hand": hand, "direction": "", "hold_ms": "",
    "flight_ms": "", "finger": finger,
})
checks.internal("every row kept", len(rowsdf) == n == 36955)

# ---- published numbers
SRC_P = "Participants"
checks.published("participants", uid.nunique(), 30, 0, SRC_P)
checks.published("female participants", sum(v == "F" for v in sex_of.values()), 17, 0, SRC_P)
checks.published("left-handed participants", sum(v == "left" for v in hand_of.values()), 3, 0, SRC_P)
checks.published("age, mean (years, paper rounds to 31)", np.mean(list(age_of.values())), 31, 0.5, SRC_P)
checks.published("age, youngest", min(age_of.values()), 20, 0, SRC_P)
checks.published("age, oldest", max(age_of.values()), 55, 0, SRC_P)
checks.published("participants who typed in Finnish", sum(v == "Fi" for v in lang_of.values()), 18, 0, SRC_P)
checks.published("participants who typed in English", sum(v == "En" for v in lang_of.values()), 12, 0, SRC_P)
checks.published("key presses in the sentences condition", n, 36955, 0, "Results (first paragraph)")
checks.published("sentences per participant (50 chosen; one participant has 49)", pd.Series(sent).groupby(uid.to_numpy()).max().median(), 50, 0, "Materials")
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    elif not ch["ok"]:
        print("  FAILED:", ch["what"], ch.get("detail", ""))

meta = dict(
    study="Feit et al. 2016",
    paper=dict(authors="Feit AM, Weir D, Oulasvirta A", year=2016,
               title="How We Type: Movement Strategies and Performance in Everyday Typing",
               venue="Proceedings of the 2016 CHI Conference on Human Factors in Computing Systems, pp. 4262-4273", doi="10.1145/2858036.2858233"),
    lab="Oulasvirta (Aalto University)", data_url="https://doi.org/10.5281/zenodo.4047775",
    license="CC BY-NC 4.0", license_url="https://creativecommons.org/licenses/by-nc/4.0/",
    changes=("Converted to the OpenMotor format with participant codes replaced (the deposit's 6-digit user ids and the questionnaire's "
             "name field are not kept). Only the 'Typing data v2.0' key logs of the sentences condition were used; the motion capture, "
             "eye tracking and video archives and the typed responses (transcribed text) were not kept. The deposit's bigram and "
             "previous-key columns were dropped as redundant. Age, sex, handedness and typing language come from Background.xlsx."),
    how=[
        "One row is one key press while a participant copied one of 50 short everyday sentences (from the Enron Mobile Email "
        "dataset, translated into Finnish for the Finnish typists) on a physical keyboard; sequence is the sentence shown, block_number "
        "counts the sentences from 1 (it restarts when the authors' input_index returns to 0), and sequence_position is the authors' "
        "input_index plus 1.",
        "key_pressed is the authors' key name (for example space, BackSpace, Shift_L, adiaeresis), interval_ms is their inter-key "
        "interval (0 for the first key of a sentence), time_s is the key press time in seconds from the participant's first key "
        "press, finger is the executing finger the authors found from the motion capture (L or R hand, then the finger), and hand "
        "is its first letter (the space bar is thumb, not S).",
        "The typed text of each sentence is not included. The deposit does not say which key presses were errors, so correct is empty. "
        "The random-letter and mixed conditions are not in this version of the deposit (its readme says so).",
    ],
    unit_note="key press while copying a sentence",
)
Rows = rowsdf
stdlib.release(ID, "keypress", Rows, meta, checks)
