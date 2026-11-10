"""Jang, Kim & Cho (2025), Data 10(12):197: lip-aperture kinematics (electromagnetic articulography, Carstens AG200) of three trisyllabic
nonce words (babaBa with stress on syllable 1, 2 or 3) said by 10 American English speakers (5 women, 5 men) at an IP or word
boundary, accented or not. Same recordings as Kim, Jang & Cho (2017), J Acoust Soc Am 142(4):EL362.

Raw files, raw_extra/Jang2025_Data/ (OSF v9t6x, CC BY 4.0):
  articulatory_data.csv  468 tokens: Speaker, Gender, Boundary (IP-final, Wd-final), Accent (Acc, Una), Stress (S1-S3), Repetition (r1-r4),
      then for C1, C2, C3 (the three [b]s): closing and opening peak velocity (pkvel), duration (dur, ms), displacement (displ)
  articulatory_data.zip  raw AG200 files of every token (binary, not converted here)
Meaning (2017 article, Methods): lip aperture = distance of upper and lower lip sensors; onset and target at 20% of peak velocity; closing duration
includes the closure up to the release; release duration includes the opening plateau up to the next closing.
Run from the site folder: python3 -I tools/std/convert/Jang2025_Data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Jang2025_Data"
PAPER = "Kim, Jang & Cho 2017, J Acoust Soc Am (same recordings)"
STIM = {"S1": "bábaba", "S2": "babába", "S3": "bababá"}


def main():
    raw = stdlib.raw_dir(ID)
    d = pd.read_csv(raw / "articulatory_data.csv", encoding="utf-8-sig")
    ck = stdlib.Checks()
    rows, n = [], {}
    for r in d.to_dict("records"):
        n.setdefault(r["Speaker"], 0)
        for k in (1, 2, 3):
            n[r["Speaker"]] += 1
            p = f"C{k}"
            rows.append({
                "dataset": ID, "study": ID, "experiment": "", "group": "", "subject": r["Speaker"], "subject_excluded_by_authors": False,
                "age": "", "sex": r["Gender"], "handedness": "", "population": "healthy", "setting": "lab",
                "device": "Carstens AG200 electromagnetic articulograph, sensors on upper and lower lips", "session": 1, "block": r["Repetition"],
                "trial": n[r["Speaker"]], "phase": "test", "time_s": "", "task": "speech", "unit": "utterance",
                "condition": f"{r['Boundary']}, {'accented' if r['Accent'] == 'Acc' else 'unaccented'}", "pert_type": "none", "pert_value": 0,
                "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
                "stimulus": STIM[r["Stress"]], "target_sound": "b", "gesture_signal": "LA", "gesture_order": k, "gesture_segment": "b",
                "phrasal_position": "phrase-final" if r["Boundary"] == "IP-final" else "phrase-medial",
                "accent": "accented" if r["Accent"] == "Acc" else "unaccented", "stress_position": int(r["Stress"][1]),
                "repetition": int(r["Repetition"][1:]),
                "closing_peak_velocity": r[f"{p}clos_pkvel"], "opening_peak_velocity": r[f"{p}open_pkvel"],
                "closure_duration_ms": r[f"{p}clos_dur"], "release_duration_ms": r[f"{p}open_dur"],
                "closing_displacement_mm": r[f"{p}clos_displ"], "opening_displacement_mm": r[f"{p}open_displ"],
            })
    ck.internal("every token of articulatory_data.csv kept with its 3 gestures, in order", len(rows) == 3 * len(d) == 1404)
    ck.internal("10 speakers, 5 women and 5 men", d.Speaker.nunique() == 10 and d.groupby("Speaker").Gender.first().value_counts().to_dict() == {"F": 5, "M": 5})
    ck.internal("no duplicate speaker x condition x repetition", not d.duplicated(["Speaker", "Boundary", "Accent", "Stress", "Repetition"]).any())
    ck.published("speakers", d.Speaker.nunique(), 10, 0, f"{PAPER}, Methods (10 speakers)")
    ck.published("women among the speakers", int((d.groupby("Speaker").Gender.first() == "F").sum()), 5, 0, f"{PAPER}, Methods")
    cm = d.groupby(["Speaker", "Stress", "Accent", "Boundary"]).C3open_pkvel.mean().groupby(["Speaker", "Boundary"]).mean().unstack()
    t = stats.ttest_rel(cm["IP-final"], cm["Wd-final"]).statistic
    ck.published("C3 release peak velocity, main effect of boundary F(1,9)", t * t, 46.91, 0.01, f"{PAPER}, Results (kinematics of C3-rel/opening)")
    for c in ck.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Each token of the authors' wide table became three rows, one per syllable's [b] lip-aperture gesture (C1 to C3); sex from Gender; labels recoded (IP-final = phrase-final, Wd-final = phrase-medial).",
        how=[
            "One row is one lip-aperture gesture (gesture_order 1 to 3 = the first to third [b] of the nonce word) of one token; the token's "
            "word, boundary, accent, stress and repetition are repeated on its three rows, and trial counts rows within the speaker.",
            "closure_duration_ms and release_duration_ms are the authors' closing and release durations, closing/opening_peak_velocity "
            "their peak velocities and closing/opening_displacement_mm their displacements. The deposit and the 2017 article do not state "
            "the units of velocity and displacement; cm/s and mm are inferred from the size of the values.",
            "The raw AG200 files of every token (articulatory_data.zip, 151 MB) were not converted.",
        ],
        unit_note="lip-aperture gesture (three per token) of one speaker",
        differences=[
            "The 2017 article analysed 467 of 480 planned tokens; the table has 468.",
            "Not reproduced: Table 2 durations of the 2017 article (for example S1 unaccented C3 release duration 145 and 81 ms, here 129 and 75) and the boundary effect on C3 release displacement (F = 31.99 there, 29.80 here); the F for peak velocity matches.",
        ],
    )
    if not ck.passed():
        sys.exit("checks failed")
    return stdlib.release(ID, "articulography", rows, meta, ck)


if __name__ == "__main__":
    main()
