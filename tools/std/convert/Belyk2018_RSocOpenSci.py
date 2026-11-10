"""Belyk, Johnson & Kotz (2018), R Soc open sci 5:171544: 34 listeners imitate 45 five-note melodies by singing and 45 by whistling.

Raw files (raw_extra/Belyk2018_RSocOpenSci/, the Zenodo copy 5001716 of Dryad doi:10.5061/dryad.504t7; the split archive xaa-xad was joined
into data.zip and unpacked on the PC with 7-Zip, audio files deleted, nothing run). Read as text:
  ex/data/subject_sNN/sing_recordings/praat_output.csv, whistle_recordings/praat_output.csv
      one line per melody: file name <subject>_trial<k>_<five note names>, then the f0 (Hz) the authors' Praat script measured from the
      centre 250 ms of each of the five imitated notes ('--undefined--' or 'NA' where no pitch was measured)
  ex/data/meta.mat.csv   sex, age, music lessons and self ratings per participant (s27 is not listed)
  ex/data/imitation_unpacker.py   (read as text, never run) the authors' note-name to Hz tables: male register A2-A3, female register A3-A4,
      whistle register A5-A6 (its female A3 is 200 Hz, a typo for 220 Hz; 220 is used)
Run from the site folder: python3 -I tools/std/convert/Belyk2018_RSocOpenSci.py
"""
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Belyk2018_RSocOpenSci"
RAW = stdlib.raw_dir(ID) / "ex/data"
SRC = "Belyk et al. 2018"
checks = stdlib.Checks()

NAMES = ["A", "Asharp", "B", "C", "Csharp", "D", "Dsharp", "E", "F", "Fsharp", "G", "Gsharp"]
MALE = {"A2": 110, "A2sharp": 116.54, "B2": 123.47, "C3": 130.81, "C3sharp": 138.59, "D3": 146.83, "D3sharp": 155.56, "E3": 164.81, "F3": 174.61,
        "F3sharp": 185, "G3": 196, "G3sharp": 207.65, "A3": 220}
FEMALE = {"A3": 220, "A3sharp": 233.08, "B3": 246.94, "C4": 261.63, "C4sharp": 277.18, "D4": 293.67, "D4sharp": 311.13, "E4": 329.63, "F4": 349.23,
          "F4sharp": 369.99, "G4": 392, "G4sharp": 415.3, "A4": 440}
WHISTLE = {"A5": 880, "A5sharp": 932.33, "B5": 987.77, "C6": 1046.5, "C6sharp": 1108.7, "D6": 1174.7, "D6sharp": 1244.5, "E6": 1318.5, "F6": 1396.9,
           "F6sharp": 1480.0, "G6": 1568.0, "G6sharp": 1661.2, "A6": 1760.0}
MALE_ONLY = set(MALE) - {"A3"}
FEMALE_ONLY = set(FEMALE) - {"A3"}


def num(v):
    return float(v) if re.fullmatch(r"-?[0-9.]+", v) else np.nan


meta = pd.read_csv(RAW / "meta.mat.csv", index_col=0)
meta = meta[meta.ID.notna()].set_index("ID")
subs = sorted(d for d in glob.glob(str(RAW / "subject_*")) if os.path.exists(d + "/sing_recordings/praat_output.csv"))
rows, parts = [], []
reg_vs_sex = []
for d in subs:
    sid = os.path.basename(d)[-3:]
    data = {}
    for cond in ("sing", "whistle"):
        L = []
        for line in open(f"{d}/{cond}_recordings/praat_output.csv", encoding="utf-8", errors="replace").read().splitlines():
            c = next(csv.reader([line]))
            m = re.fullmatch(rf"{sid}_trial(\d+)_(.+)", c[0])
            L.append((int(m.group(1)), m.group(2).split("_"), [num(v) for v in c[1:]]))
        data[cond] = sorted(L)
    notes = {n for _, nn, _ in data["sing"] for n in nn}
    reg = "male" if notes & MALE_ONLY else "female"
    assert not (notes & MALE_ONLY and notes & FEMALE_ONLY), sid
    tab = MALE if reg == "male" else FEMALE
    mt = meta.loc[sid] if sid in meta.index else None
    sex = {"female": "F", "male": "M"}.get(mt.sex, "") if mt is not None else ""
    if sex:
        reg_vs_sex.append((sex == "M") == (reg == "male"))
    na = lambda v: np.nan if v in ("NA", "") else v  # noqa: E731
    if mt is not None:
        parts.append(dict(person=sid, music_lessons=mt.lessons_binary, lesson_years=pd.to_numeric(mt.lessons_years, errors="coerce"),
                          lesson_instrument=str(mt.lessons_instr).strip(", ") if mt.lessons_instr == mt.lessons_instr else "",
                          whistle_technique=na(mt.whistle) if mt.whistle == mt.whistle else "", self_rated_good_singer=mt.good_singer,
                          self_rated_good_whistler=mt.good_whistle, easier_task=mt.easier))
    else:
        parts.append(dict(person=sid))
    n = 0
    for cond in ("sing", "whistle"):
        tbl = tab if cond == "sing" else WHISTLE
        for k, nn, hz in data[cond]:
            for j in range(5):
                n += 1
                tgt = tbl[nn[j]]
                rows.append(dict(
                    dataset=ID, study="Belyk 2018 R Soc open sci", experiment="sung and whistled melody imitation", group="", subject=sid, person=sid,
                    subject_excluded_by_authors="", age=float(mt.age) if mt is not None else "", sex=sex, handedness="", population="healthy",
                    setting="lab", device="desk microphone, sound-attenuated room; stimuli over speakers", session=1, block=cond, trial=n,
                    phase="test", time_s="", task="singing and whistling imitation", unit="trial", condition=cond, pert_type="none",
                    pert_value=0, pert_unit="", feedback="normal", excluded_by_authors="", instrument="voice" if cond == "sing" else "whistle",
                    target_pitch_hz=tgt, produced_pitch_hz=hz[j], pitch_error_cents=(1200 * math.log2(hz[j] / tgt)) if hz[j] == hz[j] else np.nan,
                    target_note=nn[j], aid="none", note_position=j + 1, melody_number=k, target_melody="_".join(nn)))
names = [c[1] for c in stdlib.columns_for("pitch_matching")]
a = pd.DataFrame(rows).reindex(columns=names)
pp = pd.DataFrame(parts)

nm = a.groupby(["subject", "condition"]).melody_number.nunique()
checks.published("participants with recordings", a.subject.nunique(), 34, 0, SRC + ", 2.2.1 Participants (thirty-four recruited)")
checks.published("melodies per participant and condition", nm.min(), 45, 0, SRC + ", 2.1 and 2.2.3 (45 melodies per task)")
checks.published("melodies per participant and condition (max)", nm.max(), 45, 0, SRC + ", 2.1 and 2.2.3")
checks.published("notes per melody", a.groupby(["subject", "condition", "melody_number"]).size().max(), 5, 0, SRC + ", 2.1 (five-note melodies)")
sg, wh = a[a.condition == "sing"], a[a.condition == "whistle"]
checks.published("lowest sung target (Hz), male range A2", sg.target_pitch_hz.min(), 110, 0.5, SRC + ", 2.1 (A2-A3 for males)")
checks.published("highest sung target (Hz), female range A4", sg.target_pitch_hz.max(), 440, 0.5, SRC + ", 2.1 (A3-A4 for females)")
checks.published("lowest whistled target (Hz), A5", wh.target_pitch_hz.min(), 880, 0.5, SRC + ", 2.1 (A5-A6)")
checks.published("highest whistled target (Hz), A6", wh.target_pitch_hz.max(), 1760, 0.5, SRC + ", 2.1 (A5-A6)")
checks.internal("the register read from the note names agrees with the participant's sex in meta.mat.csv", all(reg_vs_sex), f"{len(reg_vs_sex)} participants")
checks.internal("15300 note rows: 34 participants x 2 conditions x 45 melodies x 5 notes", len(a) == 34 * 2 * 45 * 5, str(len(a)))
checks.internal("measured pitch present for most notes", a.produced_pitch_hz.notna().mean() > 0.9, f"{a.produced_pitch_hz.notna().mean():.3f}")
checks.internal("signed sung errors have a flat median, whistled errors a sharp median, as the paper says (small tendency)",
                bool(sg.pitch_error_cents.median() < 0 < wh.pitch_error_cents.median()), f"{sg.pitch_error_cents.median():.1f}, {wh.pitch_error_cents.median():.1f}")
for c in checks.items:
    if not c["ok"]:
        print("FAILED", c)
print(len(a), a.produced_pitch_hz.notna().sum(), sg.pitch_error_cents.median(), wh.pitch_error_cents.median())
if not checks.passed():
    sys.exit(1)
s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
p = s["papers"][0]
meta_ = dict(
    paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes="Reassembled the authors' per-participant Praat output files into one row per imitated note, added the target note and frequency from the note names with the authors' note tables, computed the error in cents, and replaced participant folder names with codes. Audio files, the MBEA responses and dates are not included.",
    how=["One row is one imitated note: produced_pitch_hz is the f0 (Hz) the authors' Praat script measured over the centre 250 ms of the note, target_pitch_hz the frequency of the target note (from the note name in the stimulus file name, with the authors' note tables), and pitch_error_cents = 1200 x log2(produced / target), positive when the produced note was sharper than the target. Empty where Praat found no pitch.",
         "The deposit shares the measured pitch of every note, with no further exclusions; the paper's analysis dropped melodies with a wrong number of notes (3.6%) and notes outside each participant's producible range, and analysed 28 of the 34 participants because six could not whistle. The deposit does not say which participants or notes those were, so none is marked; participants without whistled pitch show empty values. The paper's 11,433 analysed notes therefore cannot be reproduced.",
         "Each participant has a sung block (male register A2-A3, female register A3-A4 as read from the note names) and a whistled block (A5-A6), 45 melodies of five notes each; melody_number is the melody's trial number in the authors' file names (presentation order within the block). The order of the two conditions was counterbalanced across participants and is not coded: trial runs over the sung block first, then the whistled block. The authors' female note table lists A3 as 200 Hz; 220 Hz (the equal-tempered A3 and the paper's range) is used. The error in cents is computed here from the authors' Hz values with the paper's equation (2.1); the authors' R analysis code is not in the deposit.",
         "Sex and age come from the authors' meta.mat.csv (participant s27 is not in it, so sex and age are empty for that participant). participants.csv holds the music lesson and self-rating columns."],
    unit_note="imitated note", differences=["The paper's 11,433 analysed notes and 28 analysed participants cannot be reproduced because the deposit does not identify the excluded melodies, notes and participants."])
pcols = [("music_lessons", "text", "yes/no", "Had music lessons (meta.mat.csv, lessons_binary)."),
         ("lesson_years", "number", "years", "Years of lessons (lessons_years)."),
         ("lesson_instrument", "text", "", "Instrument(s) of the lessons, as the authors list them."),
         ("whistle_technique", "text", "", "How the participant whistles (the authors' whistle column, for example lips)."),
         ("self_rated_good_singer", "text", "yes/no", "Self-identified as a good singer."),
         ("self_rated_good_whistler", "text", "yes/no", "Self-identified as a good whistler."),
         ("easier_task", "text", "singing, whistling, neither", "Which task the participant found easier.")]
stdlib.release(ID, "pitch_matching", a, meta_, checks, participants=pp, participant_columns=pcols)
