"""Bootsma et al. (2021), Front Aging Neurosci 13:643132: mirror star tracing (Apple Pencil) practised at one of three path widths
(P-LD 7 mm, P-MD 5 mm, P-HD 3 mm), 36 young and 36 older adults, tested Pre, Post and 24 h later (Retention).

Raw files raw_extra/Bootsma2021_FrontAgingNeurosci/ (DataverseNL doi 10.34894/MCN6XM, CC0): behav.sav (Behavioral_Learning.sav:
MT, ACC, DTL per participant and time point) and eeg.sav (EEG_Learning.sav: task-related power in alpha and beta over frontal,
motor and parietal regions, per hemisphere). Each value is a participant-level average over trials (the deposit has no single trials).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pyreadstat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Bootsma2021_FrontAgingNeurosci"
SRC = "Bootsma et al. 2021, Frontiers in Aging Neuroscience"
AGE = {1.0: "young", 2.0: "older"}
GRP = {1.0: "P-LD (practice path width 7 mm)", 2.0: "P-MD (practice path width 5 mm)", 3.0: "P-HD (practice path width 3 mm)"}
TIME = {1.0: "Pre", 2.0: "Post", 3.0: "Retention"}
PHASE = {"Pre": "baseline", "Post": "test", "Retention": "test"}
BEH = [("MT", "movement_time", "Average time to complete one star-tracing trial.", "mean", "s"),
       ("ACC", "bandwidth_error", "Bandwidth error: average percentage of samples in a trial that were outside the wall of the star.", "mean", "percent"),
       ("DTL", "distance_to_middle_line", "Average distance to the middle line of the star path, in pixels (the deposit's definition).", "mean", "pixels")]
EEG = [("Frontal_Alpha", "frontal"), ("Motor_Alpha", "motor"), ("Parietal_Alpha", "parietal"),
       ("Frontal_Beta", "frontal"), ("Motor_Beta", "motor"), ("Parietal_Beta", "parietal")]


def main():
    b, _ = pyreadstat.read_sav(str(stdlib.raw_dir(ID) / "behav.sav"))
    e, _ = pyreadstat.read_sav(str(stdlib.raw_dir(ID) / "eeg.sav"))
    rows = []
    cnt = {}

    def add(pid, age, grp, time, hemi, measure, desc, unit, v):
        cnt[pid] = cnt.get(pid, 0) + 1
        t = TIME[time]
        cond = f"{GRP[grp]}; {t}" + (f"; {hemi} hemisphere" if hemi else "")
        rows.append({"dataset": ID, "study": SRC, "experiment": "", "group": AGE[age], "subject": pid,
                     "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "right", "population": "healthy",
                     "setting": "lab", "device": "mirror star-tracing task with Apple Pencil on tablet; 250 Hz EEG",
                     "session": 2 if t == "Retention" else 1, "block": t, "trial": cnt[pid], "phase": PHASE[t], "time_s": "",
                     "task": "tracking", "unit": "participant summary", "condition": cond, "pert_type": "mirror reversal",
                     "pert_value": "", "pert_unit": "", "feedback": "altered", "excluded_by_authors": False,
                     "session_context": "", "measure": measure, "measure_description": desc, "statistic": "mean", "value": v,
                     "value_unit": unit, "n_trials": ""})

    for _, r in b.iterrows():
        for col, name, desc, st, unit in BEH:
            add(r.Subj_nr, r.Age, r.Group, r.Time, "", name, desc, unit, r[col])
    for _, r in e.iterrows():
        h = "contralateral" if r.Hemisphere == 1 else ("ipsilateral" if r.Hemisphere == 2 else "")
        for col, reg in EEG:
            band = col.split("_")[1].lower()
            add(r.Ppnr, r.Age, r.Group, r.Time, h, f"task_related_power_{reg}_{band}",
                f"Task-related power (%) in the {band} band over the {reg} cortex, {h} hemisphere, relative to rest (authors' TR-Power).",
                "percent", r[col])
    df = pd.DataFrame(rows)
    pers = b.drop_duplicates("Subj_nr")
    ck = stdlib.Checks()
    T1 = f"{SRC}, Participants and Table 1"
    ck.published("young participants", (pers.Age == 1).sum(), 36, 0, T1)
    ck.published("older participants", (pers.Age == 2).sum(), 36, 0, T1)
    for a in (1.0, 2.0):
        for g in (1.0, 2.0, 3.0):
            ck.published(f"{AGE[a]} {GRP[g][:4]} participants", ((pers.Age == a) & (pers.Group == g)).sum(), 12, 0, f"{SRC}, Table 1 (N per cell)")
    ck.internal("3 time points per participant in behaviour", (b.groupby("Subj_nr").size() == 3).all())
    ck.internal("EEG participants all in behaviour file", set(e.Ppnr) <= set(b.Subj_nr))
    ck.internal("row count", len(df) == len(b) * 3 + len(e) * 6, str(len(df)))
    for c in ck.items:
        print(("ok " if c["ok"] else "FAILED "), c["what"], c.get("computed", ""))
    s = next(z for z in json.load(open(stdlib.HERE / "source/extra_studies.json")) if z["id"] == ID)
    pp = s["papers"][0]
    meta = dict(study=ID, paper={k: pp[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
                license="CC0", license_url="http://creativecommons.org/publicdomain/zero/1.0",
                changes="Stacked the behavioural and EEG tables into long form (one row per participant, time point and measure), decoded the SPSS value labels into text, replaced participant numbers with codes. Values unchanged.",
                how=["One row is one measure of one participant at one time point (Pre, Post, Retention 24 h later), each the authors' average over trials: movement time (s), bandwidth error (percent of samples outside the star's path), distance to the middle line (pixels), and task-related EEG power (percent change; alpha and beta over frontal, motor and parietal cortex, contralateral and ipsilateral to the drawing hand).",
                     "Participants drew a star outline with the right hand while seeing hand and star only in a mirror (mirror-reversed vision), after practice at one path width (group condition). Which difficulty level the Pre/Post/Retention values come from is not stated in the files; the EEG file has one EEG row missing its hemisphere data for some participant time points (left empty). The deposit has no age or sex per participant (the paper's Table 1 gives group means)."],
                unit_note="measure of one participant at one time point (long form)",
                differences=["The paper's mixed-model F tests were not re-computed; checks use the group sizes."])
    return stdlib.release(ID, "summary", df, meta, ck)


if __name__ == "__main__":
    main()
