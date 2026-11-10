"""Apsvalka, Ramsey & Cross (2018), Neural Plasticity 2018:1237962: anodal tDCS over M1 (active vs sham) during observational
practice of finger sequences; 50 participants, sequence test performance before, after and 24 h after training.

Run from the site folder: python3 -I tools/std/convert/Apsvalka2018_NeuralPlast.py
Raw file (untrusted data, read only): raw_extra/Apsvalka2018_NeuralPlast/tDCSraw.csv from https://osf.io/5ryc9/ (raw data folder).
Expected values for the published checks are those of the paper's Table 1 and Results; the authors' R scripts were read as text
for the definitions of the derived scores, never run.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Apsvalka2018_NeuralPlast"
RAWFILE = stdlib.raw_dir(ID) / "tDCSraw.csv"
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(stdlib.raw_dir(ID) / "SOURCES.json"))}
checks.internal("raw file matches the downloaded hash", stdlib.sha256(RAWFILE) == src["tDCSraw.csv"]["sha256"])
raw = pd.read_csv(RAWFILE, keep_default_na=False, na_values=[""])
N = len(raw)
checks.internal("50 rows with unique participant numbers", N == 50 and raw.sID.is_unique)
checks.internal("stimulation is 0 (sham) or 1 (active)", set(raw.stimulation) <= {0, 1})
numcols = [c for c in raw.columns if c not in ("sID", "M1", "stimulation", "gender", "age", "righthanded", "belief", "correctBelief")]
checks.internal("all measure columns numeric, none missing", bool(raw[numcols].notna().all().all()) and all(np.issubdtype(raw[c].dtype, np.number) for c in numcols))
checks.internal("error rates between 0 and 1", bool(raw[[c for c in numcols if c.startswith("Err")]].apply(lambda s: s.between(0, 1)).all().all()))

group = np.where(raw.stimulation == 1, "active", "sham")
PH = {"Pre": ("pretest", "baseline"), "Post": ("posttest", "test"), "Ret": ("retention", "test")}
SEQ = {"TR": "trained sequences", "UN": "untrained sequences"}
MEAS = [  # (prefix, measure, description, statistic, unit)
    ("IT", "initiation_time", "Mean duration from the go signal to the first key press over the four sequences of that type, correct executions only", "mean", "s"),
    ("ET", "execution_time", "Mean duration from the first to the fifth key press over the four sequences of that type, correct executions only", "mean", "s"),
    ("Err", "error_rate", "Proportion of incorrect sequence executions over the four sequences of that type", "proportion", "proportion"),
    ("stdET", "execution_time_sd", "The authors' stdET column for execution time (spread of execution time; not defined further in the deposit or paper)", "sd", "s"),
]
rows = []
for i, r in raw.iterrows():
    code = f"{ID}_S{i + 1:03d}"
    base = {
        "dataset": ID, "study": ID, "experiment": f"{ID}_Exp1", "group": group[i], "subject": code,
        "subject_excluded_by_authors": False, "age": int(r.age), "sex": {"male": "M", "female": "F"}[r.gender],
        "handedness": "right", "population": "healthy", "setting": "lab", "device": "computer keyboard, left hand",
        "session": 1, "time_s": "", "task": "sequence learning", "unit": "participant summary",
        "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
        "n_trials": "",
        "session_context": "right M1 located by TMS hotspot" if r.M1 == "tms" else "right M1 located at EEG position C4 (no visible TMS twitch)",
    }
    k = 0
    def add(**kw):
        global k
        k += 1
        rows.append({**base, "trial": k, **kw})
    for ph, (blk, phase) in PH.items():
        for sq, sqn in SEQ.items():
            for pre, meas, desc, stat, unit in MEAS:
                add(block=blk, phase=phase, condition=sqn, measure=meas, measure_description=desc, statistic=stat,
                    value=float(r[f"{pre}{ph}{sq}"] if pre != "stdET" else r[f"stdET{ph}{sq}"]), value_unit=unit)
    for d in range(1, 5):
        add(block=f"training day {d}", phase="learning", condition="observation of the sequences", measure="error_detection_accuracy",
            measure_description="Proportion of observed videos in which the participant correctly judged whether the model's execution was correct",
            statistic="proportion", value=float(r[f"D{d}Acc"]), value_unit="proportion")
        for pre, meas, desc in (("max", "strongest_sensation", "Strongest intensity of discomforting sensations reported after the session (0-4)"),
                                ("stop", "sensation_lasted", "How long the sensations lasted (questionnaire scale)"),
                                ("affected", "sensation_affected_performance", "How much the sensations affected performance (questionnaire scale)")):
            add(block=f"training day {d}", phase="learning", condition="sensation questionnaire", measure=meas,
                measure_description=desc, statistic="value", value=float(r[f"{pre}{d}"]), value_unit="score")
t = pd.DataFrame(rows)
checks.internal("50 participants x (3 phases x 2 sequence types x 4 measures + 4 days x 4) = 2000 rows", len(t) == 2000, str(len(t)))

# ---------------------------------------------------------------- published checks (Table 1 and Results)
v = lambda m: t[t.measure == m]
def pstat(m, blk=None, cond=None):
    d = v(m)
    if blk: d = d[d.block == blk]
    if cond: d = d[d.condition == cond]
    return d.groupby("subject").value.mean()      # per-participant mean over the selected rows
grp = t.drop_duplicates("subject").set_index("subject").group
T1 = "Table 1, "
checks.published("sham group size", int((grp == "sham").sum()), 24, 0, T1 + "Sham (N = 24)")
checks.published("active group size", int((grp == "active").sum()), 26, 0, T1 + "Active (N = 26)")
sx = t.drop_duplicates("subject").groupby(["group", "sex"]).size()
checks.published("sham males", int(sx["sham", "M"]), 8, 0, T1 + "Gender 8 : 16")
checks.published("sham females", int(sx["sham", "F"]), 16, 0, T1 + "Gender 8 : 16")
checks.published("active males", int(sx["active", "M"]), 6, 0, T1 + "Gender 6 : 20")
checks.published("active females", int(sx["active", "F"]), 20, 0, T1 + "Gender 6 : 20")
ages = t.drop_duplicates("subject").set_index("subject").age.astype(float)
checks.published("age, all participants, mean", ages.mean(), 20.60, 0.005, "Methods, Participants (M = 20.60 years, SD = 2.40)")
checks.published("age, all participants, SD", ages.std(ddof=1), 2.40, 0.005, "Methods, Participants (M = 20.60 years, SD = 2.40)")
for g, m, s in (("sham", 20.96, 2.97), ("active", 20.27, 1.71)):
    checks.published(f"age {g}, mean", ages[grp == g].mean(), m, 0.005, T1 + "Age")
    checks.published(f"age {g}, SD", ages[grp == g].std(ddof=1), s, 0.005, T1 + "Age")
it_pre = pstat("initiation_time", "pretest"); et_pre = pstat("execution_time", "pretest"); er_pre = pstat("error_rate", "pretest") * 100
for g, (im, isd, em, esd, rm, rsd) in (("sham", (0.77, 0.25, 1.92, 0.57, 25, 13)), ("active", (0.89, 0.30, 2.02, 0.68, 30, 15))):
    m = grp == g
    checks.published(f"pretest initiation time {g}, mean (s)", it_pre[m].mean(), im, 0.005, T1 + "Pretest initiation time")
    checks.published(f"pretest initiation time {g}, SD (s)", it_pre[m].std(ddof=1), isd, 0.005, T1 + "Pretest initiation time")
    checks.published(f"pretest execution time {g}, mean (s)", et_pre[m].mean(), em, 0.005, T1 + "Pretest execution time")
    checks.published(f"pretest execution time {g}, SD (s)", et_pre[m].std(ddof=1), esd, 0.005, T1 + "Pretest execution time")
    checks.published(f"pretest error rate {g}, mean (%)", er_pre[m].mean(), rm, 0.5, T1 + "Pretest error rate (printed as integers)")
    checks.published(f"pretest error rate {g}, SD (%)", er_pre[m].std(ddof=1), rsd, 0.5, T1 + "Pretest error rate (printed as integers)")
for meas, vals in (("strongest_sensation", {"sham": (1.23, 0.49), "active": (1.46, 0.79)}),
                   ("sensation_affected_performance", {"sham": (0.16, 0.32), "active": (0.30, 0.36)}),
                   ("sensation_lasted", {"sham": (1.14, 0.48), "active": (1.79, 0.71)})):
    x = pstat(meas)
    for g, (m, s) in vals.items():
        checks.published(f"{meas} {g}, mean over four days", x[grp == g].mean(), m, 0.005, T1 + "Sensations")
        checks.published(f"{meas} {g}, SD", x[grp == g].std(ddof=1), s, 0.005, T1 + "Sensations")
acc = pstat("error_detection_accuracy") * 100
checks.published("error detection accuracy, active group, mean (%)", acc[grp == "active"].mean(), 81, 0.5, "Discussion (active M = 81%)")
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    elif not ch["ok"]:
        print("  FAILED:", ch["what"], ch.get("detail", ""))

meta = dict(
    study="Apsvalka et al. 2018",
    paper=dict(authors="Apsvalka D, Ramsey R, Cross ES", year=2018,
               title="Anodal tDCS over Primary Motor Cortex Provides No Advantage to Learning Motor Sequences via Observation",
               venue="Neural Plasticity 2018:1237962", doi="10.1155/2018/1237962"),
    lab="Cross & Ramsey (Bangor University)", data_url="https://osf.io/5ryc9/",
    license="none stated", license_url="",
    changes=("Converted from the authors' one-row-per-participant table (tDCSraw.csv) to long form, one row per participant, "
             "measure and test or day; participants are coded. The authors' belief columns (what participants guessed about "
             "their stimulation) and the per-day count columns were not all kept: belief and correctBelief were dropped."),
    how=[
        "One row is one value of one participant (the authors shared one value per participant, test and sequence type, not "
        "single executions). In a sequence test a participant pressed five-key finger sequences cued by digits with the left hand; "
        "the test was given before (pretest), after (posttest) and 24 h after (retention) four consecutive days of observational practice "
        "of four sequences, while active or sham anodal tDCS (1 mA, 20 min; sham 30 s) was applied over right M1. The deposit has values for "
        "trained and untrained sequences.",
        "Measures per test and sequence type: initiation time (go signal to first key press, s), execution time (first to fifth "
        "key press, s), error rate (proportion of incorrect executions; the paper prints it in %) and the authors' stdET column "
        "(a spread of execution time that neither the paper nor the scripts define; read it as a standard deviation with caution). "
        "Per training day: error detection accuracy while watching the videos (proportion) and three questionnaire scores "
        "(strongest sensation 0-4, how long sensations lasted, how much they affected performance).",
        "group is the authors' stimulation (sham = 0, active = 1); session_context says whether the M1 target was found with a "
        "TMS twitch or taken as C4 (nine participants). The paper reports 50 analysed participants; the deposit has no excluded rows. "
        "Sex is the authors' 'gender'. All were right-handed by self-report (variants of 'yes' in the file).",
    ],
    unit_note="one participant summary value (one measure, test or day)",
)
stdlib.release(ID, "summary", t, meta, checks)
