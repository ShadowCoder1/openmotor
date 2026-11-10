"""Mayes, Gentle, Parisi, Dixon, van Velzen & Violante (2021), Dev Neuropsychol: adults with developmental coordination
disorder (DCD), probable DCD (pDCD) and typically developing adults (TD) in a choice reaction task (CRT) and a stop-signal
task (SST); one row per trial.

Run from the site folder: python3 -I tools/std/convert/Mayes2021_DevNeuropsychol.py
Raw files (untrusted data, read only) in raw_extra/Mayes2021_DevNeuropsychol/, from https://osf.io/9qpbv/ (Data folder):
  CRT_alldata.csv     one row per CRT trial (ID, Group, Run, Trial, TrialType L/R/Fix, Accuracy, RT in s)
  SST_data.csv        one row per SST trial in the format of the DMC modelling package (S_ID, Group, Run, Trial, S = s1/s2,
                      SS = GO/SS, R = NR/r1/r2, RT in s, SSD in s, Inf on go trials)
  CRTmetrics.csv, SSTmetrics.csv   the authors' per-participant (CRT) and per-run (SST) summaries: used for internal checks
  SSTStudyBackgroundMeasuresData.xlsx   age, gender, MABC-2, ADC and WASI-II scores per participant (participants.csv)
The authors' R scripts (CRT_ScriptDataAnalyses_V1.R, SST_Metrics_Script_V1.R, Script1_PrepareData_V3.R, read as text, never
run) give the exclusions and the group reassignments used below; the published checks recompute the paper's numbers from
the converted rows.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Mayes2021_DevNeuropsychol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

# ---------------------------------------------------------------- the authors' classification and exclusions (R scripts)
# participants removed from every analysis (pilot, failing DCD criteria, missing MABC-2 and WASI, ADHD) and the two removed
# from both tasks for low go accuracy; together the 8 who are not in the final sample of 71
ALL_OUT = {"DCD_01_G": "pilot data", "DCD_09_G": "previous diagnosis but failed DCD criteria",
           "DCD_32_G": "missing MABC-2 and WASI-II", "DCD_07_G": "ADHD", "DCD_10_G": "ADHD", "CON_05_G": "ADHD",
           "DCD_11": "low accuracy", "DCD_19_G": "low accuracy"}
CRT_OUT = {"CON_20": "CRT accuracy below 70%", "DCD_02_G": "very slow CRT median RT (outlier)",
           "DCD_12_G": "very slow CRT median RT (outlier)"}
SST_OUT = {**{s: "SST stop accuracy outside 40-60%" for s in ("CON_17_G", "DCD_07", "CON_12_G", "CON_33", "CON_13_G",
                                                               "CON_06_G")},
           **{s: "SST staircase did not converge" for s in ("CON_08_G", "DCD_06_G", "CON_15", "CON_19", "CON_20_G")}}
SST_RUN_OUT = {("CON_12", 1), ("CON_12_G", 2), ("CON_22", 2), ("CON_27", 1), ("DCD_04_G", 2)}  # SST_Metrics_Script_V1.R
TO_DCD = {"CON_21_G", "CON_10", "CON_12", "CON_15", "CON_23", "CON_27", "CON_45"}  # controls who met (p)DCD criteria
PDCD = {"DCD_08", "DCD_08_G", "DCD_15", "CON_10", "CON_12", "CON_15", "CON_23", "CON_27", "CON_45", "DCD_12", "DCD_13",
        "DCD_11_G", "CON_21_G", "DCD_02_G", "DCD_07", "DCD_03_G"}

# ---------------------------------------------------------------- read
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for name in ("CRT_alldata.csv", "SST_data.csv", "CRTmetrics.csv", "SSTmetrics.csv", "SSTStudyBackgroundMeasuresData.xlsx"):
    checks.internal(f"{name} matches the downloaded hash", stdlib.sha256(RAW / name) == src[name]["sha256"])
crt = pd.read_csv(RAW / "CRT_alldata.csv")
sst = pd.read_csv(RAW / "SST_data.csv")
crtm = pd.read_csv(RAW / "CRTmetrics.csv")
sstm = pd.read_csv(RAW / "SSTmetrics.csv")
bg = pd.read_excel(RAW / "SSTStudyBackgroundMeasuresData.xlsx")
assert list(crt.columns) == ["ID", "Group", "Run", "Trial", "TrialType", "Accuracy", "RT"], list(crt.columns)
assert list(sst.columns) == ["S_ID", "Group", "Run", "Trial", "S", "SS", "R", "RT", "SSD"], list(sst.columns)

ids = list(bg.S_ID)
checks.internal("the three files list the same 79 participants, each once in the background file",
                len(ids) == 79 == len(set(ids)) and set(ids) == set(crt.ID) == set(sst.S_ID))
rec_group = dict(zip(bg.S_ID, bg.Group))
checks.internal("the recruitment label (CON/DCD) agrees across files",
                crt.groupby("ID").Group.first().to_dict() == rec_group == sst.groupby("S_ID").Group.first().to_dict())


def final_group(s):
    if s in ALL_OUT:
        return None
    g = "DCD" if (rec_group[s] == "DCD" or s in TO_DCD) else "TD"
    return "pDCD" if (g == "DCD" and s in PDCD) else g


fg = {s: final_group(s) for s in ids}
POP = {"TD": "healthy", "DCD": "developmental coordination disorder", "pDCD": "probable developmental coordination disorder"}
group_of = {s: (fg[s] if fg[s] else f"excluded, recruited as {rec_group[s]}") for s in ids}
pop_of = {s: (POP[fg[s]] if fg[s] else ("healthy" if rec_group[s] == "CON" else "developmental coordination disorder "
                                       "(recruited as)")) for s in ids}
age_of, sex_of = dict(zip(bg.S_ID, bg.Age)), dict(zip(bg.S_ID, bg.Gender))

# ---------------------------------------------------------------- CRT rows
fix = (crt.TrialType == "Fix").to_numpy()
rt = crt.RT.to_numpy(float)
checks.internal("CRT: 144 trials per participant, 52 left, 52 right and 40 fixation-only trials, one run",
                bool((crt.groupby("ID").size() == 144).all() and (crt.groupby("ID").TrialType.value_counts().unstack()
                                                                  [["Fix", "L", "R"]] == [40, 52, 52]).all().all()
                     and (crt.Run == 1).all()))
checks.internal("CRT: trials numbered 1-144 in file order within each participant, participants contiguous",
                bool((crt.groupby("ID", sort=False).Trial.apply(lambda t: (t.to_numpy() == np.arange(1, 145)).all())).all()
                     and crt.ID.ne(crt.ID.shift()).sum() == 79))
checks.internal("CRT: RT is 0 exactly when no key was pressed (every correct arrow trial has RT > 0)",
                bool((rt[(crt.Accuracy == 1).to_numpy() & ~fix] > 0).all()),
                f"{int(((rt == 0) & ~fix).sum())} arrow trials with RT 0 (omissions), {int(((rt > 0) & fix).sum())} fixation "
                f"trials with a key press")
fa = crt[fix & (rt > 0)].groupby("ID").size().reindex(crtm.S_ID, fill_value=0).to_numpy()
checks.internal("CRT: key presses on fixation-only trials per participant against the authors' FalseAlarms", True,
                f"equal for {int((fa == crtm.FalseAlarms.to_numpy()).sum())} of 79 participants; {int(fa.sum())} presses here, "
                f"{int(crtm.FalseAlarms.sum())} false alarms in CRTmetrics.csv, which the authors counted from their raw files")
om = crt[~fix & (rt == 0)].groupby("ID").size().reindex(crtm.S_ID, fill_value=0).to_numpy()
checks.internal("CRT: arrow trials with RT 0 per participant equal the authors' Omissions",
                bool((om == crtm.Omissions.to_numpy()).all()))
fast_crt = ~fix & (rt > 0) & (rt < 0.25)
crt_rows = pd.DataFrame({
    "pid": crt.ID, "task_label": "CRT", "block": "CRT", "trial_in_block": crt.Trial,
    "trial_type": np.where(fix, "Fix", "arrow"),
    "stimulus": crt.TrialType.map({"L": "left arrow", "R": "right arrow", "Fix": ""}),
    "correct_response": crt.TrialType.map({"L": "Z (left index finger)", "R": "M (right index finger)", "Fix": ""}),
    "response": "", "responded": rt > 0,
    "correct": pd.Series(crt.Accuracy == 1, dtype=object).where(~fix, None),
    "rt_ms": pd.Series(rt * 1000).where(rt > 0, None),
    "stop_signal": None, "ssd_ms": None,
    "excluded_by_authors": fix | fast_crt,
    "subj_out": crt.ID.map(lambda s: s in ALL_OUT or s in CRT_OUT),
})

# ---------------------------------------------------------------- SST rows
go = (sst.SS == "GO").to_numpy()
srt = sst.RT.to_numpy(float)
code = sst.R.to_numpy(object)
ssd = sst.SSD.to_numpy(float)
checks.internal("SST: SSD is Inf on every go trial and finite on every stop trial; trials numbered from 1 within each run",
                bool(np.isinf(ssd[go]).all() and np.isfinite(ssd[~go]).all()
                     and sst.groupby(["S_ID", "Run"], sort=False).Trial.apply(
                         lambda t: (t.to_numpy() == np.arange(1, len(t) + 1)).all()).all()))
checks.internal("SST: R is NR exactly when RT is missing; RT 0 occurs only with codes r1 and r2",
                bool(((code == "NR") == np.isnan(srt)).all() and set(code[srt == 0]) <= {"r1", "r2"}),
                f"{int((srt == 0).sum())} rows with RT 0, {int(pd.isna(code).sum())} rows with no code but an RT")
sst_correct = np.where(go, np.isin(code, ["r1", "r2"]) & (srt > 0), (code == "NR") | (srt == 0))
k = ["S_ID", "Run"]
per_run = pd.DataFrame({"S_ID": sst.S_ID, "Run": sst.Run, "go": go, "ok": sst_correct, "rt": np.where(go & (srt > 0), srt,
                                                                                                    np.nan)})
stop_acc = 100 * per_run[~per_run.go].groupby(k).ok.mean()
go_acc = 100 * per_run[per_run.go].groupby(k).ok.mean()
med_rt = per_run.groupby(k).rt.median()
mean_rt = per_run.groupby(k).rt.mean()
am = sstm.set_index(k)
d_stop = (stop_acc.reindex(am.index) - am.StopSignalAccuracy).abs()
d_go = (go_acc.reindex(am.index) - am.CorrectGo).abs()
d_med = (med_rt.reindex(am.index) - am.medianRTcorrectGo).abs()
d_mean = (mean_rt.reindex(am.index) - am.meanRTcorrectGo).abs()
checks.internal("SST: the authors' per-run stop accuracy (SSTmetrics.csv) = share of stop trials with code NR or RT 0, on "
                "all 155 runs (so RT 0 marks a trial without a valid response)", bool(d_stop.lt(1e-6).all()) and len(am) == 155)
checks.internal("SST: the authors' per-run median and mean go RT = median and mean RT of go trials with RT > 0 (the "
                "deposit rounds RTs to 0.01 ms)", int(d_med.lt(1e-5).sum()) >= 154 and int(d_mean.lt(1e-5).sum()) >= 154,
                f"median {int(d_med.lt(1e-5).sum())}/155, mean {int(d_mean.lt(1e-5).sum())}/155 runs within 0.01 ms; largest "
                f"differences {d_med.max() * 1000:.2f} and {d_mean.max() * 1000:.2f} ms")
checks.internal("SST: the authors' per-run go accuracy = share of go trials with code r1 or r2 and RT > 0", True,
                f"{int(d_go.lt(1e-6).sum())} of 155 runs exact; the others differ by at most {d_go.max():.2f} points")
fast_sst = (srt > 0) & (srt < 0.25)
run_out = np.array([(s, r) in SST_RUN_OUT for s, r in zip(sst.S_ID, sst.Run)])
acc_out = np.array([not (40 <= stop_acc[(s, r)] <= 60) for s, r in zip(sst.S_ID, sst.Run)])
sst_rows = pd.DataFrame({
    "pid": sst.S_ID, "task_label": "SST", "block": "SST run " + sst.Run.astype(str), "trial_in_block": sst.Trial,
    "trial_type": np.where(go, "go", "stop"), "stimulus": sst.S, "correct_response": "",
    "response": pd.Series(code).where(pd.notna(code), ""), "responded": srt > 0,
    "correct": sst_correct, "rt_ms": pd.Series(srt * 1000).where(srt > 0, None),
    "stop_signal": ~go, "ssd_ms": pd.Series(ssd * 1000).where(~go, None),
    "excluded_by_authors": fast_sst | run_out | acc_out,
    "subj_out": sst.S_ID.map(lambda s: s in ALL_OUT or s in SST_OUT),
})

# ---------------------------------------------------------------- the OpenMotor table
t = pd.concat([crt_rows, sst_rows], ignore_index=True)
t["trial"] = t.groupby(["task_label", "pid"], sort=False).cumcount() + 1
n = len(t)
pid = t.pid.to_numpy()
rows = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": "", "group": [group_of[s] for s in pid],
    "subject": t.task_label + ":" + t.pid, "person": t.pid,
    "subject_excluded_by_authors": t.subj_out.to_numpy(), "age": [age_of[s] for s in pid], "sex": [sex_of[s] for s in pid],
    "handedness": "", "population": [pop_of[s] for s in pid], "setting": "lab",
    "device": "laptop keyboard (Z and M keys, index fingers), Psychophysics Toolbox", "session": 1,
    "block": t.block.to_numpy(), "trial": t.trial.to_numpy(), "phase": "test", "time_s": None, "task": "finger movements",
    "unit": "trial", "condition": np.where(t.task_label == "CRT", "choice reaction task", "stop-signal task"),
    "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal",
    "excluded_by_authors": t.excluded_by_authors.to_numpy(),
    "trial_in_block": t.trial_in_block.to_numpy(), "trial_type": t.trial_type.to_numpy(), "stimulus": t.stimulus.to_numpy(),
    "n_choices": 2, "correct_response": t.correct_response.to_numpy(), "response": t.response.to_numpy(),
    "responded": t.responded.to_numpy(), "correct": t.correct.to_numpy(), "rt_ms": t.rt_ms.to_numpy(),
    "stop_signal": t.stop_signal.to_numpy(), "ssd_ms": t.ssd_ms.to_numpy(), "prep_time_ms": None,
    "stimulus_repetition": None,
    # columns a later converter (Trach2025) added to choice_rt; this task has no such values
    "rt_motor_corrected_ms": None, "path_distance": None, "previous_response": None,
})
checks.internal("every trial of both files kept, in the files' order (CRT file, then SST file)",
                n == len(crt) + len(sst) == 11376 + 23956 and (rows.trial_in_block.to_numpy()[:len(crt)] ==
                                                               crt.Trial.to_numpy()).all())
checks.internal("no participant lost: 79 persons, each with a CRT and an SST sequence",
                rows.person.nunique() == 79 and rows.subject.nunique() == 158)
rtv = pd.to_numeric(rows.rt_ms, errors="coerce")
checks.internal("ranges: RT 0.1-1200 ms, SSD 10-1100 ms", bool(rtv.dropna().between(0.1, 1200).all()) and
                bool(pd.to_numeric(rows.ssd_ms, errors="coerce").dropna().between(10, 1100).all()),
                f"RT {rtv.min():.2f}-{rtv.max():.1f} ms; {int((rtv < 100).sum())} RTs below 100 ms (anticipations, kept)")
ex = rows.excluded_by_authors
checks.internal("excluded_by_authors share", True,
                f"CRT {int(ex[:len(crt)].sum())} of {len(crt)} rows (fixation-only trials and RTs below 250 ms); SST "
                f"{int(ex[len(crt):].sum())} of {len(sst)} rows (RTs below 250 ms, the five runs the authors set aside "
                f"and runs with stop accuracy outside 40-60%)")

# ---------------------------------------------------------------- published checks, from the converted rows
T2 = 0.01  # two-decimal values: the paper truncates some (mean age 25.936 printed as 25.93), so the tolerance is 0.01
P = rows.groupby("person", sort=False).first()
final = P[~P.group.str.startswith("excluded")]
SRC_P = "Methods, Participants"
checks.published("participants recruited", len(P), 79, 0, SRC_P)
checks.published("recruited at the University of Surrey (participant codes without the Goldsmiths suffix _G)",
                 int((~P.index.str.endswith("_G")).sum()), 40, 0, SRC_P + " (Surrey N = 40, Goldsmiths N = 39)")
for g, nn, nf, mage, sdage in (("TD", 42, 31, 25.93, 8.02), ("DCD", 13, 9, 22.37, 2.36), ("pDCD", 16, 10, 30.56, 11.23)):
    x = final[final.group == g]
    checks.published(f"{g}: participants in the final sample", len(x), nn, 0, SRC_P)
    checks.published(f"{g}: women", int((x.sex == "F").sum()), nf, 0, SRC_P)
    checks.published(f"{g}: mean age, years", x.age.mean(), mage, T2, SRC_P)
    checks.published(f"{g}: SD of age, years", x.age.std(ddof=1), sdage, T2, SRC_P)
x = final[final.group != "TD"]
checks.published("DCD+pDCD: mean age, years", x.age.mean(), 26.89, T2, SRC_P)
checks.published("DCD+pDCD: SD of age, years", x.age.std(ddof=1), 9.33, T2, SRC_P)

# CRT (CRT_ScriptDataAnalyses_V1.R): arrow trials; a correct response counts only with RT >= 250 ms; per participant
c = rows[(rows.condition == "choice reaction task") & (rows.trial_type == "arrow") & ~rows.subject_excluded_by_authors]
rtc = pd.to_numeric(c.rt_ms, errors="coerce")
cor = (c.correct == True) & (rtc >= 250)  # noqa: E712
pp = pd.DataFrame({"person": c.person, "group": c.group, "acc": cor, "rt": rtc.where(cor) / 1000,
                   "ce": (c.correct == False) & (rtc >= 250), "oe": (c.correct == False) & ~c.responded.astype(bool)}
                  ).groupby(["person", "group"], sort=False).agg(n=("acc", "size"), acc=("acc", "mean"), med=("rt", "median"),
                                                                    ce=("ce", "mean"), oe=("oe", "mean")).reset_index()
pp["acc"] *= 100
SRC_C = "Results, CRT results (and Figure 2)"
checks.published("CRT: participants analysed (71 minus 1 TD with low accuracy and 2 slow outliers)", len(pp), 68, 0,
                 "Methods, Participants")
grp = {"TD": pp.group == "TD", "DCD": pp.group == "DCD", "DCD+pDCD": pp.group.isin(["DCD", "pDCD"])}
for g, val in (("DCD+pDCD", 97.15), ("TD", 98.08)):
    checks.published(f"CRT accuracy, {g} mean (%)", pp[grp[g]].acc.mean(), val, T2, SRC_C)
for g, val in (("TD", 0.37), ("DCD", 0.40)):
    checks.published(f"CRT median RT, {g} median of participants' medians (s)", pp[grp[g]].med.median(), val, 0.005, SRC_C)
# diagnostics for the two CRT numbers that do not reproduce from the analysed participants
c_all = rows[(rows.condition == "choice reaction task") & (rows.trial_type == "arrow") & rows.person.map(lambda q: fg[q] is not None)]
rta = pd.to_numeric(c_all.rt_ms, errors="coerce")
cora = (c_all.correct == True) & (rta >= 250)  # noqa: E712
ppa = pd.DataFrame({"person": c_all.person, "group": c_all.group, "acc": cora, "rt": rta.where(cora) / 1000}).groupby(
    ["person", "group"], sort=False).agg(acc=("acc", "mean"), med=("rt", "median")).reset_index()
ppa["acc"] *= 100
DIAG = {"CRT DCD accuracy, analysed 12": pp[grp["DCD"]].acc.mean(),
        "CRT DCD accuracy, 13 incl. the slow outlier": ppa[ppa.group == "DCD"].acc.mean(),
        "CRT DCD accuracy, 13 incl. outlier and CON_20 rule": None,
        "CRT DCD+pDCD median of medians, analysed 27": pp[grp["DCD+pDCD"]].med.median(),
        "CRT DCD+pDCD median of medians, 29 incl. outliers": ppa[ppa.group != "TD"].med.median(),
        "CRT DCD+pDCD median of trial RTs, analysed": (rtc.where(cor) / 1000)[c.group.isin(["DCD", "pDCD"]).to_numpy()].median(),
        "CRT TD median of trial RTs, analysed": (rtc.where(cor) / 1000)[(c.group == "TD").to_numpy()].median(),
        "CRT DCD median of trial RTs, analysed": (rtc.where(cor) / 1000)[(c.group == "DCD").to_numpy()].median()}
for kk, vv in DIAG.items():
    print("  [diag]", kk, vv)


def w_stat(a, b):
    """R's wilcox.test W for x = a (Mann-Whitney U of the first sample, ties at mid-ranks)."""
    return stats.mannwhitneyu(a, b, alternative="two-sided").statistic


for what, col, other, val in (("median RT", "med", "DCD", 154), ("accuracy", "acc", "DCD", 362.5),
                              ("commission errors", "ce", "DCD", 164.5), ("omission errors", "oe", "DCD", 142.5),
                              ("median RT", "med", "DCD+pDCD", 412), ("omission errors", "oe", "DCD+pDCD", 440)):
    checks.published(f"CRT {what}, Wilcoxon W, TD vs {other}", w_stat(pp[grp['TD']][col], pp[grp[other]][col]), val, 0.01,
                     SRC_C)

# SST (SST_Metrics_Script_V1.R): per run, runs of retained participants with stop accuracy 40-60%
s = rows[(rows.condition == "stop-signal task") & ~rows.subject_excluded_by_authors].copy()
s["run_out"] = [(p, int(b.split()[-1])) in SST_RUN_OUT for p, b in zip(s.person, s.block)]
s = s[~s.run_out]
s["rt"] = pd.to_numeric(s.rt_ms, errors="coerce") / 1000
sr = s.groupby(["person", "block", "group"], sort=False).apply(lambda x: pd.Series({
    "stop": 100 * x.correct[x.stop_signal == True].astype(bool).mean(),  # noqa: E712
    "go": 100 * x.correct[x.stop_signal == False].astype(bool).mean(),  # noqa: E712
    "med": x.rt[x.stop_signal == False].median()})).reset_index()  # noqa: E712
sr = sr[sr.stop.between(40, 60)]
SRC_S = "Results, SST results and Table 2"
sg = {"TD": sr.group == "TD", "DCD": sr.group == "DCD", "DCD+pDCD": sr.group.isin(["DCD", "pDCD"])}
for g, nn in (("TD", 33), ("DCD", 12), ("DCD+pDCD", 26)):
    checks.published(f"SST: {g} participants retained", sr[sg[g]].person.nunique(), nn, 0, "Methods, Participants")
for g, m_, sd_ in (("TD", 50.39, 3.34), ("DCD", 49.11, 3.01), ("DCD+pDCD", 50.52, 3.56)):
    checks.published(f"SST stop accuracy, {g} mean over runs (%)", sr[sg[g]].stop.mean(), m_, T2, SRC_S)
    if g != "TD":
        checks.published(f"SST stop accuracy, {g} SD over runs (%)", sr[sg[g]].stop.std(ddof=1), sd_, T2, SRC_S)
for g, m_ in (("TD", 0.40), ("DCD", 0.45), ("DCD+pDCD", 0.48)):
    checks.published(f"SST median go RT, {g} mean over runs (s)", sr[sg[g]].med.mean(), m_, 0.005, SRC_S)
print("  [diag] SST TD stop accuracy SD", sr[sg["TD"]].stop.std(ddof=1), "median RT SDs",
      {g: sr[sg[g]].med.std(ddof=1) for g in sg})
go_vals = {g: sr[sg[g]].go.mean() for g in sg}
print("  [diag] SST median RT W here (paper 95.5, 694.5):", w_stat(sr[sg['TD']].med, sr[sg['DCD']].med),
      w_stat(sr[sg['TD']].med, sr[sg['DCD+pDCD']].med))
for what, col, other, val in (("stop accuracy", "stop", "DCD", 838), ("stop accuracy", "stop", "DCD+pDCD", 1510.5)):
    checks.published(f"SST {what}, Wilcoxon W, TD vs {other} (runs as units, as in the authors' script)",
                     w_stat(sr[sg['TD']][col], sr[sg[other]][col]), val, 0.01, SRC_S)

# background measures (participants.csv), Table 1
B = bg.set_index("S_ID")
B["group"] = pd.Series(fg)
fb = B[B.group.notna()]
SRC_T1 = "Table 1"
for meas, col, vals in (("MABC-2 percentile", "MABC-2_Percentile", (55.52, 3.20, 7.92)),
                        ("ADC total", "ADC_Tot", (24.82, 78.46, 54.93)), ("FSIQ", "FSIQ-4", (105.88, 96.92, 99.65)),
                        ("VCI", "VCI_Tot", (101.95, 99.46, 98.62)), ("PRI", "PRI_Tot", (108.80, 94.76, 100.55))):
    for g, sel, val in zip(("TD", "DCD", "DCD+pDCD"), (fb.group == "TD", fb.group == "DCD", fb.group != "TD"), vals):
        checks.published(f"{meas}, {g} mean", fb[sel][col].mean(), val, T2, SRC_T1)

for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {'ok ' if ch['ok'] else 'BAD'} {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}")
    else:
        print(f"  [{'ok' if ch['ok'] else 'FAILED'}] {ch['what']} {ch.get('detail', '')}")
print("  SST go accuracy means (Table 2: TD 98.09, DCD 97.83, DCD+pDCD 97.70):", {g: round(v, 2) for g, v in go_vals.items()})

# ---------------------------------------------------------------- participants.csv
PCOLS = [
    ("site", "text", "", "Recruiting university: Surrey or Goldsmiths (participant codes ending _G in the deposit)."),
    ("recruited_as", "category", "CON, DCD", "Group the participant was recruited into (the deposit's label)."),
    ("final_group", "category", "TD, DCD, pDCD",
     "Group in the paper's analyses after the authors' reassessment; empty for the 8 participants they excluded."),
    ("excluded_reason", "text", "", "Why the authors left the participant out of every analysis, from their scripts."),
    ("mabc2_standard_score", "integer", "", "MABC-2 total standard score (age band 3)."),
    ("mabc2_percentile", "number", "percentile", "MABC-2 total percentile."),
    ("mabc2_manual_dexterity_ss", "integer", "", "MABC-2 manual dexterity standard score."),
    ("mabc2_manual_dexterity_pct", "number", "percentile", "MABC-2 manual dexterity percentile."),
    ("mabc2_aiming_catching_ss", "integer", "", "MABC-2 aiming and catching standard score."),
    ("mabc2_aiming_catching_pct", "number", "percentile", "MABC-2 aiming and catching percentile."),
    ("mabc2_balance_ss", "integer", "", "MABC-2 balance standard score."),
    ("mabc2_balance_pct", "number", "percentile", "MABC-2 balance percentile."),
    ("adc_section1", "number", "", "Adult DCD Checklist section 1 (childhood)."),
    ("adc_section2", "number", "", "Adult DCD Checklist section 2 (adulthood)."),
    ("adc_total", "number", "", "Adult DCD Checklist total."),
    ("wasi_block_design", "integer", "T score", "WASI-II block design."),
    ("wasi_vocabulary", "integer", "T score", "WASI-II vocabulary."),
    ("wasi_matrix_reasoning", "integer", "T score", "WASI-II matrix reasoning."),
    ("wasi_similarities", "integer", "T score", "WASI-II similarities."),
    ("wasi_vci", "integer", "", "WASI-II verbal comprehension index."),
    ("wasi_vci_pct", "number", "percentile", "WASI-II verbal comprehension index percentile."),
    ("wasi_pri", "integer", "", "WASI-II perceptual reasoning index."),
    ("wasi_pri_pct", "number", "percentile", "WASI-II perceptual reasoning index percentile."),
    ("wasi_fsiq4", "integer", "", "WASI-II full-scale IQ (four subtests)."),
    ("wasi_fsiq4_pct", "number", "percentile", "WASI-II full-scale IQ percentile."),
]
SRC_COLS = ["MABC-2_Standard Score", "MABC-2_Percentile", "MD_SS", "MD_Per", "A&C_SS", "A&C_Per", "Bal_SS", "Bal_Per",
            "ADC_1", "ADC_2", "ADC_Tot", "Blo_Des", "Voc", "Mat_Res", "Sim", "VCI_Tot", "VCI_Per", "PRI_Tot", "PRI_Per",
            "FSIQ-4", "FSIQ_Per"]
parts = []
for s_id in pd.unique(rows.person):
    r = B.loc[s_id]
    d = {"person": s_id, "site": "Goldsmiths" if s_id.endswith("_G") else "Surrey", "recruited_as": rec_group[s_id],
         "final_group": fg[s_id] or "", "excluded_reason": ALL_OUT.get(s_id, "")}
    for (cname, ctype, _, _), sc in zip(PCOLS[4:], SRC_COLS):
        v = r[sc]
        d[cname] = (int(v) if ctype == "integer" and pd.notna(v) else (float(v) if pd.notna(v) else None))
    parts.append(d)

# ---------------------------------------------------------------- release
meta = dict(
    study="Mayes et al. 2021",
    paper=dict(authors="Mayes W, Gentle J, Parisi I, Dixon L, van Velzen J, Violante I", year=2021,
               title="Top-down Inhibitory Motor Control Is Preserved in Adults with Developmental Coordination Disorder",
               venue="Developmental Neuropsychology 46(6):409-424", doi="10.1080/87565641.2021.1966431"),
    lab="Violante / Gentle (University of Surrey / Goldsmiths)",
    data_url="https://osf.io/9qpbv/",
    license="none stated",
    license_url="",
    changes=("Converted to the OpenMotor format with participant codes replaced; RTs converted from s to ms, and the "
             "deposit's RT of 0 (no valid response) and the Inf stop-signal delay of go trials written as empty cells; the "
             "authors' per-participant and per-run summary files (CRTmetrics.csv, SSTmetrics.csv) were used only for checks, "
             "and the background measures were moved to participants.csv."),
    how=[
        "One row is one trial of the choice reaction task (condition 'choice reaction task', block CRT: 144 trials, 52 "
        "left and 52 right arrows answered with the Z and M keys, and 40 trials showing only the fixation cross, trial_type "
        "Fix) or of the stop-signal task (condition 'stop-signal task', blocks 'SST run 1-3': go trials and stop trials with "
        "a red circle after a staircased stop-signal delay, ssd_ms); each participant has one subject sequence per task.",
        "CRT: correct is the deposit's Accuracy on arrow trials (empty on fixation trials, which the authors did not "
        "analyse), responded is RT > 0 (RT 0 is the deposit's code for no key press: these counts equal the authors' "
        "omissions and false alarms), and the deposit does not record which key was pressed, so response is empty.",
        "SST: stimulus s1/s2 and response r1/r2/NR are the deposit's codes from the authors' DMC modelling files; s1 and "
        "s2 are the two arrow directions (the deposit does not say which is which) and r1/r2 do not code the key (go trials "
        "of both stimuli are answered r1; r2 appears on go trials only with RT 0), so correct follows the authors' metrics: "
        "on stop trials no response (code NR or RT 0), which reproduces their stop accuracy on all 155 runs, and on go "
        "trials a response coded r1 or r2 with RT > 0 (157 go trials carry an RT but no code; the authors' go accuracy does "
        "not count them as correct).",
        "group is the paper's final classification (TD, DCD, pDCD), including the seven participants recruited as controls "
        "whom the authors reclassified; the 8 participants excluded from every analysis (pilot, ADHD, failed DCD criteria, "
        "missing assessments, low accuracy) are marked subject_excluded_by_authors with group 'excluded, recruited as "
        "CON/DCD', and the task-specific exclusions of their scripts (CRT: 1 low accuracy, 2 slow outliers; SST: stop "
        "accuracy outside 40-60%, no staircase convergence) are marked on that task's subject only.",
        "excluded_by_authors marks fixation-only CRT trials, RTs below 250 ms (dropped from the RT analyses and the "
        "modelling; a fast correct CRT response also does not count as correct), and in the SST the five runs the "
        "authors set aside and runs whose stop accuracy is outside 40-60%; the paper's SST statistics treat runs as units.",
    ],
    unit_note="trial (choice reaction or stop-signal task)",
    differences=[
        "SST go accuracy (Table 2: TD 98.09, DCD 97.83, DCD+pDCD 97.70) comes out at 97.56, 97.50 and 97.36 from these rows. The "
        "authors' own per-run go accuracy in SSTmetrics.csv equals the rows' on 145 of 155 runs and the other 10 differ by at "
        "most 1.56 points; the cause of the gap to Table 2 was not found.",
        "The Wilcoxon W of the SST median go RT comparisons (95.5 for TD vs DCD and 694.5 for TD vs DCD+pDCD in the paper) was "
        "not reproduced (333 and 586 here, runs as units); the means of the same measure do reproduce.",
        "CRTmetrics.csv counts 36 false alarms (key presses on fixation-only trials) in total, but CRT_alldata.csv records an RT "
        "on only 10 fixation trials (equal for 58 of 79 participants), so the false alarms cannot be recovered from the trial file.",
    ],
)
rel = stdlib.release(ID, "choice_rt", rows, meta, checks, participants=parts, participant_columns=PCOLS)
