"""Trach & McDougle (2025), Nat Hum Behav: keypress stimulus-response learning of 8 visuomotor associations (8 keys, 8 fingers),
a forced-response version of the task, and a baseline finger-to-finger motor RT task; seven experiments plus an online
(Prolific) sample; one row per trial.

Run from the site folder: python3 -I tools/std/convert/Trach2025_NatHumBehav.py
Raw files (untrusted data, read only) in raw_extra/Trach2025_NatHumBehav/, all from
https://github.com/jetrach/StructuredActionPrepVMDM (raw.githubusercontent.com/.../main/...):
  compiledData/ (the authors' trial files after their exclusions; their numbering of the experiments is shifted from the
  paper's from file 2 on):
    Experiment1.csv, motorData_Experiment1.csv          paper Experiment 1 (hierarchical mapping; 40 participants)
    Experiment2.csv, motorData_Experiment2.csv          paper 'Modified structure control' (28)
    Experiment3.csv, motorData_Experiment3.csv          paper Experiment 2 (one-feature, unstructured mapping; 33)
    Experiment4.csv, motorData_Experiment4.csv          paper Experiment 3 (three-feature, unstructured mapping; 26)
    Experiment5_learn/_forced.csv, motorData_Experiment5.csv   paper Experiment 4 (hierarchical, forced response; 36)
    Experiment6_learn/_forced.csv, motorData_Experiment6.csv   paper Experiment 5 (unstructured, forced response; 19)
    hierLearn_prolific.csv                               the authors' 'Study 1A' (Prolific replication of Experiment 1; 26)
  rawData/Experiment6_multiday/data/ (not in compiledData; the files the authors' multi-day scripts load):
    Experiment6_multiday_wexclusions.csv, Experiment6_multiday_motor.csv,
    HierLearnDay01Forced_Data_exclusions.csv, HierLearnDay08Forced_Data_exclusions.csv        paper Experiment 6 (20)
  (saved here with the prefix md_). Left out: Experiment4_scrambled_*.csv (8x28 tables of subject codes, not trials),
  modelingData_wstruct.csv (model BIC table), the per-subject files of rawData (before exclusions) and all code.
The authors' scripts (compiledData_wExclusions.Rmd, NHB_hierarchy_Expt123/Expt45/additionalAnalyses.Rmd,
NHB_Expt6_multiday_learn/forced.Rmd; read as text, never run) define the columns and the analyses reproduced below.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Trach2025_NatHumBehav"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

FILES = ["Experiment1.csv", "Experiment2.csv", "Experiment3.csv", "Experiment4.csv", "Experiment5_learn.csv",
         "Experiment5_forced.csv", "Experiment6_learn.csv", "Experiment6_forced.csv", "hierLearn_prolific.csv",
         "motorData_Experiment1.csv", "motorData_Experiment2.csv", "motorData_Experiment3.csv", "motorData_Experiment4.csv",
         "motorData_Experiment5.csv", "motorData_Experiment6.csv", "md_Experiment6_multiday_wexclusions.csv",
         "md_Experiment6_multiday_motor.csv", "md_HierLearnDay01Forced_Data_exclusions.csv",
         "md_HierLearnDay08Forced_Data_exclusions.csv"]
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for f in FILES:
    checks.internal(f"{f} matches the downloaded hash", stdlib.sha256(RAW / f) == src[f]["sha256"])


def rd(f):
    return pd.read_csv(RAW / f, low_memory=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- key coding
KEYS = "ASDFHJKL"  # fingers 1-8 = keys A S D F (left hand) H J K L (right hand)
ASCII_KEY = {ord(k): k for k in KEYS}
IDX_KEY = {i + 1: k for i, k in enumerate(KEYS)}
TREE = {"rep/rep/rep": 0, "rep/rep/sw": 2, "rep/sw/sw": 4, "sw/rep/sw": 6, "sw/sw/sw": 6}  # authors' treeDistance from combinedType


def key_from_ascii(s):
    return s.astype("Int64").map(ASCII_KEY).fillna("").astype(object)


def key_from_index(s):
    return s.astype("Int64").map(IDX_KEY).fillna("").astype(object)


def path_distance(d):
    """The authors' treeDistance; the Experiment 2 file (paper numbering) lacks the column, which the authors' script rebuilds
    from the actual-response switch codes: the same value as combinedType gives in every file that has both (checked)."""
    return d["treeDistance"] if "treeDistance" in d else d["combinedType"].map(TREE)


DEV_JS = "Lenovo Ideapad 5 laptop keyboard (keys A S D F H J K L), jsPsych 6.1.0"
DEV_PTB = "Lenovo Ideapad 5 laptop keyboard (keys A S D F H J K L), Octave 6.4.0 with Psychtoolbox 3.0.18"
DEV_MD_LAB = "Lenovo Ideapad 5 laptop keyboard (keys A S D F H J K L)"
DEV_MD_HOME = "participant's own computer keyboard (keys A S D F H J K L), online practice"
DEV_ONLINE = "participant's own computer, web browser, keyboard (keys A S D F H J K L); online (Prolific)"

PHASE = {"learn": "learning", "forced": "test", "motor": "baseline", "postmotor": "other"}
COND = {"learn": "learning task", "forced": "forced-response task", "motor": "baseline motor task",
        "postmotor": "baseline motor task after learning"}


def ascii_part(d, label, kind, device, setting, rtc_col, group_col=None, dep_streak=True):
    """Experiments with ASCII key codes (jsPsych files): kind 'learn' (trialType rl) or 'motor' (motor and postmotor)."""
    n = len(d)
    tt = d["trialType"].to_numpy()
    if kind == "learn":
        assert set(tt) == {"rl"} and set(d["phase"]) == {"test"}
        sub = np.array(["learn"] * n)
        trial_type = np.array(["learning"] * n, dtype=object)
        block = np.array(["test"] * n, dtype=object)
    else:
        assert set(tt) <= {"motor", "postmotor"} and (d["phase"].to_numpy() == tt).all()
        sub = np.array(["motor"] * n)
        trial_type = tt.astype(object)
        block = d["phase"].to_numpy().astype(object)
    ph = pd.Series(np.where(sub == "learn", "learn", tt)).map(PHASE).to_numpy()
    cond = pd.Series(np.where(sub == "learn", "learn", tt)).map(COND).to_numpy()
    out = pd.DataFrame({
        "exp": label, "pid": d["subject"].astype(str).to_numpy(), "task": sub, "session": 1, "block": block,
        "tib": d["trialNum"].to_numpy(), "phase": ph, "time_s": d["time_elapsed"].to_numpy() / 1000.0, "cond": cond,
        "ttype": trial_type, "stim": d["stimNum"].astype(str).to_numpy(),
        "corr_resp": key_from_ascii(d["corrresp"]).to_numpy(), "resp": key_from_ascii(d["key_press"]).to_numpy(),
        "responded": d["key_press"].notna().to_numpy(), "correct": (d["corr"] == 1).to_numpy(), "rt": d["rt"].to_numpy(float),
        "prep": np.nan, "rep": d["iter"].to_numpy(float) if kind == "learn" else np.nan,
        "rtc": d[rtc_col].to_numpy(float) if rtc_col else np.nan,
        "pdist": path_distance(d).to_numpy(float),
        "prev": key_from_index(d["finger_press_prev"]).to_numpy() if "finger_press_prev" in d else "",
        "group": d[group_col].astype(str).to_numpy() if group_col else "", "setting": setting, "device": device,
        "dep_streak": d["streak"].to_numpy(float) if (kind == "learn" and "streak" in d) else np.nan,
        "dep_hier": d["hierarchy"].to_numpy(object) if "hierarchy" in d else "",
    })
    # the two codings of the key in the deposit agree
    ok = (key_from_index(d["corrrespAll"]).to_numpy() == out.corr_resp.to_numpy()).all() and \
         (key_from_index(d["finger_press"]).to_numpy() == out.resp.to_numpy()).all()
    checks.internal(f"{label} {kind}: ASCII key codes agree with the authors' finger numbers 1-8 (corrrespAll, finger_press)", bool(ok))
    checks.internal(f"{label} {kind}: corr equals (key pressed == correct key) on every row",
                    bool(((out.resp == out.corr_resp).to_numpy() == out.correct.to_numpy()).all()))
    return out


def index_part(d, label, kind, device, setting, rtc_col, hier=None):
    """Octave/Psychtoolbox files (Experiments 4 and 5) with finger numbers 1-8: kind 'learn', 'forced' or 'motor'."""
    n = len(d)
    assert set(d["taskType"]) == ({"forced"} if kind == "forced" else {"free"})
    ph_ = {"learn": "learn", "forced": "forced", "motor": "motor"}[kind]
    assert set(d["phase"]) == ({"motor"} if kind == "motor" else {"learn"})
    checks.internal(f"{label} {kind}: corr equals (resp == stim); stim equals the authors' corrresp",
                    bool(((d["resp"] == d["stim"]).astype(int) == d["corr"]).all() and (d["stim"] == d["corrresp"]).all()))
    if kind == "forced":
        checks.internal(f"{label} forced: the authors' rt equals PT_actual - PT_planned (s)",
                        bool(np.allclose(d["rt"], d["PT_actual"] - d["PT_planned"], atol=0.0011)))
        rt = d["PT_actual"].to_numpy(float) * 1000.0
        prep = d["PT_planned"].to_numpy(float) * 1000.0
        tib = d["trial"].to_numpy()
        assert (d["trial"] == d["trialNum"]).all()
    else:
        checks.internal(f"{label} {kind}: rt (ms) equals rtS (s) x 1000", bool(np.allclose(d["rt"], d["rtS"] * 1000, atol=0.6)))
        rt = d["rt"].to_numpy(float)
        prep = np.full(n, np.nan)
        tib = d["trialNum"].to_numpy()
    return pd.DataFrame({
        "exp": label, "pid": d["subject"].astype(str).to_numpy(), "task": kind, "session": 1,
        "block": d["block"].astype(str).to_numpy(), "tib": tib, "phase": PHASE[ph_], "time_s": np.nan, "cond": COND[ph_],
        "ttype": {"learn": "learning", "forced": "forced response", "motor": "motor"}[kind],
        "stim": d["stim"].astype(str).to_numpy(), "corr_resp": key_from_index(d["stim"]).to_numpy(),
        "resp": key_from_index(d["resp"]).to_numpy(), "responded": True, "correct": (d["corr"] == 1).to_numpy(), "rt": rt,
        "prep": prep, "rep": d["iter"].to_numpy(float) if kind != "motor" else np.nan,
        "rtc": d[rtc_col].to_numpy(float) if rtc_col else np.nan, "pdist": path_distance(d).to_numpy(float),
        "prev": key_from_index(d["finger_press_prev"]).to_numpy() if "finger_press_prev" in d else "",
        "group": "", "setting": setting, "device": device,
        "dep_streak": d["streak"].to_numpy(float) if (kind == "learn" and "streak" in d) else np.nan,
        "dep_hier": d["group"].to_numpy(),
    })


# ---------------------------------------------------------------- read and build
EXPS = [  # (label, learn file, motor file, forced files, device, setting, group column, learn rt-corrected column)
    ("Experiment 1", "Experiment1.csv", "motorData_Experiment1.csv", DEV_JS, None),
    ("Modified structure control", "Experiment2.csv", "motorData_Experiment2.csv", DEV_JS, None),
    ("Experiment 2", "Experiment3.csv", "motorData_Experiment3.csv", DEV_JS, "type"),
    ("Experiment 3", "Experiment4.csv", "motorData_Experiment4.csv", DEV_JS, None),
]
parts = []
nfin = {}          # label -> participants in the learning file
sets = {}          # label -> {task: set of participant keys}
raw_rows = 0
learn_raw = {}
for label, lf, mf, dev, gcol in EXPS:
    dl, dm = rd(lf), rd(mf)
    raw_rows += len(dl) + len(dm)
    learn_raw[label] = dl
    pl = ascii_part(dl, label, "learn", dev, "lab", "rt_corrected_both", gcol)
    pm = ascii_part(dm, label, "motor", dev, "lab", None, gcol if gcol and gcol in dm else None)
    if gcol:   # group is the stimulus set (color or shape) of the participant, constant within participant, not recorded on postmotor rows
        gmap = dict(zip(pl.pid, pl.group))
        pm["group"] = pm.pid.map(gmap).fillna("")
    parts += [pl, pm]
    nfin[label] = pl.pid.nunique()
    sets[label] = {"learn": set(pl.pid), "motor": set(pm.pid)}
    checks.internal(f"{label}: learning and baseline-task files hold the same participants", set(pl.pid) == set(pm.pid))

# Prolific sample (the authors' Study 1A)
dp = rd("hierLearn_prolific.csv")
raw_rows += len(dp)
pp = ascii_part(dp, "Experiment 1A (Prolific)", "learn", DEV_ONLINE, "online", "rt_corrected", None)
parts.append(pp)
nfin["Experiment 1A (Prolific)"] = pp.pid.nunique()
sets["Experiment 1A (Prolific)"] = {"learn": set(pp.pid)}
learn_raw["Experiment 1A (Prolific)"] = dp
checks.internal("Experiment 1A: 169 trials without a key press have no RT and are marked not correct",
                bool(((dp.key_press.isna()) == dp.rt.isna()).sum() >= len(dp) - 1 and (dp.loc[dp.key_press.isna(), "corr"] == 0).all()),
                f"{int(dp.key_press.isna().sum())} rows without a key press, {int(dp.rt.isna().sum())} without an RT")

# Experiments 4 and 5 of the paper (Octave): learn, forced, motor
for label, nm, dev in (("Experiment 4", "5", DEV_PTB), ("Experiment 5", "6", DEV_PTB)):
    dl, df_, dm = rd(f"Experiment{nm}_learn.csv"), rd(f"Experiment{nm}_forced.csv"), rd(f"motorData_Experiment{nm}.csv")
    raw_rows += len(dl) + len(df_) + len(dm)
    learn_raw[label] = dl
    pl = index_part(dl, label, "learn", dev, "lab", "rt_corrected_both")
    pf = index_part(df_, label, "forced", dev, "lab", None)
    pm = index_part(dm, label, "motor", dev, "lab", None)
    parts += [pl, pf, pm]
    nfin[label] = pl.pid.nunique()
    sets[label] = {"learn": set(pl.pid), "forced": set(pf.pid), "motor": set(pm.pid)}
    checks.internal(f"{label}: learning, forced-response and baseline-task files hold the same participants",
                    set(pl.pid) == set(pf.pid) == set(pm.pid))

# Experiment 6 of the paper (multi-day): learning over 8 days, baseline task on days 1 and 8, forced response on days 1 and 8
ml, mm = rd("md_Experiment6_multiday_wexclusions.csv"), rd("md_Experiment6_multiday_motor.csv")
f1, f8 = rd("md_HierLearnDay01Forced_Data_exclusions.csv"), rd("md_HierLearnDay08Forced_Data_exclusions.csv")
raw_rows += len(ml) + len(mm) + len(f1) + len(f8)
LAB = "Experiment 6"
order = {s: i for i, s in enumerate(pd.unique(ml["subID"]))}
assert (ml["subID"] == ml["subject"]).all() and (mm["subID"] == mm["subject"]).all()
print("  [diag] Experiment 6 learning file: days non-decreasing within every participant in the authors' file order:",
      bool(ml.groupby("subID", sort=False).day.apply(lambda s: (np.diff(s.to_numpy()) >= 0).all()).all()))


def md_sort(d, key="subID"):
    d = d.assign(_o=d[key].map(order), _i=np.arange(len(d)))
    cols = ["_o", "day"] + (["trialNum"] if "trialNum" in d else []) + ["_i"]
    return d.sort_values(cols, kind="stable").reset_index(drop=True)


def md_part(d, kind):
    d = md_sort(d, "subID" if "subID" in d else "subject")
    n = len(d)
    day = d["day"].to_numpy()
    fin = {"learn": "corrrespAll", "motor": "corrrespAll"}
    if kind in ("learn", "motor"):
        ck, rk = key_from_index(d["corrrespAll"]), key_from_index(d["finger_press"])
        checks.internal(f"Experiment 6 {kind}: corr equals (finger pressed == correct finger); key_press is the finger number",
                        bool((((d["finger_press"] == d["corrrespAll"]).astype(int) == d["corr"]).all()) and (d["key_press"] == d["finger_press"]).all()))
        stim = d["stimNum"].astype(str).to_numpy()
        tib = d["trialNum"].to_numpy()
        rt = d["rt"].to_numpy(float)
        prep = np.full(n, np.nan)
        ph, cond, tt = PHASE[kind], COND[kind], ("learning" if kind == "learn" else "motor")
        block = np.array(["test" if kind == "learn" else "motor"] * n, dtype=object)
        rep = d["cumIter"].to_numpy(float) if kind == "learn" else np.full(n, np.nan)
    else:
        ck, rk = key_from_index(d["stim"]), key_from_index(d["resp"])
        checks.internal("Experiment 6 forced: corr equals (resp == stim); the authors' rt equals PT_actual - PT_planned (s)",
                        bool(((d["resp"] == d["stim"]).astype(int) == d["corr"]).all() and (d["stim"] == d["corrresp"]).all()
                             and np.allclose(d["rt"], d["PT_actual"] - d["PT_planned"], atol=0.0011)))
        stim = d["stim"].astype(str).to_numpy()
        tib = d["trial"].to_numpy()
        rt = d["PT_actual"].to_numpy(float) * 1000.0
        prep = d["PT_planned"].to_numpy(float) * 1000.0
        ph, cond, tt = PHASE["forced"], COND["forced"], "forced response"
        block = d["block"].astype(str).to_numpy()
        rep = d["iter"].to_numpy(float)
    return pd.DataFrame({
        "exp": LAB, "pid": d["subID"].to_numpy() if "subID" in d else d["subject"].to_numpy(), "task": kind, "session": day,
        "block": block, "tib": tib, "phase": ph, "time_s": np.nan, "cond": cond, "ttype": tt, "stim": stim,
        "corr_resp": ck.to_numpy(), "resp": rk.to_numpy(), "responded": True, "correct": (d["corr"] == 1).to_numpy(), "rt": rt,
        "prep": prep, "rep": rep,
        "rtc": d["rt_corrected_both"].to_numpy(float) if kind == "learn" else np.nan,
        "pdist": path_distance(d).to_numpy(float),
        "prev": key_from_index(d["finger_press_prev"]).to_numpy() if "finger_press_prev" in d else "",
        "group": "", "setting": np.where((day == 1) | (day == 8), "lab", "online"),
        "device": np.where((day == 1) | (day == 8), DEV_MD_LAB, DEV_MD_HOME),
        "dep_streak": d["streak"].to_numpy(float) if kind == "learn" else np.nan,
        "dep_hier": d["group"].to_numpy(),
    })


p_ml, p_mm = md_part(ml, "learn"), md_part(mm, "motor")
p_mf = md_part(pd.concat([f1, f8], ignore_index=True), "forced")
parts += [p_ml, p_mm, p_mf]
nfin[LAB] = p_ml.pid.nunique()
sets[LAB] = {"learn": set(p_ml.pid), "motor": set(p_mm.pid), "forced": set(p_mf.pid)}
checks.internal("Experiment 6: baseline-task participants are the learning participants; forced-response participants are a subset",
                set(p_mm.pid) == set(p_ml.pid) and set(p_mf.pid) < set(p_ml.pid))
checks.internal("Experiment 6: each forced-response participant has day 1 and day 8 rows",
                bool(p_mf.groupby("pid").session.apply(lambda s: set(s) == {1, 8}).all()))

import itertools  # noqa: E402

idsets = {k: set(v["prolificID"].astype(str)) | (set(v["demoID"].astype(str)) if "demoID" in v else set())
          for k, v in learn_raw.items() if "prolificID" in v}
checks.internal("no participant code of the deposit's prolificID or demoID columns appears in two of the five experiments that have "
                "them (so every experiment's participants are counted as different persons)",
                len(idsets) == 5 and all(not (idsets[a] & idsets[b]) for a, b in itertools.combinations(idsets, 2)),
                f"{len(idsets)} experiments with these columns")

# ---------------------------------------------------------------- the OpenMotor table
t = pd.concat(parts, ignore_index=True)
t["subj"] = t.exp + "|" + t.task + "|" + t.pid
t["pers"] = t.exp + "|" + t.pid
t["trial"] = t.groupby("subj", sort=False).cumcount() + 1
names = [c[1] for c in stdlib.columns_for("choice_rt")]
rows = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": t.exp.to_numpy(), "group": t.group.to_numpy(), "subject": t.subj.to_numpy(),
    "person": t.pers.to_numpy(), "subject_excluded_by_authors": False, "age": None, "sex": "", "handedness": "",
    "population": "healthy", "setting": t.setting.to_numpy(), "device": t.device.to_numpy(), "session": t.session.to_numpy(),
    "block": t.block.to_numpy(), "trial": t.trial.to_numpy(), "phase": t.phase.to_numpy(), "time_s": t.time_s.to_numpy(),
    "task": "finger movements", "unit": "trial", "condition": t.cond.to_numpy(), "pert_type": "none", "pert_value": 0,
    "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
    "trial_in_block": t.tib.to_numpy(), "trial_type": t.ttype.to_numpy(), "stimulus": t.stim.to_numpy(), "n_choices": 8,
    "correct_response": t.corr_resp.to_numpy(), "response": t.resp.to_numpy(), "responded": t.responded.to_numpy(),
    "correct": t.correct.to_numpy(), "rt_ms": t.rt.to_numpy(), "stop_signal": None, "ssd_ms": None,
    "prep_time_ms": t.prep.to_numpy(), "stimulus_repetition": t.rep.to_numpy(), "rt_motor_corrected_ms": t.rtc.to_numpy(),
    "path_distance": t.pdist.to_numpy(), "previous_response": t.prev.to_numpy(),
})
for c in names:          # columns another agent may have added to the module since: written empty
    if c not in rows:
        rows[c] = None
rows = rows[names]
n = len(rows)
checks.internal("every row of the 19 trial files kept", n == raw_rows, f"{n} rows from {raw_rows} rows in the files")
checks.internal("rows of each subject sequence are contiguous", int((rows.subject != rows.subject.shift()).sum()) == rows.subject.nunique())
checks.internal("no participant lost: 228 persons (40 + 28 + 33 + 26 + 36 + 19 + 20 + 26)",
                rows.person.nunique() == 228 == sum(nfin.values()), f"{rows.person.nunique()} persons")
rtv = pd.to_numeric(rows.rt_ms, errors="coerce")
learn_m = rows.condition == "learning task"
checks.internal("ranges: learning RT 200-3600 ms (Prolific sample 0-3000 ms), baseline-task RT 0-4600 ms, forced PT_actual 0-1750 ms",
                bool(rtv[learn_m & (rows.experiment != "Experiment 1A (Prolific)")].dropna().between(200, 3600).all()
                     and rtv[rows.experiment == "Experiment 1A (Prolific)"].dropna().between(0, 3000).all()
                     and rtv[rows.condition.str.startswith("baseline")].between(0, 4600).all()
                     and rtv[rows.condition == "forced-response task"].between(0, 1750).all()),
                f"learning {rtv[learn_m].min():.0f}-{rtv[learn_m].max():.0f}; baseline {rtv[rows.condition.str.startswith('baseline')].min():.0f}-"
                f"{rtv[rows.condition.str.startswith('baseline')].max():.0f}; forced {rtv[rows.condition == 'forced-response task'].min():.0f}-"
                f"{rtv[rows.condition == 'forced-response task'].max():.0f}")

# the authors' consecutively-correct flag (streak) can be rebuilt from the converted rows
def streak_rows(r):
    s = r.subject.to_numpy()
    tib = pd.to_numeric(r.trial_in_block).to_numpy(float)
    c = (r.correct == True).to_numpy()  # noqa: E712
    same = np.r_[False, s[1:] == s[:-1]]
    adj = np.r_[False, tib[1:] == tib[:-1] + 1]
    both = np.r_[False, c[1:] & c[:-1]]
    return (same & adj & both & c).astype(float)


lm = (t.task == "learn").to_numpy()
st = streak_rows(rows[lm])
dep = t.dep_streak.to_numpy(float)[lm]
ok_s = np.isnan(dep) | (dep == st)
pro = (t.exp[lm] == "Experiment 1A (Prolific)").to_numpy()
checks.internal("the authors' streak column (correct trial after a correct trial with the next trial number) is rebuilt exactly from "
                "correct, trial_in_block and the row order of the converted learning rows, in every experiment but 1A",
                bool(ok_s[~pro].all() and (st[np.isnan(dep)] == 0).all()),
                f"equal on {int((dep[~pro] == st[~pro]).sum())} of {int((~pro).sum())} rows outside Experiment 1A; Experiment 1A: the deposit "
                f"flags {int((~ok_s[pro]).sum())} trials that follow a trial missing from its file; the deposit leaves the flag empty on "
                f"{int(np.isnan(dep).sum())} first trials")
rows_l = rows[lm].copy()
rows_l["streak"] = st

# tree distance is a function of combinedType in every file that has both (justifies the Experiment 2 value)
tree_ok = True
for f in FILES:
    d = rd(f)
    if "treeDistance" in d and "combinedType" in d:
        e = d.combinedType.map(TREE)
        tree_ok &= bool((d.treeDistance.isna() == e.isna()).all() and (d.treeDistance.dropna() == e.dropna()).all())
checks.internal("the authors' treeDistance equals the value from combinedType (0 repeat, 2, 4, 6) in every file that has both", tree_ok)

# ---------------------------------------------------------------- published checks, from the converted rows
SRC_P = "Methods, Participants"
checks.published("Experiment 1: participants", nfin["Experiment 1"], 40, 0, SRC_P)
checks.published("Modified structure control: participants", nfin["Modified structure control"], 28, 0, SRC_P)
checks.published("Experiment 2: participants", nfin["Experiment 2"], 33, 0, SRC_P)
checks.published("Experiment 3: participants", nfin["Experiment 3"], 26, 0, SRC_P)
checks.published("Experiment 4: participants", nfin["Experiment 4"], 36, 0, SRC_P)
checks.published("Experiment 5: participants", nfin["Experiment 5"], 19, 0, SRC_P)
checks.published("Experiment 6: participants who also did the forced-response task", len(sets[LAB]["forced"]), 9, 0,
                 "Methods, Experiment 6 (subset of nine)")
checks.published("Experiment 6: participants in the Day x path distance analysis (N = denominator df + 1 of F(1,19))",
                 nfin[LAB], 20, 0, "Results, Experiment 6 (F(1,19) = 37.81)")


def paired_t(r, label, a, b):
    d = r[(r.experiment == label) & (r.streak == 1) & r.path_distance.notna()]
    avg = d.groupby(["subject", "path_distance"]).rt_motor_corrected_ms.mean().unstack("path_distance")
    x = (avg[a] - avg[b]).dropna()
    k = len(x)
    se = x.std(ddof=1) / np.sqrt(k)
    h = stats.t.ppf(0.975, k - 1) * se
    return k, x.mean() / se, x.mean() - h, x.mean() + h


def tol(s):
    return 0.5 * 10 ** -(len(s.split(".")[1]) if "." in s else 0)


rows_l["rt_motor_corrected_ms"] = pd.to_numeric(rows_l.rt_motor_corrected_ms)
rows_l["path_distance"] = pd.to_numeric(rows_l.path_distance)
PAPER = {  # experiment label -> [(x, y, t, ci_low, ci_high)]: mean of (x-distance minus y-distance) corrected RT, as printed in the Results
    # (corrected RT, consecutively correct trials, two-tailed paired t-tests; Experiments 2 and 3 print the pair as '0 vs 2' etc.)
    "Experiment 1": [(6, 4, "3.24", "17.8", "77.3"), (6, 2, "8.50", "148.6", "241.3"), (6, 0, "13.6", "315.2", "425.4"),
                     (4, 2, "7.12", "105.5", "189.3"), (4, 0, "13.11", "272.9", "372.5"), (2, 0, "13.92", "149.8", "200.8")],
    "Experiment 2": [(2, 0, "11.98", "175.9", "248.0"), (4, 0, "11.31", "197.2", "283.7"), (6, 0, "10.68", "192.4", "283.2"),
                     (4, 2, "2.39", "4.3", "52.7"), (6, 2, "2.23", "2.3", "49.4"), (6, 4, "-0.35", "-17.8", "12.5")],
    "Experiment 3": [(2, 0, "5.5", "148.3", "325.9"), (4, 0, "9.04", "241.0", "383.2"), (6, 0, "10.58", "267.3", "396.5"),
                     (4, 2, "2.19", "4.6", "145.4"), (6, 2, "2.78", "24.7", "165.0"), (6, 4, "0.76", "-34.1", "73.7")],
}
SRC_R = {"Experiment 1": "Results, Experiment 1 (two-tailed paired t-tests, t(39))",
         "Experiment 2": "Results, Experiments 2 and 3: unstructured mappings (Experiment 2, t(32))",
         "Experiment 3": "Results, Experiments 2 and 3: unstructured mappings (Experiment 3, t(25))"}
for lab, lst in PAPER.items():
    for a, b, ts, lo, hi in lst:
        k, tv, l_, h_ = paired_t(rows_l, lab, a, b)
        nm = f"{lab}: corrected RT, {a}- minus {b}-distance (consecutively correct trials, paired)"
        checks.published(f"{nm}, t({k - 1})", tv, float(ts), tol(ts), SRC_R[lab])
        checks.published(f"{nm}, 95% CI lower bound (ms)", l_, float(lo), tol(lo), SRC_R[lab])
        checks.published(f"{nm}, 95% CI upper bound (ms)", h_, float(hi), tol(hi), SRC_R[lab])

# Experiment 6: Day (1, 8) x path distance repeated-measures ANOVA on corrected RT of consecutively correct trials
from statsmodels.stats.anova import AnovaRM  # noqa: E402

md = rows_l[(rows_l.experiment == LAB) & rows_l.session.isin([1, 8]) & (rows_l.streak == 1) & rows_l.path_distance.notna()]
av = md.groupby(["subject", "session", "path_distance"]).rt_motor_corrected_ms.mean().reset_index()
cnt = av.groupby("subject").size()
av = av[av.subject.isin(cnt[cnt == 8].index)]
fit = AnovaRM(av, "rt_motor_corrected_ms", "subject", within=["session", "path_distance"]).fit().anova_table
SRC6 = "Results, Experiment 6 (Day x path distance ANOVA)"
checks.published("Experiment 6: complete participants in the Day x path distance ANOVA (df 19 = N - 1)", av.subject.nunique(), 20, 0, SRC6)
checks.published("Experiment 6: main effect of path distance, F(3,57)", fit.loc["path_distance", "F Value"], 65.6, 0.05, SRC6)
checks.published("Experiment 6: main effect of day, F(1,19)", fit.loc["session", "F Value"], 37.81, 0.005, SRC6)
checks.published("Experiment 6: day x path distance interaction, F value (the paper prints it with df 1,19; here df are 3,57)",
                 fit.loc["session:path_distance", "F Value"], 19.74, 0.005, SRC6)
print("  [diag] Experiment 6 ANOVA df:", fit[["Num DF", "Den DF"]].to_dict("index"))

# iteration counts against 'each stimulus at least 125 times' (Experiment 1 and 2), for the differences list
dl1, dl2 = learn_raw["Experiment 1"], learn_raw["Experiment 2"]
mx1 = dl1.groupby(["subject", "stimNum"]).iter.max()
mx2 = dl2.groupby(["subject", "stimNum"]).iter.max()
low1 = int((mx1.groupby("subject").min() < 125).sum())
low2 = int((mx2.groupby("subject").min() < 125).sum())
print("  [diag] highest iteration per stimulus: Experiment 1 min", int(mx1.min()), "median", int(mx1.median()),
      f"; {low1} of 40 participants below 125 on some stimulus; Experiment 2 min", int(mx2.min()), f"; {low2} of 33")

for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {'ok ' if ch['ok'] else 'BAD'} {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}")
    elif not ch["ok"]:
        print("  [FAILED]", ch["what"], ch.get("detail", ""))

# ---------------------------------------------------------------- participants.csv
G4 = {6: "color,shape,pattern", 5: "color,pattern,shape", 4: "shape,color,pattern", 3: "shape,pattern,color",
      2: "pattern,color,shape", 1: "pattern,shape,color"}   # compiledData_Dyn$hierarchy in NHB_hierarchy_additionalAnalyses.Rmd
PCOLS = [
    ("experiment", "text", "", "Experiment the participant's rows belong to. The deposit has no identifier shared between experiments, so a person who took part in two experiments would have two codes."),
    ("feature_to_level_assignment", "text", "",
     "Order in which the three stimulus features (color, shape, pattern) were assigned to the levels of the finger hierarchy, "
     "top to low: the first feature decided the hand, the second the finger couplet, the third the finger within the couplet "
     "(the authors' hierarchy column, ';' written as ','; Experiment 4 from the authors' group code as their script maps it). "
     "'shuffled' in Experiment 3, where the authors' script relabels the unstructured mapping so. Empty where the deposit has none."),
    ("stimulus_set", "category", "color, shape", "Experiment 2 only: the one feature on which that participant's eight stimuli differed."),
    ("authors_group_code", "integer", "",
     "The deposit's group column (counterbalancing code; Experiment 4 has six codes, Experiment 5 two, Experiment 6 ten). The deposit "
     "does not say what the codes stand for beyond the Experiment 4 mapping given in feature_to_level_assignment."),
]
ppl = []
for label, dl in learn_raw.items():
    pid = dl["subject"].astype(str)
    first = dl.assign(_p=pid).groupby("_p", sort=False).first()
    for p in first.index:
        r = first.loc[p]
        d = {"person": f"{label}|{p}", "experiment": label, "feature_to_level_assignment": "", "stimulus_set": "",
             "authors_group_code": None}
        if "hierarchy" in first.columns and label != "Experiment 2":
            d["feature_to_level_assignment"] = "shuffled" if label == "Experiment 3" else str(r["hierarchy"]).replace(";", ",")
        if label == "Experiment 2":
            d["stimulus_set"] = str(r["type"])
        if label == "Experiment 4":
            d["authors_group_code"] = int(r["group"])
            d["feature_to_level_assignment"] = G4[int(r["group"])]
        if label == "Experiment 5":
            d["authors_group_code"] = int(r["group"])
        ppl.append(d)
gm = ml.groupby("subID", sort=False).group.first()
for p in gm.index:
    ppl.append({"person": f"{LAB}|{p}", "experiment": LAB, "feature_to_level_assignment": "", "stimulus_set": "",
                "authors_group_code": int(gm[p])})
checks.internal("participants.csv has one row for each of the 228 persons", len(ppl) == 228 and len({x["person"] for x in ppl}) == 228)
checks.internal("hierarchy string is constant within participant in the learning files that have it",
                all((learn_raw[k].groupby("subject").hierarchy.nunique() == 1).all() for k in
                    ("Experiment 1", "Modified structure control", "Experiment 3", "Experiment 1A (Prolific)")))
checks.internal("stimulus set (type) is constant within participant in Experiment 2", bool((learn_raw["Experiment 2"].groupby("subject").type.nunique() == 1).all()))

# ---------------------------------------------------------------- release
nz = learn_raw["Experiment 1"].groupby("subject").size()
meta = dict(
    study="Trach & McDougle 2025",
    paper=dict(authors="Trach JE, McDougle SD", year=2025,
               title="Mental graphs structure the storage and retrieval of visuomotor associations",
               venue="Nat Hum Behav 9(9):1898-1912", doi="10.1038/s41562-025-02217-2"),
    lab="McDougle (Yale University)",
    data_url="https://github.com/jetrach/StructuredActionPrepVMDM",
    license="none stated",
    license_url="",
    changes=("Converted to the OpenMotor format with participant, Prolific and demographic ID columns dropped and replaced by "
             "codes. The rows are those of the authors' post-exclusion files (the repository's compiledData folder, and for "
             "Experiment 6 the multi-day files their scripts load from rawData/Experiment6_multiday/data); their preprocessing script "
             "removed participants who failed the accuracy criterion, learning trials with an RT of 200 ms or less or above the "
             "participant's mean + 3 SD, in most experiments the first three trials, baseline-task presses that were wrong or beyond "
             "3 SD, and forced-response trials with PT_actual below 0 or above 1.75 s. Those rows are not in the deposit, so "
             "subject_excluded_by_authors and excluded_by_authors are false on every row. The per-subject files of rawData (before "
             "exclusions), the three Experiment4_scrambled_*.csv tables (8x28 tables of subject codes, not trials) and "
             "modelingData_wstruct.csv (a table of model BICs) were not used. Key codes (ASCII or finger numbers 1-8) are written as "
             "the key letters; forced-response times are converted from s to ms; the authors' derived columns (z-scored and log RTs, "
             "hand, couplet and switch codes, error types, bins, early/mid/late labels, streak, previous correct response, finger "
             "distance, feature switches) and the picture file names were not carried over, except the motor-corrected RT, path "
             "distance and previous response, which the paper's analyses use and which cannot be rebuilt once trials are removed."),
    how=[
        "One row is one trial. Participants press one of eight keys, A S D F with the left hand and H J K L with the right, to "
        "the stimulus shown (eight stimuli, each with one fixed correct key; stimulus is the authors' stimulus number 1-8, which "
        "is also the number of its correct finger, 1-4 left hand, 5-8 right hand). Each participant has up to three subject "
        "sequences, one per task: condition 'learning task' (trial_type learning, the authors' trialType rl: pick the key from "
        "binary feedback; the authors' phase 'test'), 'forced-response task' (Experiments 4, 5 and 6; four beeps and the response "
        "is due at the fourth) and 'baseline motor task' (a green square marks the key to press; before learning, and also "
        "after learning, condition 'baseline motor task after learning', in Experiments 1 and 2).",
        "experiment follows the paper: 'Experiment 1' to 'Experiment 6' and 'Modified structure control'; the authors' compiledData "
        "files number them differently from file 2 on (their Experiment2.csv is the modified structure control, Experiment3.csv "
        "is the paper's Experiment 2, Experiment4.csv Experiment 3, Experiment5_* Experiment 4 and Experiment6_* Experiment 5). "
        "'Experiment 1A (Prolific)' is the authors' Study 1A of the revision (hierLearn_prolific.csv): an online replication of "
        "Experiment 1 that the paper's Methods do not describe, so it has no published numbers to check. Experiment 6 is the "
        "8-day experiment: session is the day (days 1 and 8 in the lab, days 2-7 online practice); the authors' file lists days "
        "in a non-chronological order and rows are put in day order here; its running trial counter (trial_in_block) numbers the days 2, 3, 4, 5, 6, 7, then 1, then 8 for every participant, and increases within each day. group is the stimulus set (color or shape) in "
        "Experiment 2 and empty otherwise; the counterbalancing of features over hierarchy levels is in participants.csv.",
        "In the learning task response is the key pressed (empty when the trial timed out, Experiment 1A only), correct is the "
        "authors' corr, rt_ms the authors' RT in ms (the Octave experiments store seconds, converted back to the authors' ms "
        "column), stimulus_repetition the number of presentations of that stimulus so far (the authors' iter; cumIter, over all "
        "days, in Experiment 6), and trial_in_block the authors' trial counter, which started at 1 before the authors dropped the "
        "first three trials. time_s is the experiment program's time_elapsed in s for the jsPsych experiments (1, 2, 3, modified "
        "structure control, 1A).",
        "In the forced-response task prep_time_ms is the authors' PT_planned (interval from stimulus onset to the fourth beep, "
        "100-1200 ms in steps of 100) and rt_ms their PT_actual (interval from stimulus onset to the response, 0-1750 ms): their "
        "rt column, the response relative to the beep, equals rt_ms - prep_time_ms. The deposit's timing and missed_stim_deadline "
        "flags are not documented and were not carried over.",
        "rt_motor_corrected_ms is the authors' rt_corrected_both (rt minus the participant's mean RT for the same finger "
        "transition in the baseline task); path_distance is their treeDistance (0 same key as the previous trial, 2 other key of "
        "the same couplet of fingers, 4 other couplet of the same hand, 6 other hand), computed on the full sequence, as is "
        "previous_response. The authors' streak flag (this and the previous trial correct, trial numbers consecutive) is not a "
        "column: it is rebuilt from correct, trial_in_block and row order (identical to the deposit's flag except for 20 trials of "
        "Experiment 1A that follow a trial missing from that file), and the paper's RT analyses use rows with streak = 1. Baseline-task rows are the correct first presses the authors kept (correct is always true; in Experiment 6 "
        "the deposit also holds 466 rows with attempt_number 2 or 3, correct presses that followed errors).",
    ],
    unit_note="trial (learning, forced-response or baseline motor task)",
    differences=[
        "Experiment 6: the Methods give 21 analysed participants (27 recruited, 6 excluded); the deposit's multi-day learning file "
        "has 20, matching the denominator df of the Results (F(1,19) and F(3,57) imply 20). The cause of the difference of one "
        "participant was not found.",
        "Experiment 6: the Results print the Day x path distance interaction as F(1,19) = 19.74; the converted rows give F = 19.74 "
        "with df (3,57) (as for path distance), so the printed df look like a typo.",
        f"Experiment 1: the Methods say each stimulus was shown at least 125 times; the highest iteration number of a stimulus in "
        f"the deposit is {int(mx1.min())}-{int(mx1.max())} (median {int(mx1.median())}), and {low1} of 40 participants stay below "
        f"125 on some stimulus (Experiment 2: {low2} of 33; lowest {int(mx2.min())}). The deposit's counter was not checked against "
        f"the authors' raw files.",
        "The forced-response error analyses, the model comparisons (BIC) and the mixed-model fits of the Results were not "
        "reproduced; they need smoothing and model code from the authors.",
    ],
)
stdlib.release(ID, "choice_rt", rows, meta, checks, participants=ppl, participant_columns=PCOLS)
