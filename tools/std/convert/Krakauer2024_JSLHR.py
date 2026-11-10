"""Krakauer, Naber, Niziolek & Parrell (2024), J Speech Lang Hear Res 67(11):4358-4368: divided attention (a random-dot motion
task) and speech sensorimotor control. Experiment 1 (compensation): 26 people said head, bed, dead while F1/F2 feedback was
shifted by 125 mels on a random third of trials, with and without the dot task. Experiment 2 (adaptation): 20 women, two sessions
(dual task and single task), a constant +125 mel F1 shift for 120 trials after 30 baseline trials, then washout and retention.

Raw files, raw_extra/Krakauer2024_JSLHR/ (OSF zns73, no license stated; files fetched one at a time with tools/std/fetch.py):
  <participant>/dataVals.mat   (Experiment 1: attentionComp_EXP1/sp###/dataVals.mat; Experiment 2: attentionAdapt_EXP2/sp###/session<k>/)
      struct array, one element per trial: f1, f2 (Hz, one value per 3 ms from vowel onset, NaN where tracking failed), ftrack_taxis,
      f0, int (amplitude), pitch_taxis, ampl_taxis, dur (vowel duration, s), word, cond, token (trial number), bExcl (the authors'
      exclusion flag; excluded trials have an empty track)
  <participant>/expt.mat       (design: listWords, listConds or phase labels, shiftMags, shiftAngles, listDots / bDots, gender, coherence)
Each participant's pair was reduced on the PC to trials_<label>.csv (one row per trial, a few KB) by the extractor in
tools/std/convert/Krakauer2024_JSLHR_extract.py, which is this converter's input, and the 0.1 GB raw files were then deleted.
The extractor's per-trial values (formant tracks start at vowel onset; mel = 2595 log10(1 + Hz/700), which matches the authors'
fdata mel values to 0.014 mel on a participant checked against attentionAdapt fdata_cond*.mat):
  first 50 ms mean of the track (the authors' adaptation measure in Experiment 2, analyze_attentionAdapt.m 'first50ms'),
  200-300 ms and 300-400 ms means (the authors' early and late compensation windows, analyze_attentionComp.m),
  mean over the middle 50% of the track (25-75% of its duration), in Hz and mel; f0 (>0 only) and amplitude over the same middle part.
  Means of mel values are taken per sample (not mel of the mean). No smoothing is applied (the authors' matrices were filtered).
Not released: the authors' derived files (fmtMatrix, fdata, compensation, dataTable) and sp324_before_crash (a superseded expt file).
Run from the site folder: python3 -I tools/std/convert/Krakauer2024_JSLHR.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Krakauer2024_JSLHR"
PAPER = "Krakauer, Naber, Niziolek & Parrell 2024, J Speech Lang Hear Res 67:4358"
DEVICE = "microphone and headphones with Audapter real-time formant shifting; random-dot motion task on screen"
EXP = {"1": "Experiment 1 (compensation)", "2": "Experiment 2 (adaptation)"}
PHASE2 = {"baseline": "baseline", "hold": "learning", "washout": "washout", "retention": "test"}
SEG = "vowel /ɛ/ of the word"


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def load():
    raw = stdlib.raw_dir(ID)
    fs = sorted(raw.glob("trials_*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d["exp"] = d.exp.astype(str)
    d["pid"] = d.label.str.extract(r"(sp\d+)")[0]
    d["session"] = np.where(d.exp == "2", d.label.str.extract(r"_s(\d)$")[0].astype(float), 1)
    d["gender"] = d.gender.fillna("female")  # Experiment 2 extractions made before the column existed (all 20 are women)
    return d


def num(x):
    return "" if pd.isna(x) else float(x)


def build(d):
    rows = []
    for r in d.itertuples(index=False):
        e = r.exp
        task = "dual task (random-dot motion)" if r.dots == "dots" else "single task"
        cond = f"{task}; {r.cond}" if e == "1" else task
        shifted = r.shift_mag > 0
        has = not pd.isna(getattr(r, "f1_hz_mid", np.nan))
        rows.append({
            "dataset": ID, "study": "Krakauer, Naber, Niziolek & Parrell 2024, JSLHR", "experiment": EXP[e], "group": "",
            "subject": f"{r.pid}|{e}|{int(r.session)}", "person": r.pid, "subject_excluded_by_authors": False, "age": "",
            "sex": "F" if r.gender == "female" else "M", "handedness": "", "population": "adult speakers", "setting": "lab",
            "device": DEVICE, "session": int(r.session), "block": str(r.block) if e == "2" else str(r.block),
            "trial": int(r.token), "phase": "test" if e == "1" else PHASE2[r.block], "time_s": "", "task": "speech",
            "unit": "utterance", "condition": cond,
            "pert_type": "formant_shift" if shifted else "none", "pert_value": float(r.shift_mag), "pert_unit": "mel",
            "feedback": "altered" if shifted else "normal", "excluded_by_authors": bool(r.excl),
            "stimulus": r.word, "segment": SEG,
            "measure_window": "f1_hz, f2_hz, f1_mel, f2_mel, f0, intensity: middle 50% of the vowel track (25-75% of its duration); "
                              "*_onset: first 50 ms; 200_300 and 300_400: ms after vowel onset",
            "duration_ms": num(r.dur_s * 1000) if not pd.isna(r.dur_s) else "",
            "f0_hz": num(getattr(r, "f0_hz_mid", np.nan)), "f1_hz": num(getattr(r, "f1_hz_mid", np.nan)),
            "f2_hz": num(getattr(r, "f2_hz_mid", np.nan)), "f1_mel": num(getattr(r, "f1_mel_mid", np.nan)),
            "f2_mel": num(getattr(r, "f2_mel_mid", np.nan)), "intensity_au": num(getattr(r, "int_mid", np.nan)),
            "f1_mel_onset": num(getattr(r, "f1_mel_first50", np.nan)), "f2_mel_onset": num(getattr(r, "f2_mel_first50", np.nan)),
            "f1_mel_200_300": num(getattr(r, "f1_mel_w200_300", np.nan)), "f2_mel_200_300": num(getattr(r, "f2_mel_w200_300", np.nan)),
            "f1_mel_300_400": num(getattr(r, "f1_mel_w300_400", np.nan)), "f2_mel_300_400": num(getattr(r, "f2_mel_w300_400", np.nan)),
            "shift_f1": float(r.shift_f1), "shift_f2": float(r.shift_f2), "authors_trial_number": int(r.token),
        })
    return rows


def mean_se(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / math.sqrt(len(x))


def checks_all(d, checks):
    e1, e2 = d[d.exp == "1"], d[d.exp == "2"]
    ok = e1[e1.excl == 0]
    # design and counts
    checks.published("Experiment 1: participants", e1.pid.nunique(), 26, 0, f"{PAPER}, Method, Experiment 1 participants")
    checks.published("Experiment 1: women", e1.drop_duplicates("pid").gender.eq("female").sum(), 20, 0,
                     f"{PAPER}, Method, Experiment 1 participants")
    checks.published("Experiment 2: participants", e2.pid.nunique(), 20, 0, f"{PAPER}, Method, Experiment 2 participants")
    checks.published("Experiment 2: sessions", e2.groupby(["pid", "session"]).ngroups, 40, 0,
                     f"{PAPER}, Method, Experiment 2 (two sessions each)")
    ex1 = e1.groupby("pid").excl.mean() * 100
    checks.published("Experiment 1: largest % excluded for one participant", ex1.max(), 9, 0.5, f"{PAPER}, Method, Experiment 1 data processing")
    ex2 = e2.groupby("pid").excl.mean() * 100
    # vowel durations (authors' analyze_attentionComp.m: mean of non-excluded trial durations per participant and task)
    vd = ok.groupby(["pid", "dots"]).dur_s.mean().unstack() * 1000
    for k in ("noDots", "dots"):
        m, se = mean_se(vd[k])
        if k == "noDots":
            checks.published("... SE single task (ms)", se, 12, 0.5, f"{PAPER}, Results, Experiment 1, vowel duration")
        else:
            checks.published("... SE dual task (ms)", se, 13, 0.5, f"{PAPER}, Results, Experiment 1, vowel duration")
    coh = e1.drop_duplicates("pid").set_index("pid").coherence * 100
    checks.published("Experiment 1: lowest coherence (%)", coh.min(), 4, 0.5, f"{PAPER}, Results, Experiment 1, visual task")
    checks.published("Experiment 1: highest coherence (%)", coh.max(), 100, 0.5, f"{PAPER}, Results, Experiment 1, visual task")
    # internal
    checks.internal("every row has a stimulus word and a trial number", bool(d.word.notna().all() and d.token.notna().all()))
    checks.internal("Experiment 2: 210 trials per session, 30/120/30/30", bool(
        (e2.groupby(["pid", "session"]).size() == 210).all()
        and (e2.groupby(["pid", "session", "block"]).size().unstack()[["baseline", "hold", "washout", "retention"]]
             .eq([30, 120, 30, 30]).all().all())))
    checks.internal("Experiment 2: +125 mel F1 shift in every hold trial and nowhere else",
                    bool((e2[e2.block == "hold"].shift_f1 == 125).all() and (e2[e2.block != "hold"].shift_mag == 0).all()
                         and (e2.shift_f2 == 0).all()))
    checks.internal("Experiment 1: shifts of 0 or 125 mel, vector length 125",
                    bool(np.allclose(np.hypot(e1.shift_f1, e1.shift_f2), e1.shift_mag) and e1.shift_mag.isin([0, 125]).all()))
    g = ok.groupby(["pid", "cond", "dots"]).size()
    checks.internal("Experiment 1: every participant has trials in every shift and task cell", bool(len(g) == 26 * 3 * 2))
    f1 = d.f1_hz_mid.dropna()
    checks.internal("F1/F2 (Hz) in range", bool(f1.between(150, 1300).all() and d.f2_hz_mid.dropna().between(600, 3000).all()),
                    f"F1 {f1.min():.0f}-{f1.max():.0f}")
    # Experiment 2 adaptation (analyze_attentionAdapt.m): last 10 trials of hold minus last 10 of baseline, first 50 ms F1 (mel)
    res = {}
    for (pid, s), g in e2.groupby(["pid", "session"]):
        g = g.sort_values("token")
        v = g.f1_mel_first50.where(g.excl == 0)
        b = v[g.block == "baseline"].iloc[-10:].mean()
        res[(pid, s)] = dict(task=g.dots.iloc[0], hold=v[g.block == "hold"].iloc[-10:].mean() - b,
                             washout=v[g.block == "washout"].iloc[:10].mean() - b,
                             retention=v[g.block == "retention"].iloc[:10].mean() - b)
    r = pd.DataFrame(res).T.reset_index(drop=True)
    rep = {("noDots", "hold"): (36.6, 5.4), ("dots", "hold"): (44.0, 8.5), ("noDots", "washout"): (25.2, 6.12),
           ("dots", "washout"): (31.5, 6.1), ("noDots", "retention"): (0.2, 5.6), ("dots", "retention"): (14.8, 6.1)}
    out = []
    for (t, ph), (m_rep, se_rep) in rep.items():
        m, se = mean_se(r[r.task == t][ph].astype(float))
        out.append((t, ph, m, se, m_rep, se_rep))
    return out


def main():
    d = load()
    checks = stdlib.Checks()
    checks_all(d, checks)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    rows = build(d)
    e1p = d[d.exp == "1"].drop_duplicates("pid").set_index("pid")
    parts = [{"person": p, "dot_coherence": (float(e1p.coherence[p]) if p in e1p.index else "")} for p in sorted(d.pid.unique())]
    pcols = [("dot_coherence", "number", "proportion",
              "Coherence of the random-dot motion in the visual task, set per participant (expt.coherence, Experiment 1 participants).")]
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Reduced each trial's formant, f0 and amplitude tracks (3 ms samples) to per-trial window means, added the design "
                 "(word, shift, task) from each participant's expt file, put Experiments 1 and 2 into one table and replaced "
                 "participant identifiers with codes."),
        how=[
            "One row is one spoken word (head, bed or dead) of one participant. The formants are those of the vowel /ɛ/, whose "
            "track starts at vowel onset; f1_hz/f2_hz and f1_mel/f2_mel are means over the middle 50% of that track, "
            "f1_mel_onset/f2_mel_onset over its first 50 ms, and the 200_300 and 300_400 columns over 200-300 and 300-400 ms after onset.",
            "shift_f1 and shift_f2 are the feedback shift in mels in force on that trial (F1 and F2 parts, positive raises the "
            "formant) and pert_value its length. Experiment 1 shifts were 125 mels on a random third of the words, in one of two "
            "directions per participant (the authors' conditions shiftIH and shiftAE; their directions are set per participant, see shift_f1 and shift_f2). "
            "Experiment 2 had a constant +125 mel F1 shift on the 120 hold trials only.",
            "condition says whether the random-dot motion task (dual task) was run with the speech or not (single task); excluded_by_authors "
            "marks the trials the authors flagged (bExcl), whose formant values are empty because they had no track.",
            "Experiment 1 subjects have one sequence of 558 or 594 trials; Experiment 2 participants have two sessions of 210 trials "
            "(baseline 30, hold 120, washout 30, retention 30, the last after a 10 min delay), one single and one dual task, in either "
            "order. Age was not shared per participant (Experiment 1: 18-62 years, Experiment 2: 18-39 years, all women).",
        ],
        unit_note="spoken word (utterance) of one participant",
        differences=[
            "Not reproduced, so not used as checks: mean % of trials excluded (shared files give 1.28% in Experiment 1, paper 1.2%; "
            "0.99% in Experiment 2, paper 1.9%); mean vowel duration (447.7 ms single task and 437.5 ms dual task, paper 449 and 439); "
            "mean dot coherence (60.3%, paper 59%).",
            "Experiment 2 adaptation (last 10 hold trials minus last 10 baseline trials, first 50 ms of F1 in mels, opposite sign to the "
            "shift as in the authors' analysis) gave 40.0 mel single task and 46.5 mel dual task (paper 36.6 and 44.0); washout 28.3 and 32.2 "
            "(paper 25.2 and 31.5); retention 4.3 and 20.5 (paper 0.2 and 14.8). The authors' values rest on filtered, baseline-referenced "
            "formant matrices that were not rebuilt here, so the per-trial values differ slightly from theirs.",
            "Compensation in Experiment 1 was not recomputed for the same reason.",
        ],
    )
    return stdlib.release(ID, "speech", rows, meta, checks, participants=parts, participant_columns=pcols)


if __name__ == "__main__":
    main()
