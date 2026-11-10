"""Klaus, Lametti, Shiller & McAllister (2019), J Acoust Soc Am 145(2):805-817: 40 female adults first receive perceptual
training that shifts their /ae/-/E/ category boundary toward 'had' (had-shift group) or toward 'head' (head-shift group), then
say 'head' while matching a visual formant target with a slightly higher F1 (visual-acoustic biofeedback).

Raw files, raw_extra/Klaus2019_JASA/ (OSF kme49 'Data and Scripts', fetched with tools/std/fetch.py):
  APS_production_data_cleaned_FINAL.csv   8,185 productions of 'head' (one row per token) of 40 participants: Filename (participant
        and phase), duration (s), f0_midpoint, F1_midpoint, F2_midpoint, F3_midpoint (Hz at the vowel midpoint), Subject (A = had-shift
        = Shifted_Up, B = head-shift = Shifted_Down), Phase (Pre, Intro, Ramp, Hold, Post), Count (token number in the phase),
        pre_mean and pre_sd (the participant's mean and SD of F1 in the Pre phase, Hz)
  APS_Perception_Results.csv              43 participants tested, 3 discarded for experimenter error: perceptual boundary location
        (Bias_1-4) and Acuity_1-4 at four test points, a test date and a free comment (not copied)
  APS_supplement_complete_code.Rmd         the authors' analysis, read as text: Phase names Pre-Probe, Phase 1 (100%), Phase 2 (50%),
        Phase 3 (0%), Post-Probe; Phase 1 = visual target on all 20 trials, Post-Probe = 45 trials of 'head' with no target
  APS_production_data_for_reliability.csv  a second measurement of the tokens for reliability (not used)
Run from the site folder: python3 -I tools/std/convert/Klaus2019_JASA.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Klaus2019_JASA"
SRC = "Klaus et al. 2019, J Acoust Soc Am"
RAW = stdlib.raw_dir(ID)
DEVICE = "microphone with a real-time visual display of the produced formants and a visual formant target (visual-acoustic biofeedback)"
PHASES = ["Pre", "Intro", "Ramp", "Hold", "Post"]
PHASE = {"Pre": ("baseline", "Pre-Probe (no visual target)"), "Intro": ("learning", "Phase 1 (100%): visual target on all 20 trials"),
         "Ramp": ("learning", "Phase 2 (50%): visual target on half of the trials"), "Hold": ("learning", "Phase 3 (0%): visual target withdrawn"),
         "Post": ("test", "Post-Probe (no visual target)")}
GROUP = {"Shifted_Up": "had-shift group (perceptual boundary shifted toward /æ/)",
         "Shifted_Down": "head-shift group (perceptual boundary shifted toward /ɛ/)"}

p = pd.read_csv(RAW / "APS_production_data_cleaned_FINAL.csv")
q = pd.read_csv(RAW / "APS_Perception_Results.csv")
p["f0"] = pd.to_numeric(p.f0_midpoint, errors="coerce")
order = {s: i for i, s in enumerate(dict.fromkeys(p.Subject))}
p["_o"] = p.Subject.map(order)
p["_ph"] = p.Phase.map({k: i for i, k in enumerate(PHASES)})
p = p.sort_values(["_o", "_ph", "Count"], kind="stable").reset_index(drop=True)

rows, n = [], {}
for r in p.itertuples(index=False):
    n[r.Subject] = n.get(r.Subject, 0) + 1
    ph, cond = PHASE[r.Phase]
    rows.append(dict(
        dataset=ID, study=SRC, experiment="", group=GROUP[r.Condition], subject=r.Subject, subject_excluded_by_authors=False, age="",
        sex="F", handedness="", population="healthy", setting="lab", device=DEVICE, session=1, block=r.Phase, trial=n[r.Subject],
        phase=ph, time_s="", task="speech", unit="utterance", condition=cond, pert_type="none", pert_value=0, pert_unit="",
        feedback="normal", excluded_by_authors=False,
        stimulus="head", segment="stressed vowel /ɛ/ of 'head' (EH1)", measure_window="midpoint of the vowel",
        duration_ms=r.duration * 1000, f0_hz="" if pd.isna(r.f0) else r.f0, f1_hz=r.F1_midpoint, f2_hz=r.F2_midpoint, f3_hz=r.F3_midpoint))

# participants: perceptual test results of the 40 participants whose productions were kept
disc = q[q.Discard_due_to_error == "Y"]
qq = q[q.Discard_due_to_error == "N"].set_index("Participant")
pre = p.drop_duplicates("Subject").set_index("Subject")[["pre_mean", "pre_sd"]]
parts = []
for s in order:
    t = qq.loc[s]
    parts.append(dict(person=s, bias_1=t.Bias_1, bias_2=t.Bias_2, bias_3=t.Bias_3, bias_4=t.Bias_4, acuity_1=t.Acuity_1,
                      acuity_2=t.Acuity_2, acuity_3=t.Acuity_3, acuity_4=t.Acuity_4, baseline_f1_mean_hz=pre.loc[s, "pre_mean"],
                      baseline_f1_sd_hz=pre.loc[s, "pre_sd"]))
BIAS = "Perceptual category boundary location on the authors' had-head continuum (their 'bias'), {}."
ACU = "Acuity of the perceptual identification, as the authors' file gives it (Acuity_{}), {}."
WHEN = {1: "before perceptual training", 2: "after the first block of perceptual training", 3: "after the second block of perceptual training",
        4: "after the speech motor task"}
pcols = ([(f"bias_{i}", "number", "continuum steps", BIAS.format(WHEN[i])) for i in (1, 2, 3, 4)]
         + [(f"acuity_{i}", "number", "as in the authors' file", ACU.format(i, WHEN[i])) for i in (1, 2, 3, 4)]
         + [("baseline_f1_mean_hz", "number", "Hz", "Mean F1 of the participant's productions in the Pre-Probe (the authors' pre_mean, used to normalize F1)."),
            ("baseline_f1_sd_hz", "number", "Hz", "SD of F1 of the participant's productions in the Pre-Probe (the authors' pre_sd).")])

# ---- checks
P = json.load(open(stdlib.HERE / "source/extra_studies.json", encoding="utf-8"))
S = next(e for e in P if e["id"] == ID)
df = pd.DataFrame(rows)
ck = stdlib.Checks()
ck.published("participants", df.subject.nunique(), 40, 0, "Abstract (forty female adults)")
ck.published("participants kept of those tested (43 tested, 3 discarded for experimenter error)", len(q) - len(disc), 40, 0,
             "supplementary code, perception analysis (3 participants discarded due to experimenter error)")
ck.published("trials in Phase 1 (visual target on all 20 trials), most per participant", df[df.block == "Intro"].groupby("subject").size().max(), 20, 0,
             "supplementary text, results for production (Phase 1: target presented in all 20 trials)")
ck.published("trials in the Post-Probe (45 trials of head), most per participant", df[df.block == "Post"].groupby("subject").size().max(), 45, 0,
             "supplementary text, results for production (Post-Probe elicited 45 trials)")
ck.internal("rows = tokens in the cleaned production file", len(df) == len(p) == 8185)
ck.internal("20 participants per group", sorted(df.drop_duplicates("subject").group.value_counts().tolist()) == [20, 20])
ck.internal("A participants are had-shift, B participants head-shift", bool((df.subject.str[0].map({"A": GROUP["Shifted_Up"], "B": GROUP["Shifted_Down"]}) == df.group).all()))
ck.internal("pre_mean equals the mean F1 of the Pre-Probe tokens", float(np.abs(p[p.Phase == "Pre"].groupby("Subject").F1_midpoint.mean() - pre.pre_mean).max()) < 1e-6)
ck.internal("perception rows for all kept participants, none for the discarded", set(qq.index) == set(order) and not (set(disc.Participant) & set(order)),
            f"discarded: {len(disc)}")
ck.internal("f0 undefined in some tokens only", int(df.f0_hz.eq("").sum()) == int(p.f0.isna().sum()), f"{int(p.f0.isna().sum())} tokens")

meta = dict(
    paper=dict(authors=S["papers"][0]["authors"], year=S["papers"][0]["year"], title=S["papers"][0]["title"],
               venue=S["papers"][0]["venue"], doi=S["papers"][0]["doi"]),
    lab="McAllister (NYU)", data_url=S["data_url"], license="none stated", license_url="",
    changes=("Duration was converted from seconds to milliseconds. Rows are ordered by participant, then by the experiment's phase order "
             "(Pre, Intro, Ramp, Hold, Post) and the authors' Count; trial counts the rows in that order. The F0 of 140 tokens that the "
             "authors' file marks as undefined is left empty. The authors' file name, row index and Word and TextGridLabel columns "
             "(always 'head' and EH1) were not copied; the perception file's test date and free comment were not copied, and the three "
             "participants discarded for experimenter error appear in neither table. The reliability file was not used."),
    how=["One row is one production of the word 'head' by one participant; the values are the authors' acoustic measures at the "
         "midpoint of the stressed vowel (F0, F1, F2, F3 in Hz, duration in ms).",
         "The group is the perceptual training the participant received before the speech task (boundary shifted toward /æ/ or toward /ɛ/). "
         "In the speech task a visual formant target with a slightly higher F1 was shown: on all 20 trials of Phase 1, half of the trials "
         "of Phase 2, none in Phase 3 or in the Pre and Post probes (condition and block hold the authors' phase names).",
         "participants.csv holds each participant's perceptual boundary (bias) and acuity at the four test points and the Pre-Probe F1 mean "
         "and SD the authors used to normalize F1 (F1 divided by pre_mean)."],
    unit_note="production of the word 'head'",
    differences=["The paper's mixed-model and Wilcoxon statistics and its group medians were not available to compare (the article text could "
                 "not be retrieved); the checks use the participant and trial counts of the abstract and the authors' supplementary text."])
stdlib.release(ID, "speech", rows, meta, ck, participants=parts, participant_columns=pcols)
