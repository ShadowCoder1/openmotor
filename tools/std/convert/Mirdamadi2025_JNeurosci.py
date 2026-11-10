"""Mirdamadi, Babu, Wali, Seigel, Hsiao, Lee-Miller and Block (2025), J Neurosci 45(32):e0181252025: somatosensory cortex and visuo-proprioceptive
recalibration. 'summary' module: Experiment 1 (22 people, cue-conflict and veridical sessions, with short-latency afferent inhibition, SAI) and
Experiment 2 (81 people in three cTBS groups): one row per participant, condition and measure.

Run from the site folder: python3 -I tools/std/convert/Mirdamadi2025_JNeurosci.py
Raw files (untrusted data, read only) in raw_extra/Mirdamadi2025_JNeurosci/ from https://osf.io/w5pfk/ :
  SAI_Data.xlsx (Experiment 1, sheet 'N=22'), TT1_data.xlsx (Experiment 2: sheets Recalibration, Endpoint variance,
  Endpoint variance pre post), SAI_Analysis.R (read as text). Not carried over: the JASP PDF, 'Trial Example.mp4' (video), the ID column."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Mirdamadi2025_JNeurosci"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
e1 = pd.read_excel(RAW / "SAI_Data.xlsx", sheet_name="N=22")
rec = pd.read_excel(RAW / "TT1_data.xlsx", sheet_name="Recalibration")
chg = pd.read_excel(RAW / "TT1_data.xlsx", sheet_name="Endpoint variance")
pp = pd.read_excel(RAW / "TT1_data.xlsx", sheet_name="Endpoint variance pre post")
DEV = "2D virtual reality apparatus with touchscreen and mirror; right index finger indicates target positions; TMS (Experiment 1: SAI; Experiment 2: cTBS)"
rows = []


def add(exp, group, subj, trial, block, phase, cond, pert, meas, desc, val, unit, setting="lab"):
    if pd.isna(val):
        return
    rows.append(dict(
        dataset=ID, study=ID, experiment=exp, group=group, subject=subj, subject_excluded_by_authors=False, age=None, sex=None,
        handedness=None, time_s=None, population="healthy", setting=setting, device=DEV, session=1, block=block, trial=trial, phase=phase,
        task="reaching", unit="participant summary", condition=cond, pert_type="visuo_proprioceptive_offset" if pert else "none",
        pert_value=70 if pert else 0, pert_unit="mm", feedback="altered" if pert else "normal", excluded_by_authors=False,
        measure=meas, measure_description=desc, statistic="value", value=float(val), value_unit=unit, n_trials=None))


# Experiment 1: R = cue-conflict session, S = veridical session (inferred from the recalibration values; see README)
E1 = [("R_P_realign", "proprioceptive_recalibration", "Proprioceptive recalibration (authors' R_P_realign)", "mm"),
      ("R_V_realign", "visual_recalibration", "Visual recalibration (authors' R_V_realign)", "mm"),
      ("SAI", "SAI_after", "Short-latency afferent inhibition after the task: conditioned MEP as a percentage of unconditioned MEP; below 100 means inhibition", "percent"),
      ("SAI_pre", "SAI_before", "Short-latency afferent inhibition before the task, same unit as SAI", "percent"),
      ("Vvar", "visual_y_variance", "Endpoint variance of visual-target estimates (authors' Vvar)", "mm^2"),
      ("Pvar", "proprioceptive_y_variance", "Endpoint variance of proprioceptive-target estimates (authors' Pvar)", "mm^2"),
      ("Wv", "visual_weighting_wv", "Weighting of vision (authors' Wv)", "proportion"),
      ("R_sleep", "rating_sleep", "Self-rated sleep, scale not stated in the deposit (authors' R_sleep)", "rating"),
      ("R_attention", "rating_attention", "Self-rated attention, scale not stated (authors' R_attention)", "rating"),
      ("R_fatigue", "rating_fatigue", "Self-rated fatigue, scale not stated (authors' R_fatigue)", "rating"),
      ("R_pain", "rating_pain", "Self-rated pain, scale not stated (authors' R_pain)", "rating")]
for r in e1.itertuples(index=False):
    cf = r.Cond == "R"
    cond = "cue conflict session (Cond R)" if cf else "veridical session (Cond S)"
    for k, (c, nm, d, u) in enumerate(E1):
        add("Experiment 1 (SAI)", "", f"e1_{int(r.ID)}", k + 1 + (0 if cf else 20), "cue-conflict session" if cf else "veridical session",
            "learning" if c in ("R_P_realign", "R_V_realign") else "other", cond, cf, nm, d, getattr(r, c), u)

# Experiment 2: groups 1-3 = M1, S1, sham (matched on the paper's pre-cTBS variance and weighting; see README)
G2 = {1: "M1 cTBS", 2: "S1 cTBS", 3: "Sham cTBS"}
assert (rec.Group.values == chg.Group.values).all() and (rec.Group.values == pp.Group.values).all()
for i in range(len(rec)):
    g = G2[int(rec.Group[i])]
    s = f"e2_{i}"
    for k, (c, nm, d) in enumerate([("Total", "total_recalibration", "Total recalibration, visual plus proprioceptive (cue-conflict block)"),
                                    ("Visual", "visual_recalibration", "Visual recalibration (cue-conflict block)"),
                                    ("Proprio", "proprioceptive_recalibration", "Proprioceptive recalibration (cue-conflict block)")]):
        add("Experiment 2 (cTBS)", g, s, k + 1, "cue-conflict block", "learning", "cue conflict after cTBS", True, nm, d, rec.loc[i, c], "mm")
    for k, (c, nm, d, u, blk) in enumerate([("Vvar1", "visual_y_variance_pre", "Endpoint variance of visual targets, veridical block before cTBS", "mm^2", "veridical block before cTBS"),
                                            ("Vvar2", "visual_y_variance_post", "Endpoint variance of visual targets, veridical block after cTBS", "mm^2", "veridical block after cTBS"),
                                            ("Pvar1", "proprioceptive_y_variance_pre", "Endpoint variance of proprioceptive targets, before cTBS", "mm^2", "veridical block before cTBS"),
                                            ("Pvar2", "proprioceptive_y_variance_post", "Endpoint variance of proprioceptive targets, after cTBS", "mm^2", "veridical block after cTBS"),
                                            ("Wv1", "visual_weighting_wv_pre", "Weighting of vision, before cTBS", "proportion", "veridical block before cTBS"),
                                            ("Wv2", "visual_weighting_wv_post", "Weighting of vision, after cTBS", "proportion", "veridical block after cTBS")]):
        add("Experiment 2 (cTBS)", g, s, 4 + k, blk, "baseline" if "before" in blk else "test", "veridical cues", False, nm, d, pp.loc[i, c], u)
    for k, (c, nm, d, u) in enumerate([("Change in V var", "change_in_visual_variance", "Change in visual endpoint variance (authors' 'Change in V var'; presumably post minus pre)", "mm^2"),
                                       ("Change in P var", "change_in_proprioceptive_variance", "Change in proprioceptive endpoint variance (authors' 'Change in P var')", "mm^2"),
                                       ("Change in Wv", "change_in_visual_weighting", "Change in visual weighting (authors' 'Change in Wv')", "proportion")]):
        add("Experiment 2 (cTBS)", g, s, 10 + k, "change after cTBS", "other", "veridical cues", False, nm, d, chg.loc[i, c], u)

# checks
r = e1[e1.Cond == "R"]
s_ = e1[e1.Cond == "S"]
checks.published("Experiment 1 participants", e1.ID.nunique(), 22, 0, "Methods, Participants (N = 22)")
checks.published("Experiment 1 cue-conflict proprioceptive recalibration mean (mm)", r.R_P_realign.mean(), 11.1, 0.05, "Results, Behavior")
checks.published("Experiment 1 veridical visual recalibration mean (mm)", s_.R_V_realign.mean(), -1.2, 0.05, "Results, Behavior")
checks.published("Experiment 1 veridical proprioceptive recalibration mean (mm)", s_.R_P_realign.mean(), -2.2, 0.05, "Results, Behavior")
checks.published("Experiment 2 participants", len(rec), 81, 0, "Methods, Participants (3 groups of 27)")
for gi, nm in G2.items():
    checks.published(f"Experiment 2 {nm} group size", int((rec.Group == gi).sum()), 27, 0, "Methods, Participants (N = 27 per group)")
for gi, nm, v in [(1, "M1", 36.8), (2, "S1", 38.8), (3, "sham", 32.7)]:
    checks.published(f"Experiment 2 {nm} visual recalibration mean (mm)", rec[rec.Group == gi].Visual.mean(), v, 0.05, "Results, Experiment 2 cue conflict block")
checks.published("Experiment 2 sham proprioceptive recalibration mean (mm)", rec[rec.Group == 3].Proprio.mean(), 9.6, 0.05, "Results, Experiment 2 cue conflict block")
for gi, nm, v in [(1, "M1", 56.1), (2, "S1", 53.8), (3, "sham", 60.5)]:
    checks.published(f"Experiment 2 {nm} pre-cTBS proprioceptive variance (mm^2)", pp[pp.Group == gi].Pvar1.mean(), v, 0.05, "Results, Veridical blocks")
checks.internal("Experiment 2 total = visual + proprioceptive within 0.01 mm", bool(((rec.Total - rec.Visual - rec.Proprio).abs() < 0.01).all()))
checks.internal("Experiment 1: 2 sessions per person", bool((e1.groupby("ID").size() == 2).all()))
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']}: {c.get('computed', '')}")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Block (Indiana University)", data_url="https://osf.io/w5pfk/", license="none stated on the OSF project",
    license_url="https://osf.io/w5pfk/",
    changes="Converted to the OpenMotor summary format: the four sheets reshaped to one row per participant, session or group, and measure; "
            "participant IDs dropped and participants coded; Experiment 2 group numbers named from the paper.",
    how=["One row is one measure of one participant. Recalibration is in mm (positive = in the direction of the imposed 70 mm visuo-proprioceptive "
         "offset); endpoint variances are mm^2 and visual weighting (Wv) is a 0-1 proportion; SAI is the conditioned-over-unconditioned MEP in percent. "
         "Experiment 1 gives each of 22 people a cue-conflict and a veridical session; Experiment 2 gives 81 people cTBS to M1, S1 or sham before the task.",
         "Experiment 2 sheets are matched by row (the same Group column in all three, and one person per row); trial only orders the measures of a person.",
         "Condition R of Experiment 1 is read as the cue-conflict session and S as the veridical session because the recalibration values (visual 34.7 "
         "mm for R, -1.2 mm for S) match the paper's cue-conflict and veridical means; the authors' R script calls S the reference/sham level. "
         "Experiment 2 groups 1, 2, 3 are M1, S1, sham: matched to the paper's pre-cTBS visual variance and weighting (72.2/91.6/68.5 mm^2 and "
         "0.39/0.46/0.41), which fix the order."],
    unit_note="participant summary", differences=[
        "The paper prints S1 and M1 proprioceptive recalibration as 'cm'; the values match millimetres from the file and are treated as mm.",
        "The paper's Experiment 1 cue-conflict visual recalibration (34.7 mm) is 34.650 mm from the file, which rounds to 34.6; not used as a check.",
        "Ratings of sleep, attention, fatigue and pain have no stated scale in the deposit. Age and sex are not in the deposit (paper: Experiment 1 mean age 21.6 yr, 12 men; Experiment 2 38 men).",
        "The Group-to-site mapping and R/S meanings are inferred as described above; the video example in the deposit is not carried over."])
stdlib.release(ID, "summary", rows, meta, checks)
