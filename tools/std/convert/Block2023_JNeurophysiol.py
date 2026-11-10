"""Block and Liu (2023), J Neurophysiol 129(5):1249-1258: visuo-proprioceptive recalibration and reaches made with the other hand.
'summary' module: one row per participant and measure (recalibration values, right-hand reach distance on each trial before and after
the calibration task, left-index-finger position estimates on each trial).

Run from the site folder: python3 -I tools/std/convert/Block2023_JNeurophysiol.py
Raw files (untrusted data, read only) in raw_extra/Block2023_JNeurophysiol/ from https://osf.io/zy49x/ :
  RPg_structure.mat  (MATLAB struct array RPg, 2 groups x 16 participants)   RPg_final.m (the authors' analysis script, read as text)
Not carried over: 'perception' (16 x 9) and columns 3 and 4 of 'VR' (no description in the deposit or the script), the derived 'calc',
'stats', 'first4', 'last4' and 'yby4' fields."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Block2023_JNeurophysiol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
RPg = sio.loadmat(RAW / "RPg_structure.mat", struct_as_record=False, squeeze_me=True)["RPg"]
GNAME = {0: "Mismatch", 1: "Veridical"}
TT = {0: "V", 1: "P", 2: "VP"}
DEV = "robotic manipulandum (right hand) with 2D VR display and mirror; left index finger on touchscreen for the calibration task"
rows = []


def add(g, p, trial, block, phase, measure, desc, value, unit, nt=None):
    if value is None or np.isnan(value):
        return
    rows.append(dict(
        dataset=ID, study=ID, experiment=ID, group=GNAME[g], subject=f"g{g}p{p}", subject_excluded_by_authors=False,
        age=None, sex=None, handedness=None, time_s=None, population="healthy", setting="lab", device=DEV,
        session=1, block=block, trial=trial, phase=phase, task="reaching", unit="participant summary",
        condition="visuo-proprioceptive 70 mm mismatch" if g == 0 else "veridical visuo-proprioceptive cues",
        pert_type="visuo_proprioceptive_offset" if g == 0 else "none", pert_value=70 if g == 0 else 0, pert_unit="mm",
        feedback="altered" if g == 0 else "normal", excluded_by_authors=False,
        measure=measure, measure_description=desc, statistic="value", value=float(value), value_unit=unit, n_trials=nt))


for g in range(2):
    G = RPg[g]
    for p in range(G.VR.shape[0]):
        k = 0
        for j, (nm, d) in enumerate([("visual_recalibration", "Visual recalibration after the calibration task (column 1 of the authors' VR; the script treats column 2 as proprioceptive and the sum as total)"),
                                     ("proprioceptive_recalibration", "Proprioceptive recalibration (column 2 of the authors' VR)")]):
            k += 1
            add(g, p, k, "calibration task", "learning", nm, d, G.VR[p, j], "mm")
        for t in range(G.Vel5DisPre.shape[1]):
            k += 1
            add(g, p, k, "right-hand reaches before calibration", "baseline", "reach_distance",
                "Distance of a right-hand reach toward a 100 mm target on one trial (authors' Vel5DisPre; metres as in the script's plot label). trial counts rows of this person in file order.",
                G.Vel5DisPre[p, t], "m")
        for t in range(G.Vel5DisPost.shape[1]):
            k += 1
            add(g, p, k, "right-hand reaches after calibration", "test", "reach_distance",
                "Distance of a right-hand reach toward a 100 mm target on one trial (authors' Vel5DisPost; metres).", G.Vel5DisPost[p, t], "m")
        for t in range(3):
            y = G.VRtbyt.target[t].y
            for c in range(y.shape[1]):
                k += 1
                add(g, p, k, f"calibration task, {TT[t]} targets", "learning", f"finger_y_position_{TT[t]}_target",
                    f"Forward (y) position in mm of the finger-pointed estimate on one {TT[t]} trial (authors' VRtbyt.target({t+1}).y; target type order "
                    "inferred: the script plots targets 1 and 2 as the V and P types, target 3 has twice the trials as the VP mismatch targets).",
                    y[p, c], "mm")

# published numbers
VR0 = RPg[0].VR
n0, n1 = VR0.shape[0], RPg[1].VR.shape[0]
checks.published("participants in the Mismatch group", n0, 16, 0, "Methods, Participants (n = 16)")
checks.published("participants in the Veridical group", n1, 16, 0, "Methods, Participants (n = 16)")
checks.published("proprioceptive recalibration, Mismatch group mean (mm)", np.nanmean(VR0[:, 1]), 17.4, 0.05, "Results, first paragraph")
checks.published("proprioceptive recalibration, Mismatch group SE (mm)", np.nanstd(VR0[:, 1], ddof=1) / np.sqrt(n0), 4.0, 0.05, "Results, first paragraph")
checks.published("visual recalibration, Mismatch group mean (mm)", np.nanmean(VR0[:, 0]), 38.1, 0.05, "Results, first paragraph")
checks.published("visual recalibration, Mismatch group SE (mm)", np.nanstd(VR0[:, 0], ddof=1) / np.sqrt(n0), 6.1, 0.05, "Results, first paragraph")
pre = [np.nanmean(RPg[g].Vel5DisPre[:, -5:], axis=1) for g in range(2)]
post1 = np.nanmean(RPg[0].Vel5DisPost[:, :5], axis=1)
tt = stats.ttest_ind(pre[0], pre[1])
checks.published("pre-calibration reach distance, Mismatch vs Veridical t(30)", tt.statistic, 0.26, 0.005, "Results, Reaching (t30 = 0.26)")
ydif = (post1 - pre[0]) * -1000
total = VR0[:, 0] + VR0[:, 1]
checks.published("change in undershoot vs proprioceptive recalibration, paired t(15)", stats.ttest_rel(ydif, VR0[:, 1]).statistic, 0.46, 0.005, "Results, Reaching (t15 = 0.46)")
checks.internal("rows = values carried over", len(rows) == 3936, str(len(rows)))
checks.internal("20 pre and 20 post reaches per person", RPg[0].Vel5DisPre.shape[1] == 20 and RPg[0].Vel5DisPost.shape[1] == 20)
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']}: {c.get('computed', '')}")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Block (Indiana University)", data_url="https://osf.io/zy49x/", license="none stated on the OSF project (the paper is CC BY 4.0)",
    license_url="https://osf.io/zy49x/",
    changes="Converted to the OpenMotor summary format: the MATLAB struct array RPg read with scipy and reshaped to one row per participant, "
            "measure and trial; participants coded; groups named Mismatch (struct 1) and Veridical (struct 2) as in the script's plot titles.",
    how=["One row is one value of one participant: two recalibration values in mm (visual, proprioceptive), 20 right-hand reach distances before "
         "and 20 after the calibration task (metres; reaches start at 0 and the target is 100 mm away), and the finger-pointed forward position "
         "in mm on every V, P and VP trial of the calibration task. trial numbers a person's rows in file order; block says which part they come from.",
         "Mismatch participants received a gradually imposed 70 mm forward visuo-proprioceptive offset; Veridical participants saw aligned cues.",
         "Group order (1 Mismatch, 2 Veridical) and the V/P/VP order of the three target structures are inferred from the authors' script (plot titles, "
         "colours and trial counts), not stated in a readme."],
    unit_note="participant summary", differences=[
        "Age and sex are not in the deposit (paper: Mismatch 19.9 +/- 1.2 yr, 4 men; Veridical 20.5 +/- 1.8 yr, 6 men, of 16 each).",
        "The paper's t(15) = -4.32 for change in undershoot vs total recalibration gives -4.33 from the file (computed as in the authors' script); not used as a check.",
        "The deposit's 'perception' matrix and VR columns 3 and 4 are not documented and were left out."])
stdlib.release(ID, "summary", rows, meta, checks)
