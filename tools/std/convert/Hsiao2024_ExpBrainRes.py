"""Hsiao and Block (2024), Exp Brain Res 242(9):2249-2261: explicit knowledge of a visuo-proprioceptive cue conflict, 3 groups x 18
participants x 3 blocks. 'summary' module: one row per participant, block and measure (recalibration, weighting, variance, movement
time) plus one row per participant for age, handedness score and the post-task questionnaire.

Run from the site folder: python3 -I tools/std/convert/Hsiao2024_ExpBrainRes.py
Raw file (untrusted data, read only) in raw_extra/Hsiao2024_ExpBrainRes/ from https://osf.io/gqxra/ : CSg_structure.mat. Column meanings come
from the project wiki 'Data Structure' and the script CSg_Final.m (read as text). The subject-code column of 'perception' is not carried over."""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hsiao2024_ExpBrainRes"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
S = sio.loadmat(RAW / "CSg_structure.mat", struct_as_record=False, squeeze_me=True)["CSg_structure"]
GROUP = ["Group 1 (no information about the offset)", "Group 2 (medium information: instructional cues)",
         "Group 3 (high information: instructional and extra visual cues)"]
BLOCK = ["Block 1 (veridical cues)", "Block 2 (mismatch: recalibration block)", "Block 3 (veridical cues)"]
RAWM = [("visual_recalibration", "Visual recalibration", "mm"), ("proprioceptive_recalibration", "Proprioceptive recalibration", "mm"),
        ("total_recalibration", "Total recalibration (visual plus proprioceptive)", "mm"),
        ("visual_weighting_wv", "Mean weighting of vision (wv)", "proportion"),
        ("visual_y_variance", "y-dimension variance of visual-target estimates", "mm^2"),
        ("proprioceptive_y_variance", "y-dimension variance of proprioceptive-target estimates", "mm^2"),
        ("movement_time", "Average movement time, leaving the start box until the endpoint was recorded (unit not stated; values of 0.2-0.6 suggest seconds)", "s")]
DEV = "2D virtual reality apparatus: touchscreen, mirror and rear projection; right index finger indicates target positions"
rows = []


def base(g, p, age, sex, trial, block, phase, cond):
    return dict(dataset=ID, study=ID, experiment=ID, group=GROUP[g], subject=f"g{g}p{p}", subject_excluded_by_authors=False,
                age=age, sex=sex, handedness=None, time_s=None, population="healthy", setting="lab", device=DEV, session=1,
                block=block, trial=trial, phase=phase, task="reaching", unit="participant summary", condition=cond,
                pert_type="visuo_proprioceptive_offset" if "mismatch" in cond else "none", pert_value=70 if "mismatch" in cond else 0,
                pert_unit="mm", feedback="altered" if "mismatch" in cond else "normal", excluded_by_authors=False,
                statistic="value", n_trials=None)


def num(x):
    return None if np.isnan(x) else float(x)


for g in range(3):
    per = S[g].perception
    for p in range(per.shape[0]):
        age = num(per[p, 1])
        sex = {0: "F", 1: "M"}.get(int(per[p, 2])) if not np.isnan(per[p, 2]) else None
        t = 0
        for b in range(3):
            raw = S[g].data[b].raw
            cond = "70 mm mismatch" if b == 1 else "veridical cues"
            for m, (nm, d, u) in enumerate(RAWM):
                v = num(raw[p, m])
                t += 1
                if v is None:
                    continue
                r = base(g, p, age, sex, t, BLOCK[b], "learning" if b == 1 else "baseline" if b == 0 else "test", cond)
                r.update(measure=nm, measure_description=d, value=v, value_unit=u)
                rows.append(r)
        QUES = [(3, "edinburgh_handedness_score", "Edinburgh Handedness Inventory score", "score"),
                (4, "noticed_offset", "Noticed the visuo-proprioceptive offset (1 yes, 0 no)", "0/1"),
                (5, "estimated_offset_direction", "Estimated offset direction (0 forward, 1 back, 2 left, 3 right; empty = could not tell)", "code"),
                (6, "estimated_offset_amount", "Estimated offset amount", "cm"),
                (7, "sleep_quality", "Quality of sleep the night before (1-10, 10 high)", "rating"),
                (8, "attention_level", "Attention throughout the task (1-10, 10 high)", "rating"),
                (9, "fatigue", "Fatigue after the study (1-10, 10 high)", "rating")]
        for c, nm, d, u in QUES:
            v = num(per[p, c])
            t += 1
            if v is None:
                continue
            r = base(g, p, age, sex, t, "questionnaire after the task", "other", "none")
            r.update(measure=nm, measure_description=d, value=v, value_unit=u)
            rows.append(r)

for g in range(3):
    checks.published(f"participants in group {g+1}", S[g].perception.shape[0], 18, 0, "OSF wiki 'Data Structure': N=18 participants per group")
checks.published("participants in total", sum(S[g].perception.shape[0] for g in range(3)), 54, 0, "OSF wiki 'Data Structure': 3 groups of 18")
checks.internal("raw matrices have 18 rows and 7 columns in every group and block",
                all(S[g].data[b].raw.shape == (18, 7) for g in range(3) for b in range(3)))
checks.internal("sex coded 0/1 only", set(np.unique(np.concatenate([S[g].perception[:, 2] for g in range(3)]))) <= {0.0, 1.0})
checks.internal("total = visual + proprioceptive recalibration within 0.01 mm",
                all(np.nanmax(np.abs(S[g].data[b].raw[:, 2] - S[g].data[b].raw[:, 0] - S[g].data[b].raw[:, 1])) < 0.01 for g in range(3) for b in range(3)))
checks.internal("Group 3 mean block-2 total recalibration is the lowest of the three groups (abstract: Group 3 recalibrated significantly less)",
                np.nanmean(S[2].data[1].raw[:, 2]) < min(np.nanmean(S[0].data[1].raw[:, 2]), np.nanmean(S[1].data[1].raw[:, 2])))
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']}: {c.get('computed', '')}")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Block (Indiana University)", data_url="https://osf.io/gqxra/", license="none stated on the OSF project",
    license_url="https://osf.io/gqxra/",
    changes="Converted to the OpenMotor summary format: the MATLAB struct array CSg_structure read with scipy and reshaped to one row per "
            "participant, block and measure; the subject code column dropped and participants coded; age and sex taken from the 'perception' matrix.",
    how=["One row is one measure of one participant. For each of three blocks: visual, proprioceptive and total recalibration (mm), mean visual "
         "weighting (wv, 0-1), y-dimension variance of visual and proprioceptive estimates (mm^2, unit from the paper's convention for endpoint variance, "
         "not stated in the wiki) and average movement time (unit not stated, read as seconds). Then one row each for the questionnaire items.",
         "A 70 mm visuo-proprioceptive offset was gradually imposed in block 2; blocks 1 and 3 are the veridical blocks used for weighting and variance "
         "(this is how the authors' script uses them). Group 1 got no information about the offset, Group 2 instructional cues, Group 3 instructional "
         "plus extra visual cues (abstract).",
         "The matrices of the three groups and the perception matrix are matched by row, as the wiki says each row is one participant."],
    unit_note="participant summary", differences=[
        "The paper's full text was not accessible, so no numbers from the paper's results were reproduced; the checks are the group sizes stated in "
        "the deposit's wiki plus internal checks, and the direction of the group difference stated in the abstract.",
        "Which block is baseline, mismatch and washout is inferred from CSg_Final.m (block 2 is analysed for recalibration, blocks 1 and 3 for variance and weighting)."])
stdlib.release(ID, "summary", rows, meta, checks)
