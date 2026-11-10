"""Gastrock, Henriques & 't Hart (2026), Discover Neuroscience 21:33: EEG dynamics of movement preparation and error processing in
motor adaptation (30 deg rotation) and de novo learning (mirror reversal). 32 healthy adults, one row per reach, reach90 format.

Run from the site folder: python3 -I tools/std/convert/Gastrock2026_DiscovNeurosci.py
Raw file (untrusted data, read only) in raw_extra/Gastrock2026_DiscovNeurosci/ from https://osf.io/q5n6j/ : behavior.zip (data/behavior).
Used: the main folders p000 to p031, files p###-<k>-<block>_selected.txt (the authors' trial and sample selection; 19 columns, positions in cm),
7 blocks per participant. Not converted: the raw per-trial csv files, the familiarization folder (fam/), p001_lefthand and the three *_unfinished
folders (raw csv in pixels, no authors' selection flags, runs that are not part of the 32-person analysis), and the EEG folder.
The authors' analysis code (github.com/rgastrock/MirRevAdaptEEG: ana/learningRates.R getTrialReachAngleAtFeedback, ana/prepSelection.R, ana/shared.R,
exp/denovo_adaptation.py) was read as text. Preprint read for the design: bioRxiv 10.64898/2025.12.11.693740.

Hand angle: the authors' reach deviation 'at feedback': the angle between the line start->target and the line start->hand at the first selected
sample of step 5 (the moment the hand crosses the target distance and the cursor freezes), computed as atan2 of the mouse position rotated by
minus the target angle. The positions are in the PsychoPy frame of the experiment (y up, 0 deg right), so positive is counter-clockwise, and the
30 deg rotation of the file (column rotation = 30) turns the cursor counter-clockwise: cursor direction minus mouse direction is +30 in the data
(checked below), so full compensation is a hand angle of -30. The mirror axis is read from the data (cursor y = -mouse y for a mirror across the
horizontal line, cursor x = -mouse x for a mirror across the vertical line).
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Gastrock2026_DiscovNeurosci"
RAW = stdlib.raw_dir(ID)
ck = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]
z = zipfile.ZipFile(RAW / "behavior.zip")
ZN = z.namelist()
SEL = ["step", "time_ms", "mousex", "mousey", "cursorx", "cursory", "homex", "homey", "rotation", "participant", "targetangle", "targetx", "targety",
       "trial", "trialsel", "samplesel", "interp", "maxvel", "unsure"]
PHASE = {"aligned": "baseline", "random": "other", "rotation": "learning", "mirror": "learning", "washout": "washout"}


def ang(x, y, t):
    """angle of (x, y) relative to the target direction t (deg), counter-clockwise positive, in (-180, 180]."""
    a = np.degrees(np.arctan2(y, x)) - t
    return (a + 180) % 360 - 180


def num(x):
    return "" if x is None or x != x else x


rows, parts = [], []
probe = dict(rot_diff=[], mirror_flip=[], mirror_comp=[], ranges=[])
for pid in range(32):
    files = sorted((n for n in ZN if re.fullmatch(rf"behavior/p{pid:03d}/p{pid:03d}-\d+-\w+_selected\.txt", n)),
                   key=lambda n: int(re.search(r"-(\d+)-", n).group(1)))
    assert len(files) == 7, (pid, len(files))
    order = "rotation_first" if "rotation" in files[2] else "mirror_first"
    assert (order == "rotation_first") == (pid % 2 == 0), "the authors' rule: even ids do the rotation first"
    mir_axis = None
    k = 0
    for bi, n in enumerate(files):
        taskno, bname = re.search(r"-(\d+)-(\w+)_selected", n).groups()
        kind = re.sub(r"\d+$", "", bname)
        d = pd.read_csv(io.BytesIO(z.read(n)), sep=r"\s+", header=None, names=SEL)
        assert (d.participant == pid).all()
        tr = sorted(d.trial.unique())
        assert tr == list(range(len(tr))), "gaps in the authors' trial numbers"
        # perturbation facts from the samples of the reach (step 4)
        r4 = d[(d.step == 4) & (np.hypot(d.mousex, d.mousey) > 1)]
        if kind == "mirror":
            fy = np.isclose(r4.cursory, -r4.mousey, atol=0.02).mean()
            fx = np.isclose(r4.cursorx, -r4.mousex, atol=0.02).mean()
            assert max(fx, fy) > 0.95 and min(fx, fy) < 0.05, (fx, fy)
            axis = 0 if fy > fx else 90   # reflection line: y negated = mirror across the horizontal line (0 deg); x negated = vertical line (90 deg)
            mir_axis = axis if mir_axis is None else mir_axis
            assert mir_axis == axis
            probe["mirror_flip"].append((fx, fy))
        elif kind in ("rotation", "random", "aligned", "washout"):
            dd = ang(r4.cursorx, r4.cursory, 0) - ang(r4.mousex, r4.mousey, 0)
            dd = (dd + 180) % 360 - 180
        for t in tr:
            s = d[d.trial == t]
            first = s.iloc[0]
            tang = first.targetangle % 360
            tdist = float(np.hypot(first.targetx, first.targety))
            selected = bool(first.trialsel == 1)
            s5 = s[(s.step == 5) & (s.samplesel == 1)]
            ok = selected and len(s5) > 0
            rot = first.rotation
            r = dict.fromkeys(names, "")
            r.update(dataset=ID, experiment=f"{ID}_{kind}", study=ID, subject=f"p{pid:03d}", person=f"p{pid:03d}", id_type="lab", person_link="assumed_unique",
                     prior_exposure="unknown", subject_excluded_by_authors="False", handedness="right", device="tablet", setting="lab",
                     population="healthy", session="0", trial=str(k), trial_orig=str(int(t)), block=bname, phase=PHASE[kind],
                     break_before="True" if t == 0 else "False", trial_kind="reach", target_angle=num(tang), target_distance_norm="1",
                     target_visible_during_move="True", target_event="none", target_jump="0", probe_kind="none", instr_goal="cursor_to_target",
                     instr_report="none", hand_used="right", hand_measure="fixed_distance", fb_visible="online", fb_form="cursor",
                     excluded_by_authors="False" if ok else "True")
            if kind in ("rotation", "mirror"):
                r.update(instr_strategy="allowed", instr_disclosure="exists")
            if kind == "mirror":
                r.update(fb_mode="mirror", online_cursor="perturbed", fb_endpoint_transform_only="False", mirror_axis_deg=str(mir_axis))
            elif kind in ("rotation", "random"):
                r.update(fb_mode="rotated", online_cursor="perturbed", fb_endpoint_transform_only="False", rotation=str(int(rot)))
            else:
                r.update(fb_mode="veridical", online_cursor="veridical", fb_endpoint_transform_only="False", rotation="0")
            if ok:
                f5 = s5.iloc[0]
                ha = float(ang(f5.mousex, f5.mousey, tang))
                ca = float(ang(f5.cursorx, f5.cursory, tang))
                r.update(hand_angle=num(ha), hand_angle_fb=num(ha), hand_extent=num(float(np.hypot(f5.mousex, f5.mousey)) / tdist),
                         cursor_error=num(ca))
                if kind == "rotation":
                    probe["rot_diff"].append(((ca - ha) + 180) % 360 - 180)
                if kind == "mirror":
                    ideal = ((2 * (90 if mir_axis == 90 else 0) - 2 * tang) + 180) % 360 - 180
                    probe["mirror_comp"].append(abs(ideal))
            rows.append(r)
            k += 1
    parts.append(dict(person=f"p{pid:03d}", perturbation_order=order, mirror_axis_deg=mir_axis))

t = pd.DataFrame(rows)
P = pd.DataFrame(parts)
n_ok = (t.excluded_by_authors == "False").sum()

# ---------------------------------------------------------------- published checks (expected values read from the paper)
S = "Methods, Participants"
ck.published("participants", t.person.nunique(), 32, 0, S)
per = t.groupby(["person", "block"]).size().unstack()
for blk, n in (("aligned", 48), ("random0", 48), ("random1", 48), ("washout0", 48), ("washout1", 48)):
    ck.published(f"{blk} trials per participant (all 32 participants have the same number)", per[blk].mean() if per[blk].nunique() == 1 else -1, n, 0, "Fig. 1B caption: 48 aligned, 48 random, 90 training, 48 washout trials")
pert = t[t.block.isin(["rotation", "mirror"])].groupby(["person", "block"]).size()
ck.published("rotation and mirror training trials per participant (all equal)", pert.mean() if pert.nunique() == 1 else -1, 90, 0, "Fig. 1B caption: 90 training trials per perturbation; Methods, Perturbed trials: 15 blocks of six")
al = t[t.block == "aligned"]
ck.published("distinct target directions in the aligned block of one participant (all participants)", al.groupby("person").target_angle.nunique().mean() if al.groupby("person").target_angle.nunique().nunique() == 1 else -1, 12, 0, "Methods, Aligned trials: 12 target locations")
tp = t[t.block.isin(["rotation", "mirror"])]
ck.published("distinct training targets per perturbation and participant (all equal)", tp.groupby(["person", "block"]).target_angle.nunique().mean() if tp.groupby(["person", "block"]).target_angle.nunique().nunique() == 1 else -1, 6, 0, "Methods, Perturbed trials: each block included six targets")
rr = t[t.block.isin(["random0", "random1"])]
rv = rr.rotation.astype(int)
ck.published("distinct random rotation values (+-15, +-25, +-35)", rv.nunique(), 6, 0, "Methods, Random rotation trials")
ck.published("random rotation values are +-15, +-25, +-35 only", int(set(rv.unique()) == {-35, -25, -15, 15, 25, 35}), 1, 0, "Methods, Random rotation trials")
ck.published("each random rotation value appears 8 times per random block and participant", rr.groupby(["person", "block"]).rotation.apply(lambda x: x.astype(int).value_counts().nunique() == 1 and x.astype(int).value_counts().iloc[0] == 8).mean(), 1, 0, "Methods, Random rotation trials: each magnitude and direction once per shuffled set of six, eight blocks")
ck.published("fixed rotation size (deg, counter-clockwise)", t[t.block == "rotation"].rotation.astype(int).median(), 30, 0, "Methods, Perturbed trials: cursor rotated 30 degrees counterclockwise")
ck.published("participants with even ids who did the rotation before the mirror (of 16 even ids)", int(((P.perturbation_order == "rotation_first") & (P.person.str[1:].astype(int) % 2 == 0)).sum()), 16, 0, "Fig. 1C: even numbered participant IDs experienced the rotation before the mirror reversal")
ck.published("distinct full-compensation levels for the mirror (15, 30, 45)", len(set(np.round(probe["mirror_comp"], 0))), 3, 0, "Methods, Amount of compensation: full compensation for the mirror could be 15, 30, or 45 degrees")
ck.published("smallest and largest mirror compensation (deg)", min(probe["mirror_comp"]) + max(probe["mirror_comp"]), 60, 0.01, "Methods, Amount of compensation: 15 and 45 degrees")
# internal
rd = np.array(probe["rot_diff"])
ck.internal("cursor direction = hand direction + 30 deg on every rotation trial (counter-clockwise rotation, checked on the data)", bool(np.abs(rd - 30).max() < 2), f"max abs deviation {np.abs(rd - 30).max():.2f}")
rotl = t[(t.block == "rotation") & (t.excluded_by_authors == "False")].copy()
rotl["h"] = rotl.hand_angle.astype(float)
late = rotl[rotl.trial_orig.astype(int) >= 75].h.mean()
ck.internal("late rotation training hand angle is negative (compensation for a counter-clockwise rotation)", -40 < late < -15, f"mean {late:.1f}")
wo = t[(t.block.isin(["washout0", "washout1"])) & (t.excluded_by_authors == "False")]
ck.internal("mirror axis reads 0 or 90 deg for every participant", set(P.mirror_axis_deg) <= {0, 90})
ck.internal("rows kept: 32 participants x (5 x 48 + 2 x 90 trials)", len(t) == 32 * (48 * 5 + 90 * 2), str(len(t)))
ck.internal("fraction of trials flagged by the authors", True, f"{(t.excluded_by_authors == 'True').mean() * 100:.2f}% of {len(t)} trials flagged as not selected or without a reach sample")

pcols = [("perturbation_order", "text", "rotation_first, mirror_first", "Which perturbation came first; read from the authors' file names (even participant ids do the rotation first, odd ids the mirror first, per the paper's Fig. 1C)."),
         ("mirror_axis_deg", "number", "0 or 90 degrees", "Mirror reflection line used for this participant, read from the cursor and hand positions in the data: 0 = mirror across the horizontal line, 90 = across the vertical line (cursor y or cursor x negated).")]
paper = dict(authors="Gastrock RQ, Henriques DYP, 't Hart BM", year=2026, title="EEG dynamics of movement preparation and error processing in motor adaptation and De Novo learning",
             venue="Discover Neuroscience 21(1):33", doi="10.1186/s13064-026-00260-x")
meta = dict(
    paper=paper, lab="Henriques & 't Hart (York University)", data_url="https://osf.io/q5n6j/", license="none stated", license_url="",
    changes=("Converted to the OpenMotor reaching format; participant folders renamed to codes; hand and cursor angles computed from the authors' selected samples as "
             "the authors' own reach deviation at feedback; positions, EEG and the raw sample files not copied."),
    how=["One row is one reach of a stylus on a digitizing tablet (right hand, cursor seen on a vertical monitor, a circular stencil limited the movement). Each participant did 7 blocks: "
         "48 aligned (12 targets), 48 random-rotation trials (rotations of +-15, +-25, +-35 deg), 90 trials of a fixed 30 deg counter-clockwise rotation or of a mirror reversal, 48 washout trials, "
         "another random block, 90 trials of the other perturbation and 48 washout trials; even-numbered participants did the rotation first. trial is the order over all 7 blocks (from 0); trial_orig is the authors' trial number within the block.",
         "hand_angle is the authors' reach deviation at feedback onset (angle between the start-to-target line and the start-to-hand line at the first selected sample after the hand crossed the target distance), "
         "counter-clockwise positive. The sign is inferred rather than stated: the positions are in the experiment's y-up frame, the file's rotation column (30) turns the cursor counter-clockwise (cursor angle minus hand angle is +30 on every rotation trial), "
         "and late rotation training shows hand angles near -30. hand_angle_fb equals hand_angle and cursor_error is the cursor's direction at the same sample relative to the target. "
         "In mirror blocks rotation is empty and mirror_axis_deg gives the reflection line (0 or 90 deg), read from the cursor and hand positions. Random-rotation trials are phase 'other' with their rotation in the rotation column.",
         "excluded_by_authors marks trials the authors' selection (trialselected flag) removed or that have no reach sample after the hand reached the target distance. Targets lie about 4.9 in the file's units (cm, converted by the authors with 28.04 pixels per cm), whereas the paper states 4.58 cm, so hand_extent is relative to the target distance in the file. "
         "Not included: the familiarization trials (fam folder), the left-hand run and three unfinished runs (raw files in pixels, outside the analysed 32), the EEG recordings, and reaction and movement times.",
         "Age and sex are given in the paper only as group statistics (13 women, mean age 21.6, SD 3.3)."],
    unit_note="reach",
    differences=["The paper reports behaviour as figures and Bayesian t-tests on baseline-corrected percent compensation (for example BF10 > 1e8 for the first versus last block of fixed-rotation training, BF10 = 0.447 for the mirror); "
                 "these were not recomputed, so the published checks are the design numbers of Methods and Fig. 1 (participants, trials per block, targets, rotation sizes, mirror compensation levels). "
                 "The OSF project states no license; the article itself is CC BY-NC-ND 4.0, which does not cover the data."],
)
stdlib.release(ID, "reach90", t.to_dict("records"), meta, ck, participants=P.to_dict("records"), participant_columns=pcols)
