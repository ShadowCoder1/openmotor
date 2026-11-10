"""Hillman, McClure & McDougle (2025), J Neurophysiol: the same 31 participants did a motor working memory (MWM) task and
then a 45 deg visuomotor rotation (VMR) task with catch (exclusion) trials, on a Kinarm End-Point robot. One row per
reach, in the 90-column reaching format, two experiments: _MWM and _VMR (person codes link them).

Run from the site folder: python3 -I tools/std/convert/Hillman2025_JNeurophysiol.py
Raw files (untrusted data, read only) in raw_extra/Hillman2025_JNeurophysiol/, from
https://github.com/hhillman231/MWM_MotorLearning_public (commit e089baf):
  mwm.csv   recall reaches of the MWM task (31 participants, 2 blocks of 48 trials, trials with |angular error| > 90 deg
            removed by the authors), with each participant's VMR metrics, age, sex and handedness repeated on every row
  vmr.csv   reaches of the VMR task (255 trials per participant; outlier trials removed by the authors)
The authors' fig2-5.Rmd were read as text to learn the variables and the metrics, never run (their helper
functions_formats.R is not in the repository).

Signs. vmr.csv stores angles mirrored for the participants with rotDir = +1 so that compensation is positive for everyone:
handAng = -rotDir * wrap(atan2(handy, handx) - atan2(targy, targx)) and targAng = -rotDir * atan2(targy, targx) (checked on
every row below). Multiplying back by -rotDir gives counter-clockwise-positive angles in the frame of the raw x, y
coordinates (x right, y away from the body), the OpenMotor convention. The cursor was rotated by +45 * rotDir deg in that
frame (cursAng = handAng - 45 on rotation trials in the authors' mirrored frame). mwm.csv angles are already in the raw
frame: ang_err = wrap(hand_ang - target_ang), both from atan2 of the x, y coordinates.
"""
import json
import math
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hillman2025_JNeurophysiol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
TARGET_RADIUS_DEG = math.degrees(math.atan(0.5 / 9.0))  # VMR: 0.5 cm radius target at 9 cm (Methods, Visuomotor Rotation Task)


def wrap(a):
    """Degrees into (-180, 180]."""
    w = (a + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


def tf(x):
    return "True" if x else "False"


# ---------------------------------------------------------------- read
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for fn in ("mwm.csv", "vmr.csv"):
    checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])
m = pd.read_csv(RAW / "mwm.csv")
v = pd.read_csv(RAW / "vmr.csv")
ids = list(dict.fromkeys(m.id))
checks.internal("31 participants in both files, same ids, rows sorted by participant and trial",
                ids == list(dict.fromkeys(v.id)) == list(range(1, 32))
                and (m.sort_values(["id", "trial"], kind="stable").index == m.index).all()
                and (v.sort_values(["id", "trial"], kind="stable").index == v.index).all())
DEMO = ["age", "sex", "dom_hand", "hand_score"]
VM = ["total", "lastcatch", "explicit", "implicit"]
checks.internal("age, sex, handedness and the VMR metrics are constant within participant in mwm.csv, and the "
                "demographics are the same in vmr.csv",
                (m.groupby("id")[DEMO + VM].nunique() == 1).all().all() and (v.groupby("id")[DEMO].nunique() == 1).all().all()
                and (m.groupby("id")[DEMO].first().astype(str).values == v.groupby("id")[DEMO].first().astype(str).values).all())
demo = m.groupby("id")[DEMO + VM].first()
rotdir = v.groupby("id").rotDir.first()
checks.internal("rotation direction constant within participant (15 participants +1, 16 participants -1)",
                (v.groupby("id").rotDir.nunique() == 1).all() and Counter(rotdir) == Counter({1: 15, -1: 16}))

# ---------------------------------------------------------------- VMR: schedule and signs
sched = v.groupby("trial")[["phase", "type", "rot", "online"]].agg(lambda s: tuple(sorted(set(s))))
checks.internal("VMR schedule is the same for everyone: trial number fixes the phase, rotation and cursor (255 trials)",
                (sched.map(len) == 1).all().all() and list(sched.index) == list(range(1, 256)))
PHASE_OF = {t: sched.loc[t, "phase"][0] for t in sched.index}
ROT_ON = {t: sched.loc[t, "rot"][0] == 45 for t in sched.index}
CURSOR = {t: sched.loc[t, "online"][0] == 1 for t in sched.index}
counts = Counter(PHASE_OF.values())
checks.internal("39 baseline, 160 rotation (learn1-5: 20, 40, 40, 40, 20), 4 x 4 catch and 40 washout trials (Methods)",
                counts == Counter({"baseline": 39, "learn1": 20, "learn2": 40, "learn3": 40, "learn4": 40, "learn5": 20,
                                   "catch1": 4, "catch2": 4, "catch3": 4, "catch4": 4, "washout": 40}))
checks.internal("cursor shown (online = 1) on baseline and rotation trials only; rotation 45 exactly on learn1-5",
                all(CURSOR[t] == (PHASE_OF[t] == "baseline" or PHASE_OF[t].startswith("learn")) for t in PHASE_OF)
                and all(ROT_ON[t] == PHASE_OF[t].startswith("learn") for t in PHASE_OF))
std_t = np.degrees(np.arctan2(v.targy, v.targx))
std_h = np.degrees(np.arctan2(v.handy, v.handx))
rd = v.id.map(rotdir).astype(float)
d1 = np.abs(((v.targAng + rd * std_t) + 180) % 360 - 180).max()
d2 = np.abs((((-rd * v.handAng) - (std_h - std_t)) + 180) % 360 - 180).max()
checks.internal("targAng = -rotDir * atan2(targy, targx) and handAng = -rotDir * (hand direction - target direction) "
                "on every row", d1 < 1e-9 and d2 < 1e-9, f"max differences {d1:.1e}, {d2:.1e} deg")
cur = (v.cursAng + rd * std_h + 180) % 360 - 180
checks.internal("cursor 45 deg from the hand on rotation trials (cursAng = handAng - 45 in the authors' mirrored frame), "
                "on the hand otherwise", (np.abs(cur[v.rot == 45] + 45).max() < 1e-9) and (np.abs(cur[v.rot == 0]).max() < 1e-9))
checks.internal("handAngErr = 45 - handAng on rotation trials and -handAng otherwise (the cursor error, mirrored)",
                np.abs(v.handAngErr + v.handAng - v.rot).max() < 1e-9)
checks.internal("hand angle taken where the hand crossed the 9 cm target ring (hand x, y at 8.8-9.1 cm from home)",
                np.sqrt(v.handx ** 2 + v.handy ** 2).between(8.7, 9.1).all())
late = v[v.phase == "learn5"].groupby("id").handAng.mean()
checks.internal("all 31 participants compensate in late learning (mean handAng in learn5 > 15 deg), in both rotation "
                "directions", (late > 15).all())
checks.internal("the speed rule of the Methods: the authors' fb code is 1 exactly on trials with RT > 500 ms",
                ((v.fb == 1) == (v.rt > 500)).all())

# ---------------------------------------------------------------- MWM: structure and signs
m["off"] = m.trial - m.block_trial
checks.internal("MWM: the authors' trial = block offset + block_trial within each participant's block",
                (m.groupby(["id", "block"]).off.nunique() == 1).all())
th = np.degrees(np.arctan2(m.handy, m.handx))
tt = np.degrees(np.arctan2(m.targety, m.targetx))
e1 = np.abs(((th - m.hand_ang) + 180) % 360 - 180).max()
e2 = np.abs(((tt - m.target_ang) + 180) % 360 - 180).max()
e3 = np.abs((((m.hand_ang - m.target_ang) + 180) % 360 - 180) - m.ang_err).max()
checks.internal("MWM: hand_ang and target_ang are atan2 of the x, y coordinates and ang_err = hand_ang - target_ang "
                "(counter-clockwise positive)", max(e1, e2, e3) < 1e-9, f"max differences {e1:.1e}, {e2:.1e}, {e3:.1e}")
tdist = np.sqrt(m.targetx ** 2 + m.targety ** 2)
checks.internal("MWM: targets 9 cm from home, dist = |hand x, y|, ext_err = dist - target distance, dist_err = "
                "Euclidean distance hand - target, abs_ang_err = |ang_err|",
                tdist.between(8.999, 9.001).all() and np.abs(np.sqrt(m.handx ** 2 + m.handy ** 2) - m.dist).max() < 1e-9
                and np.abs(m.dist - tdist - m.ext_err).max() < 1e-9
                and np.abs(np.sqrt((m.handx - m.targetx) ** 2 + (m.handy - m.targety) ** 2) - m.dist_err).max() < 1e-9
                and np.abs(m.ang_err.abs() - m.abs_ang_err).max() < 1e-12)
pk = m[["targ1x", "targ2x", "targ3x", "targ4x"]].to_numpy()[np.arange(len(m)), m.probe.to_numpy() - 1]
pky = m[["targ1y", "targ2y", "targ3y", "targ4y"]].to_numpy()[np.arange(len(m)), m.probe.to_numpy() - 1]
checks.internal("MWM: the recalled target is the probed movement of the four encoded ones, all four distinct",
                np.abs(pk - m.targetx).max() == 0 and np.abs(pky - m.targety).max() == 0
                and (m[["targ1x", "targ2x", "targ3x", "targ4x"]].round(4).astype(str)
                     + m[["targ1y", "targ2y", "targ3y", "targ4y"]].round(4).astype(str).values).nunique(axis=1).eq(4).all())
checks.internal("MWM: every kept trial has |angular error| < 90 deg (the authors' exclusion rule)", (m.ang_err.abs() < 90).all())
mwm_dirs = np.array(sorted(set(round(float(a) % 360, 6) for a in m.target_ang)))
checks.internal("MWM: twelve target directions evenly spaced from 15 to 165 deg (Methods)",
                len(mwm_dirs) == 12 and np.abs(mwm_dirs - np.linspace(15, 165, 12)).max() < 0.01)
checks.internal("MWM: ms = mt + 1 on every row (ms is not kept)", ((m.ms - m.mt) == 1).all())

# ---------------------------------------------------------------- rows
names = [c[1] for c in stdlib.reach90_columns()]
SEX = {"Female": "F", "Male": "M"}
MWM_TID = {a: i for i, a in enumerate(mwm_dirs)}
rows = []


def base(i, exp):
    d = demo.loc[i]
    row = dict.fromkeys(names, "")
    row.update(dataset=ID, study=ID, experiment=f"{ID}_{exp}", subject=f"{ID}_{exp}_{i}", person=f"{ID}_{i}",
               id_type="lab", person_link="verified", group="", prior_exposure="unknown",
               subject_excluded_by_authors="False", age=int(d.age), sex=SEX[d.sex], handedness=d.dom_hand,
               device="kinarm", setting="lab", population="healthy", session=0)
    return row


def enc(x, y):
    return "%.2f" % (math.degrees(math.atan2(y, x)) % 360)


# MWM (done first by every participant)
n_mwm_missing = 0
for i in ids:
    g = m[m.id == i]
    j = 0
    for bl in dict.fromkeys(g.block):
        gb = g[g.block == bl].set_index("block_trial")
        off = int(gb.off.iloc[0])
        for bt in range(1, int(gb.index.max()) + 1):
            row = base(i, "MWM")
            row.update(trial=j, trial_orig=off + bt, block=f"block{bl}", phase="other", break_before=tf(bt == 1),
                       target_distance_norm=1, target_visible_during_move="False", target_event="none", target_jump=0,
                       probe_kind="none", instr_goal="hand_to_target", instr_strategy="unspecified",
                       instr_disclosure="none", instr_report="none", fb_cued="True", hand_measure="endpoint",
                       fb_visible="none", fb_mode="none", fb_form="none", online_cursor="none",
                       fb_endpoint_transform_only="False")
            if bt in gb.index:
                r = gb.loc[bt]
                ta = float(r.target_ang) % 360
                seq = "/".join(enc(r[f"targ{k}x"], r[f"targ{k}y"]) for k in (1, 2, 3, 4))
                row.update(trial_kind="reach", target_angle=ta, target_id=MWM_TID[round(ta, 6)],
                           cue_elements=f"recall_prompt:{int(r.probe)}|{'switch_hands' if r.switch == 1 else 'wait'}"
                                        f"|encoded_deg:{seq}",
                           hand_used="left" if r.switch == 1 else "right", hand_angle=wrap(float(r.ang_err)),
                           hand_extent=float(r.dist) / math.hypot(r.targetx, r.targety),
                           rt_ms=int(r.rt), mt_ms=int(r.mt), excluded_by_authors="False")
            else:
                row.update(trial_kind="missing", excluded_by_authors="True")
                n_mwm_missing += 1
            rows.append(row)
            j += 1

# VMR
n_vmr_missing = Counter()
CATCH_START = {t for t in PHASE_OF if PHASE_OF[t].startswith("catch") and PHASE_OF.get(t - 1, "") != PHASE_OF[t]}
WASH_START = min(t for t in PHASE_OF if PHASE_OF[t] == "washout")
VMR_TID = {0.0: 0, 60.0: 1, 120.0: 2, 180.0: 3}
for i in ids:
    g = v[v.id == i].set_index("trial")
    r_dir = int(rotdir[i])
    for t in range(1, int(g.index.max()) + 1):
        ph = PHASE_OF[t]
        fb = CURSOR[t]
        rot = 45.0 * r_dir if ROT_ON[t] else 0.0
        no_strategy = ph.startswith("catch") or ph == "washout"
        row = base(i, "VMR")
        row.update(trial=t - 1, trial_orig=t, block=ph,
                   phase="baseline" if ph == "baseline" else "learning" if ph.startswith("learn")
                   else "probe" if ph.startswith("catch") else "aftereffect",
                   break_before=tf(t == 1 or t in CATCH_START or t == WASH_START),
                   target_size_deg=TARGET_RADIUS_DEG, target_distance_norm=1, target_visible_during_move="True",
                   target_event="none", target_jump=0, probe_kind="none",
                   instr_goal="hand_to_target" if no_strategy else "cursor_to_target",
                   instr_strategy="remove" if no_strategy else "unspecified", instr_disclosure="none",
                   instr_report="none", fb_cued="True", hand_used="right", hand_measure="fixed_distance",
                   fb_visible="online" if fb else "none",
                   fb_mode=("rotated" if ROT_ON[t] else "veridical") if fb else "none",
                   fb_form="cursor" if fb else "none",
                   online_cursor=("perturbed" if ROT_ON[t] else "veridical") if fb else "none",
                   fb_endpoint_transform_only="False", rotation=rot if fb else "", fb_delay_ms=0 if fb else "",
                   outcome_channels="cursor" if fb else "")
        if t in g.index:
            r = g.loc[t]
            ta = (-r_dir * float(r.targAng)) % 360
            ha = wrap(-r_dir * float(r.handAng))
            row.update(trial_kind="reach", target_angle=ta, target_id=VMR_TID[round(ta)], hand_angle=ha,
                       hand_angle_fb=ha, rt_ms=int(r.rt), mt_ms=int(r.mt), excluded_by_authors="False",
                       cursor_error=wrap(ha + rot) if fb else "")
        else:
            row.update(trial_kind="missing", excluded_by_authors="True")
            n_vmr_missing[i] += 1
        rows.append(row)

rm = [r for r in rows if r["experiment"].endswith("MWM")]
rv = [r for r in rows if r["experiment"].endswith("VMR")]
checks.internal("every row the authors shared is kept, in order (2847 MWM and 7841 VMR reaches)",
                sum(r["trial_kind"] == "reach" for r in rm) == len(m) == 2847
                and sum(r["trial_kind"] == "reach" for r in rv) == len(v) == 7841
                and [r["trial_orig"] for r in rv if r["trial_kind"] == "reach"] == v.trial.tolist()
                and [r["trial_orig"] for r in rm if r["trial_kind"] == "reach"] == m.trial.tolist())
checks.internal("VMR target directions 0, 60, 120 and 180 deg; hand angles recover the authors' raw hand x, y direction",
                set(VMR_TID) == {round(float(r["target_angle"])) for r in rv if r["trial_kind"] == "reach"}
                and max(abs(wrap(float(r["hand_angle"]) + float(r["target_angle"]) - a)) for r, a in
                        zip([r for r in rv if r["trial_kind"] == "reach"], std_h)) < 1e-9)
checks.internal("cursor_error equals rotDir * the authors' handAngErr on every cursor trial",
                max(abs(float(r["cursor_error"]) - rotdir[int(r["subject"].rsplit("_", 1)[1])] * e)
                    for r, e in zip([r for r in rv if r["trial_kind"] == "reach"], v.handAngErr)
                    if r["cursor_error"] != "") < 1e-9)
checks.internal("placeholder rows only for gaps in the authors' trial numbers before each participant's last trial: "
                f"{n_mwm_missing} MWM, {sum(n_vmr_missing.values())} VMR",
                n_mwm_missing == 96 and sum(n_vmr_missing.values()) == 51)
checks.internal("hand_angle within (-180, 180], target_angle within [0, 360)",
                all(-180 < float(r["hand_angle"]) <= 180 and 0 <= float(r["target_angle"]) < 360
                    for r in rows if r["trial_kind"] == "reach"))

# ---------------------------------------------------------------- published numbers, from the converted rows
persons = OrderedDict((r["person"], r) for r in rows)
ages = [int(r["age"]) for r in persons.values()]
checks.published("participants analysed", len(persons), 31, 0, "Results, Motor Working Memory Task (n = 31)")
checks.published("females", sum(r["sex"] == "F" for r in persons.values()), 17, 0, "Methods, Participants")
checks.published("mean age", np.mean(ages), 18.8, 0.05, "Methods, Participants (18.8 +/- 0.75 years)")
checks.published("SD of age", np.std(ages, ddof=1), 0.75, 0.005, "Methods, Participants")
checks.published("right-handed participants", sum(r["handedness"] == "right" for r in persons.values()), 27, 0,
                 "Methods, Participants (Edinburgh Handedness Inventory)")
checks.published("VMR trials removed for hand angles beyond 3 SD of the block mean",
                 sum(r["trial_kind"] == "missing" for r in rv), 51, 0,
                 "Methods, Analysis (51 trials, 0.6% of all data)")

# VMR metrics as in the authors' fig3-5.Rmd, on the compensation-positive scale (-rotation sign * hand angle)
vm = {}
for i in ids:
    rr = [r for r in rv if r["person"] == f"{ID}_{i}" and r["trial_kind"] == "reach"]
    s = -float(rotdir[i])

    def ha(r):
        return s * float(r["hand_angle"])

    tot = []
    for tid in range(4):
        x = [ha(r) for r in rr if r["block"] == "learn5" and r["target_id"] == tid]
        tot += x[-3:]
    imp = []
    for tid in range(4):
        x = [ha(r) for r in rr if r["block"] == "washout" and r["target_id"] == tid]
        imp += x[:3]
    lc = np.mean([ha(r) for r in rr if r["block"] == "catch4"])
    vm[i] = dict(total=np.mean(tot), lastcatch=lc, explicit=np.mean(tot) - lc, implicit=np.mean(imp))
dv = max(abs(vm[i][k] - demo.loc[i, k]) for i in ids for k in VM)
checks.internal("total, last catch, explicit and implicit learning recomputed from the converted VMR rows equal the "
                "authors' per-participant values in mwm.csv", dv < 1e-9, f"max difference {dv:.1e} deg")


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))


R3 = "Results, Visuomotor Rotation Task"
for k, sd, t, d in (("total", 5.51, 36.63, 6.58), ("explicit", 7.52, 8.51, 1.53), ("implicit", 6.36, 17.37, 3.12)):
    x = np.array([vm[i][k] for i in ids])
    checks.published(f"VMR {k} learning: SD across participants", x.std(ddof=1), sd, 0.005, R3)
    checks.published(f"VMR {k} learning: t against 0", tstat(x), t, 0.005, R3)
    checks.published(f"VMR {k} learning: Cohen's d", x.mean() / x.std(ddof=1), d, 0.005, R3)
checks.published("VMR implicit learning (early aftereffect): mean", np.mean([vm[i]["implicit"] for i in ids]), 19.83, 0.005, R3)

# MWM: per participant, condition and recalled movement, as in fig2.Rmd (paired switch vs same)
mw = pd.DataFrame([dict(id=int(r["person"].rsplit("_", 1)[1]), switch=r["hand_used"] == "left",
                        probe=int(r["cue_elements"].split("|")[0].split(":")[1]), err=float(r["hand_angle"]))
                   for r in rm if r["trial_kind"] == "reach"])


def iqr(x):
    q = np.percentile(x, [25, 75])
    return q[1] - q[0]


R1 = "Results, Motor Working Memory Task"
for stat, f, rep in (("mean absolute angular error", lambda x: np.mean(np.abs(x)), {1: (1.34, .24), 2: (1.26, .23), 3: (2.21, .40), 4: (4.62, .83)}),
                     ("IQR of angular error", iqr, {1: (0.84, .15), 2: (0.56, .10), 3: (1.46, .26), 4: (2.28, .41)})):
    for p, (t, d) in rep.items():
        a = mw[mw.probe == p].groupby(["id", "switch"]).err.apply(f).unstack()
        diff = (a[True] - a[False]).to_numpy()
        checks.published(f"MWM {stat}, movement {p} recalled: paired t, switch vs same hand", tstat(diff), t, 0.005, R1)
        checks.published(f"MWM {stat}, movement {p} recalled: Cohen's d", diff.mean() / diff.std(ddof=1), d, 0.005, R1)


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


expl = np.array([vm[i]["explicit"] for i in ids])
impl = np.array([vm[i]["implicit"] for i in ids])
RC = "Results, Cross-Task Correlations"
sw = mw[mw.switch]
for name, f, rp, rs in (("IQR of angular error", iqr, -0.44, -0.46), ("mean absolute angular error", lambda x: np.mean(np.abs(x)), -0.41, None)):
    x = sw.groupby("id").err.apply(f).reindex(ids).to_numpy()
    checks.published(f"Pearson r, explicit learning vs switch-hand MWM {name}", pearson(x, expl), rp, 0.005, RC + " (Fig. 4)")
    if rs is not None:
        checks.published(f"Spearman r, explicit learning vs switch-hand MWM {name}", spearman(x, expl), rs, 0.005, RC)
same = mw[~mw.switch]
for name, f, rp, rs in (("IQR of angular error", iqr, -0.46, -0.27), ("mean absolute angular error", lambda x: np.mean(np.abs(x)), -0.53, -0.46)):
    a = same.assign(g=np.where(same.probe == 4, "p4", "p13")).groupby(["id", "g"]).err.apply(f).unstack().reindex(ids)
    x = (a.p4 - a.p13).to_numpy()
    checks.published(f"Pearson r, implicit learning vs same-hand {name} (movement 4 - movements 1-3)", pearson(x, impl), rp, 0.005,
                     RC + " (Fig. 5A-B)")
    checks.published(f"Spearman r, implicit learning vs same-hand {name} (movement 4 - movements 1-3)", spearman(x, impl), rs,
                     0.005, RC)
for name, f, rp, rs in (("IQR of angular error", iqr, 0.38, 0.40), ("mean absolute angular error", lambda x: np.mean(np.abs(x)), 0.24, 0.18)):
    a = mw[mw.probe == 4].groupby(["id", "switch"]).err.apply(f).unstack().reindex(ids)
    x = (a[True] - a[False]).to_numpy()
    checks.published(f"Pearson r, implicit learning vs movement-4 {name}, switch - same", pearson(x, impl), rp, 0.005,
                     RC + " (text; Fig. 5C-D plots same - switch, r of opposite sign)")
    checks.published(f"Spearman r, implicit learning vs movement-4 {name}, switch - same", spearman(x, impl), rs, 0.005, RC)

# ---------------------------------------------------------------- release
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
tot_mean = np.mean([vm[i]["total"] for i in ids])
exp_mean = np.mean([vm[i]["explicit"] for i in ids])
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="McDougle (Yale)",
    data_url="https://github.com/hhillman231/MWM_MotorLearning_public",
    license="none stated",
    license_url="",
    changes="converted to the OpenMotor format, participant codes replaced, VMR angles mirrored back to counter-clockwise "
            "positive, and placeholder rows added for trials the authors removed.",
    unit_note="reach of one participant sequence (MWM recall reach or VMR trial)",
    how=[
        "Two experiments with the same 31 participants (person links them): _MWM, the motor working memory task done "
        "first (the robot moved the right hand through four movements to invisible targets 9 cm away, then the "
        "participant recalled one of them with the same hand after waiting, hand_used right, or with the left hand after "
        "switching hands, hand_used left), and _VMR, a 45 deg visuomotor rotation task with four 4-trial catch blocks and "
        "a 40-trial washout, both without cursor and with the instruction to reach straight to the target.",
        "MWM rows: target_angle is the recalled movement's end point, hand_angle the authors' angular error of the recall "
        "end point (counter-clockwise positive), hand_extent the end point distance over the target distance (9 cm); "
        "cue_elements gives the recall prompt (movement 1-4), the instruction (wait or switch_hands) and the directions "
        "of the four encoded movements in order.",
        "VMR rows: hand_angle is the hand direction where the hand crossed the 9 cm target ring; the authors stored it "
        "mirrored so that compensation is positive, and it was mirrored back with their rotDir, so rotation is +45 or -45 "
        "(counter-clockwise or clockwise) and late-learning hand angles have the opposite sign.",
        "Rows with trial_kind missing mark trials absent from the authors' files before each participant's last trial "
        "(51 VMR trials, the authors' outliers; 96 MWM trials, see the differences below); trials after a "
        "participant's last shared trial are not added. Not kept: the per-participant VMR metrics repeated in mwm.csv "
        "(reproduced exactly from these rows), the Edinburgh score, the undocumented vel, headAng, velAng and fb columns.",
    ],
    differences=[
        f"The text gives mean total learning 36.21 deg and explicit learning 11.51 deg; the shared data give "
        f"{tot_mean:.2f} and {exp_mean:.2f} (the same values the authors' mwm.csv stores), while the reported SDs, t "
        "values and Cohen's d reproduce exactly. The t tests are reported with 29 degrees of freedom but reproduce with "
        "all 31 participants (30 df).",
        "The Methods report 98 MWM trials excluded (|angular error| > 90 deg); the shared file lacks 129 of 31 x 96 "
        "trials, 96 of them before each participant's last shared trial (added here as placeholder rows) and 33 at "
        "block ends, so the excluded trials cannot be told apart from trials that were not recorded.",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks)
