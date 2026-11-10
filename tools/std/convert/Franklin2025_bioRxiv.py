"""Franklin, Nietschmann, Papadopoulou & Franklin (2025), bioRxiv: long-term development of a motor memory. vBOT planar
manipulandum, curl force field, 9 weekly-ish sessions (week 1-8, then week 23), 25 cm forward reaches, channel trials to
measure force compensation. One row per trial. Experiment 1: 10 participants; Experiment 2: 18 participants (same session
structure in the deposit's data; the preprint version read describes only Experiment 1).

Run from the site folder: python3 -I tools/std/convert/Franklin2025_bioRxiv.py
Raw files (untrusted data, read only) in raw_extra/Franklin2025_bioRxiv/ from https://doi.org/10.6084/m9.figshare.28550636:
  LonglearnDATA_25.mat   FF(participant, day, condition) trial structs, Experiment 1 (10 participants)
  Longlearn2DATA_25.mat  same, Experiment 2 (18 participants)
The authors' plotting scripts (df_plotLLdata_25.m, df_plotLL2data.m) were read as text to learn the variables, never run. The
5 GB / 9 GB trace files (LonglearnDATAtraces_25.mat, Longlearn2DATAtraces_25.mat: force and velocity traces of every trial) are
not converted (size). The deposit's Read me.pdf could not be text-extracted (embedded fonts).
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Franklin2025_bioRxiv"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
EXPS = [("Experiment1", "LonglearnDATA_25.mat", 10), ("Experiment2", "Longlearn2DATA_25.mat", 18)]
NCOND = [12, 6, 6, 7, 6, 6, 9, 6, 9]  # conditions per day (the authors' DayNumConds / LastCond)
# (label, phase) by day type, from the trial counts of each block and the paper's Methods (week 1: 310 baseline null, 300 CCW,
# 40 decay channel, 200 CCW, 30 CW, 40 channel, 300 CCW; weeks 2-3,5,6,8: 5 retention channel, 300 CCW, 40 decay, 400 CCW; week 4:
# plus 100 CCW with left-hand tests; weeks 7 and 23: 5 channel, 300 CCW, 40 decay, 200 CCW, 30 CW, 40 channel, 300 CCW)
D1 = {1: ("baseline_right", "baseline"), 2: ("baseline_left_hand", "baseline"), 3: ("baseline_right_2", "baseline"),
      4: ("baseline_null", "baseline"), 5: ("ccw_training_1a", "learning"), 6: ("ccw_training_1b", "learning"),
      7: ("decay_channel", "aftereffect"), 8: ("ccw_training_2", "learning"), 9: ("cw_field", "other"),
      10: ("spontaneous_recovery_channel", "aftereffect"), 11: ("ccw_training_3a", "learning"), 12: ("ccw_training_3b", "learning")}
DN = {1: ("retention_channel", "test"), 2: ("ccw_training_1a", "learning"), 3: ("ccw_training_1b", "learning"),
      4: ("decay_channel", "aftereffect"), 5: ("ccw_training_2a", "learning"), 6: ("ccw_training_2b", "learning"),
      7: ("left_hand_transfer", "transfer")}
D79 = {1: ("retention_channel", "test"), 2: ("ccw_training_1a", "learning"), 3: ("ccw_training_1b", "learning"),
       4: ("decay_channel", "aftereffect"), 5: ("ccw_training_2", "learning"), 6: ("cw_field", "other"),
       7: ("spontaneous_recovery_channel", "aftereffect"), 8: ("ccw_training_3a", "learning"), 9: ("ccw_training_3b", "learning")}


def label(day, c):
    if day == 1:
        return D1[c]
    if day in (7, 9):
        return D79[c]
    return DN[c]


TTYPE = {1: "null", 2: "channel", 3: "field", 5: "field"}
f = lambda a, i: (None if a is None else a[i])  # noqa: E731


def arr(x, name, n, scale=1.0, strict=True):
    a = np.atleast_1d(getattr(x, name)).astype(float)
    if a.shape != (n,):
        return None
    return a * scale


rows = []
store = {}
ntrial = {}
for exp, fn, ns in EXPS:
    FF = sio.loadmat(RAW / fn, squeeze_me=True, struct_as_record=False)["FF"]
    checks.internal(f"{exp}: array is {ns} participants x 9 days x 12 condition slots", FF.shape == (ns, 9, 12))
    misaligned = 0
    for s in range(ns):
        trial = 0
        for d in range(9):
            for c in range(1, NCOND[d] + 1):
                x = FF[s, d, c - 1]
                n = int(x.CondTrialNum)
                lab, phase = label(d + 1, c)
                ft = np.atleast_1d(x.FieldType).astype(int)
                cols = {k: arr(x, k, n) for k in ("MPE", "MPECHAN", "LFC", "LMPE", "FC", "FCAng", "FCAngzero", "FCzero", "ABSerr",
                                                  "SIGerr", "ABSFE", "SIGFE", "duration", "RT", "peakvel", "peakvelpercent",
                                                  "success", "TargetAngle", "LeftHand")}
                if cols["FCzero"] is None:
                    misaligned += 1
                assert all(cols[k] is not None for k in cols if k != "FCzero"), (exp, s, d, c)
                for t in range(n):
                    trial += 1
                    left = cols["LeftHand"][t] == 1
                    tt = TTYPE[ft[t]]
                    fcv = cols["LFC"][t] if left else cols["FC"][t]
                    mp = cols["LMPE"][t] if left else cols["MPE"][t]
                    on = ft[t] in (3, 5)
                    g = lambda v: "" if v != v else v  # noqa: E731
                    rows.append(dict(
                        dataset=ID, study=ID, experiment=exp, group="", subject=f"{exp}_{s + 1}", person=f"{exp}_{s + 1}",
                        subject_excluded_by_authors=False, age="", sex="", handedness="right" if exp == "Experiment1" else "",
                        population="healthy", setting="lab", device="vBOT planar robotic manipulandum", session=d + 1,
                        block=f"day{d + 1}_cond{c}_{lab}", trial=trial, phase=phase, time_s="", task="reaching", unit="trial",
                        condition=lab, pert_type="force_field" if on else "none",
                        pert_value=(15 if ft[t] == 3 else -15) if on else 0, pert_unit="N s/m" if on else "", feedback="normal",
                        excluded_by_authors=False, trial_type=tt, trial_in_condition=t + 1, block_in_condition="",
                        mpe_cm=g(mp) if tt != "channel" else "",
                        force_compensation_pct=g(100 * fcv) if tt == "channel" else "",
                        force_compensation_sym_pct="", movement_duration_s=g(cols["duration"][t]), lateral_deviation_mm="",
                        target_error_mm="", reaction_time_s=g(cols["RT"][t]), peak_speed_m_s=g(cols["peakvel"][t]),
                        peak_speed_pct=g(cols["peakvelpercent"][t]), speed_goal_met=bool(cols["success"][t]),
                        target_direction_deg=cols["TargetAngle"][t], hand="left" if left else "right", force_field_code=int(ft[t]),
                        force_compensation_zero_pct=g(100 * cols["FCzero"][t]) if cols["FCzero"] is not None else "",
                        force_compensation_angle_pct=g(100 * cols["FCAng"][t]),
                        force_compensation_angle_zero_pct=g(100 * cols["FCAngzero"][t]),
                        mpe_channel_cm=g(cols["MPECHAN"][t]), path_error_abs=g(cols["ABSerr"][t]),
                        path_error_signed=g(cols["SIGerr"][t]), force_error_abs_pct=g(cols["ABSFE"][t]),
                        force_error_signed_pct=g(cols["SIGFE"][t])))
        ntrial[(exp, s)] = trial
    store[exp] = (ns, misaligned)

# ---------------------------------------------------------------- checks
import collections  # noqa: E402

per_day = collections.defaultdict(set)
for (exp, s), nt in ntrial.items():
    pass
day_tr = collections.defaultdict(collections.Counter)
for r in rows:
    if r["subject"] == "Experiment1_1":
        day_tr[1][r["session"]] += 1
d1 = day_tr[1]
SRC = "Methods, Experimental paradigm (session structure)"
checks.published("week 1 trials", d1[1], 1220, 0, SRC)
checks.published("week 2 trials", d1[2], 745, 0, SRC)
checks.published("week 4 trials", d1[4], 845, 0, SRC)
checks.published("week 7 trials", d1[7], 915, 0, SRC)
checks.published("week 23 (9th session) trials", d1[9], 915, 0, SRC)
checks.published("week 1 baseline null trials (first four blocks)", sum(1 for r in rows if r["subject"] == "Experiment1_1"
                 and r["session"] == 1 and r["phase"] == "baseline"), 310, 0, SRC)
checks.published("participants, experiment 1", len({r["subject"] for r in rows if r["experiment"] == "Experiment1"}), 10, 0,
                 "Methods, Participants")
checks.published("sessions", len({r["session"] for r in rows}), 9, 0, "Methods (eight weekly sessions and a ninth 15 weeks later)")
cw = [r for r in rows if r["subject"] == "Experiment1_1" and r["session"] == 1 and r["block"].startswith("day1_cond9_")]
checks.published("week 1 clockwise-field block (30 CW trials, 3 of them channel trials)", len(cw), 30, 0, SRC)
ang = sorted({float(r["target_direction_deg"]) for r in rows if r["trial_type"] == "channel"} - {90.0})
checks.published("generalization channel directions (-60, -40, -20, 20, 40, 60 deg from training): count", len(ang), 6, 0,
                 "Methods, Generalisation")
checks.published("largest generalization angle from the 90 deg training direction", max(abs(a - 90) for a in ang), 60, 0, "Methods")
e1 = [r for r in rows if r["experiment"] == "Experiment1" and r["session"] == 1 and r["trial_type"] == "channel"
      and r["hand"] == "right" and float(r["target_direction_deg"]) == 90 and r["force_compensation_pct"] != ""]
late = [float(r["force_compensation_pct"]) for r in e1 if r["block"].startswith("day1_cond12")][-10:]
sub_late = {}
for r in e1:
    if r["block"].startswith("day1_cond12"):
        sub_late.setdefault(r["subject"], []).append(float(r["force_compensation_pct"]))
m_late = np.mean([np.mean(v[-5:]) for v in sub_late.values()])
checks.published("force compensation at the end of week 1 (mean of the last 5 channel trials; 'about 80%', read from Fig 1B)",
                 m_late, 80, 12, "Results, Fig 1B")
pred = []
for s in range(1, 11):
    v = [float(r["force_compensation_pct"]) for r in rows if r["subject"] == f"Experiment1_{s}" and r["session"] == 8
         and r["trial_type"] == "channel" and r["hand"] == "right" and r["force_compensation_pct"] != ""
         and float(r["target_direction_deg"]) == 90 and r["condition"].startswith("ccw_training")]
    pred.append(np.mean(v[-10:]))
checks.published("predictive force compensation in week 8 ('over 95% for weeks 4-8', read from Fig 3B)", np.mean(pred), 97, 8,
                 "Results, Fig 3B")
checks.internal("both experiments have all 9 sessions with the same trials per session for every participant",
                len({v for (e, s), v in ntrial.items()}) == 1 and list(ntrial.values())[0] == 7620,
                f"{sorted(set(ntrial.values()))}")
checks.internal("experiment 2 has 18 participants", len({r["subject"] for r in rows if r["experiment"] == "Experiment2"}) == 18)
checks.internal("channel trials have a force compensation, other trials do not (right-hand 90 deg or left hand)",
                all((r["force_compensation_pct"] != "") <= (r["trial_type"] == "channel") for r in rows))
mp = [float(r["mpe_cm"]) for r in rows if r["mpe_cm"] != ""]
checks.internal("maximum perpendicular error within +-30 cm (a few failed trials reach 20+ cm)", max(abs(v) for v in mp) < 30, f"range {min(mp):.2f} to {max(mp):.2f}")
fcv = [float(r["force_compensation_pct"]) for r in rows if r["force_compensation_pct"] != ""]
checks.internal("force compensation values are fractions x 100 (median of channel trials 0-200)", -50 < np.median(fcv) < 150,
                f"median {np.median(fcv):.1f}, range {min(fcv):.0f} to {max(fcv):.0f}")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, lab="Franklin (TU Munich)",
    data_url="https://doi.org/10.6084/m9.figshare.28550636", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor force-field reaching format, participant codes replaced, force compensation "
            "multiplied by 100 to percent, values of the left-hand tests (the authors' LFC, LMPE) put in the same columns "
            "with hand = left; force and velocity traces not included.",
    unit_note="trial",
    how=[
        "One row is one trial: Experiment 1 has 10 participants and Experiment 2 has 18, each with 9 sessions (7620 trials "
        "per participant: week 1 has 1220, weeks 2, 3, 5, 6 and 8 have 745, week 4 has 845, weeks 7 and 23 have 915). "
        "block names the authors' condition slot (day, condition number) and a label derived from its trial count and the paper's session structure.",
        "trial_type is read from the authors' FieldType code: 1 null, 2 channel, 3 counter-clockwise curl field (15 N s/m, the "
        "main training field), 5 clockwise curl field (30 trials in sessions 1, 7 and 9, used for spontaneous recovery). The "
        "codes are not defined in the deposit; they follow from the block lengths and the paper (30 CW trials, 10% channel trials, 40 channel trials).",
        "force_compensation_pct is the authors' FC (a fraction in the file; 1 = full compensation) times 100, on channel "
        "trials; mpe_cm is the maximum perpendicular error on non-channel trials. The sign of mpe_cm is the authors'.",
        "The 'Experiment 2' deposit data have the same session structure as Experiment 1 (18 participants); the preprint "
        "version read for this conversion describes only the 10-participant experiment, so Experiment 2 is described here from the deposit alone. "
        "Whether any Experiment 2 participant also took part in Experiment 1 is not stated, so each experiment's participants are coded as different people.",
        "The authors' FCzero is stored with the wrong length for a few conditions (one to four trials short); it is left empty "
        "for those participants' conditions. Columns whose meaning the deposit does not define (path_error_*, force_error_*, "
        "peak_speed_pct, FCAng variants) are kept as stored and say so in columns.csv.",
    ],
    differences=["The paper reports the compensation levels only in figures and as 'about 80%' (end of week 1) and 'over 95%' "
                 "(weeks 4-8), so those checks use wide tolerances; its retention and decay statistics need the force traces, which are not converted."],
)
stdlib.release(ID, "force_field_reaching", rows, meta, checks)
