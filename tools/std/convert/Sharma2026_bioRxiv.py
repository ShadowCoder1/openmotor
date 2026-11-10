"""Sharma, Hillman & McDougle (2026), bioRxiv: temporal indexing of motor memory. A visuomotor dual-adaptation task with a
'set-go' preparation delay (500 or 1500 ms) deterministically paired with the sign of a 15 deg clamped cursor error, a 150-trial
no-feedback test phase with three extra, unseen delays, KINARM End-Point robot, right hand, one target per participant.
One row per trial, 90-column reaching format, two experiments: _InternalMonitoring (the go cue is a colour change, so the delay
has to be timed internally) and _LoomingCue (a growing target signals the delay).

Run from the site folder: python3 -I tools/std/convert/Sharma2026_bioRxiv.py
Raw files (untrusted data, read only) in raw_extra/Sharma2026_bioRxiv/, from
https://github.com/apoorvas980/Temporal-Indexing-of-Motor-Memory (data/ folder, MIT license):
  internallymonitored_model.csv   trial-level data, internal-monitoring experiment (22 participants x 570 trials)
  externallycued_model.csv        trial-level data, looming-cue experiment (22 participants x 570 trials; no rtGoSig column)
  behavioral_implicit_lme.csv,    the training trials of the same two experiments re-expressed for the authors' mixed models
  behavioral_explicit_lme.csv     (baseline-subtracted hand angle, rotation of the current and two previous trials); they
                                  restate the model files, cannot be matched to their participants, and are not converted.
The authors' data/README.md, context_analysis_lme.m and timing_models.m were read as text, never run.

Signs. The authors' model files are already aligned across the counterbalanced subgroups (target left or right, delay-to-rotation
assignment): in every participant the 500 ms delay goes with rotDir = +1 and the 1500 ms delay with rotDir = -1 in training, and
handAng has the sign of rotDir (after a +1 trial the hand moves to negative angles). The physical direction (clockwise or
counter-clockwise) and the target side were counterbalanced and cannot be recovered, so target_angle is left empty and hand_angle,
clamp and cursor_error are in the authors' aligned frame, not in the counter-clockwise-positive screen frame.
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

ID = "Sharma2026_bioRxiv"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
LICENSE_TEXT = 'MIT License\n\nCopyright (c) 2026 apoorvas980\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n'
TARGET_RADIUS_DEG = math.degrees(math.atan(0.5 / 9.0))  # 0.5 cm radius target at 9 cm (Methods, Task Design)
EXPS = (("InternalMonitoring", "internallymonitored_model.csv"), ("LoomingCue", "externallycued_model.csv"))
PHASE = {1: ("baseline", "baseline"), 2: ("training", "learning"), 3: ("test", "transfer")}  # authors' label, OpenMotor phase


def tf(x):
    return "True" if x else "False"


# ---------------------------------------------------------------- read and check the structure
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
T = {}
for exp, fn in EXPS:
    checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])
    T[exp] = pd.read_csv(RAW / fn)
    t = T[exp]
    ids = list(dict.fromkeys(t.id))
    ok_order = all((g.subTrial.to_numpy() == np.arange(1, 571)).all() for _, g in t.groupby("id", sort=False)) \
        and t.id.ne(t.id.shift()).sum() == t.id.nunique()
    checks.internal(f"{exp}: 22 participants x 570 trials, subTrial 1-570 in order, each participant's rows contiguous",
                    len(t) == 22 * 570 and len(ids) == 22 and ok_order)
    sched = t.groupby(["phase", "rotDir", "goDelay"]).size().to_dict()
    checks.internal(f"{exp}: schedule of the Methods: 20 baseline trials (10 per delay), 400 training trials (200 per delay: rotDir +1 "
                    "with 500 ms, -1 with 1500 ms for everyone), 150 test trials (30 per delay of 500, 750, 1000, 1250, 1500 ms)",
                    sched == {(1, 0, 500): 220, (1, 0, 1500): 220, (2, 1, 500): 4400, (2, -1, 1500): 4400,
                              **{(3, 0, d): 660 for d in (500, 750, 1000, 1250, 1500)}}
                    and t.groupby("phase").subTrial.agg(["min", "max"]).to_dict("index")
                    == {1: {"min": 1, "max": 20}, 2: {"min": 21, "max": 420}, 3: {"min": 421, "max": 570}})
    checks.internal(f"{exp}: the sign alignment is applied: every participant has rotDir +1 at 500 ms and -1 at 1500 ms in training",
                    (t[(t.phase == 2) & (t.goDelay == 500)].groupby("id").rotDir.agg(lambda s: set(s)) == {1}).all()
                    and (t[(t.phase == 2) & (t.goDelay == 1500)].groupby("id").rotDir.agg(lambda s: set(s)) == {-1}).all())
    ha = t.handAng.dropna()
    checks.internal(f"{exp}: hand angles of kept trials lie within the authors' +-30 deg rule", ha.abs().max() <= 30.0,
                    f"max |angle| {ha.abs().max():.2f} deg")
rt_ok = "rtGoSig" in T["InternalMonitoring"] and "rtGoSig" not in T["LoomingCue"]
checks.internal("rtGoSig exists in the internal-monitoring file only; it is empty exactly where handAng is empty",
                rt_ok and (T["InternalMonitoring"].rtGoSig.isna() == T["InternalMonitoring"].handAng.isna()).all())
rt = T["InternalMonitoring"].rtGoSig.dropna()
checks.internal("reaction times relative to the go signal are in milliseconds (kept trials: median 300-450, no value below -200, the authors' early-start limit)",
                300 < rt.median() < 450 and rt.min() >= -200 and rt.max() <= 2000,
                f"median {rt.median():.0f}, range {rt.min():.0f} to {rt.max():.0f} ms")

# ---------------------------------------------------------------- rows
names = [c[1] for c in stdlib.reach90_columns()]
rows = []
for exp, fn in EXPS:
    t = T[exp]
    for k, pid in enumerate(dict.fromkeys(t.id)):
        g = t[t.id == pid]
        for j, r in enumerate(g.itertuples(index=False)):
            st = int(r.subTrial)
            ph = int(r.phase)
            label, std_phase = PHASE[ph]
            fb_on = (ph == 1 and st <= 10) or ph == 2  # veridical cursor on baseline trials 1-10, clamped cursor in training
            clamp = 15.0 * int(r.rotDir) if ph == 2 else None
            ha = float(r.handAng)
            kept = not math.isnan(ha)
            row = dict.fromkeys(names, "")
            row.update(
                dataset=ID, study=ID, experiment=f"{ID}_{exp}", subject=f"{ID}_{exp}_{pid}", person=f"{ID}_{exp}_{pid}",
                id_type="lab", person_link="assumed_unique", group="", prior_exposure="unknown",
                subject_excluded_by_authors="False", device="kinarm", setting="lab", population="healthy", handedness="right",
                trial=j, trial_orig=st, session=0, block=label, phase=std_phase, trial_kind="reach",
                break_before=tf(st in (1, 21, 421)), target_size_deg=TARGET_RADIUS_DEG, target_distance_norm=1,
                target_visible_during_move="True", target_event="none", target_jump=0, probe_kind="none",
                instr_goal="hand_to_target", instr_strategy="forbidden" if ph == 2 else "unspecified",
                instr_disclosure="none" if ph == 1 else "exists", instr_report="none",
                cue_elements="looming_target" if exp == "LoomingCue" else "", target_to_go_ms=int(r.goDelay),
                fb_cued="True", hand_used="right", hand_measure="peak_velocity", excluded_by_authors=tf(not kept),
                fb_visible="online" if fb_on else "none",
                fb_mode=("clamped" if ph == 2 else "veridical") if fb_on else "none",
                fb_form="cursor" if fb_on else "none",
                online_cursor=("perturbed" if ph == 2 else "veridical") if fb_on else "none",
                fb_endpoint_transform_only="False", fb_delay_ms=0 if fb_on else "", outcome_channels="cursor" if fb_on else "",
            )
            if ph == 2:
                row["clamp"] = clamp
                row["cursor_error"] = clamp
            if kept:
                row["hand_angle"] = ha
                if ph == 1 and fb_on:
                    row["cursor_error"] = ha
                if exp == "InternalMonitoring":
                    row["rt_ms"] = float(r.rtGoSig)
            rows.append(row)

rx = {e: [r for r in rows if r["experiment"].endswith(e)] for e, _ in EXPS}
checks.internal("every row the authors shared is kept, in order (2 x 12540 rows); rows without a hand angle are kept and flagged as "
                "excluded_by_authors",
                [len(rx[e]) for e, _ in EXPS] == [12540, 12540]
                and all([r["trial_orig"] for r in rx[e]] == T[e].subTrial.tolist() for e, _ in EXPS)
                and all((np.array([r["excluded_by_authors"] == "True" for r in rx[e]]) == T[e].handAng.isna().to_numpy()).all() for e, _ in EXPS))

# ---------------------------------------------------------------- published numbers, from the converted rows
df = pd.DataFrame([dict(exp=r["experiment"].rsplit("_", 1)[1], person=r["person"], ph=r["block"], delay=r["target_to_go_ms"],
                        ha=float(r["hand_angle"]) if r["hand_angle"] != "" else np.nan,
                        rt=float(r["rt_ms"]) if r["rt_ms"] != "" else np.nan) for r in rows])
share = {e: df[df.exp == e].ha.isna().mean() for e, _ in EXPS}
checks.published("internal-monitoring experiment: share of trials removed before the analyses", 100 * share["InternalMonitoring"], 12.01,
                 0.005, "Methods, Statistical Analysis (12.01% of trials)")
tstat = lambda x: x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))  # noqa: E731
res = {}
for e, rep_diff, rep_g in (
        ("InternalMonitoring", (1.578, 2.283, 3.242),
         {500: (-1.025, 1.291, -3.722), 750: (-0.340, 0.851, -1.872), 1000: (0.088, 0.719, 0.575), 1250: (0.771, 1.064, 3.397),
          1500: (0.748, 0.857, 4.094)}),
        ("LoomingCue", (3.699, 3.543, 4.896),
         {500: (-1.854, 1.442, -6.031), 750: (-0.541, 0.632, -4.010), 1000: (0.637, 0.503, 5.939), 1250: (0.847, 0.849, 4.678),
          1500: (1.026, 1.144, 4.206)})):
    d = df[df.exp == e]
    base = d[d.ph == "baseline"].groupby(["person", "delay"]).ha.mean().unstack()
    test = d[d.ph == "test"].groupby(["person", "delay"]).ha.mean().unstack()
    mu_all = d[d.ph == "test"].groupby("person").ha.mean()
    diff = (test[1500] - base[1500]) - (test[500] - base[500])
    lab = "internal-monitoring" if e == "InternalMonitoring" else "looming-cue"
    S = "Results, " + ("Temporal Indexing under high uncertainty" if e == "InternalMonitoring" else "Temporal Indexing under reduced uncertainty")
    checks.published(f"{lab}: participants (t tests with 21 degrees of freedom)", d.person.nunique(), 22, 0, S)
    for nm, val, (rm, rs, rt_) in (("test-phase context effect (1500 - 500 ms, baseline-subtracted): ", diff, rep_diff),):
        checks.published(f"{lab}: {nm}mean", val.mean(), rm, 0.0006, S)
        checks.published(f"{lab}: {nm}SD", val.std(ddof=1), rs, 0.0006, S)
        checks.published(f"{lab}: {nm}t(21)", tstat(val), rt_, 0.0006, S)
    for dl, (rm, rs, rt_) in rep_g.items():
        val = test[dl] - mu_all
        checks.published(f"{lab}: aligned generalization at {dl} ms (test-phase mean minus the participant's test mean): mean", val.mean(), rm, 0.0006, S)
        checks.published(f"{lab}: aligned generalization at {dl} ms: SD", val.std(ddof=1), rs, 0.0006, S)
        checks.published(f"{lab}: aligned generalization at {dl} ms: t(21)", tstat(val), rt_, 0.0006, S)
    res[e] = base
rt_tr = df[(df.exp == "InternalMonitoring") & (df.ph == "training")].groupby(["person", "delay"]).rt.median().unstack()
rt_m500, rt_m1500, rt_t = rt_tr[500].mean(), rt_tr[1500].mean(), tstat(rt_tr[1500] - rt_tr[500])

# ---------------------------------------------------------------- release
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={**{k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, "venue": "bioRxiv (preprint)"},
    lab="McDougle (Yale)",
    data_url="https://github.com/apoorvas980/Temporal-Indexing-of-Motor-Memory",
    license="MIT",
    license_url="https://github.com/apoorvas980/Temporal-Indexing-of-Motor-Memory/blob/main/LICENSE",
    license_text=LICENSE_TEXT,
    changes="converted to the OpenMotor format, participant codes replaced, and the authors' rotDir re-expressed as the sign of the "
            "clamped cursor error; nothing else was recoded (hand angles keep the authors' sign alignment).",
    unit_note="trial of one participant sequence",
    how=[
        "Two experiments, 22 participants each, 570 trials per participant: 20 baseline trials (10 with a veridical cursor, 10 "
        "without), 400 training trials with a cursor clamped 15 deg to one side of the target and 150 test trials without feedback. "
        "A 'set' cue (the target appears) is followed after 500 or 1500 ms (test: also 750, 1000, 1250 ms; target_to_go_ms) by the "
        "'go' cue; in training the 500 ms delay always goes with one sign of the clamp and the 1500 ms delay with the other.",
        "_InternalMonitoring: the go cue is the target turning green, so the delay has to be timed internally (rt_ms is the "
        "authors' rtGoSig, the reaction time relative to the go signal). _LoomingCue (the authors' externally-cued file): the target "
        "grows during the delay and the participant starts when it reaches full size (cue_elements looming_target); the file has no "
        "reaction times.",
        "hand_angle is the authors' handAng, the hand direction relative to the target at peak velocity, in degrees. The authors "
        "shared the data already aligned across the counterbalanced subgroups (target on the left or right, which delay goes with "
        "which clamp direction), so the 500 ms delay has clamp +15 and the 1500 ms delay -15 for everyone, hand_angle has the same "
        "sign convention as clamp, and the physical direction is not recoverable: target_angle is empty and signs are not "
        "counter-clockwise positive.",
        "Rows the authors had removed before sharing (hand angle empty: failed to start, started over 200 ms early, over 500 ms "
        "movement time or beyond +-30 deg) are kept with excluded_by_authors True and no hand_angle. Not converted: the two "
        "behavioral_*_lme.csv files, which restate the training trials for the authors' mixed models.",
    ],
    differences=[
        f"The Methods give 4.36% of trials removed in the looming-cue experiment; the shared file has {100 * share['LoomingCue']:.2f}% without a "
        "hand angle (the 12.01% of the internal-monitoring experiment reproduces).",
        f"The Results give mean participant median reaction times of 408.64 and 347.57 ms for the 500 and 1500 ms delays (t(21) = -8.15) "
        f"in the internal-monitoring experiment; the median rtGoSig of the training trials gives {rt_m500:.2f} and {rt_m1500:.2f} ms "
        f"(t(21) = {rt_t:.2f}), so the authors' RT measure is not recovered. The looming-cue reaction times (275.00 and 286.02 ms) "
        "cannot be checked because that file has no reaction times.",
        "The Methods call the training feedback 'a fixed cursor rotation of +15 or -15 deg', the Figure 1 legend a fixed ('clamped') "
        "cursor trajectory 15 deg off the reach direction that participants ignored; it is coded as an error clamp. The model-fitting "
        "text also describes the error as the rotation minus the hand angle.",
        "The third experiment (discrete numeric cues, 9 participants after exclusions) and the participants excluded in the paper "
        "(2 and 4 in the first two experiments) are not in the deposit.",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks)
