"""Morris, Cruz, Kainth & Ferris (2026) J Neurophysiol: electrocortical alpha response to intermittent visual occlusions while
standing and walking on a treadmill at 0.4, 0.8, 1.2 and 1.6 m/s. The EEG recordings (EDF, brain and neck-muscle signals) are
not converted: this release holds the gait the authors recorded with insole force sensors.

Raw file (OpenNeuro ds007222, GitHub mirror of the BIDS tree): participants.tsv and sub-XX/eeg/*_events.tsv, one table per
participant and speed with heel-strike (lhs, rhs), toe-off (lto, rto) and occlusion (close_oc, open_oc) times in seconds from the
start of the file. One row of the output is one stride of the left foot (left heel strike to the next one).
Run from the site folder: python3 -I tools/std/convert/Morris2026_JNeurophysiol.py
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

DS = "Morris2026_JNeurophysiol"
MODULE = "walking"
SPEEDS = {"00": 0.0, "04": 0.4, "08": 0.8, "12": 1.2, "16": 1.6}
DEVICE = ("Woodway PPS 70 Bari-Mill treadmill; Loadsol insole force sensors for heel-strike and toe-off times; liquid crystal "
          "goggles (Senaptec Strobe) alternating opaque 1.5 s and transparent 8-10 s; safety harness; 120-channel EEG")
DRIFT = {"sub-05", "sub-06", "sub-07"}  # README, Missing data: loadsol data missing, incomplete or drifting


def strides(ev, oc_pairs, win):
    h = {k: ev[ev.trial_type == k].onset.to_numpy() for k in ("lhs", "rhs", "lto", "rto")}
    lhs = h["lhs"]
    rows = []
    for i in range(len(lhs) - 1):
        a, b = lhs[i], lhs[i + 1]
        st = h["lto"][(h["lto"] > a) & (h["lto"] < b)]
        r = dict(start=a, stride_left=b - a, stance_left=st[0] - a if len(st) == 1 else np.nan)
        r["swing_left"] = r["stride_left"] - r["stance_left"]
        rh = h["rhs"][(h["rhs"] >= a) & (h["rhs"] < b)]
        rn = h["rhs"][(h["rhs"] >= b) & (h["rhs"] < (lhs[i + 2] if i + 2 < len(lhs) else b))]
        r.update(stride_right=np.nan, stance_right=np.nan, swing_right=np.nan)
        if len(rh) == 1 and len(rn) == 1:
            r["stride_right"] = rn[0] - rh[0]
            sr = h["rto"][(h["rto"] > rh[0]) & (h["rto"] < rn[0])]
            if len(sr) == 1:
                r["stance_right"] = sr[0] - rh[0]
                r["swing_right"] = r["stride_right"] - r["stance_right"]
        r["occluded"] = any(c <= a <= o for c, o in oc_pairs)
        r["in_trial"] = win[0] <= a <= win[1]
        rows.append(r)
    return rows


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(DS) / "ds007222_main.zip")
    names = {i.filename.split("/", 1)[1]: i.filename for i in z.infolist() if not i.is_dir()}
    part = pd.read_csv(io.BytesIO(z.read(names["participants.tsv"])), sep="\t")
    part.columns = [c.strip() for c in part.columns]
    assert len(part) == 15
    ck = stdlib.Checks()
    out, occl_dur, gaps = [], [], []
    have = {}  # subject -> set of walking speeds with gait events
    strides_by = {}
    for p in part.itertuples():
        sub = p.participant_id
        code = f"{DS}_S{int(sub[4:]):03d}"
        for sp, v in SPEEDS.items():
            ev = pd.read_csv(io.BytesIO(z.read(names[f"{sub}/eeg/{sub}_task-treadmill{sp}flow00_events.tsv"])), sep="\t")
            c = ev[ev.trial_type == "close_oc"].onset.to_numpy()
            o = ev[ev.trial_type == "open_oc"].onset.to_numpy()
            pairs = []
            for ci in c:
                oo = o[o > ci]
                if len(oo):
                    pairs.append((ci, oo[0]))
            occl_dur += [b - a for a, b in pairs if b - a < 3]
            gaps += [pairs[k + 1][0] - pairs[k][1] for k in range(len(pairs) - 1)]
            s = ev[ev.trial_type == "start"].onset.to_numpy()
            e = ev[ev.trial_type == "stop"].onset.to_numpy()
            cand = [(a, e[e > a][0]) for a in s if (e > a).any()]
            win = max(cand, key=lambda t: t[1] - t[0])
            rows = strides(ev, pairs, win)
            if rows:
                have.setdefault(sub, set()).add(sp)
            for k, r in enumerate(rows):
                out.append(dict(
                    dataset=DS, study=DS, experiment="visual occlusions at treadmill speeds", group="", subject=code,
                    subject_excluded_by_authors=False, age=float(p.age), sex=p.sex, handedness="", population="healthy",
                    setting="lab", device=DEVICE, session=1, block=f"{v:g} m/s", phase="test" if r["in_trial"] else "other",
                    time_s=r["start"], task="walking", unit="stride", condition=f"treadmill {v:g} m/s",
                    pert_type="visual_occlusion" if r["occluded"] else "none", pert_value=1.5 if r["occluded"] else 0.0,
                    pert_unit="s" if r["occluded"] else "", feedback="masked" if r["occluded"] else "normal",
                    excluded_by_authors=False, fast_leg="unknown", belt_speed_fast_mps=v, belt_speed_slow_mps=v,
                    walking_trial=int(sp) // 4 + 1 if sp != "00" else 1, stride_start_s=r["start"],
                    stride_time_left_s=r["stride_left"], stride_time_right_s=r["stride_right"],
                    stance_time_left_s=r["stance_left"], stance_time_right_s=r["stance_right"],
                    swing_time_left_s=r["swing_left"], swing_time_right_s=r["swing_right"],
                    stance_pct_left=100 * r["stance_left"] / r["stride_left"], swing_pct_left=100 * r["swing_left"] / r["stride_left"],
                    stance_pct_right=100 * r["stance_right"] / r["stride_right"], swing_pct_right=100 * r["swing_right"] / r["stride_right"],
                ))
    df = pd.DataFrame(out)
    df["trial"] = df.groupby("subject").cumcount() + 1
    df["walking_trial"] = df["block"].map({"0 m/s": 1, "0.4 m/s": 2, "0.8 m/s": 3, "1.2 m/s": 4, "1.6 m/s": 5})

    # ---- internal checks
    walk = df[(df.belt_speed_fast_mps > 0) & (df.phase == "test")]
    mean_stride = walk.groupby("belt_speed_fast_mps").stride_time_left_s.median()
    ck.internal("median left stride time falls with each faster speed", mean_stride.is_monotonic_decreasing, str(mean_stride.round(3).to_dict()))
    ck.internal("left stride times between 0.4 and 4 s", df.stride_time_left_s.between(0.4, 4).mean() > 0.97,
                f"{df.stride_time_left_s.between(0.4, 4).mean():.4f} of strides")
    ck.internal("transparent intervals between occlusions are 8-10 s (within 0.1 s) in at least 95 percent of cases",
                np.mean([(8 - 0.1 <= g <= 10 + 0.1) for g in gaps]) > 0.95, f"{np.mean([(7.9 <= g <= 10.1) for g in gaps]):.3f} of {len(gaps)}")
    ck.internal("participants without any gait events: sub-05 only has none at any speed",
                sorted(set(part.participant_id) - set(have)) == ["sub-05"], str(sorted(set(part.participant_id) - set(have))))

    # ---- published numbers
    ck.published("participants", len(part), 15, 0, "Abstract")
    ck.published("men", int((part.sex == "M").sum()), 5, 0, "Abstract")
    ck.published("women", int((part.sex == "F").sum()), 10, 0, "Abstract")
    ck.published("occlusion length, mean (s)", np.mean(occl_dur), 1.5, 0.05, "Methods, Experimental design")
    full = [s for s, v in have.items() if {"04", "08", "12", "16"} <= v and s not in DRIFT]
    ck.published("participants with step-duration data at all four speeds, minus the three the README lists with missing or drifting insole data",
                 len(full), 12, 0, "Methods, Kinematic Analysis")

    meta = dict(
        study=DS,
        paper=dict(authors="Morris JR, Cruz K, Kainth R, Ferris DP", year=2026,
                   title="Walking speed does not modulate electrocortical alpha response to intermittent visual occlusions",
                   venue="J Neurophysiol 136(1):165-179", doi="10.1152/jn.00432.2025"),
        lab="Ferris (University of Florida)", data_url="https://openneuro.org/datasets/ds007222", license="CC0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Read the heel-strike, toe-off and occlusion times of the BIDS events tables and turned them into one row per left-foot "
                 "stride (durations are differences of the authors' event times); age and sex come from participants.tsv, race/ethnicity "
                 "is left out. The EEG and EMG recordings (EDF) and the trial-wise events of the stationary condition are not converted."),
        how=["One row is one stride of the left foot of one participant, from a left heel strike to the next, in the order recorded, "
             "for each treadmill speed (0.4, 0.8, 1.2, 1.6 m/s; the standing file has gait events for one participant only, shown at 0 m/s). "
             "Times are in seconds from the start of each file, which holds 5 s of padding before and after the 3 minute trial: phase is "
             "test between the file's start and stop markers (the longest start-to-stop pair when a file has several) and other in the padding.",
             "Stance time is the interval from the left heel strike to the left toe-off inside the stride, swing time the rest of the stride, "
             "and the right-foot columns are filled only when exactly one right heel strike falls in this stride and in the next, with one "
             "right toe-off between the right heel strikes; otherwise they stay empty (insole events are missing or irregular for several "
             "participants). The authors' own step-duration measure is not in the deposit.",
             "pert_type is visual_occlusion, with value 1.5 (the length of an occlusion in seconds) and feedback masked, on strides whose left "
             "heel strike falls inside an occlusion (opaque goggles); occlusions come every 8-10 s of transparency in all conditions. The "
             "random order of the speeds is not shared, so trial counts the rows of a participant in speed order. Participant sub-05 has no gait "
             "events at all and sub-07 only at 1.6 m/s, so participants.tsv lists 15 people but the table holds 14."],
        unit_note="stride of the left foot (left heel strike to the next)",
        differences=["participants.tsv gives age 14 for one participant while the paper and README say all participants were 18-29; the table "
                     "keeps the deposit's value.",
                     "The paper's step-duration results (Fig. 6, 12 participants) report no numbers that can be reproduced from the events; the "
                     "check of the 12 participants uses the count only. The trial windows between the start and stop markers do not all last exactly "
                     "the 3 minutes the paper gives."],
    )
    df = df.reindex(columns=[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"])
    stdlib.release(DS, MODULE, df, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.5g} vs {c['reported']}  ({c['source']})")
        elif c.get("detail"):
            print(f"  [internal] {c['what'][:70]}: {c['detail'][:300]}")
    print(len(df), df.subject.nunique())


if __name__ == "__main__":
    main()
