"""Liu et al. (2025) Imaging Neuroscience: young adults walking on an uneven-terrain treadmill (OpenNeuro ds004625,
"Mind in Motion Young Adults Walking Over Uneven Terrain", CC0), one row per stride from the authors' gait events.

Raw files (raw_extra/Liu2025_ImagingNeurosci_young/): participants.tsv, the events.tsv of all 512 recordings (32 participants
x 8 conditions x 2 runs), and the BIDS description files (read as text). The 142 MB EEG/EMG/IMU/force .fdt files are not
converted (67 GB in total); the events files hold the authors' foot strikes and toe-offs (RHS, LHS, RTO, LTO) from the
insole force sensors.
Run from the site folder: python3 -I tools/std/convert/Liu2025_ImagingNeurosci_young.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Liu2025_ImagingNeurosci_young"
raw = stdlib.raw_dir(DS)
TASKS = [("flat", "flat terrain", 0, None), ("low", "low terrain", 1, None), ("med", "medium terrain", 2, None),
         ("high", "high terrain", 3, None), ("0p25", "flat, fixed speed 0.25 m/s", 0, 0.25), ("0p5", "flat, fixed speed 0.5 m/s", 0, 0.5),
         ("0p75", "flat, fixed speed 0.75 m/s", 0, 0.75), ("1p0", "flat, fixed speed 1.0 m/s", 0, 1.0)]
DEVICE = ("Uneven-terrain slat-belt treadmill (polyurethane disks on the belt) with a safety harness; 120-channel dual-layer EEG, "
          "neck EMG, sacral IMU; gait events from loadsol insole force sensors (20 N threshold)")
ck = stdlib.Checks()

parts = pd.read_csv(raw / "participants.tsv", sep="\t")
rows, counts = [], dict(strides=0, no_to=0, recs=0)
bad_alt = 0
for _, p in parts.iterrows():
    sid = p.participant_id
    n_in_subject = 0
    for task, label, level, fixed in TASKS:
        for run in (1, 2):
            ev = pd.read_csv(raw / f"{sid}_task-{task}_run-{run}_events.tsv", sep="\t")
            counts["recs"] += 1
            t0 = float(ev.loc[ev.value == "TrialStart", "onset"].iloc[0])
            t1 = float(ev.loc[ev.value == "TrialEnd", "onset"].iloc[0])
            speed = fixed if fixed else float(p.speed)
            recs = []
            for foot, side in (("R", "right"), ("L", "left")):
                hs = np.sort(ev.loc[ev.value == f"{foot}HS", "onset"].to_numpy())
                to = np.sort(ev.loc[ev.value == f"{foot}TO", "onset"].to_numpy())
                for a, b in zip(hs[:-1], hs[1:]):
                    if a < t0 or b > t1:
                        continue
                    inside = to[(to > a) & (to < b)]
                    r = dict(side=side, start=a, stride=b - a)
                    if len(inside) == 1:
                        r.update(stance=inside[0] - a, swing=b - inside[0])
                    else:
                        counts["no_to"] += 1
                        bad_alt += 1
                    recs.append(r)
            for r in sorted(recs, key=lambda x: x["start"]):
                s = r["side"]
                n_in_subject += 1
                d = dict(
                    dataset=DS, study=DS, experiment="", group="", subject=sid, subject_excluded_by_authors=False,
                    age=float(p.age), sex=p.gender, handedness={"R": "right", "L": "left"}[p.handedness], population="healthy",
                    setting="lab", device=DEVICE, session=1, block=task, trial=n_in_subject, phase="test", task="walking",
                    unit="stride", condition=label, pert_type="uneven_terrain", pert_value=float(level),
                    pert_unit="terrain level (0 flat, 1 low, 2 medium, 3 high)", feedback="normal", excluded_by_authors=False,
                    fast_leg="unknown", belt_speed_fast_mps=speed, belt_speed_slow_mps=speed, walking_trial=run,
                    stride_start_s=r["start"], walking_context="treadmill", surface_incline_deg=0.0)
                d[f"stride_time_{s}_s"] = r["stride"]
                if "stance" in r:
                    d[f"stance_time_{s}_s"] = r["stance"]
                    d[f"swing_time_{s}_s"] = r["swing"]
                    d[f"stance_pct_{s}"] = 100 * r["stance"] / r["stride"]
                    d[f"swing_pct_{s}"] = 100 * r["swing"] / r["stride"]
                rows.append(d)
df = pd.DataFrame(rows).reindex(columns=[c[1] for c in stdlib.columns_for("walking") if c[1] != "person"])
for c in ("stride_start_s", "stride_time_left_s", "stride_time_right_s", "stance_time_left_s", "stance_time_right_s",
          "swing_time_left_s", "swing_time_right_s"):
    df[c] = df[c].round(6)

nsub = df.subject.nunique()
ck.internal("32 participants, 16 recordings each (8 conditions x 2 runs): 512 events files read", nsub == 32 and counts["recs"] == 512)
ck.internal("every stride has exactly one toe-off of the same foot between its two heel strikes, except a small minority",
            counts["no_to"] / max(1, len(df)) < 0.01, f"{counts['no_to']} of {len(df)} strides without a single toe-off")
good = df.dropna(subset=["stance_time_left_s"])
st = df.stride_time_left_s.fillna(df.stride_time_right_s)
share = float(((st > 0.4) & (st < 6)).mean())
ck.internal("stride times between 0.4 and 6 s for 99% of strides", share > 0.99,
            f"{share:.5f}; quantiles 0.1/50/99.9%: {st.quantile(.001):.3f} {st.median():.3f} {st.quantile(.999):.3f}")
ck.internal("left and right strides both present for every participant",
            all(g.stride_time_left_s.notna().any() and g.stride_time_right_s.notna().any() for _, g in df.groupby("subject")))
pp = df.drop_duplicates("subject")
spd = parts.speed
ck.published("left-handed participants (28 right, 3 left among the 31 analysed)", (parts.handedness == "L").sum(), 3, 0, "Table 1")
ck.published("women among the participants (16 of the 31 analysed)", (parts.gender == "F").sum(), 16, 0, "Table 1")
ck.published("mean age, years (24, SD 4)", parts.age.mean(), 24, 0.5, "Table 1")
ck.published("SD of age, years", parts.age.std(), 4, 0.5, "Table 1")
ck.published("mean treadmill speed on all terrains, m/s (0.70)", spd.mean(), 0.70, 0.01, "Table 1")
ck.published("SD of treadmill speed, m/s (0.16)", spd.std(), 0.16, 0.01, "Table 1")
ck.published("mean treadmill speed, m/s (0.7, SD 0.2; 35 recruited)", spd.mean(), 0.7, 0.05, "Methods 2.1")

meta = dict(
    paper=dict(authors="Liu C, Pliner EM, Salminen J, Downey RJ, Hwang J, Roy A, Swearinger R, Richer N, Hass CJ, Clark DJ, Manini TM, Cruz-Almeida Y, Seidler RD, Ferris DP",
               year=2025, title="Age differences in electrocortical dynamics during uneven terrain walking",
               venue="Imaging Neuroscience 3:IMAG.a.1039", doi="10.1162/imag.a.1039"),
    lab="Ferris (University of Florida), Mind in Motion", data_url="https://openneuro.org/datasets/ds004625",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("From the authors' events files (foot strikes and toe-offs) the stride, stance and swing times of each stride were "
             "computed as differences of event times; strides before the trial-start marker or after the trial-end marker are "
             "left out. The EEG, EMG, IMU and force recordings, electrode locations and channel files are not included."),
    how=["One row is one stride of one foot (heel strike to the next heel strike of the same foot, from the authors' foot-strike "
         "events detected with 20 N insole force thresholds) in one 3-minute walking trial; left strides fill the _left columns "
         "and right strides the _right columns, and rows are in time order within each trial. Stance is that foot's heel strike to "
         "its toe-off, swing the toe-off to the next heel strike (seconds), and the percentages are of the stride time.",
         "Each participant walked two 3-minute trials on each of four terrains (flat, low, medium, high; pert_value 0 to 3) at one "
         "self-selected speed (participants.tsv), and two trials on flat terrain at each fixed speed 0.25, 0.5, 0.75 and 1.0 m/s. "
         "Both belts ran at the same speed. The deposit does not give the order the trials were walked in, so trial counts the "
         "rows in the order terrains, then fixed speeds, runs 1 and 2; walking_trial is the run number.",
         "The terrain levels are the authors' disk mixes (all 1.3 cm; half 1.3 and half 2.5 cm; 20/30/50% of 1.3/2.5/3.8 cm disks "
         "in Downey et al. 2022)."],
    unit_note="stride (of one foot)",
    differences=["The deposit holds 32 young adults and the paper analysed 31 (three were excluded for gelling and technical "
                 "problems, and the paper's text and Table 1 count 35 recruited and 4 excluded); the deposit does not say which "
                 "participant was left out of the analysis, so no one is flagged as excluded. Table 1 values (age 24 +- 4, 16 women, "
                 "3 left-handed, speed 0.70 +- 0.16 m/s) are reproduced from all 32 participants within rounding.",
                 "The paper's gait outcomes (step-duration variability, sacral excursion variability) come from the insole and IMU "
                 "recordings, which are not part of the converted file, so they are not reproduced."],
)
stdlib.release(DS, "walking", df, meta, ck)
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
print(df.groupby("subject").size().describe())
