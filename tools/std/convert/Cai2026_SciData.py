"""Cai, Fu, Wang, Lu, Guo & Xu (2026) Sci Data 13(1):933, IACKD: 15 right-handed adults moved the open hand (Leap Motion, palm centre) to guide a
ball on a screen to a target frame; a red ball moved in the hand's direction (congruent), a yellow ball opposite (incongruent). Each run
holds 40 or 60 trials of 1 s "pre-move" then execution; EEG was recorded in parallel (not carried: EEG needs its own format).
Deposit: OpenNeuro ds006840 v1.0.0 (CC0), files sub-XX/sourcedata/beh/*_leap.tsv, *_ball.tsv, sub-XX/eeg/*_events.tsv, sub-XX_scans.tsv (used only to
order the runs in time; the acquisition times are not carried).
Run from the site folder: python3 -I tools/std/convert/Cai2026_SciData.py"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Cai2026_SciData"
MODULE = "timeseries"
D = stdlib.raw_dir(ID)
TMP = Path("tmp_" + ID)
shutil.rmtree(TMP, ignore_errors=True)
(TMP / "samples").mkdir(parents=True)
DEVICE = "Leap Motion hand tracker (palm centre, nominal 170 Hz) and a 24-inch 1920x1080 60 Hz monitor with a ball cursor (PsychoPy)"
SIGNALS = {
    "leap": dict(kind="hand position: palm centre (Leap Motion), x to the right, y up, z perpendicular to the sensor plane",
                 cols=["x_mm", "y_mm", "z_mm"], units="mm;mm;mm", rate=170.0),
    "ball": dict(kind="controlled ball on the screen: horizontal position, trial index, boundary hit, cue colour and target direction",
                 cols=["ball_x_pix", "trial", "hit", "ball_color", "move_direct"], units="px (screen, 0 = centre, +-960 = screen edges);index from 0;1 = target boundary touched;red = congruent, yellow = incongruent;left or right", rate=60.0),
}
rows, extra = [], []
stat = dict(ev_short=0, trials=0, premove=[], cong_ok=True, leap_dt=[], ball_dt=[], runs={}, nan=0, nonmono=0, combos=set())

for s in range(1, 16):
    sid = f"sub-{s:02d}"
    sc = pd.read_csv(D / f"{sid}_scans.tsv", sep="\t", encoding="utf-8-sig")
    order = []
    for _, r in sc.sort_values("acq_time").iterrows():
        m = re.search(r"acq-(left|right)_run-(\d+)", r.filename)
        order.append((m.group(1), int(m.group(2))))
    k = 0
    for hand, run in order:
        ev = pd.read_csv(D / f"{sid}_task-ihc_acq-{hand}_run-{run:02d}_events.tsv", sep="\t")
        evname = f"{sid}_{hand}_run{run:02d}_events.csv"
        e2 = ev.rename(columns={"onset": "onset_s", "duration": "duration_s"})
        e2.to_csv(TMP / evname, index=False, lineterminator="\r\n", float_format="%.8g")
        extra.append((f"events/{evname}", str(TMP / evname)))
        stat["premove"] += list(ev[ev.trial_type == "premove"].duration)
        lab = ev.dropna(subset=["congruency", "ball_color"])
        stat["unlabelled"] = stat.get("unlabelled", 0) + len(ev) - len(lab)
        cm = lab.congruency.map({"congruent": "red", "incongruent": "yellow"}) == lab.ball_color
        stat["cong_ok"] &= bool(cm.all())
        ntr = int((ev.trial_type == "premove").sum())
        ntb = pd.read_csv(D / f"{sid}_task-ihc_run-{run:02d}_hand-{hand}_ball.tsv", sep="\t", usecols=["trial"]).trial.nunique()
        for sig in ("leap", "ball"):
            k += 1
            t = pd.read_csv(D / f"{sid}_task-ihc_run-{run:02d}_hand-{hand}_{sig}.tsv", sep="\t")
            assert (t.hand == hand).all()
            t = t.drop(columns="hand").rename(columns={"time": "time_s"})
            if sig == "ball":
                t["ball_color"] = t.ball_color.str.strip('"')
                t["move_direct"] = t.move_direct.str.strip('"')
                t["hit"] = t.hit.astype(int)
                stat["combos"] |= set(zip(t.ball_color, t.move_direct))
                assert t.trial.nunique() >= ntr, (sid, hand, run, t.trial.nunique(), ntr)
                stat["trials"] += t.trial.nunique()
                stat["ev_short"] += t.trial.nunique() - ntr
                stat["ball_dt"].append(np.median(np.diff(t.time_s)))
            else:
                stat["leap_dt"].append(np.median(np.diff(t.time_s)))
            stat["nan"] += int(t.isna().sum().sum())
            stat["nonmono"] += int((np.diff(t.time_s) <= 0).sum())
            fn = f"{sid}_{hand}_run{run:02d}_{sig}.csv"
            t.to_csv(TMP / "samples" / fn, index=False, lineterminator="\r\n", float_format="%.8g")
            extra.append((f"samples/{fn}", str(TMP / "samples" / fn)))
            sg = SIGNALS[sig]
            stat["runs"].setdefault(sid, set()).add((hand, run))
            rows.append(dict(
                dataset=ID, study="Cai et al. 2026 Sci Data", experiment="", group="", subject=sid, person=sid, subject_excluded_by_authors=False,
                age=np.nan, sex="", handedness="right", population="healthy", setting="lab", device=DEVICE, session=1,
                block=f"{hand} hand, run {run}", trial=k, phase="test", time_s=np.nan, task="finger movements", unit="recording",
                condition=f"{hand} hand; congruent (red) and incongruent (yellow) trials mixed",
                pert_type="ball_direction_reversal_on_incongruent_trials", pert_value=np.nan,
                pert_unit="per trial: yellow ball moves opposite to the hand (see ball_color in the ball samples)", feedback="altered",
                excluded_by_authors=False, recording=k, samples_file=f"samples/{fn}", signal_kind=sg["kind"], channels=";".join(sg["cols"]),
                channel_units=sg["units"], sample_rate_hz=sg["rate"], n_samples=len(t), duration_s=float(t.time_s.iloc[-1] - t.time_s.iloc[0]),
                events="", processing="none stated (source data of the deposit, sourcedata/beh)",
                trial_info=f"hand={hand};run={run};n_trials={ntb};n_trials_in_events_file={ntr};trial_events_file=events/{evname} (onsets on the EEG clock, not on time_s of this file)"))
df = pd.DataFrame(rows)

ck = stdlib.Checks()
ck.published("participants", df.subject.nunique(), 15, 0, "Abstract")
ck.published("trials recorded", stat["trials"], 7040, 0, "Abstract (7,040 trials from 15 participants)")
ck.published("sessions (hand x run) of a standard participant, s02", len(stat["runs"]["sub-02"]), 8, 0, "Methods: configuration A, 8 sessions (4 left, 4 right)")
ck.published("sessions of s04", len(stat["runs"]["sub-04"]), 12, 0, "Methods: configuration B, 12 sessions (6 left, 6 right)")
ck.published("sessions of s05", len(stat["runs"]["sub-05"]), 12, 0, "Methods: configuration B, 12 sessions")
ck.published("pre-movement stage (s)", np.mean(stat["premove"]), 1.0, 0.01, "Methods: 1-s pre-movement stage")
ck.published("ball frame rate (Hz, from the median sample interval)", 1 / np.mean(stat["ball_dt"]), 60, 0.1, "Methods: 60 Hz monitor")
ck.published("Leap Motion rate (Hz, from the median sample interval; paper says 170 nominal)", 1 / np.mean(stat["leap_dt"]), 170, 5, "Data records: 170 Hz")
ck.internal("red ball = congruent and yellow = incongruent in every labelled event of every events file", stat["cong_ok"])
ck.internal("all four cue combinations (red/yellow x left/right) occur", len(stat["combos"]) == 4, str(sorted(stat["combos"])))
ck.internal("no missing samples and time increases within every recording", stat["nan"] == 0 and stat["nonmono"] == 0, f"nan {stat['nan']}, non-increasing {stat['nonmono']}")
ck.internal("events files lack 11 of the 7,040 trials (in 10 runs)", stat["ev_short"] == 11, str(stat["ev_short"]))
ck.internal("256 recordings (128 hand runs x leap and ball)", len(df) == 256, str(len(df)))
meta = dict(
    paper=dict(authors="Cai M, Fu R, Wang Y, Lu B, Guo S, Xu F", year=2026,
               title="Intention-Action Conflict EEG-Hand Kinematics Dataset for Unimanual Control under Congruent and Incongruent Conditions",
               venue="Sci Data 13(1):933", doi="10.1038/s41597-026-07146-x"),
    lab="Fu (Yanshan University)", data_url="https://openneuro.org/datasets/ds006840", license="CC0",
    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("For each run and hand the deposit's two behavioural tables were carried as two recordings: the Leap Motion palm-centre trajectory (time, x, y, z) and the "
             "controlled-ball table (time, ball position, trial index, hit, cue colour, target direction), values unchanged except 8 significant digits, True/False written as 1/0 and "
             "the quotes around ball_color and move_direct removed. The constant hand column is in the recording's block. The trial-event tables (premove, execution, hit with "
             "congruency, direction, cue colour) come from the BIDS events files of the EEG and are in events/ (onset and duration in seconds on the EEG clock). Left out: all EEG "
             "(BrainVision), derivatives, scans acquisition times (used only to put the runs in time order), and participants.tsv (age, sex, handedness are 'n/a' there). "
             "Handedness 'right' for all and the group description are from the paper."),
    how=["One row is one recording: either the hand position (Leap Motion, mm, palm centre of the open hand moved mainly by the elbow) or the on-screen ball of one run (40 or 60 trials) of "
         "one hand. In each trial a 1 s pre-move cue is followed by execution: the participant guides the ball to a target frame left or right of the centre; a red ball moves in the "
         "same direction as the hand (congruent), a yellow ball in the opposite direction (incongruent). Trials are invalid if the boundary is not reached in 6 s.",
         "Rows are in time order of the runs (from the scan list), the leap row before the ball row of each run. time_s in the sample files is the device's own clock for that "
         "table; the hand, ball and EEG-event clocks are different, and the authors align them with a linear mapping in their scripts (not applied here).",
         "Participants: 15 healthy right-handed adults (11 men, 4 women, aged 23 to 27, mean 24.3; per-person values are not shared). Participants s04 and s05 did 12 sessions of 40 trials "
         "(6 runs per hand), s01 8 sessions of 40 trials, the others 8 sessions of 60 trials."],
    unit_note="recording (one run of one hand: either the hand trajectory or the ball trajectory, 40 or 60 trials)",
    differences=["The paper states 200 Hz for the Leap Motion in the acquisition section and 170 Hz elsewhere; the deposit's JSON says 170 and the median sample interval gives about "
                 "167 Hz, so sample_rate_hz is the nominal 170 (ball 60).",
                 "The paper keeps 6,671 of the 7,040 trials after its own quality control (artifact and kinematic rejection); the deposit does not mark which trials, so no row is marked "
                 "excluded_by_authors. The paper's completion times (about 300 ms) and its synchronization residuals come from its aligned derivatives and preprocessing "
                 "and were not reproduced here.",
                 "In 10 runs the EEG events file lists fewer trials than the ball table (11 trials in all, the last ones of the run, counts per run in trial_info); the ball and hand tables are complete.",
                 "Some event rows in the EEG events files have no congruency, colour or direction label (kept as empty cells in events/).",
                 "The paper says every participant did 480 trials (7,200 in all) but 7,040 were recorded; the data give 320 for the pilot participant s01 and 480 for the other 14.",
                 "The paper's 'sub-XX' and 's1-s15' naming and the BIDS participants table (age, sex all 'n/a') differ; only the sub-XX names appear in the deposit and are replaced by codes."])
try:
    stdlib.release(ID, MODULE, df, meta, ck, extra_files=extra)
finally:
    shutil.rmtree(TMP, ignore_errors=True)
for c in ck.items:
    print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""), c.get("detail", ""))
