"""Nardi, Arora, Faisal and Haar (2026), iScience 29(10):117523: real-world pool (cut shot) with a gradual 5 degree visuomotor rotation, 32
participants, an error-feedback and a reward-feedback session, ten blocks of 25 shots per session; 'game' module, one row per block.

Run from the site folder: python3 -I tools/std/convert/Nardi2026_iScience.py
Raw files (untrusted data, read only) in raw_extra/Nardi2026_iScience/ from https://doi.org/10.6084/m9.figshare.31228423 (CC BY 4.0):
  all_subjects_full.csv       one row per participant, session and block (carried over in full)
  tbt_between.csv, tbt_within.csv   the authors' derived statistics (not carried over, see README)
The authors' notebook (.ipynb) was not downloaded."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Nardi2026_iScience"
SRC = "Methods and Results (iScience 29(10):117523)"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
d = pd.read_csv(RAW / "all_subjects_full.csv")
nan = lambda v: None if pd.isna(v) else float(v)  # noqa: E731

first = d.sort_values(["Subject", "Round"]).groupby("Subject")["Mode"].first()
PHASE = lambda b: "baseline" if b <= 3 else ("learning" if b <= 9 else "washout")  # noqa: E731
rows = []
for k, r in enumerate(d.itertuples(index=False)):
    b = int(r.Block)
    rot = 4 <= b <= 9
    rows.append(dict(
        dataset=ID, study=ID, experiment=ID, group=f"{first[r.Subject].lower()} session first", subject=f"{r.Subject}",
        subject_excluded_by_authors=False, age=None, sex=None, handedness="right", time_s=None, population="healthy", setting="lab",
        device="physical pool table, cue and ball in an HTC Vive virtual scene, OptiTrack cue tracking, 7-channel EEG",
        session=int(r.Round), block=f"block {b}", trial=int(1 + (d.Subject.iloc[:k + 1] == r.Subject).sum() - 1) if False else 0,
        phase=PHASE(b), task="sports and games", unit="trial", condition=f"{r.Mode.lower()} feedback",
        pert_type="gradual_visuomotor_rotation" if rot else "none", pert_value=5 if rot else 0, pert_unit="deg",
        feedback="altered", excluded_by_authors=False, game="pool cut shot (VR)", attempt=b, shots_in_block=25,
        block_success_rate=float(r.SuccessRate), block_real_success_rate=None,
        mean_cue_ball_angle_deg=float(r.MeanCueBallAngle), corrected_variability_deg=float(r.Variability),
        pmbr_percent=nan(r.PMBR), baseline_pmbr_percent=nan(r.BaselinePMBR), target_pocket=r.Pocket.lower()))
cnt = {}
for r in rows:  # trial = order of the block row within the participant, from 1
    cnt[r["subject"]] = cnt.get(r["subject"], 0) + 1
    r["trial"] = cnt[r["subject"]]

nsub = d.Subject.nunique()
checks.published("participants", nsub, 32, 0, f"Participants, {SRC}")
per = d.groupby("Subject").size()
checks.published("blocks per participant (2 sessions x (3 baseline + 6 rotation + 1 washout))", int(per.iloc[0]), 20, 0, f"Design, {SRC}")
checks.published("every participant has 20 blocks", int((per == 20).sum()), 32, 0, f"Design, {SRC}")
eeg = d.dropna(subset=["PMBR"]).groupby("Mode").Subject.nunique()
checks.published("participants with usable EEG, error session", int(eeg["Error"]), 30, 0, f"EEG exclusions, {SRC}")
checks.published("participants with usable EEG, reward session", int(eeg["Reward"]), 28, 0, f"EEG exclusions, {SRC}")
checks.published("participants who did the error session first", int((first == "Error").sum()), 16, 0, f"Design (two groups of 16), {SRC}")
checks.internal("success rates are multiples of 1/25", bool(((d.SuccessRate * 25 - (d.SuccessRate * 25).round()).abs() < 1e-6).all()))
checks.internal("each participant has one error and one reward session", bool((d.groupby("Subject").Mode.nunique() == 2).all()))
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']}")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Haar (Imperial College London)", data_url="https://doi.org/10.6084/m9.figshare.31228423", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="Converted to the OpenMotor format: participant numbers replaced by codes, authors' column names mapped to the game format, the "
            "session mode written as the condition, phases assigned from the block number (blocks 1-3 baseline, 4-9 rotation, 10 washout).",
    how=["One row is one block of 25 pool cut shots (the authors share block summaries, not shots): block_success_rate is the proportion of "
         "successful shots (in reward sessions during rotation, the proportion of shots that showed the rewarded trajectory), mean_cue_ball_angle_deg the block mean of the cue ball "
         "direction error in degrees relative to the ideal angle (sign as the authors give it, not restated in the deposit), "
         "corrected_variability_deg the trend-corrected variability, pmbr_percent and baseline_pmbr_percent the post-movement beta rebound "
         "over C3 and its baseline (empty where EEG was unusable).",
         "Each participant did an error-feedback and a reward-feedback session (session 1 or 2 is the order, group says which came first), "
         "10 blocks each; the rotation was introduced gradually from block 4 (0.2 degrees every 5 shots) up to 5 degrees, and pert_value is that final size, given on blocks 4-9. "
         "trial counts the participant's 20 block rows. target_pocket is the pocket aimed at in that session."],
    unit_note="block of 25 pool shots",
    differences=["The deposit's derived tables (intertrialvariability.csv, tbt_between.csv, tbt_within.csv: inter-trial variability decay and "
                 "success-dependent trial-to-trial change statistics) are not carried over; they are computed from shot-level data that "
                 "the authors did not share.",
                 "Per-participant age and sex are not in the deposit (the paper reports 11 women and 21 men, mean age 25.9)."])
stdlib.release(ID, "game", rows, meta, checks)
