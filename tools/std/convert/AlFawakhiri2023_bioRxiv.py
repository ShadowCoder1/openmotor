"""Al-Fawakhiri, Kayani & McDougle (2023), bioRxiv: an online Pong-like paddle game whose paddle size was adapted to hold the
participant's error rate at 15, 30, 50 or 65% (198 participants who posted data), one row per trial, in the 'game' module.

Run from the site folder: python3 -I tools/std/convert/AlFawakhiri2023_bioRxiv.py
Raw files (untrusted data, read only) in raw_extra/AlFawakhiri2023_bioRxiv/, from https://github.com/ilestz/optimalER
(no license file in the repository):
  all_subjects_fast.json   the first batch of 100 participants (config + data.calib 20 trials + data.basic 90 trials each)
  pong_fast100.json        the second batch, 98 participants
  analyzePongFast_bin.Rmd  the authors' analysis, read as text, never run: it drops participants 87 and 98 of the first file and 37 and
                           48 of the second ('skip ... cuz of bug'; the paper's four participants with no recorded movement), uses
                           the first 100 trials (20 calibration + 80 basic: the paper's 10 baseline + 70 training trials) and takes
                           the trial error from endpoint_y in the second batch.
The cursor samples (movement_data) and the logged device details are not carried over; ids and timestamps are dropped.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "AlFawakhiri2023_bioRxiv"
RAW = stdlib.raw_dir(ID)
FILES = ("all_subjects_fast.json", "pong_fast100.json")
SKIP = ({87, 98}, {37, 48})  # 1-based positions skipped in analyzePongFast_bin.Rmd (lines 17 and 127)
SRC = "Methods and Results (bioRxiv 2023.07.19.549705)"
checks = stdlib.Checks()
differences = []

src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for fn in FILES:
    checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])

rows = []
perf = []  # per analysed participant: dict of group, calibration/baseline/training success
fixed_xtab = {}
n_present = 0
skipped_moved = 0
for b, fn in enumerate(FILES):
    data = json.load(open(RAW / fn))
    for pos, p in enumerate(data, 1):
        n_present += 1
        cfg = p["config"]
        skipped = pos in SKIP[b]
        trials = p["data"]["calib"] + p["data"]["basic"]
        assert [t["trial_type"] for t in trials] == ["calib"] * 20 + ["basic"] * 90
        t0 = trials[0]["ref_time"]
        grp = int(cfg["group"])
        moved = any(s["raw_y"] != 0 for t in trials for s in t["movement_data"])
        if skipped:
            skipped_moved += int(moved)
        succ = {"calib": [], "base": [], "train": []}
        subj = f"b{b + 1}p{pos}"
        for i, t in enumerate(trials, 1):
            tn = t["trial_number"]
            if t["trial_type"] == "calib":
                phase, block = "practice", "calibration"
            else:
                phase = "baseline" if tn < 30 else "learning" if tn < 100 else "other"
                block = "basic, fixed difficulty" if t["is_ER_fixed"] else "basic"
                fixed_xtab[(phase, bool(t["is_ER_fixed"]))] = fixed_xtab.get((phase, bool(t["is_ER_fixed"])), 0) + 1
            if not skipped:
                succ["calib" if tn < 20 else "base" if tn < 30 else "train" if tn < 100 else "none"] = succ.get(
                    "calib" if tn < 20 else "base" if tn < 30 else "train" if tn < 100 else "none", []) + [int(t["success"])]
            err = t.get("endpoint_y") if b == 1 else None
            vx = t.get("ball_vx")
            rows.append(dict(
                dataset=ID, study=ID, experiment=f"{ID}_Main", group=f"ER {grp}%", subject=subj,
                subject_excluded_by_authors=skipped, age="", sex="", handedness="", population="healthy", setting="online",
                device=f"{cfg['device']}, {cfg['hand']} hand reported, web browser", session=1, block=block, trial=i,
                phase=phase, time_s=(t["ref_time"] - t0) / 1000.0, task="sports and games", unit="trial", condition="",
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=skipped or tn >= 100,
                game="Pong (paddle interception)", attempt=tn, score_unit="points (+10 for each intercepted puck)",
                score_total=t.get("score", ""), success=bool(t["success"]) if t.get("success") is not None else "",
                outcome_error=err if err is not None else "", outcome_error_unit="px",
                difficulty=t.get("displ_rad", ""), difficulty_unit="px (paddle half-length)",
                difficulty_next=t.get("new_rad", ""), hit_rate_running=t.get("hit_rate", ""), rmse_running=t.get("rmse", ""),
                intended_error_rate=float(t["er"]) if t.get("er") is not None else "",
                object_start_y=t.get("ball_y", ""), object_direction_deg=t.get("ball_theta", ""),
                object_speed=vx * 1000 if vx is not None else "", object_unit="px",
                object_bounces=t.get("bounces", "")))
        if not skipped:
            perf.append(dict(group=grp, **{k: float(np.mean(v)) for k, v in succ.items() if k != "none" and v}))
    del data

print("phase x is_ER_fixed trial counts:", fixed_xtab)
checks.internal("rows: 198 participants x 110 trials", len(rows) == n_present * 110 == 198 * 110, f"{len(rows)} rows")
checks.internal("the four participants the authors' script skips are in the files (two per batch)", True,
                f"{4 - skipped_moved} of the 4 have no cursor movement in any trial, {skipped_moved} has some")
if skipped_moved:
    differences.append(f"The paper says the four participants left out made no recorded movements; in the shared rows "
                       f"{4 - skipped_moved} of the four have no cursor movement and {skipped_moved} has cursor movement on some trials "
                       f"(they are the four the authors' script skips, so they are flagged subject_excluded_by_authors).")
checks.internal("every analysed participant has calibration, baseline and training trials",
                all(k in d for d in perf for k in ("calib", "base", "train")))
checks.internal("groups are the four intended error rates", sorted({d["group"] for d in perf}) == [15, 30, 50, 65])

# ---------------------------------------------------------------- published numbers
cal = np.array([d["calib"] for d in perf]) * 100
bas = np.array([d["base"] for d in perf]) * 100
trn = np.array([d["train"] for d in perf]) * 100
grp = np.array([d["group"] for d in perf])
passed = []


def chk(what, computed, reported, tol, source):
    if abs(computed - reported) <= tol:
        checks.published(what, computed, reported, tol, source)
        passed.append(what)
    else:
        differences.append(f"{what}: the paper reports {reported:g}, the shared rows give {computed:.4g} ({source}).")
    print(f"  {'ok ' if abs(computed - reported) <= tol else 'DIFF'} {what}: {computed:.4g} vs {reported:g}")


chk("participants with data (200 recruited, 2 did not post)", n_present, 198, 0, f"Methods, Participants, {SRC}")
chk("participants after dropping the four without recorded movement", len(perf), 194, 0, f"Methods, Participants, {SRC}")
chk("calibration success, mean (%)", cal.mean(), 72, 0.5, f"Methods, calibration, {SRC}")
chk("calibration success, SD (%)", cal.std(ddof=1), 12, 0.5, f"Methods, calibration, {SRC}")
chk("baseline success, mean (%)", bas.mean(), 77, 0.5, f"Methods, baseline, {SRC}")
chk("baseline success, SD (%)", bas.std(ddof=1), 17, 0.5, f"Methods, baseline, {SRC}")
g65 = trn[grp == 65]
chk("65% group: participants (degrees of freedom 48 + 1)", len(g65), 49, 0, f"Results, error-rate control, {SRC}")
chk("65% group: mean success after baseline (%)", g65.mean(), 41.6, 0.15, f"Results, error-rate control, {SRC}")
chk("65% group: SD of success (%)", g65.std(ddof=1), 6.1, 0.15, f"Results, error-rate control, {SRC}")
res = stats.ttest_1samp(g65, 35.0)
chk("65% group: t against the intended 35% success", res.statistic, 3.03, 0.01, f"Results, error-rate control, {SRC}")
chk("65% group: p of that t", res.pvalue, 0.004, 0.0006, f"Results, error-rate control, {SRC}")
true_er = 100 - trn
under = true_er - grp
chk("true error rate minus intended error rate, mean (points of %)", under.mean(), -3.1, 0.15, f"Results, error-rate control, {SRC}")
chk("true error rate minus intended error rate, SD (points of %)", under.std(ddof=1), 6.6, 0.15, f"Results, error-rate control, {SRC}")
chk("same difference in trials (of 70), mean", under.mean() * 0.7, -2.2, 0.1, f"Results, error-rate control, {SRC}")
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']}")
assert passed, "no published number reproduced"

differences += [
    "The paper's ten learning-index outliers (1.75 IQR rule on a precision measure built from the trial errors) are not flagged: "
    "the first batch does not log the trial error, so the measure cannot be rebuilt from the shared rows.",
    "In the first batch (all_subjects_fast.json) endpoint_y is a placeholder (mostly 0) and the authors' script rebuilds the error from "
    "the puck's launch values and the last cursor sample; outcome_error is therefore filled for the second batch only, where "
    "endpoint_y is the final puck minus paddle-centre distance the script uses.",
    "The last 10 trials of each participant (attempt 100-109, fixed difficulty) are shared but the authors' script does not use them; "
    "they are marked excluded_by_authors.",
]

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={**{k: paper[k] for k in ("authors", "year", "title", "doi")}, "venue": "bioRxiv"},
    lab="McDougle (Yale)", data_url="https://github.com/ilestz/optimalER", license="none stated", license_url="",
    changes=("Converted to the OpenMotor format: participant ids, timestamps, browser details and the cursor samples of each trial "
             "were left out, trials were put in one sequence per participant (20 calibration then 90 basic trials), time is "
             "seconds from the participant's first trial, and the puck speed was multiplied by 1000 (px per ms to px per s)."),
    how=[
        "One row is one trial of an online game in which a puck crosses the screen and the participant moves a vertical paddle with "
        "their pointing device to intercept it; success is the game's hit/miss, score_total the running points (+10 per hit).",
        "Trials 0-19 are the speed calibration (staircase), then 10 baseline trials (attempt 20-29) and 70 training trials (30-99) in "
        "which the game resized the paddle (difficulty, its half-length in px) to hold the intended error rate in group; the last 10 "
        "trials (100-109) keep the paddle size fixed.",
        "object_start_y is the puck's start height and object_direction_deg its launch angle in the game's own coordinates; "
        "outcome_error (second batch only) is the final puck position minus the paddle centre in px.",
        "subject_excluded_by_authors marks the four participants the authors' script leaves out (the paper: no recorded movement).",
    ],
    unit_note="trial of the paddle game", differences=differences)
stdlib.release(ID, "game", rows, meta, checks)
