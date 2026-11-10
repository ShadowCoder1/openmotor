"""Hill et al. (2025), eLife: age-dependent predictors of reinforcement motor learning (385 children and adults, four tasks).
Raw (untrusted, read only): raw_extra/Hill2025_eLife/AllData_1.mat (Mendeley Data g9z67hkyz5): per task (CP, DP, CD, DD) matrices
of participants x [7 descriptor columns + 150 trials] for target position, reward and end X position, plus summaryData.
Trials 1-20 baseline, 21-120 learning, 121-130 success clamp, 131-140 fail clamp, 141-150 single target (Readme)."""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hill2025_eLife"
m = sio.loadmat(stdlib.raw_dir(ID) / "AllData_1.mat", squeeze_me=True, struct_as_record=False)
B, R, T, S = (m[k] for k in ("buttonPositions_AllTasks", "rewardData_AllTasks", "trialEndData_AllTasks", "summaryData_AllTasks"))
inlab = set(int(x) for x in m["inLab_CP"])
BLOCKS = [("baseline", 0, 20, "baseline"), ("learning", 20, 120, "learning"), ("success_clamp", 120, 130, "other"),
          ("fail_clamp", 130, 140, "other"), ("single_target", 140, 150, "test")]
TASKNAME = {"CP": "continuous_probabilistic", "DP": "discrete_probabilistic", "CD": "continuous_deterministic",
            "DD": "discrete_deterministic"}
BROWSER = {0: "Chrome", 1: "Edge", 2: "Safari", 3: "Firefox", 4: "other"}
DEVICE = {0: "mouse", 1: "trackpad", 2: "touch screen"}
rows, parts = [], []
agegrp = {}
for k in ("CP", "DP", "CD", "DD"):
    b, r, t, s = getattr(B, k), getattr(R, k), getattr(T, k), getattr(S, k)
    for i in range(b.shape[0]):
        pid = int(b[i, 0])
        assert r[i, 0] == pid == t[i, 0] == s[i, 0]
        age = float(b[i, 1]) / 12.0
        agegrp[pid] = int(b[i, 2])
        base = dict(dataset=ID, study=ID, experiment=k, group="adult" if b[i, 2] == 6 else "child", subject=f"{k}_{pid}",
                    person=f"{k}_{pid}", subject_excluded_by_authors=False, age=age, sex="M" if b[i, 3] == 0 else "F",
                    population="healthy", setting="lab" if (k == "CP" and pid in inlab) else "online",
                    handedness="", time_s="", device=DEVICE[int(b[i, 4])], session=1, task="reaching", unit="trial", condition=TASKNAME[k],
                    feedback="altered", excluded_by_authors=False,
                    hand_used="left" if b[i, 6] == 1 else "right", browser=BROWSER[int(b[i, 5])])
        n = 0
        for name, lo, hi, ph in BLOCKS:
            for j in range(lo, hi):
                n += 1
                tx = b[i, 7 + j]
                tx = "" if (name in ("success_clamp", "fail_clamp") or (name == "learning" and k in ("CP", "CD")) or
                            (name == "single_target" and False)) else float(tx)
                clamp = name in ("success_clamp", "fail_clamp")
                rows.append({**base, "block": name, "trial": n, "phase": ph, "target_x_gu": tx,
                             "endpoint_x_gu": float(t[i, 7 + j]), "reward": int(r[i, 7 + j]),
                             "pert_type": "reward_clamp" if clamp else "none",
                             "pert_value": (1 if name == "success_clamp" else 0) if clamp else 0,
                             "pert_unit": "reward" if clamp else ""})
        parts.append(dict(person=f"{k}_{pid}", baseline_precision_mean=s[i, 7], baseline_precision_sd=s[i, 8],
                          learning_distance_mean=s[i, 9], learning_distance_sd=s[i, 10], success_clamp_sd=s[i, 11],
                          fail_clamp_sd=s[i, 12], single_target_precision_mean=s[i, 13], single_target_precision_sd=s[i, 14],
                          pathlength_ratio_mean=s[i, 15], pathlength_ratio_sd=s[i, 16], reaction_time_s=s[i, 17],
                          stationary_time_s=s[i, 18], movement_time_s=s[i, 19], total_play_min=s[i, 20],
                          screen_refresh_hz=s[i, 21], input_sampling_hz=s[i, 22]))

checks = stdlib.Checks()
persons = {r["person"]: r["group"] for r in rows}
per_task = {k: len({p for p in persons if p.startswith(k + "_")}) for k in TASKNAME}
kids = lambda k: len([p for p, g in persons.items() if p.startswith(k + "_") and g == "child"])
adults = lambda k: len([p for p, g in persons.items() if p.startswith(k + "_") and g == "adult"])
SRC = "Introduction and Methods (participants); expected values read from the paper"
checks.published("participants in all four tasks", len(persons), 385, 0, SRC)
checks.published("children aged 3-17", sum(1 for g in persons.values() if g == "child"), 298, 0, SRC)
checks.published("adults", sum(1 for g in persons.values() if g == "adult"), 87, 0, SRC)
checks.published("continuous probabilistic task: children", kids("CP"), 111, 0, "Results, continuous probabilistic task (F(1,109) implies 111)")
checks.published("continuous probabilistic task: adults", adults("CP"), 33, 0, "Results, participants")
checks.published("discrete probabilistic task: children", kids("DP"), 106, 0, "Results, discrete probabilistic task")
checks.published("discrete probabilistic task: adults", adults("DP"), 33, 0, "Results, discrete probabilistic task")
checks.published("trials per participant (20+100+10+10+10)", len(rows) / len(persons), 150, 0, "Methods, trial structure")
checks.published("age range lower bound is 3 years (youngest participant, years)", min(r["age"] for r in rows) // 1, 3, 0, "Methods, participants (3-17 year-olds)")
checks.internal("success clamp always rewarded, fail clamp never", all(r["reward"] == 1 for r in rows if r["block"] == "success_clamp")
                and all(r["reward"] == 0 for r in rows if r["block"] == "fail_clamp"))
checks.internal("one row per trial, no missing values in endpoints", all(r["endpoint_x_gu"] == r["endpoint_x_gu"] for r in rows))

pc = [("baseline_precision_mean", "number", "game units", "Mean of the baseline signed error (authors' value)."),
      ("baseline_precision_sd", "number", "game units", "SD of the baseline signed error: the authors' baseline precision."),
      ("learning_distance_mean", "number", "game units", "Mean absolute distance from the 100% reward zone over the last 10 learning trials."),
      ("learning_distance_sd", "number", "game units", "SD of that distance."),
      ("success_clamp_sd", "number", "game units", "SD of X over trials 2-10 of the success clamp."),
      ("fail_clamp_sd", "number", "game units", "SD of X over trials 2-10 of the fail clamp."),
      ("single_target_precision_mean", "number", "game units", "Mean signed error on single-target trials."),
      ("single_target_precision_sd", "number", "game units", "SD of signed error on single-target trials."),
      ("pathlength_ratio_mean", "number", "", "Mean path length ratio."), ("pathlength_ratio_sd", "number", "", "SD of path length ratio."),
      ("reaction_time_s", "number", "seconds", "Reaction time."), ("stationary_time_s", "number", "seconds", "Stationary time."),
      ("movement_time_s", "number", "seconds", "Movement time."), ("total_play_min", "number", "minutes", "Total game play time."),
      ("screen_refresh_hz", "number", "Hz", "Screen refresh sampling rate."), ("input_sampling_hz", "number", "Hz", "Input device sampling rate.")]
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Bastian (Kennedy Krieger Institute / Johns Hopkins)", data_url="https://doi.org/10.17632/g9z67hkyz5.1",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor format; participant codes replaced; age converted from months to years; the authors' coded "
            "sex, device, browser and hand written as words; the authors' matrices of 150 trials turned into one row per trial.",
    unit_note="trial of one participant (one swipe)",
    how=["One row is one trial of a web game in which a participant moved a penguin from the bottom to the far edge of the screen (24 game units, GU) "
         "with a mouse, trackpad or touch screen and was rewarded depending on the X position where it crossed the far edge. There is no hand "
         "direction, so the 90-column reaching format is not used; the endpoint is endpoint_x_gu (0 = centre, negative = left) and the reward is 0 or 1.",
         "Each participant did 150 trials: baseline (20, one random discrete target, target_x_gu), learning (100), success clamp (10, always rewarded), "
         "fail clamp (10, never rewarded) and single target (10, centre). Experiment is the task: CP continuous probabilistic, DP discrete probabilistic, "
         "CD continuous deterministic, DD discrete deterministic. In discrete tasks the learning target_x_gu is the position of the 100% reward target "
         "(-12 GU) as the authors share it; continuous tasks have no target on learning trials.",
         "Clamps are coded pert_type reward_clamp (pert_value 1 success, 0 fail). group is child (under 18) or adult. setting is lab for the 16 "
         "participants the authors list as tested in person (continuous probabilistic task); everyone else was online. participants.csv holds the authors' summary values."],
    differences=["Block boundaries (baseline 1-20, learning 21-120, clamps, single target) are taken from the authors' Readme; target_x_gu in the discrete "
                 "learning block is the single value the authors store (the correct button), not the seven targets' positions."],
)
stdlib.release(ID, "reward_swipe", rows, meta, checks, participants=parts, participant_columns=pc)
