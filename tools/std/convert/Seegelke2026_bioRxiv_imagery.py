"""Seegelke & Heed (2026), bioRxiv: motor imagery and execution in history-dependent biases in reaching. Three mouse
experiments (PsychoPy files A061 = Experiment 1, A054 = Experiment 2, A069 = Experiment 3): a prime reach (executed, imagined,
or withheld) followed by a probe reach to the same target, with or without an obstacle. One row per reach, 90-column format.

Run from the site folder: python3 -I tools/std/convert/Seegelke2026_bioRxiv_imagery.py
Raw files (untrusted data, read only) in raw_extra/Seegelke2026_bioRxiv_imagery/ from https://doi.org/10.5281/zenodo.22251643:
  A061_data.csv, A054_data.csv, A069_data.csv and *_SubjInfo.xlsx (demographics, notes, imagery questionnaires)
Not converted: the *_Data_complete_*.mat trajectory files (large), the imagery questionnaires, and Data_Roberts_2025_EBR.xlsx (data of
a different paper, Roberts et al. 2025, that the authors re-analysed).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
import Seegelke2026_bioRxiv_imagery_mouse as M  # noqa: E402

ID = "Seegelke2026_bioRxiv_imagery"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

EXPS = [  # file, experiment label, non-go trial type, excluded participants, paper experiment number
    ("A061", "Experiment1_execution_imagery_mixed_blocks", "imagery", {7: "did not complete the experiment (stopped after 9 blocks) and did not follow instructions"}, 1),
    ("A054", "Experiment2_execution_nomovement_mixed_blocks", "no_movement", {28: "did not follow instructions"}, 2),
    ("A069", "Experiment3_execution_imagery_separate_blocks", "imagery", {}, 3),
]


def subnum(x):
    if isinstance(x, str):
        d = "".join(ch for ch in x if ch.isdigit())
        return int(d)
    return int(x)


rows = []
stats = {}
for fid, exp, label, excl, k in EXPS:
    df = pd.read_csv(RAW / f"{fid}_data.csv", low_memory=False)
    df = M.prepare(df)
    di = pd.read_excel(RAW / f"{fid}_SubjInfo.xlsx", sheet_name="Demographics")
    di.columns = ["id", "age", "sex", "hand", "ehi"]
    demo = {}
    for _, r in di.iterrows():
        n = subnum(r["id"])
        if fid == "A054" and n == 31:
            n = 999  # the deposit's 31st participant carries code 999 in the data file and S31 in the demographics
        demo[n] = (float(r["age"]), {"m": "M", "f": "F"}.get(str(r["sex"]).lower(), "other"),
                   {"r": "right", "l": "left", "a": "ambi"}.get(str(r["hand"]).lower(), ""))
    checks.internal(f"{fid}: every participant in the data has demographics", set(df.subID.unique()) <= set(demo))
    # sign check from raw coordinates (original screen frame, y up): hand direction minus target direction
    sub = df.dropna(subset=["x_pos_HandMovOnset_Probe", "y_pos_HandOutOfStart_Probe", "ReachDiff2_sig_Probe"])
    dx = sub.x_pos_HandOutOfStart_Probe - sub.x_pos_HandMovOnset_Probe
    dy = sub.y_pos_HandOutOfStart_Probe - sub.y_pos_HandMovOnset_Probe
    tx = 0.0 - sub.x_pos_HandMovOnset_Probe  # hand positions are stored in the frame rotated so that the target is at (0, 8)
    ty = 8.0 - sub.y_pos_HandMovOnset_Probe
    calc = M.wrap(np.degrees(np.arctan2(dy, dx)) - np.degrees(np.arctan2(ty, tx)))
    err = np.abs(M.wrap(calc - sub.ReachDiff2_sig_Probe))
    checks.internal(f"{fid}: ReachDiff2_sig (hand_angle) equals the counter-clockwise angle (in the stored target-up frame) of the first "
                    "movement relative to the direction to the target, recomputed from the stored positions, modulo 360",
                    float(np.mean(err < 1e-3)) > 0.999,
                    f"{len(sub)} probes, max diff {float(err.max()):.2e} deg")
    if fid == "A069":
        rr = M.trial_rows(df, ID, exp, label, "E3_S", demo, excl, order_key=lambda g: {
            e: i for i, e in enumerate(sorted(g.expName.unique(), key=lambda e: g[g.expName == e].expStart.iloc[0]))})
    else:
        rr = M.trial_rows(df, ID, exp, label, f"E{k}_S", demo, excl)
    rows += rr
    stats[fid] = (df, demo, excl, rr)

# order_key plumbing above returns a callable per group; trial_rows maps expName with it
# ---------------------------------------------------------------- checks against the paper
P = {"A061": dict(n=31, f=15, m=16, age=(23.0, 3.48), rng=(18, 35), stop=773, slow=(152, 168), obs=(161, 296), onset=(127, 198), multi=806, total=2479, rtmt=471),
     "A054": dict(n=30, f=18, m=12, age=(24.4, 6.06), rng=(18, 50), stop=358, slow=(47, 81), obs=(89, 264), onset=(144, 178), multi=666, total=1756, rtmt=268),
     "A069": dict(n=30, f=27, m=3, age=(22.8, 1.77), rng=(20, 26), stop=374, slow=(35, 289), obs=(176, 280), onset=(426, 482), multi=771, total=2632, rtmt=717)}
SRC = "Methods, Participants and Data processing and analysis"
for fid, exp, label, excl, k in EXPS:
    df, demo, excl, rr = stats[fid]
    p = P[fid]
    an = [s for s in df.subID.unique() if s not in excl]
    ages = [demo[s][0] for s in an]
    sexes = [demo[s][1] for s in an]
    checks.published(f"Experiment {k}: participants analysed", len(an), p["n"], 0, SRC)
    checks.published(f"Experiment {k}: female participants", sexes.count("F"), p["f"], 0, SRC)
    checks.published(f"Experiment {k}: male participants", sexes.count("M"), p["m"], 0, SRC)
    checks.published(f"Experiment {k}: mean age", np.mean(ages), p["age"][0], 0.06, SRC)
    checks.published(f"Experiment {k}: SD of age", np.std(ages, ddof=1), p["age"][1], 0.006, SRC)
    checks.published(f"Experiment {k}: youngest age", min(ages), p["rng"][0], 0, SRC)
    checks.published(f"Experiment {k}: oldest age", max(ages), p["rng"][1], 0, SRC)
    m = df[(~df._practice) & (~df.subID.isin(excl))]
    checks.published(f"Experiment {k}: trials where the cursor left the start on a withheld/imagery trial", int(m.Error_stop_signal.sum()), p["stop"], 0, SRC)
    checks.published(f"Experiment {k}: too-slow trials, prime", int(m.Error_too_slow_prime.sum()), p["slow"][0], 0, SRC)
    checks.published(f"Experiment {k}: too-slow trials, probe", int(m.Error_too_slow_probe.sum()), p["slow"][1], 0, SRC)
    checks.published(f"Experiment {k}: obstacle hits, prime", int(m.Error_obstacle_prime.sum()), p["obs"][0], 0, SRC)
    checks.published(f"Experiment {k}: obstacle hits, probe", int(m.Error_obstacle_probe.sum()), p["obs"][1], 0, SRC)
    checks.published(f"Experiment {k}: movement onset not detected, prime", int(m.Error_MovOnsetDetection_Prime.sum()), p["onset"][0], 0, SRC)
    checks.published(f"Experiment {k}: movement onset not detected, probe", int(m.Error_MovOnsetDetection_Probe.sum()), p["onset"][1], 0, SRC)
    checks.published(f"Experiment {k}: trials with more than two velocity peaks (not already excluded)",
                     int((m._multipeak & ~m._err).sum()), p["multi"], 0, SRC)
    tr = {(r["subject"], r["trial_orig"]) for r in rr if r["excluded_by_authors"] == "True" and r["phase"] != "practice"
          and r["subject_excluded_by_authors"] == "False" and r["trial_kind"] in ("reach", "other")}
    ex1 = {(r["subject"], r["trial_orig"]) for r in rr if r["excluded_by_authors"] == "True" and r["phase"] != "practice"
           and r["subject_excluded_by_authors"] == "False"}
    # trials excluded for the error / multi-peak reasons: re-derive from the frame (the rt/mt rule comes second in the paper)
    first = (m._err | m._multipeak)
    checks.published(f"Experiment {k}: trials excluded before the RT/MT rule", int(first.sum()), p["total"], 0, SRC)
    checks.published(f"Experiment {k}: trials removed by RT < 100 ms or MT > 1000 ms", int((m._rtmt & ~first).sum()), p["rtmt"], 0, SRC)
    checks.internal(f"Experiment {k}: converted rows flag every analysed-trial exclusion (rows excluded = first + RT/MT rules)",
                    len(ex1) == int((first | m._rtmt).sum()), f"{len(ex1)} trials flagged")
    checks.internal(f"Experiment {k}: two rows per trial", len(rr) == 2 * len(df))

nrows = len(rows)
checks.internal("hand angles within (-180, 180]", all(-180 < float(r["hand_angle"]) <= 180 for r in rows if r.get("hand_angle")))
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, lab="Heed (University of Salzburg)",
    data_url="https://doi.org/10.5281/zenodo.22251643", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor 90-column reaching format; one row per prime or probe reach; participant codes replaced; "
            "imagery questionnaires, trajectory files and the Roberts et al. 2025 table not included.",
    unit_note="reach (prime or probe)",
    how=[
        "Each trial has a prime and a probe reach to the same one of 8 targets (22.5 to 337.5 deg, 8 cm); both are rows, with the "
        "same trial_orig. On imagery / no-movement trials the prime is not a reach (trial_kind other, red_target in cue_elements); "
        "obstacle in cue_elements means an obstacle stood between start and target. block names the block, the role and the trial type "
        "(execution, imagery or no_movement). Experiment 1 mixes execution and imagery trials, Experiment 2 mixes execution and "
        "no-movement trials, Experiment 3 presents them in separate blocks (order by the recorded start time of each part).",
        "hand_angle is the authors' ReachDiff2_sig: the angle between the first movement (from movement onset to where the cursor left "
        "the start) and the direction from the cursor at movement onset to the target, counter-clockwise positive. The sign was "
        "recomputed from the stored positions (which are in a frame rotated so that the target is straight up). That the rotation preserves handedness is assumed (a rigid rotation to the 'up' target); the deposit gives no original-frame hand positions to confirm it, and the sign convention is therefore inferred. The paper uses its absolute value ('initial reach error').",
        "excluded_by_authors marks the practice block and the trials the authors removed: a withheld-movement error, too-slow reaction, obstacle hit, "
        "undetected movement onset or more than two velocity peaks (their Error_Any and peak counts), and reaches with RT < 100 ms or MT > 1000 ms; "
        "these rules reproduce the paper's exclusion counts exactly. Reaches made on withheld trials are not given a hand angle.",
        "The data file for Experiment 2 lists 31 participants (one with code 999); the demographics sheet lists the 31st as S31, which is how the two were matched. "
        "Participant overlap between the three experiments is not stated, so each experiment's participants are coded as different people.",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks)
