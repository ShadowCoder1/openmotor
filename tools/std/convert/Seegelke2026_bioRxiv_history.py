"""Seegelke & Heed (2026), bioRxiv: motor planning and execution establish distinct feedforward and feedback motor histories.
Ten experiments in the deposit, all converted to the 90-column reaching format, one row per reach (prime or probe):
  mouse (PsychoPy) obstacle avoidance: A038 = Experiment Set 1 OA (stop-signal), A054 = Set 4 OA (no-go), A072 = Set 4 OA control;
  Kinarm hand selection (HS): A039 = Set 1 (stop-signal), A056 = Set 2 (mechanical block), B098 = Set 3 (passive), A096 = Set 4 (no-go);
  Kinarm obstacle avoidance (OA): A073 = Set 2 (mechanical block), A074 = Set 3 (passive), A080 = Set 3 control (vision on / off).
The Kinarm rows are built by Seegelke2026_bioRxiv_history_kinarm.py, the mouse rows by Seegelke2026_bioRxiv_history_mouse.py.
The .mat files of the deposit (full trajectories, force channels) are not converted: they hold continuous signals for plots.

Run from the site folder: python3 -I tools/std/convert/Seegelke2026_bioRxiv_history.py
Raw files (untrusted data, read only) in raw_extra/Seegelke2026_bioRxiv_history/ from https://doi.org/10.5281/zenodo.22257057.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
import Seegelke2026_bioRxiv_history_mouse as M  # noqa: E402
import Seegelke2026_bioRxiv_history_kinarm as K  # noqa: E402

ID = "Seegelke2026_bioRxiv_history"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

EXPS = [  # file, experiment label, non-go label, prefix, excluded, set name
    ("A038", "SetA_OA_stop_signal", "stop_signal", "S1_", {}, "Experiment Set 1 OA task"),
    ("A054", "SetD_OA_nogo", "nogo", "S4_", {28: "did not follow instructions"}, "Experiment Set 4 OA task"),
    ("A072", "SetD_OA_control_separate_blocks", "nogo", "S4c_", {}, "Experiment Set 4 OA control task"),
]
# paper (Data exclusion): stop-signal errors, slow prime/probe, obstacle prime/probe, onset prime/probe, multi-peak, total, RT/MT rule
P = {"A038": (2302, 153, 174, 127, 230, 149, 166, 768, 3851, 187), "A054": (358, 47, 81, 89, 264, 144, 178, 666, 1756, 268),
     "A072": (349, 18, 134, 183, 294, 272, 258, 683, 2084, 626)}
DEMO = {"A038": dict(n=30, f=24, age=(22.5, 4.5), ehi=(84.3, 20.3)), "A054": dict(n=30, f=18, age=(24.4, 6.1), ehi=(90.0, 14.0)),
        "A072": dict(n=30, f=21, age=(22.4, 3.2), ehi=(72.2, 21.1))}
SRC = "Methods, Specific task setups and procedures; Data exclusion"

rows = []
keep = {}
for fid, exp, label, pre, excl, sname in EXPS:
    df = M.prepare(pd.read_csv(RAW / f"{fid}_data.csv", low_memory=False))
    di = pd.read_excel(RAW / f"{fid}_SubjInfo.xlsx", sheet_name="Demographics").iloc[:, :5]
    di.columns = ["id", "age", "sex", "hand", "ehi"]
    demo, ehi = {}, {}
    for _, r in di.iterrows():
        n = int("".join(c for c in str(r["id"]) if c.isdigit()))
        if fid == "A054" and n == 31:
            n = 999
        demo[n] = (float(r["age"]), {"m": "M", "f": "F"}.get(str(r["sex"]).lower(), "other"),
                   {"r": "right", "l": "left", "a": "ambi"}.get(str(r["hand"]).lower(), ""))
        ehi[n] = float(r["ehi"])
    checks.internal(f"{fid}: every participant in the data has demographics", set(df.subID.unique()) <= set(demo))
    sub = df.dropna(subset=["x_pos_HandMovOnset_Probe", "y_pos_HandOutOfStart_Probe", "ReachDiff2_sig_Probe"])
    dx = sub.x_pos_HandOutOfStart_Probe - sub.x_pos_HandMovOnset_Probe
    dy = sub.y_pos_HandOutOfStart_Probe - sub.y_pos_HandMovOnset_Probe
    err = np.abs(M.wrap(M.wrap(np.degrees(np.arctan2(dy, dx)) - np.degrees(np.arctan2(8.0 - sub.y_pos_HandMovOnset_Probe, -sub.x_pos_HandMovOnset_Probe)))
                        - sub.ReachDiff2_sig_Probe))
    checks.internal(f"{fid}: ReachDiff2_sig equals the counter-clockwise angle of the first movement relative to the direction to the "
                    "target (stored target-up frame), recomputed from the stored positions", float(np.mean(err < 1e-3)) > 0.999)
    if fid == "A072":
        def order_key(g):
            first = "go" if g.Order.iloc[0] == "Go-Stop" else "stop"
            return {f"A072_Experiment_{first}": 0, f"A072_Experiment_{'stop' if first == 'go' else 'go'}": 1}
        rr = M.trial_rows(df, ID, exp, label, pre + "S", demo, excl, order_key=order_key)
    else:
        rr = M.trial_rows(df, ID, exp, label, pre + "S", demo, excl)
    rows += rr
    keep[fid] = (df, demo, ehi, excl, rr)

for fid, exp, label, pre, excl, sname in EXPS:
    df, demo, ehi, excl, rr = keep[fid]
    p, d = P[fid], DEMO[fid]
    an = [s for s in df.subID.unique() if s not in excl]
    ages = [demo[s][0] for s in an]
    checks.published(f"{sname}: participants analysed", len(an), d["n"], 0, SRC)
    checks.published(f"{sname}: female participants", sum(demo[s][1] == "F" for s in an), d["f"], 0, SRC)
    checks.published(f"{sname}: mean age", np.mean(ages), d["age"][0], 0.06, SRC)
    checks.published(f"{sname}: SD of age", np.std(ages, ddof=1), d["age"][1], 0.06, SRC)
    if fid != "A072":  # the paper's control-task EHI (72.2) does not reproduce from the demographics sheet (62.7); listed in differences
        checks.published(f"{sname}: mean EHI score", np.mean([ehi[s] for s in an]), d["ehi"][0], 0.06, SRC)
    m = df[(~df._practice) & (~df.subID.isin(excl))]
    first = m._err | m._multipeak
    for what, val, exp_ in (("withheld-trial movement errors", int(m.Error_stop_signal.sum()), p[0]),
                            ("too-slow prime", int(m.Error_too_slow_prime.sum()), p[1]), ("too-slow probe", int(m.Error_too_slow_probe.sum()), p[2]),
                            ("obstacle hit prime", int(m.Error_obstacle_prime.sum()), p[3]), ("obstacle hit probe", int(m.Error_obstacle_probe.sum()), p[4]),
                            ("onset not detected prime", int(m.Error_MovOnsetDetection_Prime.sum()), p[5]),
                            ("onset not detected probe", int(m.Error_MovOnsetDetection_Probe.sum()), p[6]),
                            ("more than two velocity peaks (not already excluded)", int((m._multipeak & ~m._err).sum()), p[7]),
                            ("trials excluded before the RT/MT rule", int(first.sum()), p[8]),
                            ) + ((("trials removed by the RT/MT rule", int((m._rtmt & ~first).sum()), p[9]),) if fid != "A072" else ()):
        checks.published(f"{sname}: {what}", val, exp_, 0, SRC + " (Data exclusion)")
    flagged = {(r["subject"], r["trial_orig"]) for r in rr if r["excluded_by_authors"] == "True" and r["phase"] != "practice"
               and r["subject_excluded_by_authors"] == "False"}
    checks.internal(f"{fid}: converted rows flag exactly the removed trials", len(flagged) == int((first | m._rtmt).sum()))
    checks.internal(f"{fid}: two rows per trial", len(rr) == 2 * len(df))


# ---------------------------------------------------------------- Kinarm experiments (one file at a time)
OA_R = lambda: None
KIN = [  # file, experiment label, kind, person prefix, config, paper facts
    dict(fid="A039", exp="SetA_HS_stop_signal", kind="hs", pre="K1H_S", sname="Experiment Set 1 HS task", n=39, f=25, age=(25.6, 7.6),
         ehi={"r": (33, 81.3, 18.7), "l": (4, -66.3, 34.9), "b": (2, 31.0, 43.8)},
         cfg=dict(nongo="stop_signal", nongo_cue="stop_signal", excl_col="Error_Any_New", block_col="BlockNo",
                  rt_cols={"prime": "RT_Prime", "probe": "RT_Probe"}, mt_cols={"prime": "MT_Prime", "probe": "MT_Probe"}),
         counts=[("moved out of the start on stop trials", "Error_Stop_Trial", 3499), ("too-slow prime", "Error_Slow_Prime_New", 191),
                 ("too-slow probe", "Error_Slow_Probe_New", 177), ("wrong hand prime", "Error_Hand_Prime", 861),
                 ("wrong hand probe", "Error_Hand_Probe", 1365), ("onset not detected prime", "Error_Prime_Movement", 0),
                 ("onset not detected probe", "Error_Probe_Movement", 27)], total=5761),
    dict(fid="A056", exp="SetB_HS_mechanical_block", kind="hs", pre="K2H_S", sname="Experiment Set 2 HS task", n=35, f=13, age=(26.3, 8.5),
         ehi={"r": (31, 95.0, 7.7), "l": (3, -57.7, 62.7), "b": (1, 20.0, None)},
         cfg=dict(nongo="mechanical_block", nongo_cue="mechanical_block", excl_col="Error_Any_New", block_col="BlockNo",
                  rt_cols={"prime": "RT_Prime", "probe": "RT_Probe"}, mt_cols={"prime": "MT_Prime", "probe": "MT_Probe"}),
         counts=[("too-slow prime", "Error_Slow_Prime_New", 59), ("too-slow probe", "Error_Slow_Probe_New", 55),
                 ("wrong hand prime", "Error_Hand_Prime", 1458), ("wrong hand probe", "Error_Hand_Probe", 1405),
                 ("onset not detected prime", "Error_Prime_Movement", 0), ("onset not detected probe", "Error_Probe_Movement", 15)], total=2843),
    dict(fid="A096", exp="SetD_HS_nogo", kind="hs", pre="K4H_S", sname="Experiment Set 4 HS task", n=29, f=24, age=(24.1, 5.5),
         ehi={"r": (25, 86.3, 15.5), "l": (3, -90.0, 17.3), "b": (1, 0.0, None)},
         cfg=dict(nongo="nogo", nongo_cue="red_stop_circle", excl_col="Error_Any_New", block_col="BlockNo",
                  rt_cols={"prime": "RT_Prime", "probe": "RT_Probe"}, mt_cols={"prime": "MT_Prime", "probe": "MT_Probe"}),
         counts=[("moved on no-go trials", "Error_Stop_Trial", 85), ("too-slow prime", "Error_Slow_Prime_New", 88),
                 ("too-slow probe", "Error_Slow_Probe_New", 131), ("wrong hand prime", "Error_Hand_Prime", 970),
                 ("wrong hand probe", "Error_Hand_Probe", 1316), ("onset not detected prime", "Error_Prime_Movement", 0),
                 ("onset not detected probe", "Error_Probe_Movement", 19)], total=None),
    dict(fid="B098", exp="SetC_HS_passive", kind="hs", pre="K3H_S", sname="Experiment Set 3 HS passive task", n=28, f=15, age=(23.8, 3.0),
         ehi=None, ehi_all=(98.4, 4.7),
         cfg=dict(nongo=None, hidden_targets=True, excl_col="Error_Any", block_col="BlockNo",
                  rt_cols={"prime": "RT_Prime", "probe": "RT_Probe"}, mt_cols={"prime": "MT_Prime", "probe": "MT_Probe"}),
         counts=[("too-slow prime", "Error_Slow_Prime", 46), ("too-slow probe", "Error_Slow_Probe", 72),
                 ("wrong hand prime", "Error_Hand_Prime", 290), ("wrong hand probe", "Error_Hand_Probe", 864),
                 ("force exerted on passive trials", "Error_Force", 175)], total=2733),
    dict(fid="A073", exp="SetB_OA_mechanical_block", kind="oa", pre="K2O_S", sname="Experiment Set 2 OA task", n=30, f=15, age=(25.0, 2.4),
         ehi={"r": (25, 92.2, 16.8), "l": (4, -81.6, 21.7), "b": (1, 47.4, None)},
         cfg=dict(nongo="mechanical_block", excl_col="Error_Any", block_col="BlockNumber", practice_col="Session", practice_value="Training",
                  rt_cols={"prime": "RT_Prime_vel", "probe": "RT_Probe_vel"}, mt_cols={"prime": "MT_Prime_vel", "probe": "MT_Probe_vel"},
                  target_radius_deg=True),
         counts=[("too-slow prime", "Error_SlowPrime", 815), ("too-slow probe", "Error_SlowProbe", 987),
                 ("obstacle hit prime", "Error_ObsHitPrime", 113), ("obstacle hit probe", "Error_ObsHitProbe", 339)], total=2415),
    dict(fid="A074", exp="SetC_OA_passive", kind="oa", pre="K3O_S", sname="Experiment Set 3 OA passive task", n=30, f=23, age=(22.4, 4.4),
         ehi={"r": (28, 87.4, 13.3), "l": (2, -95.0, 7.1)}, group_col="ColorAssociation",
         cfg=dict(nongo=None, passive_types=True, feedback_col="Feedback", excl_col="Error_Any", block_col="BlockNumber", group_col="ColorAssociation",
                  rt_cols={"prime": "RT_Prime_vel", "probe": "RT_Probe_vel"}, mt_cols={"prime": "MT_Prime_vel", "probe": "MT_Probe_vel"},
                  target_radius_deg=True),
         counts=[("too-slow prime", "Error_SlowPrime", 1107), ("too-slow probe", "Error_SlowProbe", 900),
                 ("obstacle hit prime", "Error_ObsHitPrime", 77), ("obstacle hit probe", "Error_ObsHitProbe", 118)], total=4196),
    dict(fid="A080", exp="SetC_OA_vision_control", kind="oa", count_all=True, pre="K3C_S", sname="Experiment Set 3 OA control task", n=30, f=27, age=(21.5, None),
         ehi={"r": (24, None, None), "l": (6, None, None)},
         cfg=dict(nongo=None, vision_off_value="vision_off", excl_col="Error_Any", block_col=None, practice_col="Session",
                  practice_value="Training",
                  rt_cols={"prime": "RT_Prime_vel", "probe": "RT_Probe_vel"}, mt_cols={"prime": "MT_Prime_vel", "probe": "MT_Probe_vel"},
                  target_radius_deg=True),
         counts=[("too-slow prime", "Error_SlowPrime", 696), ("too-slow probe", "Error_SlowProbe", 69),
                 ("obstacle hit prime", "Error_ObsHitPrime", 154), ("obstacle hit probe", "Error_ObsHitProbe", 156)], total=1338),
]
OA_RADIUS = float(np.degrees(np.arctan(0.005 / 0.1)))  # 0.5 cm target radius at 10 cm (TargetPos_r1_vis / start-target distance)
frames = []
TOT = {}
for E in KIN:
    fid, kind, cfg = E["fid"], E["kind"], dict(E["cfg"])
    if cfg.get("target_radius_deg"):
        cfg["target_radius_deg"] = OA_RADIUS
    df = pd.read_csv(RAW / f"{fid}_data.csv", low_memory=False)
    sheets = pd.read_excel(RAW / f"{fid}_SubjInfo.xlsx", sheet_name=None)
    di = sheets["Demographics"].iloc[:, :5]
    di.columns = ["id", "age", "sex", "hand", "ehi"]
    demo, ehi, hand_of = {}, {}, {}
    for _, r in di.iterrows():
        n_ = int("".join(c for c in str(r["id"]) if c.isdigit()))
        demo[n_] = (float(r["age"]), {"m": "M", "f": "F"}.get(str(r["sex"]).lower(), "other"),
                    {"r": "right", "l": "left", "b": "ambi"}.get(str(r["hand"]).lower(), ""))
        ehi[n_] = float(str(r["ehi"]).replace(",", "."))
        hand_of[n_] = str(r["hand"]).lower()
    notes = next(v for k, v in sheets.items() if k != "Demographics")
    excl = {}
    for _, r in notes.iterrows():
        txt = str(r.iloc[1])
        if txt.upper().startswith("EXCLUDE"):
            excl[int("".join(c for c in str(r.iloc[0]) if c.isdigit()))] = txt.split(":", 1)[-1].strip().replace(">", "over ").replace("errros", "errors")
    if "BlockNumber" in df.columns:
        df["BlockNumber"] = df.groupby("subjNr").BlockNumber.transform(lambda x: x.ffill().bfill())
    checks.internal(f"{fid}: every participant in the data has demographics", set(df.subjNr.astype(int).unique()) <= set(demo))
    if kind == "oa":
        w = K.wrap
        okf = []
        for role in ("prime", "probe"):
            R = role.capitalize()
            a = -df[f"ReachError100_TC_{R}_sig_adj"]
            h = K.hand_angles(df, role)
            m = a.notna() & h.notna()
            okf.append(float((np.abs(w(h - a))[m] < 1e-3).mean()))
        checks.internal(f"{fid}: hand_angle equals minus the authors' ReachError100_TC_sig_adj (prime and probe), recomputed from the stored directions",
                        min(okf) > 0.999)
    else:
        med = float(np.nanmedian(np.abs(pd.concat([K.hand_angles(df, "prime"), K.hand_angles(df, "probe")]))))
        checks.internal(f"{fid}: median absolute initial direction error is under 15 degrees (sign and frame are plausible)", med < 15)
    rr = K.kinarm_rows(df, ID, E["exp"], kind, E["pre"], demo, excl, cfg)
    checks.internal(f"{fid}: two rows per trial", len(rr) == 2 * len(df))
    # published numbers: analysed participants only
    an = [s for s in sorted(df.subjNr.astype(int).unique()) if s not in excl]
    ages = [demo[s][0] for s in an]
    S_ = "Methods, Participants; Data exclusion"
    checks.published(f"{E['sname']}: participants analysed", len(an), E["n"], 0, S_)
    checks.published(f"{E['sname']}: female participants", sum(demo[s][1] == "F" for s in an), E["f"], 0, S_)
    checks.published(f"{E['sname']}: mean age", np.mean(ages), E["age"][0], 0.06, S_)
    if E["age"][1] is not None:
        checks.published(f"{E['sname']}: SD of age", np.std(ages, ddof=1), E["age"][1], 0.06, S_)
    if E.get("ehi"):
        for hk, (hn, hm, hs) in E["ehi"].items():
            v = [ehi[s] for s in an if hand_of[s] == hk]
            nm = {"r": "right-handed", "l": "left-handed", "b": "ambidextrous"}[hk]
            checks.published(f"{E['sname']}: number of {nm} participants", len(v), hn, 0, S_)
            if hm is not None:
                checks.published(f"{E['sname']}: mean EHI of {nm} participants", np.mean(v), hm, 0.06, S_)
            if hs is not None:
                checks.published(f"{E['sname']}: SD of EHI of {nm} participants", np.std(v, ddof=1), hs, 0.06, S_)
    if E.get("ehi_all"):
        v = [ehi[s] for s in an]
        checks.published(f"{E['sname']}: mean EHI", np.mean(v), E["ehi_all"][0], 0.06, S_)
        checks.published(f"{E['sname']}: SD of EHI", np.std(v, ddof=1), E["ehi_all"][1], 0.06, S_)
    m = df[df.subjNr.astype(int).isin(an)]
    if cfg.get("practice_col") and not E.get("count_all"):
        m = m[m[cfg["practice_col"]] != cfg["practice_value"]]
    for what, col, exp_ in E["counts"]:
        checks.published(f"{E['sname']}: {what}", int(m[col].sum()), exp_, 0, "Methods, Data exclusion")
    if kind == "oa":
        mp = (m.num_peak_vel_Prime > 2) | (m.num_peak_vel_Probe > 2)
        tot = int(((m[cfg["excl_col"]] == 1) | (m["Error_Trial"] == 1) | mp).sum())
    else:
        tot = int((m[cfg["excl_col"]] == 1).sum())
    TOT[fid] = tot
    if E["total"] is not None:
        checks.published(f"{E['sname']}: trials excluded in total", tot, E["total"], 0, "Methods, Data exclusion")
    frames.append(pd.DataFrame(rr))
    del df, rr
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json", encoding="utf-8")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, lab="Heed (University of Salzburg)",
    data_url="https://doi.org/10.5281/zenodo.22257057", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted the per-trial CSV files of all 10 experiments (3 mouse, 7 Kinarm) to the OpenMotor 90-column reaching format; "
            "one row per prime or probe movement; participant codes replaced. The .mat files with full trajectories are not included.",
    unit_note="reach (prime or probe)",
    how=[
        "Ten experiments, one file each. Mouse obstacle avoidance (PsychoPy): A038 (Experiment Set 1, stop-signal), A054 (Set 4, no-go), "
        "A072 (Set 4 control, no-go and go trials in separate blocks). Kinarm hand selection: A039 (Set 1, stop-signal), A056 (Set 2, "
        "mechanical block), B098 (Set 3, passive movements, targets visible or hidden), A096 (Set 4, no-go). Kinarm obstacle avoidance: "
        "A073 (Set 2, mechanical block), A074 (Set 3, passive prime with or without cursor feedback), A080 (Set 3 control, active, "
        "cursor, target and obstacle visible or hidden after the go cue). experiment names the set and task.",
        "Each trial has a prime and a probe movement, both are rows with the same trial_orig. When the prime is not a reach (stop-signal, "
        "no-go, mechanical block, passive) its row has trial_kind other (passive: proprio_probe) and no hand_angle, rt or mt; block names "
        "block, role and trial type. cue_elements lists obstacle, hand_color (hand selection: target colour tells the hand), stop_signal, "
        "red_stop_circle, mechanical_block, targets_hidden, vision_off_at_go_cue. On A074 fb_visible is none when the file says NoFB, "
        "on A080 vision_off trials the cursor, target and obstacle disappear at the go cue (fb_visible none).",
        "Hand selection: hand_used is the hand the colour cue asked for (wrong-hand movements are errors and are flagged). In the "
        "obstacle-avoidance Kinarm files the hand is not recorded, so hand_used is empty. target_angle is the direction from the start "
        "to the target in the robot frame (x right, y away from the body, counter-clockwise positive).",
        "hand_angle is the direction of the hand 100 ms after it left the start minus the direction from the hand's start position to "
        "the target centre, counter-clockwise positive, from the authors' V_HandStart_100_Ori and V_HandStartTarget_Ori columns. In the "
        "obstacle-avoidance files this is exactly minus the authors' ReachError100_TC_sig_adj (they define the error as target minus "
        "hand), checked in the converter; in the hand-selection files, which lack that column, the same formula is applied. The 100 ms "
        "point and the sign were chosen by us. For the mouse files hand_angle is the authors' ReachDiff2_sig (first movement), recomputed "
        "from the stored positions; that this rotation preserves handedness is assumed, so the sign is inferred.",
        "excluded_by_authors marks the practice block or session and the trials removed by the paper's rules. Mouse: errors, more than "
        "two velocity peaks, RT < 100 or > 1200 ms, MT > 1000 ms. Kinarm hand selection: the authors' Error_Any_New (Error_Any in B098) flag "
        "with the participants the authors excluded. Kinarm obstacle avoidance: the authors' Error_Any flag, passive trials skipped because "
        "the matching active trial had an error (Error_Trial), and more than two velocity peaks (this last rule is not flagged in the file "
        "but reproduces the paper's totals exactly). Participants the authors excluded (high error rates, technical problems, "
        "incomplete sessions) are marked subject_excluded_by_authors with their reason. A054 is the same data file as in the deposit of "
        "the companion imagery paper.",
    ],
)
meta["differences"] = [
    "Set 4 control task (mouse): the paper reports a mean EHI of 72.2; the demographics sheet gives 62.7 for the 30 participants. "
    "The RT/MT rule removes 624 trials from the deposit's values where the paper reports 626 (the other nine counts per experiment match exactly).",
    "Set 4 hand selection (A096): the paper prints 2543 excluded trials (11.0%); the authors' flags give 2453, which is 11.0% of the 22,272 "
    "trials, so the printed total is probably a transposition. All category counts match, and the total is not used as a check.",
    "Set 3 OA control (A080): the paper's exclusion counts reproduce only when the 64 practice trials per participant are included (they "
    "are in the rows, flagged as practice). The SD of age is 1.87 (paper 1.8), and the EHI of the 6 left-handed participants is -70.0 with "
    "SD 26.9 (paper -85.8, SD 5.6; right-handed 93.5, paper 93.7); these are not used as checks.",
    "Kinarm obstacle avoidance (A073, A074, A080): the paper's totals include trials with more than two velocity peaks, which the "
    "deposit's Error_Any flag does not contain; we flag them. Other paper sub-counts (non-smooth trajectories, movement time) are not "
    "separate columns and are not checked one by one.",
    "In the Kinarm files the deposit records no hand for the obstacle-avoidance task, and for passive prime "
    "movements our rows leave hand_angle, rt and mt empty."]
import itertools
def _all_rows():
    yield from rows
    for fr in frames:
        cols = list(fr.columns)
        for tup in fr.itertuples(index=False, name=None):
            yield dict(zip(cols, tup))
stdlib.release(ID, "reach90", _all_rows(), meta, checks)
