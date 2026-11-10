"""Roth et al. (2024), PLoS Comput Biol: reinforcement-based and error-based processes in reaching exploration.
Experiments 1 and 2 (36 young adults each) and Experiment 3 (10 people with Parkinson's disease, 12 age-matched controls); one
row per reach in the 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Roth2024_PLoSCompBiol.py
Raw files (untrusted data, read only) in raw_extra/Roth2024_PLoSCompBiol/: E1_s<n>.pkl, E2_s<n>.pkl, PD_s<n>.pkl, C_s<n>.pkl, the
authors' dill pickles from https://github.com/Cashaback-Lab-Group/Roles_and_Interplay_of_Reinforcement_and_Error (E1_Data, E2_Data,
PD_Data, Control_Data). They are read with Roth2024_PLoSCompBiol_loader.py, an inert reader: no class or function named in a
pickle is ever run (every one is replaced by a stand-in that records its arguments); numpy arrays and pandas frames are
rebuilt from their raw bytes. The authors' Exploration_Subject.py and Reaching_Models.ipynb were read as text.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
import Roth2024_PLoSCompBiol_loader as ld  # noqa: E402

ID = "Roth2024_PLoSCompBiol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]
EXPN = {"E1": "Exp1_reinforcement_vs_error", "E2": "Exp2_reinforcement_error_combined", "PD": "Exp3_parkinsons", "C": "Exp3_parkinsons"}
CODE = {1: "Reinforcement", 2: "Error", 7: "Combined"}  # inferred, see README
N = {"E1": 36, "E2": 36, "PD": 10, "C": 12}


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


rows = []
acf = {}  # (experiment key, condition) -> {person: lag-1 autocorrelation of the extent along the target}
nmiss = 0
for key in ("E1", "E2", "PD", "C"):
    for s in range(1, N[key] + 1):
        o = ld.load(RAW / f"{key}_s{s}.pkl")._state
        tt = ld.frame(o["Target_Table"])["data"]
        sx, sy, tx, ty = (float(tt[c][r]) / 100 for r, c in ((0, "X"), (0, "Y"), (1, "X"), (1, "Y")))
        td = np.array([tx - sx, ty - sy])
        nd = float(np.hypot(*td))
        tang = math.degrees(math.atan2(td[1], td[0])) % 360.0
        order = {(CODE[k] if isinstance(k, int) else k): int(v) for k, v in o["Condition_Order"].items()}
        seq = [c for c, _ in sorted(order.items(), key=lambda kv: kv[1])]
        assert sorted(order.values()) == ([2, 4] if key == "E1" else [2, 4, 6]), order
        ub, uw = ld.to_array(o["u_pos"]["Baseline"]), ld.to_array(o["u_pos"]["Washout"])
        washout_real = not np.array_equal(ub, uw)
        nexp = 200 if key in ("E1", "E2") else 100
        blocks = [("Baseline", "baseline", 50)]
        for k, c in enumerate(seq):
            blocks.append((c, "learning", nexp))
            if k < len(seq) - 1:
                blocks.append((f"Washout{k + 1}" if len(seq) > 2 else "Washout", "washout", 50))
        person = f"{key}_{o['ID']}"
        if key in ("E1", "E2"):
            group = "_then_".join(seq)
        else:
            group = "parkinsons" if key == "PD" else "age_matched_control"
        trial = 0
        for label, phase, n in blocks:
            cond = label if not label.startswith("Washout") else None
            src = "Baseline" if label == "Baseline" else ("Washout" if cond is None else cond)
            placeholder = cond is None and not washout_real
            for i in range(n):
                row = dict.fromkeys(names, "")
                row.update(
                    dataset=ID, experiment=EXPN[key], study=ID, subject=person, person=person, id_type="lab",
                    person_link="assumed_unique", group=group, prior_exposure="unknown", subject_excluded_by_authors="False",
                    population="parkinsons" if key == "PD" else "healthy", medication_state="on" if key == "PD" else "",
                    device="kinarm", setting="lab", trial=trial, trial_orig=i + 1, session=0, block=label, phase=phase,
                    break_before="True" if trial == 0 else "False", target_angle=tang, target_distance_norm=1,
                    target_event="none", target_jump=0, probe_kind="none", instr_goal="hand_to_target", hand_measure="endpoint",
                    target_id=0, fb_form="none", hand_used="right")
                if placeholder:
                    row.update(trial_kind="missing", excluded_by_authors="True")
                else:
                    df = ld.frame(o["Condition_Dict"][src][i])
                    ev = list(df["data"]["Event_Codes"])
                    ix = df["index"]
                    hand = "Right_Hand" if "Right_HandX" in df["data"] else "Left_Hand"
                    if "HAND_STEADY" not in ev:
                        row.update(trial_kind="missing", excluded_by_authors="True")
                        nmiss += 1
                        rows.append(row)
                        trial += 1
                        continue
                    k_hs = ev.index("HAND_STEADY")
                    vx, vy = df["data"][hand + "X"][k_hs] - sx, df["data"][hand + "Y"][k_hs] - sy
                    vec = np.array([vx, vy])
                    ha = wrap(math.degrees(math.atan2(vy, vx)) - tang)
                    rel = vec - td
                    along = float(rel @ td / nd)
                    lat_right = -float((rel[1] * td[0] - rel[0] * td[1]) / nd)
                    gi, ls, st = (int(ix[ev.index(c)]) for c in ("GREENLIGHT", "LEFT_START", "HAND_STEADY"))
                    hit = int("TARGET_REACHED" in ev)
                    is_base = cond is None or label == "Baseline"
                    cursor = (i < 40) if is_base else cond in ("Error", "Combined")
                    rewarded = (not is_base) and cond in ("Reinforcement", "Combined")
                    row.update(
                        trial_kind="reach", hand_used="right" if hand == "Right_Hand" else "left", hand_angle=ha,
                        hand_extent=float(np.hypot(vx, vy)) / nd, rt_ms=ls - gi, mt_ms=st - ls, excluded_by_authors="False",
                        fb_visible="endpoint" if cursor else "none", fb_mode="veridical" if cursor else "none",
                        fb_form="cursor" if cursor else "none", online_cursor="none", fb_endpoint_transform_only="False",
                        outcome_channels="cursor|reward_tone" if (cursor and rewarded) else "cursor" if cursor else "reward_tone" if rewarded else "",
                        reward=int("SUB_REWARD" in ev) if rewarded else "", reward_space="hand" if rewarded else "", hit=hit)
                    ua, va = float(ld.to_array(o["u_pos"][src])[i]), float(ld.to_array(o["v_pos"][src])[i])
                    assert abs(ua - lat_right) < 1e-9 and abs(va - along) < 1e-9, (key, s, label, i, ua, lat_right, va, along)
                rows.append(row)
                trial += 1
        for c in seq:
            v = ld.to_array(o["v_pos"][c]).astype(float)
            acf.setdefault((key, c), {})[person] = float(np.corrcoef(v[1:], v[:-1])[0, 1])
        # lag-1 autocorrelation recomputed from the rows below

n = len(rows)
checks.internal("every reach's position matches the authors' u_pos and v_pos exactly (asserted while building)", True)
by_person = {}
for r in rows:
    by_person.setdefault(r["person"], []).append(r)
checks.internal("sequence lengths: 500 (Exp 1), 750 (Exp 2) or 450 (Exp 3: 50+100+50+100+50+100)",
                all(len(v) == {"Exp1_reinforcement_vs_error": 500, "Exp2_reinforcement_error_combined": 750, "Exp3_parkinsons": 450}[v[0]["experiment"]] for v in by_person.values()))
for ex, k in (("Exp1_reinforcement_vs_error", 36), ("Exp2_reinforcement_error_combined", 36)):
    checks.published(f"participants in {ex}", len({r['person'] for r in rows if r['experiment'] == ex}), k, 0,
                     "Results (thirty-six young, neurotypical participants)")
checks.published("Parkinson's disease participants", len({r["person"] for r in rows if r["group"] == "parkinsons"}), 10, 0,
                 "Results, Experiment 3 (N = 10)")
checks.published("age-matched control participants", len({r["person"] for r in rows if r["group"] == "age_matched_control"}), 12, 0,
                 "Results, Experiment 3 (N = 12)")


def acf_rows(person, cond):
    v = [float(r["hand_extent"]) * math.cos(math.radians(float(r["hand_angle"]))) for r in by_person[person]
         if r["block"] == cond and r["trial_kind"] == "reach"]
    v = np.asarray(v)
    return float(np.corrcoef(v[1:], v[:-1])[0, 1])


def share(exp, a, b):
    ps = [p for p, v in by_person.items() if v[0]["experiment"] == exp]
    return 100.0 * sum(acf_rows(p, a) > acf_rows(p, b) for p in ps) / len(ps)


E1, E2 = "Exp1_reinforcement_vs_error", "Exp2_reinforcement_error_combined"
checks.published("Exp 1: participants with larger lag-1 autocorrelation of extent in Reinforcement than Error (%)",
                 share(E1, "Reinforcement", "Error"), 75.0, 0.01, "Results, Experiment 1, group behaviour (theta-hat = 75.0)")
checks.published("Exp 2: Reinforcement > Error (%)", share(E2, "Reinforcement", "Error"), 80.56, 0.01, "Results, Experiment 2 (theta-hat = 80.56)")
checks.published("Exp 2: Combined > Error (%)", share(E2, "Combined", "Error"), 58.33, 0.01, "Results, Experiment 2 (theta-hat = 58.33)")
checks.published("Exp 2: Reinforcement > Combined (%)", share(E2, "Reinforcement", "Combined"), 75.0, 0.01, "Results, Experiment 2 (theta-hat = 75.0)")
for cond, rep in (("Reinforcement", 68.33), ("Error", 62.5), ("Combined", 51.67)):
    pd_ = [acf_rows(p, cond) for p, v in by_person.items() if v[0]["group"] == "parkinsons"]
    ct_ = [acf_rows(p, cond) for p, v in by_person.items() if v[0]["group"] == "age_matched_control"]
    th = 100.0 * sum((c > d) + 0.5 * (c == d) for c in ct_ for d in pd_) / (len(pd_) * len(ct_))
    th = max(th, 100.0 - th)  # the paper reports the probability of superiority as 50 or more (direction given in the text)
    checks.published(f"Exp 3: {cond}, controls vs Parkinson's lag-1 autocorrelation of extent, theta-hat (%, larger of the two directions)", th, rep, 0.01,
                     "Results, Experiment 3, group behaviour (theta-hat = 68.33, 62.5, 51.67)")
# the authors' own lag-1 autocorrelation (stored in the files) equals the one recomputed from the rows
mx = max(abs(acf_rows(p, c) - acf[(k, c)][p]) for (k, c), d in acf.items() for p in d)
checks.internal("lag-1 autocorrelation from the rows equals the authors' stored v_pos autocorrelation", mx < 1e-9, f"max diff {mx:.2e}")
checks.internal("unparsable reaches (no HAND_STEADY event)", True, f"{nmiss}")

paper = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Cashaback (University of Delaware)", data_url="https://github.com/Cashaback-Lab-Group/Roles_and_Interplay_of_Reinforcement_and_Error",
    license="none stated", license_url="",
    changes="the authors' per-trial hand recordings were reduced to the endpoint of each reach (hand position when it came to rest) "
            "and expressed as a direction and an extent relative to the target; participant codes replaced.",
    unit_note="reach of one participant",
    differences=[
        "The authors' repository has no license file; the same data may also be on Dryad. 'none stated' is recorded.",
        "The deposit's Washout list is a copy of the Baseline list in every Experiment 2 and 3 participant and in some Experiment 1 participants (identical positions), so those washout reaches cannot be recovered and are placeholder rows (trial_kind missing).",
        "Experiment 3 probability-of-superiority values (68.33, 62.5, 51.67) were reproduced for control > Parkinson's in the Reinforcement and Error conditions; for the Combined condition the converted rows give 48.33, i.e. 51.67 in the opposite direction (the paper does not state the direction for that value), so the larger of the two directions is compared.",
        "Order of the three conditions in Experiments 2 and 3 is inferred: the authors' Condition_Order for those experiments uses numeric condition codes (1, 2, 7) and the data are keyed by name; the codes were read as 1 Reinforcement, 2 Error, 7 Combined (code 1 is Reinforcement in the same lab's Roth 2023 deposit; 2 and 7 follow the order of the condition names). Reach values do not depend on this reading, only which block came when."],
    how=[
        "One row is one reach of one participant toward a rectangular target 17 cm from the start (the robot frame's 135 deg; start and target positions from the authors' Target_Table, in cm), made while holding the handle of a KINARM robot. The paper says the target was 15 cm away at 45 deg left of the sagittal plane; the stored coordinates are used. Each sequence is 50 baseline reaches, then blocks of 200 (Experiments 1 and 2) or 100 (Experiment 3) reaches in each feedback condition (Reinforcement: target expands, pleasant sound and a small payment when the hand stops inside the target; Error: a cursor shows the final hand position; Combined: both) separated by 50 washout reaches. The block label is the authors' condition name; group gives the order of the conditions in Experiments 1 and 2, and the clinical group in Experiment 3.",
        "hand_angle is the direction of the final hand position (the sample at the HAND_STEADY event) from the start, minus the target direction, wrapped to (-180, 180] and counter-clockwise positive; hand_extent is the distance from the start in units of the target distance. The authors' lateral position (u) and extent error along the target (v) were recomputed from the raw samples and match the stored values exactly. rt_ms is GREENLIGHT to LEFT_START and mt_ms LEFT_START to HAND_STEADY (event indices in ms).",
        "hit is 1 when the authors' TARGET_REACHED event occurred (also in baseline, where the target is a 0.5 cm circle); reward is 1 when the SUB_REWARD event occurred, and is only filled in the Reinforcement and Combined conditions where a reward was given (the Error condition has SUB_REWARD events too, which only mark hits). The first 40 baseline (and washout) reaches showed a cursor at the final hand position, the last 10 did not (paper, Methods). The minor axis of the experimental target was scaled to each participant's baseline variability and the stored value is a rounded integer, so the reward zone columns are empty.",
        "Parkinson's disease participants took their usual medication (paper, Methods; medication_state on). Ages are given in the paper only as a group mean and a table not linked to the file numbers, so age is empty."],
)
stdlib.release(ID, "reach90", rows, meta, checks)
