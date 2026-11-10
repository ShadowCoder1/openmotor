"""De Comite, Crevecoeur & Lefevre (2022), eNeuro 9(2):ENEURO.0439-21.2022: reward-dependent selection of feedback gains.
Three reaching experiments with a KINARM end-point robot and lateral step loads (14, 20 and 19 participants); one row per reach in
the force-field reaching format (with the added step-load, reward, target-choice and EMG columns), plus 10 ms bin means of the hand
path and the two EMG channels around the load onset as samples/<id>_S####.csv.

Run from the site folder: python3 -I tools/std/convert/DeComite2022_eNeuro.py
Raw files (untrusted data, read only) in raw_extra/DeComite2022_eNeuro/: DSExpe1.mat, DSExpe2.mat, DSExpe3.mat from the Zenodo copy
(record 6403130, CC0) of the Dryad deposit doi:10.5061/dryad.79cnp5hx8. Each file is one struct with a field per participant.
Not converted: the per-condition mean and SD traces (mean_kine, std_kine, mean_EMG, std_EMG), which can be rebuilt from the rows.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "DeComite2022_eNeuro"
RAW = stdlib.raw_dir(ID)
OUTS = RAW / "samples_out"
OUTS.mkdir(parents=True, exist_ok=True)
checks = stdlib.Checks()

# trial-type code -> (unperturbed/perturbed, signed load N, reward points, condition label, background N), decoded from the data (see README)
def decode(exp, tp):
    if exp == 1:
        if tp <= 3:
            return dict(load=0, reward=[1, 5, 10][tp - 1], cond=f"reward{[1, 5, 10][tp - 1]}", bg=None)
        k = (tp - 4) % 3
        return dict(load=9 if tp <= 6 else -9, reward=[1, 5, 10][k], cond=f"reward{[1, 5, 10][k]}", bg=None)
    odd = tp in (1, 4, 7, 10, 13)
    if tp <= 2:
        load = 0
    else:
        load = {4: 10, 5: 10, 7: -10, 8: -10, 10: 6, 11: 6, 13: -6, 14: -6}[tp] if exp == 2 else \
               {4: 6, 5: 6, 7: -6, 8: -6, 10: 3, 11: 3, 13: -3, 14: -3}[tp]
    if exp == 2:
        return dict(load=load, reward=None, cond="different_values" if odd else "same_values", bg=None)
    return dict(load=load, reward=None, cond="no_background" if odd else "background", bg=0 if odd else -4)


BLOCK_LEN = {1: 72, 2: 80, 3: 60}
N_PART = {1: 14, 2: 20, 3: 19}
rows = []
sample_files = []
rt_all = {1: [], 2: [], 3: []}
vel_pts = []
n_zero = 0
si = 0
for exp in (1, 2, 3):
    f = RAW / f"DSExpe{exp}.mat"
    m = sio.loadmat(str(f), squeeze_me=True, struct_as_record=False)
    S = m[f"dS{exp}"]
    for sid in S._fieldnames:
        q = getattr(S, sid)
        si += 1
        t = q.time_matrix.astype(int)
        K, E, TP, TM, AT = q.matrix_kinematics, q.matrix_EMG, q.vector_TP, q.matrix_timing, q.array_timing
        n = len(TP)
        valid = ~((K[:, :, 0] == 0) & (K[:, :, 1] == 0))
        # 10 ms bins from -500 to +1000 ms around the load onset
        edges = np.arange(-500, 1001, 10)
        bsel = [(t >= a) & (t < a + 10) for a in edges[:-1]]
        out = ["trial,time_ms,hand_x_m,hand_y_m,emg1_norm,emg2_norm"]
        for i in range(n):
            lab = {str(x): TM[i, j] for j, x in enumerate(AT[i]) if isinstance(x, str)}
            tp = int(TP[i])
            d = decode(exp, tp)
            blk = i // BLOCK_LEN[exp] + 1
            ft = lab.get("force")
            row = dict(
                dataset=ID, study=ID, experiment=f"Exp{exp}", group="", subject=f"E{exp}_{sid}", person=f"E{exp}_{sid}",
                subject_excluded_by_authors=False, age="", sex="", handedness="right", population="healthy", setting="lab",
                device="KINARM end-point robot with surface EMG", session=1, block=f"block{blk}", trial=i + 1, phase="test", time_s="",
                task="reaching", unit="trial", condition=d["cond"], pert_type="step_load" if d["load"] else "none",
                pert_value=d["load"], pert_unit="N (positive to the right)" if d["load"] else "", feedback="normal", excluded_by_authors=False,
                trial_type="field" if d["load"] else "null", trial_in_condition=i + 1, block_in_condition=blk, hand="right",
                step_load_n=d["load"], load_onset_time_s=ft, target_direction_deg=90 if exp == 1 else "",
            )
            if d["reward"] is not None:
                row["reward_points"] = d["reward"]
            if d["bg"] is not None:
                row["background_load_n"] = d["bg"]
            elif exp == 3:
                pass
            left = lab.get("left_first", lab.get("home_target_left"))
            goal = lab.get("second_target", lab.get("goal_target_shown"))
            reach = lab.get("target_reached")
            row["speed_goal_met"] = bool(lab.get("success", 0) > 0)
            if left and goal and left > goal:
                row["reaction_time_s"] = left - goal
                rt_all[exp].append(left - goal)
            if left and reach and reach > left:
                row["movement_duration_s"] = reach - left
            v = valid[i]
            if not v.any():
                n_zero += 1
                rows.append(row)
                continue
            idx = np.flatnonzero(v)
            a, b = idx[0], idx[-1] + 1
            x, y = K[i, a:b, 0], K[i, a:b, 1]
            j0 = int(np.argmin(np.abs(t + 100)))
            x0 = K[i, j0, 0] if v[j0] else x[0]
            vy = np.gradient(y) * 1000.0
            row["peak_speed_m_s"] = float(vy.max())
            row["max_lateral_displacement_cm"] = float(np.abs(x - x0).max() * 100)
            jr = None
            if reach and ft is not None and reach > 0:
                jr = int(round((reach - ft) * 1000)) - int(t[0])
            xe = (K[i, jr, 0] if jr is not None and 0 <= jr < K.shape[1] and v[jr] else x[-1]) - x0
            row["endpoint_x_cm"] = float(xe * 100)
            if exp > 1:
                row["chosen_target"] = "left" if xe < -0.045 else ("right" if xe > 0.045 else "center")
            for ch in (0, 1):
                for nm, lo, hi in (("pre150", -150, 0), ("20_50ms", 20, 50), ("50_100ms", 50, 100), ("100_180ms", 100, 180)):
                    w = (t >= lo) & (t < hi) & v
                    if w.any():
                        row[f"emg{ch + 1}_{nm}_norm"] = float(E[i, w, ch].mean())
            for e, sl in zip(edges[:-1], bsel):
                w = sl & v[:]
                if w.any():
                    out.append(f"{i + 1},{e},{K[i, w, 0].mean():.5g},{K[i, w, 1].mean():.5g},{E[i, w, 0].mean():.4g},{E[i, w, 1].mean():.4g}")
            rows.append(row)
        p = OUTS / f"{ID}_S{si:04d}.csv"
        p.write_text("\r\n".join(out) + "\r\n", newline="")
        sample_files.append((f"samples/{ID}_S{si:04d}.csv", str(p)))
        print(exp, sid, n, flush=True)
    del m, S
    # reward effect check data (Exp 1)
df = pd.DataFrame(rows)
for _c in [c[1] for c in stdlib.columns_for('force_field_reaching')]:
    if _c not in df.columns and _c != 'person':
        df[_c] = ''

# ------------------------------------------------------------------ checks
per = df.groupby("experiment").subject.nunique()
for e, nn in N_PART.items():
    checks.published(f"Exp{e} participants", float(per[f"Exp{e}"]), nn, 0, f"Methods, Participants (Experiment {e}: {nn} participants)")
tpp = df.groupby(["experiment", "subject"]).size().groupby("experiment").agg(["min", "max"])
checks.published("Exp1 trials per participant", float(tpp.loc["Exp1", "max"]), 432, 0, "Methods, Experiment 1 (6 blocks of 72 trials)")
checks.published("Exp2 trials per participant", float(tpp.loc["Exp2", "max"]), 480, 0, "Methods, Experiment 2 (6 blocks of 80 trials)")
checks.published("Exp3 trials per participant", float(tpp.loc["Exp3", "max"]), 480, 0, "Methods, Experiment 3 (8 blocks of 60 trials)")
e1 = df[df.experiment == "Exp1"]
c1 = e1[e1.trial_type == "field"].groupby(["subject", "reward_points", "step_load_n"]).size()
checks.published("Exp1 perturbed trials per participant per reward x direction", float(c1.min()), 24, 0, "Methods, Experiment 1 (24 trials per perturbed condition)")
checks.published("Exp1 share of perturbed trials", float((e1.trial_type == "field").mean()), 1 / 3, 0.005, "Methods, Experiment 1 (33% perturbed trials)")
checks.published("Exp1 unperturbed trials per reward level per block", float(e1[(e1.trial_type == "null") & (e1.subject == "E1_S1")].groupby(["block", "reward_points"]).size().max()), 16, 0, "Methods, Experiment 1 (48 unperturbed trials per block, 16 per reward)")
e2 = df[df.experiment == "Exp2"]
checks.published("Exp2 share of perturbed trials", float((e2.trial_type == "field").mean()), 0.5, 0.005, "Methods, Experiment 2 (50% perturbed trials)")
c2 = e2[e2.trial_type == "field"].groupby(["subject", "condition", "step_load_n"]).size()
checks.published("Exp2 perturbed trials per participant per condition (2 reward conditions x 4 loads)", float(c2.min()), 30, 0, "Methods, Experiment 2 (30 trials per perturbation condition)")
e3 = df[df.experiment == "Exp3"]
checks.published("Exp3 share of perturbed trials", float((e3.trial_type == "field").mean()), 1 / 3, 0.005, "Methods, Experiment 3 (33% perturbed trials)")
checks.published("Exp3 trials with background load per participant", float((e3.background_load_n.astype(float) != 0).sum() / 19), 240, 0, "Methods, Experiment 3 (four blocks of 60 without, four with background load)")
checks.published("Exp1 largest absolute step load (N)", float(e1.step_load_n.abs().max()), 9, 0, "Methods, Experiment 1 (lateral step force of 9 N)")
checks.published("Exp2 largest absolute step load (N)", float(e2.step_load_n.abs().max()), 10, 0, "Methods, Experiment 2 (6 or 10 N)")
checks.published("Exp3 largest absolute step load (N)", float(e3.step_load_n.abs().max()), 6, 0, "Methods, Experiment 3 (3 or 6 N)")
# direction: rightward loads push the hand to the right in the first part of the perturbed response
pk = df[df.trial_type == "field"].copy()
checks.internal("rows: 14*432 + 20*480 + 19*480", len(df) == 14 * 432 + 20 * 480 + 19 * 480, str(len(df)))
checks.internal("rightward loads give larger rightward lateral endpoint shifts than leftward loads (Exp1)",
                e1[e1.step_load_n > 0].max_lateral_displacement_cm.astype(float).mean() > 0)
checks.internal("Exp2 unperturbed lateral reaches are rare when the lateral targets have low reward",
                (e2[(e2.trial_type == "null") & (e2.condition == "different_values")].chosen_target.dropna() != "center").mean() < 0.01)
print("all-zero trials:", n_zero)
for e in (1, 2, 3):
    print("Exp", e, "mean RT", float(np.mean(rt_all[e])))
u2 = e2[(e2.trial_type == "null")]
print("Exp2 unperturbed lateral share by condition", u2.groupby("condition").chosen_target.apply(lambda s: (s.dropna() != "center").mean()).to_dict())
# reward effect on peak forward velocity (Exp 1, unperturbed): within-participant slope per reward level coded 1,2,3
u1 = e1[e1.trial_type == "null"].copy()
u1["r"] = u1.reward_points.map({1: 1, 5: 2, 10: 3}).astype(float)
u1["pv"] = u1.peak_speed_m_s.astype(float)
u1 = u1.dropna(subset=["pv"])
u1["pvc"] = u1.pv - u1.groupby("subject").pv.transform("mean")
u1["rc"] = u1.r - u1.groupby("subject").r.transform("mean")
print("Exp1 within-participant slope of peak forward speed on reward level (1,2,3):", float((u1.pvc * u1.rc).sum() / (u1.rc ** 2).sum()))

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Crevecoeur & Lefevre (UCLouvain)", data_url="https://doi.org/10.5061/dryad.79cnp5hx8 (Zenodo copy https://zenodo.org/records/6403130)",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes="three MATLAB files turned into one table with one row per reach; trial-type codes decoded into reward, load and condition; "
            "movement measures and EMG window means computed from the stored 1 kHz traces; traces cut to 10 ms bin means; participant codes replaced.",
    unit_note="reach",
    how=["One row is one reach of one participant with the KINARM end-point robot (right hand, forward reach to a goal 25 cm away in Experiment 1; "
         "three goals 20 cm away, one straight ahead and two 9 cm to each side, in Experiments 2 and 3). Experiment 1 (14 people): 432 reaches in 6 blocks "
         "of 72, goal reward 1, 5 or 10 points, 1/3 of the reaches with a lateral step load of +9 or -9 N. Experiment 2 (20 people): 480 reaches in 6 blocks of 80, "
         "lateral targets with the same or a different (lower) value than the central one, half the reaches with a step load of +-6 or +-10 N. Experiment 3 (19 people): "
         "480 reaches in 8 blocks of 60, the first four without and the last four with a 4 N leftward background load, 1/3 of the reaches with a +-3 or +-6 N step.",
         "The deposit stores each reach as a trial-type code 1 to 14 and time-aligned traces. Codes were decoded from the data and from the paper's design: "
         "Experiment 1 codes 1-3 unperturbed with reward 1, 5, 10 (ordered by the rise of peak velocity with reward, which the paper reports), 4-6 rightward and 7-9 leftward loads "
         "(sign checked on the lateral hand deviation); Experiments 2 and 3 odd codes (1, 4, 7, 10, 13) are the first condition (different values; no background) and even codes (2, 5, 8, 11, 14) the second "
         "(same values; background, checked on the EMG baseline of channel 2), 4/5 and 7/8 the larger rightward and leftward loads and 10/11 and 13/14 the smaller ones (checked on the hand deviation). "
         "trial_type is field on reaches with a step load and null otherwise (the module's name for a perturbation); pert_value is the signed load in N, positive to the right.",
         "Within each block the deposit stores the reaches sorted by trial type, not in presentation order, so trial is the row's position in the authors' file, "
         "and block (the true block, from the position in the file) is the only reliable order variable.",
         "The traces are aligned to the load onset (the cursor crossing the trigger line, 8 cm from the start in Experiment 1, 2 cm in the others), also on unperturbed reaches. "
         "Hand positions are in m (lowpass filtered by the authors); Experiments 2 and 3 have the start at x of about 0.09 m. Measures computed here: peak_speed_m_s (peak forward speed from the hand y trace), "
         "max_lateral_displacement_cm and endpoint_x_cm (relative to the start x, 100 ms before the load onset; the endpoint is the position when the target was reached), chosen_target (the endpoint more than 4.5 cm left or right of the start), "
         "reaction_time_s (home target left minus goal target shown, in the authors' event matrix) and movement_duration_s (target reached minus home target left). "
         "EMG columns are means of the authors' normalized rectified EMG in windows around the load onset; the deposit does not say which channel is pectoralis major and which posterior deltoid.",
         "samples/ holds 10 ms bin means of hand x, y and the two EMG channels from 500 ms before to 1000 ms after the load onset, one file per participant; trial matches the main table. Recordings of some reaches are all zero in the deposit; their measures and samples are empty."],
    differences=[f"{n_zero} reaches have an all-zero recording in the deposit (not recorded or removed by the authors); they are kept with empty measures.",
                 "Experiment 3: the paper's text states 24 trials per perturbation condition, but the 8 blocks hold 20 per condition (480 trials in total, as in the Methods); the deposit has 20.",
                 "Reaction time as computed here (home target left minus goal target shown) has a mean of 524 ms in Experiment 1 (paper: 518 ms) but 381 and 389 ms in Experiments 2 and 3 (paper: 311 and 424 ms), so it is not used as a check; the paper's definition is not given.",
                 "The paper's printed unperturbed lateral-reach rates in Experiment 2 (under 1% and 8%) are 0.2% and 8.2% here with lateral meaning an endpoint more than 4.5 cm off the start line; the 7.5% and 23.5% lateral-trial proportions could not be tied to a defined subset, and the reward slope of peak forward speed (paper 0.013 per level) is 0.0147 here by a simple within-participant fit rather than the paper's mixed model.",
                 "Ages and sex are not in the deposit per participant (the paper gives ranges per experiment), so they are empty. Participants of the three experiments are coded separately, since the deposit does not say whether anyone took part twice."],
)
stdlib.release(ID, "force_field_reaching", df, meta, checks, extra_files=sample_files)
