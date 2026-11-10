"""Wolf, Wagner & Schutz (2019) J Vis: competition between salience and informational value for saccade adaptation.

Raw files: raw_extra/Wolf2019_JVis/Exp1.zip, Exp2.zip, Exp3.zip (Zenodo 3241428). Trial tables
(exp1_adaptation.txt, exp2_adaptation.txt, exp3_indRand_adapt.txt, exp3_orthRand_adapt.txt, columns in the deposit's
columnDescription_*.xlsx), the "timeCourse" tables (participant, recoded vertical endpoint, recoded positions of the
informative and the salient target, saccade-onset timestamp) and, for the three Experiment 1 control groups, the
timeCourse tables alone. The pre/post performance files hold one proportion per participant and cell and are not released.
Run from the site folder: python3 -I tools/std/convert/Wolf2019_JVis.py
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Wolf2019_JVis"
MODULE = "saccade"
DEVICE = "EyeLink 1000 (right eye, 1000 Hz), VIEWPixx monitor at 60 cm, keyboard"
ZIPS = {z: zipfile.ZipFile(stdlib.raw_dir(DS) / f"{z}.zip") for z in ("Exp1", "Exp2", "Exp3")}


def rd(z, name):
    return pd.read_csv(io.BytesIO(ZIPS[z].read(name)), header=None, na_values=["NaN"])


# (zip, trial table, timecourse table, experiment label, group label, kind, authors' excluded participants)
MAIN = [
    ("Exp1", "Exp1/exp1_adaptation.txt", "Exp1/exp1_adaptation_timeCourse.txt", "exp1", "double-target, sequential judgments", "two", {8}),
    ("Exp2", "Exp2/exp2_adaptation.txt", "Exp2/exp2_adaptation_timeCourse.txt", "exp2", "double-target, simultaneous judgments", "one", set()),
    ("Exp3", "Exp3/exp3_indRand_adapt.txt", "Exp3/exp3_indRand_adaptation_timeCourse.txt", "exp3", "random steps, sequential judgments", "two", {1, 4}),
    ("Exp3", "Exp3/exp3_orthRand_adapt.txt", "Exp3/exp3_orthRand_adaptation_timeCourse.txt", "exp3", "random steps, simultaneous judgments", "one", {1, 6}),
]
CONTROLS = [("Exp1/exp1_notask_adaptation_timeCourse.txt", "no-task"), ("Exp1/exp1_infonly_adaptation_timeCourse.txt", "informative-only")]

COLS = ["target_amplitude_deg", "target_direction_deg", "step_x_deg", "step_y_deg", "step_delay_ms", "distractor_x_deg",
        "distractor_y_deg", "distractor_delay_ms", "saccade_amplitude_x_deg", "saccade_amplitude_y_deg", "saccade_latency_ms",
        "saccade_duration_ms", "saccade_peak_velocity_deg_s", "blink", "eye", "response", "response_correct",
        "vertical_endpoint_recoded_deg", "n_saccades", "trial_info"]


def fnum(v):
    return "" if pd.isna(v) else f"{v:g}"


def frame(d, code, exp, group, kind, excl_p, sign, inf, sal, tsec):
    """d: trial table rows of one participant; sign/inf/sal: orientation of the vertical axis and the recoded target positions."""
    n = len(d)
    A = d.reset_index(drop=True)
    adapt = A[3].to_numpy() == 1
    # columns differ between the 25-column ("two": vertical and horizontal component) and the 23-column ("one") tables
    if kind == "two":
        i = dict(lat=19, pv=20, dur=21, ax=22, ay=23, end=24, ns=17, nb=18)
        resp = [f"vertical={fnum(a)};horizontal={fnum(b)}" if not (pd.isna(a) and pd.isna(b)) else ""
                for a, b in zip(A[12], A[13])]
        corr = A[16]
        info = [f"ecc={e:g};contrast_vertical={c1:g};contrast_horizontal={c2:g};sf_vertical_cpd={s1:g};sf_horizontal_cpd={s2:g};"
                f"orient_dev_vertical_deg={o1:g};orient_dev_horizontal_deg={o2:g};correct_vertical={fnum(k1)};correct_horizontal={fnum(k2)}"
                for e, c1, c2, s1, s2, o1, o2, k1, k2 in zip(A[2], A[6], A[7], A[8], A[9], A[10], A[11], A[14], A[15])]
    else:
        i = dict(lat=17, pv=18, dur=19, ax=20, ay=21, end=22, ns=15, nb=16)
        resp = ["" if pd.isna(r) else ("orthogonal" if r == 1 else "not orthogonal") for r in A[13]]
        corr = A[14]
        info = [f"ecc={e:g};contrast_first={c1:g};contrast_second={c2:g};sf_first_cpd={s1:g};sf_second_cpd={s2:g};"
                f"orient_dev_first_deg={o1:g};orient_dev_second_deg={o2:g};orient_difference_deg={od:g}"
                for e, c1, c2, s1, s2, o1, o2, od in zip(A[2], A[6], A[7], A[8], A[9], A[10], A[11], A[12])]
    end = A[i["end"]]
    out = pd.DataFrame({
        "dataset": DS, "study": DS, "experiment": exp, "group": group, "subject": f"{code}",
        "subject_excluded_by_authors": bool(excl_p), "age": "", "sex": "", "handedness": "", "population": "healthy",
        "setting": "lab", "device": DEVICE, "session": "", "block": np.where(adapt, "adaptation", "pre-adaptation"),
        "trial": A[1].astype(int), "phase": np.where(adapt, "learning", "baseline"), "time_s": tsec, "task": "eye movements",
        "unit": "saccade", "condition": group,
        "pert_type": np.where(adapt, "intrasaccadic_step", "none"),
        "pert_value": np.where(adapt, sign * inf, 0.0), "pert_unit": "deg",
        "feedback": "normal", "excluded_by_authors": end.isna().to_numpy(),
        "target_amplitude_deg": A[2], "target_direction_deg": 0.0, "step_x_deg": 0.0,
        "step_y_deg": sign * inf, "step_delay_ms": np.where(adapt, 0.0, np.nan), "distractor_x_deg": np.where(adapt, 0.0, np.nan),
        "distractor_y_deg": np.where(adapt, sign * sal, np.nan), "distractor_delay_ms": np.where(adapt, 0.0, np.nan),
        "saccade_amplitude_x_deg": A[i["ax"]], "saccade_amplitude_y_deg": A[i["ay"]], "saccade_latency_ms": A[i["lat"]],
        "saccade_duration_ms": A[i["dur"]], "saccade_peak_velocity_deg_s": A[i["pv"]] * 1000.0,
        "blink": [("" if pd.isna(b) else bool(b > 0)) for b in A[i["nb"]]], "eye": "right", "response": resp,
        "response_correct": [("" if pd.isna(c) else bool(c == 1)) for c in corr],
        "vertical_endpoint_recoded_deg": end, "n_saccades": A[i["ns"]], "trial_info": info,
    })
    return out


def main():
    ck = stdlib.Checks()
    rows = []
    stats = {}
    for z, tn, tcn, exp, group, kind, excl in MAIN:
        d, t = rd(z, tn), rd(z, tcn)
        assert len(d) == len(t) and (d[0].to_numpy() == t[0].to_numpy()).all()
        end_col = 24 if kind == "two" else 22
        for p in sorted(d[0].unique()):
            m = (d[0] == p).to_numpy()
            dp, tp = d[m], t[m]
            # orientation of the vertical axis: the authors' endpoint is amplitude minus baseline mean, times +1 or -1
            ok = dp[[end_col, end_col - 1]].dropna()
            sl = np.polyfit(ok[end_col - 1], ok[end_col], 1)[0]
            sign = 1.0 if sl > 0 else -1.0
            assert abs(abs(sl) - 1) < 1e-3, (tn, p, sl)
            ck.internal(f"{exp}/{group[:20]} p{p}: recoded endpoint = +-(vertical amplitude minus a constant); target positions are 0 in baseline, +2/-2 recoded after",
                        bool(set(tp[2].unique()) <= {-2, 0, 2} and set(tp[3].unique()) <= {-2, 0, 2}))
            ts = tp[4].to_numpy(dtype=float)
            tsec = (ts - np.nanmin(ts)) / 1000.0
            code = f"{exp}|{group}|{p}"
            rows.append(frame(dp, code, exp, group, kind, int(p) in excl, sign, tp[2].to_numpy(dtype=float), tp[3].to_numpy(dtype=float), tsec))
            stats[(exp, group, int(p))] = (tp[1].to_numpy(dtype=float), dp[1].to_numpy(), int(p) in excl, sign, tp[2].to_numpy(dtype=float))
    for tcn, grp in CONTROLS:
        t = rd("Exp1", tcn)
        for p in sorted(t[0].unique()):
            tp = t[t[0] == p].reset_index(drop=True)
            n = len(tp)
            trial = np.arange(1, n + 1)
            adapt = trial > 50
            ts = tp[4].to_numpy(dtype=float)
            tsec = (ts - np.nanmin(ts)) / 1000.0
            r = pd.DataFrame({
                "dataset": DS, "study": DS, "experiment": "exp1", "group": f"control: {grp}", "subject": f"exp1|control {grp}|{p}",
                "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
                "setting": "lab", "device": DEVICE, "session": "", "block": np.where(adapt, "adaptation", "pre-adaptation"),
                "trial": trial, "phase": np.where(adapt, "learning", "baseline"), "time_s": tsec, "task": "eye movements",
                "unit": "saccade", "condition": f"control: {grp}",
                "pert_type": np.where(adapt, "intrasaccadic_step", "none"),
                "pert_value": np.where(adapt, tp[2], 0.0), "pert_unit": "deg", "feedback": "normal",
                "excluded_by_authors": tp[1].isna().to_numpy(), "target_amplitude_deg": 12.5, "target_direction_deg": 0.0,
                "step_x_deg": 0.0, "step_y_deg": tp[2], "step_delay_ms": np.where(adapt, 0.0, np.nan),
                "distractor_x_deg": np.where(adapt, 0.0, np.nan), "distractor_y_deg": np.where(adapt, tp[3], np.nan),
                "distractor_delay_ms": np.where(adapt, 0.0, np.nan), "eye": "right", "vertical_endpoint_recoded_deg": tp[1],
            })
            rows.append(r)
            stats[("exp1", f"control: {grp}", int(p))] = (tp[1].to_numpy(dtype=float), trial, False, 1.0, tp[2].to_numpy(dtype=float))
    sal_bytes, inf_bytes = ZIPS["Exp1"].read("Exp1/exp1_salonly_adaptation_timeCourse.txt"), ZIPS["Exp1"].read("Exp1/exp1_infonly_adaptation_timeCourse.txt")
    ck.internal("the deposit's salient-only time-course file is a byte-identical copy of the informative-only file (left out, see Known differences)",
                sal_bytes == inf_bytes)
    out = pd.concat(rows, ignore_index=True)
    # rows of a recoded participant: the authors' endpoint is on the recoded axis, the amplitude and positions on the raw axis
    ck.internal("every row the authors shared is kept: 3200 + 2800 + 2000 + 2000 double-target rows and 2 x 1600 control rows",
                len(out) == 3200 + 2800 + 2000 + 2000 + 3200, str(len(out)))

    # ---- checks with the paper -----------------------------------------------------------------------------------
    def per_part(exp, group, keep=lambda s: True):
        return {k[2]: v for k, v in stats.items() if k[0] == exp and k[1].startswith(group) and keep(v)}
    e1 = per_part("exp1", "double-target")
    e2 = per_part("exp2", "double-target")
    ck.published("Exp 1: participants recorded", len(e1), 16, 0, "Methods, Participants")
    ck.published("Exp 1: participants analysed (one excluded)", sum(1 for v in e1.values() if not v[2]), 15, 0, "Methods, Participants")
    valid = 100 * (1 - out[(out['experiment'] == 'exp1') & out['group'].str.startswith('double')]['excluded_by_authors'].mean())
    ck.published("Exp 1: % of trials considered for the final analyses (double-target task, all 16 recorded participants)", valid, 90.5, 0.1,
                 "Methods, Eye movement and data analysis")
    ck.published("Exp 2: participants analysed", len(e2), 14, 0, "Experiment 2, Methods")
    disc = [100 * (1 - np.isfinite(v[0]).mean()) for v in e2.values()]
    ck.published("Exp 2: mean % of trials discarded per participant", np.mean(disc), 13.4, 0.1, "Experiment 2, Methods")
    ck.published("Exp 2: smallest % discarded", min(disc), 4, 1, "Experiment 2, Methods (range 4%-25%)")
    ck.internal("Exp 2: largest % discarded is 28 where the paper's range ends at 25 (see Known differences)", abs(max(disc) - 28) < 0.01, f"{max(disc):.1f}")
    late2 = [np.nanmean(v[0][150:200]) for v in e2.values()]
    ck.published("Exp 2: mean vertical endpoint, trials 151-200 (deg)", np.mean(late2), 0.35, 0.015, "Experiment 2, Results (M = 0.35 deg)")
    ck.published("Exp 2: SD of that endpoint across participants (deg)", np.std(late2, ddof=1), 0.27, 0.015, "Experiment 2, Results (SD = 0.27 deg)")
    ck.published("Exp 2: participants with a positive mean late endpoint", sum(1 for x in late2 if x > 0), 13, 0, "Experiment 2, Results (13 of 14 toward the informative target)")
    for g, rep_m, rep_sd in (("control: no-task", 0.73, 0.43), ("control: informative-only", 1.11, 0.45)):
        late = [np.nanmean(v[0][150:200]) for v in per_part("exp1", g).values()]
        ck.published(f"Exp 1 {g}: participants", len(late), 8, 0, "Methods, Participants")
        ck.published(f"Exp 1 {g}: mean |vertical endpoint| trials 151-200 (deg)", abs(np.mean(late)), rep_m, 0.015, "Results, Figure 3B")
        ck.published(f"Exp 1 {g}: SD across participants (deg)", np.std(late, ddof=1), rep_sd, 0.015, "Results, Figure 3B")
    for grp, name in (("random steps, sequential judgments", "sequential"), ("random steps, simultaneous judgments", "simultaneous")):
        e3 = {k[2]: v for k, v in stats.items() if k[0] == "exp3" and k[1] == grp}
        keep = {p: v for p, v in e3.items() if not v[2]}
        ck.published(f"Exp 3 {name}: participants analysed", len(keep), 8, 0, "Experiment 3, Methods")
        ck.published(f"Exp 3 {name}: mean % of valid trials", np.mean([100 * np.isfinite(v[0]).mean() for v in keep.values()]),
                     83 if name == "sequential" else 84, 1.5, "Experiment 3, Methods (83% and 84% correct trials)")
    # axis direction: after the step, the raw vertical amplitude moves toward the informative target on average (sign check)

    meta = dict(
        study=DS,
        paper=dict(authors="Wolf C, Wagner I, Schutz AC", year=2019, title="Competition between salience and informational value for saccade adaptation",
                   venue="J Vis 19(14):26", doi="10.1167/19.14.26"),
        lab="Schutz (University of Marburg)",
        data_url="https://doi.org/10.5281/zenodo.3241428",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Joined the authors' trial tables with their time-course tables (same rows) to add the recoded target positions "
                 "and the saccade timestamps, gave participants codes, converted peak velocity from deg/ms to deg/s and stated "
                 "target and distractor positions on the vertical axis of the raw saccade amplitudes. The pre/post performance files "
                 "(one proportion per participant and cell) are not included."),
        how=["One row is one trial of one participant: a rightward saccade to a 12.5 deg eccentric plaid (informative target: 9 cpd, "
             "contrast 0.3; salient target: 1.5 cpd, contrast 0.6) that, in adaptation trials (51-200), was split during the saccade "
             "into its two components, one stepped up and the other down by 2 deg.",
             "step_y_deg gives the vertical offset of the informative target and distractor_y_deg that of the salient target, both on "
             "the vertical axis of the raw vertical amplitude (up positive) and with the response values of the authors; in "
             "experiments 1 and 2 the authors' vertical_endpoint_recoded_deg flips this axis for participants whose informative target "
             "stepped down, so positive always means toward the informative target, and it is baseline corrected by subtracting the "
             "mean of trials 1-50; in experiment 3 (random step direction) it is not flipped.",
             "excluded_by_authors marks the trials the authors set to NaN (fixation, latency 80-400 ms, horizontal amplitude, saccade "
             "count criteria); subject_excluded_by_authors marks participants named in the deposit's column descriptions (exp1 #8, "
             "exp3 sequential #1 and #4, exp3 simultaneous #1 and #6). The three Experiment 1 control groups come from the "
             "time-course files only: recoded endpoint and the two target positions, without other trial values.",
             "Participants are coded per experiment and group; the deposit does not say whether anyone took part in more than one."],
        unit_note="trial (one primary saccade)",
        differences=["Experiment 2: the paper reports that 4%-25% of trials were discarded per participant (mean 13.4%); the deposit's trials reproduce the mean and the lower end, but one participant has 28% of trials without a vertical endpoint. Experiment 1 control groups: the deposit's salient-only file (exp1_salonly_adaptation_timeCourse.txt) is a byte-identical copy of the informative-only file, and its values reproduce the paper's informative-only mean and SD (1.11 and 0.45 deg) but not the salient-only ones (0.95 and 0.28 deg), so the salient-only group's data are not in the deposit and only one copy is released, as informative-only. In the control files both target positions are listed in every adaptation trial although only one target was shown. The deposit holds 14 participants for Experiment 2 and 10 per Experiment 3 group, where the paper analysed 8 per Experiment 3 group: the authors' excluded participants are flagged, not removed, in Experiment 3 and Experiment 1 (16 recorded, 15 analysed). Participants excluded in Experiment 2 are absent from the deposit."],
    )
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in out.columns and c != "person":
            out[c] = ""
    out = out[[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"]]
    stdlib.release(DS, MODULE, out, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
