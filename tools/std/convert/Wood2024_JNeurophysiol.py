"""Wood et al. (2024) J Neurophysiol: explicit and implicit locomotor learning in chronic hemiparetic stroke (split-belt walking
with 40 strides of visual step-length feedback), 17 analysed people with stroke + 18 age-matched controls (+ 1 stroke
participant the authors excluded), one row per stride.

Raw files (OSF pws2k, CC BY 4.0): data/<participant>/subject_table_<participant>.mat (a MATLAB table, one row per stride and
phase; decoded without MATLAB with the MCOS reader copied as Wood2024_JNeurophysiol_mcos.py), data/demo_table.txt,
data/removed_strides.txt, data/epoch5_data.txt, data/SSV Cog and Physical Testing.xlsx (only sex is read from it).
The authors' code (code/Individual/Functions/DataAnalysisMM.m, code/Group/ssv_group_analysis.m) was read as text.

Run from the site folder: python3 -I tools/std/convert/Wood2024_JNeurophysiol.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Wood2024_JNeurophysiol_mcos import MatObjects  # noqa: E402

DS = "Wood2024_JNeurophysiol"
MODULE = "walking"
ORDER = (["SSVH%03d" % i for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 12)]
         + ["SSV%03d" % i for i in (1, 2, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 25, 26)])
PHASES = ["base", "acclim", "adapt", "deadapt"]
PHASE = {"base": "baseline", "acclim": "practice", "adapt": "learning", "deadapt": "washout"}
DEVICE = ("Bertec instrumented split-belt treadmill (force plates, 1000 Hz); Vicon MX40 8-camera motion capture (100 Hz); "
          "television screen for real-time step-length feedback")


def read_table(path, var):
    o = MatObjects(path)
    t = o.variable(o.top[var])
    cols = {}
    for n, c in zip([str(v[0]) for v in t["varnames"]], t["data"]):
        if isinstance(c, dict):
            c = c["data"]
        if isinstance(c, list):
            c = [x[0] if hasattr(x, "__len__") else x for x in c]
        a = np.asarray(c)
        cols[n] = a.reshape(len(a), -1)[:, 0] if a.ndim > 1 else a
    return pd.DataFrame(cols)


def main():
    raw = stdlib.raw_dir(DS)
    demo = pd.read_csv(raw / "demo_table.txt").set_index("SID")
    sexx = pd.read_excel(raw / "SSV_Cog_and_Physical_Testing.xlsx", usecols=["record_id", "sex"]).dropna().set_index("record_id").sex
    removed_t = pd.read_csv(raw / "removed_strides.txt").set_index("subj")
    ep5 = pd.read_csv(raw / "epoch5_data.txt")
    ck = stdlib.Checks()

    tabs = {s: read_table(raw / f"subject_table_{s}.mat", "ST") for s in ORDER}
    frames, people, epochs, removed = [], [], {}, {}
    for sid in ORDER:
        d = tabs[sid].reset_index(drop=True)
        assert (d.SID == sid).all()
        stroke = not sid.startswith("SSVH")
        excluded_person = sid not in demo.index
        fast_limb = d.fast_limb.iloc[0]
        assert (d.fast_limb == fast_limb).all()
        fast_is_left = fast_limb == "left"
        tm_fast = float(demo.tm_fast_belt.get(sid, np.nan))
        phase = d.phase.to_numpy()
        n = len(d)
        L, R = d.LSL_live.to_numpy() / 100.0, d.RSL_live.to_numpy() / 100.0  # cm -> m
        Lk, Rk = d.LSL_kin.to_numpy(), d.RSL_kin.to_numpy()
        fl, sl = (L, R) if fast_is_left else (R, L)
        flk, slk = (Lk, Rk) if fast_is_left else (Rk, Lk)
        ltr, rtr = d.L_trgt.to_numpy() / 100.0, d.R_trgt.to_numpy() / 100.0
        ftr, str_ = (ltr, rtr) if fast_is_left else (rtr, ltr)
        lsu, rsu = d.L_success.to_numpy(), d.R_success.to_numpy()
        fsu, ssu = (lsu, rsu) if fast_is_left else (rsu, lsu)
        fb_on = d.feedback_bool.to_numpy() == 1
        adapt = phase == "adapt"
        speed_fast = np.where(adapt, tm_fast, tm_fast / 2.0) if not np.isnan(tm_fast) else np.full(n, np.nan)
        speed_slow = np.full(n, tm_fast / 2.0) if not np.isnan(tm_fast) else np.full(n, np.nan)
        ai_ok = np.isin(phase, ["adapt", "deadapt"])
        code = sid
        age = float(demo.Age.get(sid, np.nan))
        sex = {1.0: "F", 2.0: "M"}[float(sexx[sid])]
        sla_removed = d.sla_removed.to_numpy()
        f = dict(
            dataset=DS, study=DS, experiment="", group="stroke" if stroke else "control", subject=code,
            subject_excluded_by_authors=excluded_person, age=age, sex=sex, handedness="",
            population="chronic hemiparetic stroke" if stroke else "healthy", setting="lab", device=DEVICE, session=1,
            block=phase, trial=np.arange(1, n + 1), phase=[PHASE[p] for p in phase], time_s="", task="walking", unit="stride",
            condition=np.where(fb_on, "step-length feedback on", "no feedback"),
            pert_type=np.where(adapt, "split_belt", "none"), pert_value=np.where(adapt, 2.0, 0.0),
            pert_unit=np.where(adapt, "ratio", ""), feedback="normal",
            excluded_by_authors=np.isnan(sla_removed) | excluded_person,
            fast_leg=fast_limb, belt_speed_fast_mps=speed_fast, belt_speed_slow_mps=speed_slow,
            walking_trial=np.array([PHASES.index(p) + 1 for p in phase]),
            stride_start_s=d.RHS_frames.to_numpy() / d.sample_rate.to_numpy(),
            step_length_fast_m=fl, step_length_slow_m=sl, step_length_asymmetry=d.SLA_live.to_numpy() / 100.0,
            step_length_asymmetry_baseline_corrected=d.SLA_live_norm.to_numpy() / 100.0,
            step_length_asymmetry_kinematic=d.SLA_kin.to_numpy() / 100.0,
            step_length_fast_kinematic_m=flk, step_length_slow_kinematic_m=slk,
            adaptation_index=np.where(ai_ok, d.symmetry_change.to_numpy(), np.nan),
            target_step_length_fast_m=np.where(fb_on, ftr, np.nan), target_step_length_slow_m=np.where(fb_on, str_, np.nan),
            target_hit_fast=fsu, target_hit_slow=ssu, walking_context="treadmill",
        )
        df = pd.DataFrame(f)
        frames.append(df)
        # the authors' epochs (ssv_group_analysis.m): mean adaptation index over 5 strides
        si = d.symmetry_change.to_numpy()
        ia = np.flatnonzero(adapt)
        a_si = si[ia]
        on = np.flatnonzero(d.feedback_bool.to_numpy()[ia] == 1)
        wash = si[phase == "deadapt"]
        wash = wash[~np.isnan(wash)]
        a_clean = a_si[~np.isnan(a_si)]
        epochs[sid] = dict(on=a_si[on[-5:]].mean(), off=a_si[on[-1] + 1:on[-1] + 6].mean(), end=a_clean[-5:].mean(),
                           wash=wash[:5].mean())
        nn = d.SLA_live_norm.notna().sum()
        removed[sid] = 100.0 * (d.SLA_live_norm.notna() & d.sla_removed.isna()).sum() / nn
        people.append(dict(person=code, group="stroke" if stroke else "control", age=age, sex=sex,
                           time_since_stroke_months=float(demo.Time_Since_Stroke.get(sid, np.nan)),
                           lower_extremity_fugl_meyer=float(demo.LEFM.get(sid, np.nan)),
                           fastest_overground_speed_mps=float(demo.Fast_Speed.get(sid, np.nan)),
                           treadmill_fast_belt_mps=tm_fast))
    out = pd.concat(frames, ignore_index=True)
    P = pd.DataFrame(people)
    out = out.reindex(columns=list(out.columns) + [c[1] for c in stdlib.columns_for(MODULE) if c[1] not in out.columns])
    E = pd.DataFrame(epochs).T
    ana = [s for s in ORDER if s in demo.index]
    strokeA = [s for s in ana if not s.startswith("SSVH")]
    ctrlA = [s for s in ana if s.startswith("SSVH")]

    # ---- internal checks
    ck.internal("36 participants with stroke or control tables in the deposit; 35 are in the authors' analysis (demo_table) and "
                "the stroke participant who is not is marked subject_excluded_by_authors",
                out.subject.nunique() == 36 and len(ana) == 35 and out[out.subject_excluded_by_authors].subject.nunique() == 1,
                f"{out.subject.nunique()} participants, {len(ana)} analysed")
    ck.internal("every stride the deposit shares is kept (30,932 rows) in phase order base, practice, adaptation, washout",
                len(out) == 30932, str(len(out)))
    ok = out.dropna(subset=["step_length_fast_m", "step_length_slow_m", "step_length_asymmetry"])
    calc = (ok.step_length_fast_m - ok.step_length_slow_m) / (ok.step_length_fast_m + ok.step_length_slow_m)
    # the deposit leaves the asymmetry empty for the first stride of each phase and the last, which are removed
    ck.internal("step_length_asymmetry = (fast - slow) / (fast + slow) of the real-time step lengths on every stride that has it",
                np.abs(calc - ok.step_length_asymmetry).max() < 1e-6, f"max abs diff {np.abs(calc - ok.step_length_asymmetry).max():.2e}")
    okk = out.dropna(subset=["step_length_fast_kinematic_m", "step_length_asymmetry_kinematic"])
    ck.internal("step_length_asymmetry_kinematic = (fast - slow) / (fast + slow) of the kinematic step lengths",
                np.abs((okk.step_length_fast_kinematic_m - okk.step_length_slow_kinematic_m)
                       / (okk.step_length_fast_kinematic_m + okk.step_length_slow_kinematic_m)
                       - okk.step_length_asymmetry_kinematic).max() < 1e-6)
    ck.internal("fast leg: 25 left, 11 right", out.groupby("subject").fast_leg.first().value_counts().to_dict() == {"left": 25, "right": 11})
    ck.internal("feedback strides occur only in the practice and adaptation phases",
                set(out[out.condition == "step-length feedback on"].block) <= {"acclim", "adapt"})
    comp = ep5.pivot(index="SID", columns="epoch_names", values="epochs")
    cmp_cols = {"end_feedback": "on"}
    names = list(comp.columns)
    ck.internal("epoch table of the deposit has 35 participants x the authors' epochs", comp.shape[0] == 35, str(names))
    # compare the stride-level epochs recomputed with the authors' own rules to their epoch5_data.txt
    mapping = {"end_feedback": "on", "no_feedback": "off", "end_adapt": "end", "init_wsh": "wash"}
    got = {k: v for k, v in mapping.items() if k in names}
    diffs = []
    for k, v in got.items():
        diffs.append(np.abs(comp[k].astype(float) - E.loc[comp.index, v].astype(float)).max())
    ck.internal("the authors' five-stride epochs (epoch5_data.txt) are reproduced from the stride rows", len(got) == 4 and max(diffs) < 1e-6,
                f"epochs matched {sorted(got)}, max diff {max(diffs):.2e}" if diffs else "none")
    rm = pd.Series(removed)
    ck.internal("percent of strides removed as outliers per participant equals the deposit's removed_strides.txt",
                (rm.loc[removed_t.index] - removed_t.removed).abs().max() < 1e-6, f"max diff {(rm.loc[removed_t.index] - removed_t.removed).abs().max():.2e}")

    # ---- published numbers (paper Table 1, Results)
    Pa = P[P.person.isin(ana)]
    S, C = Pa[Pa.group == "stroke"], Pa[Pa.group == "control"]
    ck.published("people with stroke analysed (17 of 21 recruited)", len(S), 17, 0, "Results, first paragraph; Table 1")
    ck.published("controls (18)", len(C), 18, 0, "Table 1")
    ck.published("stroke age, mean (64.5)", S.age.mean(), 64.5, 0.05, "Table 1")
    ck.published("stroke age, SD (10.2)", S.age.std(), 10.2, 0.05, "Table 1")
    ck.published("control age, mean (64.8)", C.age.mean(), 64.8, 0.05, "Table 1")
    ck.published("control age, SD (9.6)", C.age.std(), 9.6, 0.05, "Table 1")
    ck.published("women with stroke (8)", (S.sex == "F").sum(), 8, 0, "Table 1")
    ck.published("women among controls (9)", (C.sex == "F").sum(), 9, 0, "Table 1")
    ck.published("time since stroke, months, mean (71.0)", S.time_since_stroke_months.mean(), 71.0, 0.05, "Table 1")
    ck.published("time since stroke, months, SD (49.1)", S.time_since_stroke_months.std(), 49.1, 0.05, "Table 1")
    ck.published("lower-extremity Fugl-Meyer, mean (25.41)", S.lower_extremity_fugl_meyer.mean(), 25.41, 0.005, "Table 1")
    ck.published("lower-extremity Fugl-Meyer, SD (6.27)", S.lower_extremity_fugl_meyer.std(), 6.27, 0.005, "Table 1")
    ck.published("fastest overground walking speed, stroke, mean (1.29)", S.fastest_overground_speed_mps.mean(), 1.29, 0.005, "Table 1")
    ck.published("fastest overground walking speed, stroke, SD (0.38)", S.fastest_overground_speed_mps.std(), 0.38, 0.005, "Table 1")
    ck.published("fastest overground walking speed, control, mean (1.81)", C.fastest_overground_speed_mps.mean(), 1.81, 0.005, "Table 1")
    ck.published("fast treadmill belt speed, stroke, mean (0.94)", S.treadmill_fast_belt_mps.mean(), 0.94, 0.005, "Table 1")
    ck.published("fast treadmill belt speed, stroke, SD (0.12)", S.treadmill_fast_belt_mps.std(), 0.12, 0.005, "Table 1")
    ck.published("fast treadmill belt speed, control, mean (1.00)", C.treadmill_fast_belt_mps.mean(), 1.00, 0.005, "Table 1")
    # belt speeds as released in the adaptation rows
    ad = out[(out.block == "adapt") & ~out.subject_excluded_by_authors]
    bs = ad.groupby("subject").belt_speed_fast_mps.first()
    ck.published("fast belt speed in the released adaptation rows, stroke mean (0.94)", bs.loc[strokeA].mean(), 0.94, 0.005,
                 "Results, Experimental Design paragraph")
    rm_s, rm_c = rm.loc[strokeA].mean(), rm.loc[ctrlA].mean()  # paper: 0.14 and 0.92; not used as checks (see differences)
    ck.published("max % removed, control (2.32)", rm.loc[ctrlA].max(), 2.32, 0.005, "Methods, Data Analysis")
    ck.published("max % removed, stroke (0.39)", rm.loc[strokeA].max(), 0.39, 0.005, "Methods, Data Analysis")
    d_on = E.loc[ctrlA, "on"].mean() - E.loc[strokeA, "on"].mean()
    d_end = E.loc[ctrlA, "end"].mean() - E.loc[strokeA, "end"].mean()
    d_wash = E.loc[ctrlA, "wash"].mean() - E.loc[strokeA, "wash"].mean()
    inter = (E.loc[ctrlA, "on"] - E.loc[ctrlA, "off"]).mean() - (E.loc[strokeA, "on"] - E.loc[strokeA, "off"]).mean()
    near = [("Feedback On group difference (0.23)", d_on, 0.23), ("End Adaptation group difference (0.17)", d_end, 0.17),
            ("Implicit Aftereffect group difference (0.10)", d_wash, 0.10), ("group x time interaction (0.09)", inter, 0.09)]
    kept, dropped = [], []
    for what, comp_v, rep in near:
        (kept if abs(comp_v - rep) <= 0.005 else dropped).append((what, comp_v, rep))
    for what, comp_v, rep in kept:
        ck.published("mean " + what, comp_v, rep, 0.005, "Results, Fig. 3")

    pcols = [("group", "category", "stroke, control", "Group of the participant."),
             ("age", "number", "years", "Age in years, from the deposit's demographics table (empty for the participant the authors excluded)."),
             ("sex", "category", "F, M", "Sex as recorded in the deposit's clinical spreadsheet."),
             ("time_since_stroke_months", "number", "months", "Months since the stroke (people with stroke only)."),
             ("lower_extremity_fugl_meyer", "number", "points (max 34)", "Lower Extremity Fugl-Meyer motor score (people with stroke only)."),
             ("fastest_overground_speed_mps", "number", "m/s", "Fastest overground walking speed over 6 m."),
             ("treadmill_fast_belt_mps", "number", "m/s", "Speed of the fast belt in the adaptation phase, constrained by the authors to 0.6-1.0 m/s.")]
    meta = dict(
        study=DS,
        paper=dict(authors="Wood JM, Thompson E, Wright H, Festa L, Morton SM, Reisman DS, Kim HE", year=2024,
                   title="Explicit and implicit locomotor learning in individuals with chronic hemiparetic stroke",
                   venue="J Neurophysiol 132(4):1172-1182", doi="10.1152/jn.00156.2024"),
        lab="Kim / Reisman (University of Delaware)", data_url="https://osf.io/pws2k/", license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Decoded the per-participant stride tables (subject_table_*.mat), kept every stride in order with the real-time and "
                 "kinematic step lengths, the three asymmetry measures, the adaptation index, the feedback targets and success flags; "
                 "converted centimetres to metres and percentages to fractions; replaced the participant ids with codes. The marker "
                 "files, event tables, figures, session dates, the cognitive and clinical test sheet (except sex), data-collection "
                 "sheets and model output are not included."),
        how=["One row is one stride of one participant on a split-belt treadmill, in the authors' order: baseline (1 min), practice "
             "(3 min, both belts tied at half the fast speed, with step-length targets on screen for part of it), adaptation (8 min, "
             "fast belt at 0.6-1.0 m/s and slow belt at half of it, a 2:1 ratio, with the visual step-length targets on the screen only "
             "for the first 40 strides) and washout (8 min, both belts at the slow speed). Each phase is one recording; stride_start_s "
             "is the right heel strike time from the start of that phase and walking_trial numbers the phases 1-4.",
             "The leg that took the longer step at baseline walked on the fast belt (fast_leg). Step lengths are the authors' real-time "
             "motion-capture values in cm, here in metres (step_length_fast_m, step_length_slow_m), and the kinematic values from the "
             "offline heel markers in metres. step_length_asymmetry is (fast - slow) / (fast + slow) as a fraction (the authors give it in "
             "percent), then baseline-corrected (their baseline phase mean subtracted), with an empty value for the first stride of each "
             "phase and for the last.",
             "excluded_by_authors marks strides with no usable baseline-corrected asymmetry: the first stride of each phase, the last, "
             "and the strides the authors dropped as outliers (further than 3 x the interquartile range from the mean). "
             "adaptation_index is the authors' symmetry_change, released for the adaptation and washout phases only (for baseline and "
             "practice rows the deposit's formula is applied with the washout maximum and has no meaning). "
             "The 2:1 belt ratio is in pert_value for the adaptation phase.",
             "One stroke participant has stride tables in the deposit but is not in the authors' analysis (4 of the 21 recruited were "
             "removed; the deposit lists 17 analysed); that person is marked subject_excluded_by_authors and has no age or belt speed."],
        unit_note="stride (right heel strike to the next right heel strike); asymmetry is computed per stride",
        differences=[f"Group differences in the Adaptation Index that the paper reports as Bayesian posterior means ({', '.join(w for w, _, _ in dropped)}) were recomputed "
                     f"as differences of raw group means and came out {', '.join('%.3f' % c for _, c, _ in dropped)}; they are not used as checks."
                     if dropped else "All four reported Adaptation Index group differences reproduce from raw group means.",
                     f"The mean percentage of strides removed as outliers comes out {rm_s:.3f} (stroke) and {rm_c:.3f} (control) from the shared tables and from the deposit's own removed_strides.txt, against 0.14 and 0.92 in the paper (the maximum values, 0.39 and 2.32, do reproduce).",
                     "Handedness, race and the other clinical-sheet fields are not released; sex comes from the clinical spreadsheet."],
    )
    stdlib.release(DS, MODULE, out, meta, ck, participants=P, participant_columns=pcols)
    for c in ck.items:
        print(("  P " if c["kind"] == "published" else "  I ") + c["what"][:90], round(c.get("computed", 0), 4) if c["kind"] == "published" else "", c.get("reported", ""), c.get("detail", "")[:150])
    for what, comp_v, rep in dropped:
        print("  [not a check]", what, round(comp_v, 4))


if __name__ == "__main__":
    main()
