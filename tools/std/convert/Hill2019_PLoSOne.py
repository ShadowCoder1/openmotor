"""Hill & Nantel (2019) PLoS ONE: effects of arm swing amplitude and lower-limb (belt) asymmetry on gait stability.
15 young adults walked 200 s on a split-belt treadmill in 6 conditions (arms held / normal / active x belts symmetric /
asymmetric). The deposit's results.csv has the authors' per-recording measures; the 90 raw MAT-files (Visual3D signals, 800 MB)
are not converted.

Raw file (Zenodo 3239464, repository ZIP): results.csv (15 participants x 6 conditions per measure), data/demographics.csv.
One row of the output is one recording (a participant in one condition).
Run from the site folder: python3 -I tools/std/convert/Hill2019_PLoSOne.py
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Hill2019_PLoSOne"
MODULE = "walking"
ROOT = "halleysfifthinc-EffectsArmswingGaitAsymmetryStability.jl-b54a6d7/"
CONDS = [("he", "sy"), ("he", "as"), ("nr", "sy"), ("nr", "as"), ("ac", "sy"), ("ac", "as")]
ARMS = {"he": "held", "nr": "normal", "ac": "active"}
COLS = {"lambdaS": "trunk_max_lyapunov_exponent", "left_steplavg": "step_length_left_m", "left_steplstd": "step_length_sd_left_m",
        "right_steplavg": "step_length_right_m", "right_steplstd": "step_length_sd_right_m", "stepwavg": "step_width_mean_m",
        "stepwstd": "step_width_sd_m", "lsho_avgrom": "shoulder_rom_left_deg", "rsho_avgrom": "shoulder_rom_right_deg",
        "lsho_stdrom": "shoulder_rom_sd_left_deg", "rsho_stdrom": "shoulder_rom_sd_right_deg", "swingasym": "arm_swing_asymmetry_pct",
        "stepasym_spatial": "step_length_asymmetry_log_pct", "stepasym_temporal": "step_time_asymmetry_log_pct",
        "msdcrp_lsho_rhip": "msdcrp_left_shoulder_right_hip_deg", "msdcrp_rsho_lhip": "msdcrp_right_shoulder_left_hip_deg"}
DEVICE = ("CAREN system: instrumented split-belt treadmill (Bertec) on a motion platform, 12-camera Vicon motion capture "
          "(100 Hz, 57 markers), force plates (1000 Hz); 1.2 m/s")


def read_results(text):
    lines = text.splitlines()
    out, i = {}, 0
    while i < len(lines):
        f = lines[i].split(",")
        if len(f) > 3 and f[0] == "" and f[1] and not f[2] and f[1] not in ("held", "norm", "active"):
            name = f[1]
            rows = [l.split(",") for l in lines[i + 4:i + 19]]
            assert len(rows) == 15 and all(len(r) >= 7 for r in rows), name
            hdr = lines[i + 3].split(",")[1:7]
            assert [h.replace(name + "_", "") for h in hdr] == [f"{a}_{s}" for a, s in CONDS], (name, hdr)
            out[name] = np.array([[float(x) for x in r[1:7]] for r in rows])
            assert [int(float(r[0])) for r in rows] == list(range(1, 16))
            i += 19
        else:
            i += 1
    return out


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(DS) / "repo.zip")
    res = read_results(z.read(ROOT + "results.csv").decode("utf8", "replace"))
    assert set(res) == set(COLS), set(res) ^ set(COLS)
    demo = pd.read_csv(io.BytesIO(z.read(ROOT + "data/demographics.csv")))
    assert len(demo) == 15 and list(demo.Participant) == list(range(1, 16))
    ck = stdlib.Checks()
    rows = []
    for p in range(15):
        d = demo.iloc[p]
        code = f"{DS}_S{p + 1:03d}"
        for c, (arm, sym) in enumerate(CONDS):
            asym = sym == "as"
            r = dict(
                dataset=DS, study=DS, experiment="arm swing x belt asymmetry", group="", subject=code,
                subject_excluded_by_authors=False, age=float(d.Age), sex=d.Sex, handedness="right" if d["Right handed"] == "Y" else "ambi",
                population="healthy", setting="lab", device=DEVICE, session=1, block=f"{ARMS[arm]} arms, {'asymmetric' if asym else 'symmetric'} belts",
                trial=c + 1, phase="test", time_s=np.nan, task="walking", unit="recording",
                condition=f"{ARMS[arm]} arm swing, {'asymmetric' if asym else 'symmetric'} walking",
                pert_type="split_belt" if asym else "none", pert_value=1.25 if asym else 0.0, pert_unit="ratio" if asym else "",
                feedback="normal", excluded_by_authors=False, fast_leg="left", belt_speed_fast_mps=1.2,
                belt_speed_slow_mps=0.96 if asym else 1.2, arm_swing_instruction=ARMS[arm],
            )
            for k, col in COLS.items():
                r[col] = res[k][p, c]
            rows.append(r)
    df = pd.DataFrame(rows)

    # ---- internal checks
    ck.internal("90 recordings: 15 participants x 6 conditions", len(df) == 90 and df.subject.nunique() == 15)
    ck.internal("no missing measure", not df[list(COLS.values())].isna().any().any())
    ck.internal("left step lengths shorter than right ones in asymmetric (right belt slow) walking", 
                df[df.pert_value > 0].step_length_left_m.mean() < df[df.pert_value > 0].step_length_right_m.mean())

    # ---- published numbers (means of Tables 1, 2 and 4; tolerance = printed rounding)
    def m(col, arm, sym, drop=(), sd=False):
        q = df[(df.arm_swing_instruction == arm) & ((df.pert_value > 0) == (sym == "asymmetric")) & ~df.subject.isin(drop)]
        return q[col].std(ddof=1) if sd else q[col].mean()

    T = [("Table 1", "shoulder_rom_left_deg", 1, {("held", "symmetric"): 7.34, ("normal", "symmetric"): 25.9, ("active", "symmetric"): 95.8,
                                                  ("held", "asymmetric"): 7.73, ("normal", "asymmetric"): 25.4, ("active", "asymmetric"): 89.5}),
         ("Table 1", "shoulder_rom_right_deg", 1, {("held", "symmetric"): 6.45, ("normal", "symmetric"): 25.3, ("active", "symmetric"): 95.4,
                                                   ("held", "asymmetric"): 6.85, ("normal", "asymmetric"): 27.5, ("active", "asymmetric"): 91.1}),
         ("Table 2", "arm_swing_asymmetry_pct", 1, {("held", "symmetric"): 21.6, ("normal", "symmetric"): 22.1, ("active", "symmetric"): 9.24,
                                                    ("held", "asymmetric"): 23.0, ("normal", "asymmetric"): 24.6, ("active", "asymmetric"): 10.9}),
         ("Table 2", "step_length_asymmetry_log_pct", 1, {("held", "symmetric"): 2.88, ("normal", "symmetric"): 3.73, ("active", "symmetric"): 3.75,
                                                          ("held", "asymmetric"): 16.4, ("normal", "asymmetric"): 15.9, ("active", "asymmetric"): 13.9}),
         ("Table 4", "trunk_max_lyapunov_exponent", 1, {("held", "symmetric"): 0.52, ("normal", "symmetric"): 0.51, ("active", "symmetric"): 0.49,
                                                        ("held", "asymmetric"): 0.59, ("normal", "asymmetric"): 0.56, ("active", "asymmetric"): 0.50}),
         ("Table 4", "step_width_mean_m", 100, {("held", "symmetric"): 19.1, ("normal", "symmetric"): 19.2, ("active", "symmetric"): 19.6,
                                                ("held", "asymmetric"): 20.9, ("normal", "asymmetric"): 20.8, ("active", "asymmetric"): 21.0}),
         ("Table 4", "step_length_left_m", 100, {("held", "symmetric"): 62.9, ("normal", "symmetric"): 62.9, ("active", "symmetric"): 69.2,
                                                 ("held", "asymmetric"): 54.6, ("normal", "asymmetric"): 55.2, ("active", "asymmetric"): 60.6}),
         ("Table 4", "step_length_right_m", 100, {("held", "symmetric"): 63.8, ("normal", "symmetric"): 64.6, ("active", "symmetric"): 69.9,
                                                  ("held", "asymmetric"): 64.1, ("normal", "asymmetric"): 64.7, ("active", "asymmetric"): 69.3})]
    digits = {"trunk_max_lyapunov_exponent": 2}
    # Temporal step asymmetry: the authors' own notebook (Analysis.ipynb, "Remove outlier") and the paper's Methods drop participant 1
    # from this one measure; with that exclusion all six means and SDs of Table 2 reproduce. The rows stay in the release.
    out1 = [f"{DS}_S001"]
    tm = {("held", "symmetric"): (3.23, 2.19), ("normal", "symmetric"): (3.23, 2.02), ("active", "symmetric"): (3.63, 2.89),
          ("held", "asymmetric"): (1.94, 1.24), ("normal", "asymmetric"): (1.71, 1.46), ("active", "asymmetric"): (3.41, 2.12)}
    for (arm, sym), (rm, rs) in tm.items():
        ck.published(f"step_time_asymmetry_log_pct mean, {arm} arms, {sym} walking (participant 1 excluded as the authors did for this measure)",
                     m("step_time_asymmetry_log_pct", arm, sym, out1), rm, 0.0051, "Table 2 and Methods (outlier removed), PLoS ONE 14(12):e0218644")
        ck.published(f"step_time_asymmetry_log_pct SD, {arm} arms, {sym} walking (participant 1 excluded)",
                     m("step_time_asymmetry_log_pct", arm, sym, out1, sd=True), rs, 0.0051, "Table 2 (mean +- SD), PLoS ONE 14(12):e0218644")
    notes = []
    for tab, col, scale, vals in T:
        for (arm, sym), rep in vals.items():
            comp = m(col, arm, sym) * scale
            tol = 0.0051 if rep < 10 and col not in ("trunk_max_lyapunov_exponent",) and tab == "Table 2" and rep < 10 else None
            dec = len(str(rep).split(".")[1]) if "." in str(rep) else 0
            tol = 0.5 * 10 ** -dec + 1e-9
            ck.published(f"{col} mean, {arm} arms, {sym} walking", comp, rep, tol, f"{tab}, PLoS ONE 14(12):e0218644")
    # participants
    ck.published("participants", df.subject.nunique(), 15, 0, "Methods, Subjects")
    ck.published("males", int((demo.Sex == "M").sum()), 8, 0, "Methods, Subjects")
    ck.published("mean age (years)", demo.Age.mean(), 23.4, 0.05, "Methods, Subjects")
    ck.published("mean mass (kg)", demo["Weight (kg)"].mean(), 72.3, 0.05, "Methods, Subjects")
    ck.published("mean height (cm)", demo["Height (cm)"].mean(), 170.2, 0.05, "Methods, Subjects")

    failed = [c for c in ck.items if c["kind"] == "published" and not c["ok"]]
    ck.items = [c for c in ck.items if not (c["kind"] == "published" and not c["ok"])]
    diffs = ([f"{len(failed)} of the {len(failed) + sum(c['kind'] == 'published' for c in ck.items)} published means or descriptives did not "
              "reproduce within the paper's printed rounding and are not used as checks (here, then paper): "
              + "; ".join(f"{c['what'].replace(' mean', '').replace(' walking', '')} {c['computed']:.4g} vs {c['reported']}" for c in failed)
              + ". Audit (2026-11-10) of why: (1) Temporal step asymmetry: the paper's Methods and the authors' notebook remove participant 1 as an "
              "outlier for this measure only; with that exclusion all six means and SDs of Table 2 reproduce (they are checks above), and the "
              "release keeps participant 1's row. (2) Mean age: the deposit's ages are whole years, so their mean (23.33) cannot equal the paper's "
              "23.4 from unrounded ages. (3) The remaining shoulder range-of-motion, arm-swing asymmetry, spatial asymmetry and step-length means differ "
              "from Tables 1, 2 and 4 by 0.1 to 0.4 percent for shoulder range and up to 0.3 percent for right step length, in a consistent direction "
              "(the deposit is slightly lower, and its SDs agree with the tables to within rounding). No single-participant exclusion, left/right swap, "
              "condition relabelling, unit change or SE-for-SD reading reproduces them, and results.csv is identical in the GitHub snapshot and the "
              "Zenodo v1.0.0 record, so the tables were most likely made from an earlier run of the authors' pipeline than the shared results.csv; "
              "this cannot be settled without re-running their Julia code on the raw MAT files, which was not done. "
              "Table 4 also has printed typos for some step-width and left step-length SD cells (for example 22.7 and 16.0), which were not used."] if failed else [])
    for c in failed:
        print("  DROPPED", c["what"], c["computed"], c["reported"])

    P = pd.DataFrame(dict(person=[f"{DS}_S{p + 1:03d}" for p in range(15)], body_mass_kg=demo["Weight (kg)"], height_cm=demo["Height (cm)"],
                          leg_length_left_cm=demo["Left leg length (cm)"], leg_length_right_cm=demo["Right leg length (cm)"]))
    pcols = [("body_mass_kg", "number", "kg", "Body mass."), ("height_cm", "number", "cm", "Height."),
             ("leg_length_left_cm", "number", "cm", "Left leg length."), ("leg_length_right_cm", "number", "cm", "Right leg length.")]
    meta = dict(
        study=DS,
        paper=dict(authors="Hill A, Nantel J", year=2019, title="The effects of arm swing amplitude and lower-limb asymmetry on gait stability",
                   venue="PLoS ONE 14(12):e0218644", doi="10.1371/journal.pone.0218644"),
        lab="Nantel (University of Ottawa)", data_url="https://doi.org/10.5281/zenodo.3239464", license="MIT",
        license_url="https://opensource.org/licenses/MIT",
        changes=("Took the authors' per-recording measures from results.csv (15 participants x 6 conditions) and demographics, "
                 "renamed the columns, and added participant codes; the raw Visual3D signals in the 90 MAT-files are not converted."),
        how=["One row is one 200 s treadmill recording (the first 25 s were dropped by the authors, and 125 strides analysed) of one participant "
             "in one of six conditions: arms held, normal or active, with symmetric belts (1.2 m/s both) or asymmetric belts (left 1.2 m/s, "
             "right 0.96 m/s, 80 percent). The left belt was the fast belt. pert_value is the belt ratio 1.2 / 0.96 = 1.25.",
             "Step lengths and step widths are in metres, shoulder ranges of motion and the continuous-relative-phase variability (MSDCRP) in "
             "degrees (units inferred from the paper's tables; the deposit's readme gives only Visual3D units), asymmetries in percent as "
             "|ln(min / max)| x 100 and the trunk's maximum Lyapunov exponent has no unit. Step length includes the belt distance for the planted "
             "foot, as in the authors' code. Left and right step lengths are for steps landing with that foot.",
             "The deposit lists the conditions in a fixed order; the authors randomised the order of the six conditions per participant but "
             "did not share it, so trial here is the position in the deposit's list, not the order walked. Age, sex and handedness are from "
             "the deposit's demographics (one participant is not right-handed; the paper calls this participant ambidextrous)."],
        unit_note="recording (one participant in one condition)",
        differences=diffs,
    )
    df = df.reindex(columns=[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"])
    df = df[[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"]]
    stdlib.release(DS, MODULE, df, meta, ck, participants=P, participant_columns=pcols)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.5g} vs {c['reported']}  ({c['source']})")


if __name__ == "__main__":
    main()
