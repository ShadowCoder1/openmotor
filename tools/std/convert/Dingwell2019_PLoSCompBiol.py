"""Dingwell & Cusumano (2019) PLoS Comput Biol 15(3):e1006850: lateral foot placement, body position, heading and step
width for every step of 3-minute treadmill trials, 13 able-bodied adults and 8 adults with a transtibial amputation, with
no perturbation (NOP), visual-field perturbations (VIS) and platform perturbations (PLA).

Raw files: raw_extra/Dingwell2019_PLoSCompBiol/{Able_Bodied,Trans_Tibial_Amputees}_CSV_Files.zip and the two READMEs (Zenodo
record 4937771, a copy of the Dryad deposit 10.5061/dryad.p254480, CC0). The MAT versions hold the same numbers.
Each subject CSV has 15 trials x 5 columns (zL, zR, zB, DzB, w) named in the first two rows; one row is one step.
The published paper analysed only the 13 able-bodied participants' NOP trials; the VIS and PLA trials and the amputee group
come from the parent study (README lists its other papers).
Run from the site folder: python3 -I tools/std/convert/Dingwell2019_PLoSCompBiol.py"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Dingwell2019_PLoSCompBiol"
MODULE = "walking"
GROUPS = {"AB": ("Able-bodied", "healthy young adults"), "TT": ("Transtibial amputation", "transtibial amputation")}
COND = {"NOP": ("No perturbation", "none", "normal", "no perturbation"),
        "VIS": ("Visual field perturbation", "visual_field", "altered", "visual field perturbation"),
        "PLA": ("Platform perturbation", "platform_translation", "normal", "platform perturbation")}


def main():
    raw = stdlib.raw_dir(DS)
    ck = stdlib.Checks()
    frames = []
    for pre, zname in (("AB", "Able_Bodied_CSV_Files.zip"), ("TT", "Trans_Tibial_Amputees_CSV_Files.zip")):
        zf = zipfile.ZipFile(raw / zname)
        for fn in sorted(n for n in zf.namelist() if n.endswith(".csv")):
            sid = re.fullmatch(rf"({pre}\d\d)\.csv", fn).group(1)
            df0 = pd.read_csv(zf.open(fn), header=None, skiprows=2)
            names = pd.read_csv(zf.open(fn), header=None, nrows=1).iloc[0]
            assert df0.shape[1] == 75
            for k in range(15):
                tn = names.iloc[5 * k]
                m = re.fullmatch(r"(NOP|VIS|PLA)(\d)", tn)
                X = df0.iloc[:, 5 * k:5 * k + 5].dropna(how="all").to_numpy(float)
                cname, ptype, fb, cond = COND[m.group(1)]
                f = pd.DataFrame(dict(
                    dataset=DS, study=DS, experiment="", group=GROUPS[pre][0], subject=f"{DS}_{sid}",
                    subject_excluded_by_authors=False, age=np.nan, sex="", handedness="", population=GROUPS[pre][1],
                    setting="lab", device="Motorized treadmill in a virtual-reality environment; Vicon motion capture",
                    session=1, block=tn, trial=0, phase="other", time_s=np.nan, task="walking", unit="step",
                    condition=cname, pert_type=ptype, pert_value=np.nan if ptype != "none" else 0.0, pert_unit="",
                    feedback=fb, excluded_by_authors=False, walking_trial=int(m.group(2)) + 5 * list(COND).index(m.group(1)),
                    lateral_foot_position_left_m=X[:, 0], lateral_foot_position_right_m=X[:, 1],
                    lateral_body_position_m=X[:, 2], lateral_heading_m=X[:, 3], step_width_m=X[:, 4]))
                f["_pre"], f["_cond"], f["_tn"] = pre, m.group(1), tn
                frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["trial"] = df.groupby("subject", sort=False).cumcount() + 1
    ntr = df.groupby(["_pre", "subject", "_tn"]).size().rename("n").reset_index()
    ab = ntr[(ntr._pre == "AB") & ntr._tn.str.startswith("NOP")]

    ck.internal("13 able-bodied and 8 transtibial participants (README)",
                df.groupby("_pre").subject.nunique().to_dict() == {"AB": 13, "TT": 8})
    ck.internal("15 trials per participant", (ntr.groupby("subject").size() == 15).all())
    ck.internal("no missing values in the five step columns",
                np.isfinite(df[["lateral_foot_position_left_m", "lateral_foot_position_right_m", "lateral_body_position_m",
                                "lateral_heading_m", "step_width_m"]]).all().all())
    err = (df.lateral_foot_position_right_m - df.lateral_foot_position_left_m - df.step_width_m).abs()
    ck.internal("step width = zR - zL (README: w is step width)", err.max() < 1e-6, f"max |diff| {err.max():.2e}")
    ck.published("able-bodied participants", ab.subject.nunique(), 13, 0, "Methods, Participants")
    ck.published("able-bodied NOP trials (13 x 5 = 65)", len(ab), 65, 0, "Fig 2 caption: 65 total trials")
    ck.published("steps per NOP trial analysed (first 290 steps) - fewest steps in a trial at least 290", min(ab.n.min(), 290), 290, 0,
                 "Methods: first 290 consecutive steps of each trial")
    ck.published("treadmill half-width (all foot positions inside +-0.885 m)",
                 max(df[df._pre == "AB"].lateral_foot_position_left_m.abs().max(), df[df._pre == "AB"].lateral_foot_position_right_m.abs().max()) < 0.885, 1, 0,
                 "Fig 5 caption: lateral boundary +-0.885 m (1.77 m treadmill)")

    out = df.drop(columns=["_pre", "_cond", "_tn"])
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in out.columns:
            out[c] = np.nan
    meta = dict(
        study=DS,
        paper=dict(authors="Dingwell JB, Cusumano JP", year=2019,
                   title="Humans use multi-objective control to regulate lateral foot placement when walking",
                   venue="PLoS Comput Biol 15(3):e1006850", doi="10.1371/journal.pcbi.1006850"),
        lab="Dingwell (Penn State)", data_url="https://doi.org/10.5061/dryad.p254480",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Read the CSV versions (the MATLAB files hold the same numbers) from the Zenodo copy of the Dryad deposit, "
                 "made one row per step, replaced the file names AB01-AB13 and TT01-TT10 with participant codes and "
                 "stacked each participant's 15 trials in file order."),
        how=["One row is one step of one 3-minute treadmill trial: the lateral positions of the left and right foot, the "
             "body (midpoint of the feet) and its step-to-step change (heading), and the step width, all in metres from the "
             "centre of the treadmill (the authors' zL, zR, zB, DzB and w). Each foot placement appears on two consecutive "
             "rows (README).",
             "Each participant walked 5 trials in each of 3 conditions: no perturbation, visual-field perturbations and "
             "platform perturbations (the README refers to the authors' papers for their definitions; sizes are not in the "
             "deposit, so pert_value is empty). walking_trial numbers the trials 1-15 in file order (NOP1-5, VIS1-5, PLA1-5).",
             "The deposit shares no age, sex or handedness per participant, only group summaries (able-bodied 10 men and 3 "
             "women, 24.8 +- 6.92 years; transtibial amputation 8 men, 30.4 +- 7.15 years)."],
        unit_note="step (one footfall to the next); trials are 3 minutes long",
        differences=["The paper analysed only the first 290 steps of the 13 able-bodied participants' no-perturbation trials. "
                     "Every trial in the deposit holds exactly those 290 steps. The rows here keep every row the authors shared, which includes the perturbed trials and "
                     "the 8 participants with a transtibial amputation (README: 10 TT files named TT01-TT10, 8 present, numbering not "
                     "consecutive)."],
    )
    stdlib.release(DS, MODULE, out, meta, ck)
    for c in ck.items:
        print(("  P " if c["kind"] == "published" else "  I ") + c["what"][:80], c.get("computed", ""), c.get("reported", ""), c.get("detail", "")[:150])


if __name__ == "__main__":
    main()
