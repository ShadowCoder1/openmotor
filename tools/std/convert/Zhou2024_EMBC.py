"""Zhou et al. 2024 (EMBC): ten early-rehabilitation stroke patients, two visits, foot-worn IMU gait parameters per stride.
Deposit (Zenodo 10534055): processed.zip holds, per patient and visit, left and right foot stride tables (core_params) and the
authors' aggregates; raw.zip adds participant_info.csv. The raw and interim IMU signals are not converted."""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Zhou2024_EMBC"
MODULE = "walking"
DEVICE = "Wearable inertial measurement units on the feet (stride parameters from foot IMUs)"
PCOLS = [("height_cm", "number", "cm", "Height; empty where the deposit has none."), ("weight_kg", "number", "kg", "Body mass; empty where the deposit has none."),
         ("fac_visit1", "integer", "0-5", "Functional Ambulation Category at visit 1."), ("fac_visit2", "integer", "0-5", "Functional Ambulation Category at visit 2."),
         ("days_since_stroke_visit1", "integer", "days", "Days since the stroke at visit 1."), ("days_since_stroke_visit2", "integer", "days", "Days since the stroke at visit 2.")]


def main():
    ck = stdlib.Checks()
    zp = zipfile.ZipFile(stdlib.raw_dir(DS) / "processed.zip")
    zr = zipfile.ZipFile(stdlib.raw_dir(DS) / "raw.zip")
    info = pd.read_csv(io.BytesIO(zr.read("raw/participant_info.csv")), encoding="cp1252", encoding_errors="replace")
    info.columns = [c.lstrip("﻿ï»¿").strip() for c in info.columns]
    info = info.rename(columns={info.columns[0]: "sub"})
    info = info.set_index("sub")
    subs = sorted({n.split("/")[1] for n in zp.namelist() if n.startswith("processed/imu") and n.count("/") >= 2 and not n.startswith("__")})
    frames, agg_diff, n_agg = [], [], 0
    for sub in subs:
        for vi, v in enumerate(("visit1", "visit2"), 1):
            parts = []
            for side in ("left", "right"):
                d = pd.read_csv(io.BytesIO(zp.read(f"processed/{sub}/{v}/{side}_foot_core_params.csv")))
                agg = pd.read_csv(io.BytesIO(zp.read(f"processed/{sub}/{v}/{side}_foot_aggregate_params.csv"))).iloc[0]
                excl = d.is_outlier | d.turning_interval | d.interrupted
                g = d[~excl]
                for col, a in (("stride_length", "stride_length_avg"), ("stride_time", "stride_time_avg"), ("swing_time", "swing_time_avg"),
                               ("stance_time", "stance_time_avg"), ("clearance", "clearance_avg"), ("speed", "speed_avg")):
                    agg_diff.append(abs(g[col].mean() - agg[a]))
                n_agg += 1
                f = pd.DataFrame(dict(
                    dataset=DS, study=DS, experiment="", group="", subject=sub, subject_excluded_by_authors=False,
                    age=float(info.loc[sub, "age"]), sex=info.loc[sub, "sex"], population="stroke, early rehabilitation", setting="clinic",
                    device=DEVICE, session=vi, block=f"visit {vi}", phase="other", task="walking", unit="stride",
                    condition="walking", pert_type="none", pert_value=0.0, pert_unit="", feedback="normal",
                    excluded_by_authors=excl.to_numpy(), walking_context="overground", walking_trial=1,
                    stride_start_s=d.timestamp.to_numpy(), walking_speed_mps=d.speed.to_numpy(),
                    **{f"stride_time_{side}_s": d.stride_time.to_numpy(), f"swing_time_{side}_s": d.swing_time.to_numpy(),
                       f"stance_time_{side}_s": d.stance_time.to_numpy(), f"stance_pct_{side}": (d.stance_ratio * 100).to_numpy(),
                       f"stride_length_{side}_m": d.stride_length.to_numpy(), f"foot_clearance_{side}_m": d.clearance.to_numpy()},
                ))
                f["_order"] = d.stride_index.to_numpy()
                f["_side"] = side
                parts.append(f)
            p = pd.concat(parts, ignore_index=True)
            p = p.sort_values(["stride_start_s", "_side"], kind="stable").drop(columns=["_order", "_side"], errors="ignore")
            p["trial"] = np.arange(1, len(p) + 1)
            frames.append(p)
    df = pd.concat(frames, ignore_index=True)
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in df.columns and c != "person":
            df[c] = np.nan
    df["sex"] = df["sex"].astype(str)

    persons = pd.DataFrame([dict(person=s, height_cm=info.loc[s, "height_cm"], weight_kg=info.loc[s, "weight_kg"],
                                 fac_visit1=info.loc[s, "FAC_visit1"], fac_visit2=info.loc[s, "FAC_visit2"],
                                 days_since_stroke_visit1=info.loc[s, "visit1_days_since_stroke"],
                                 days_since_stroke_visit2=info.loc[s, "visit2_days_since_stroke"]) for s in subs])
    ck.internal("10 patients, 2 visits each, left and right foot tables for every visit",
                len(subs) == 10 and df.groupby("subject").session.nunique().eq(2).all(), f"{len(subs)} patients")
    ck.internal("every stride row of the deposit's left and right foot tables is kept (none lost)",
                len(df) == sum(len(pd.read_csv(io.BytesIO(zp.read(n)))) for n in zp.namelist()
                               if n.endswith("_core_params.csv") and not n.startswith("__MACOSX")), f"{len(df)} rows")
    ck.internal("each stride row fills exactly one foot", (pd.to_numeric(df.stride_time_left_s, errors="coerce").notna() ^ pd.to_numeric(df.stride_time_right_s, errors="coerce").notna()).all())
    ck.internal("stride time 0.5-3 s on non-excluded strides",
                pd.to_numeric(df.loc[~df.excluded_by_authors, ["stride_time_left_s", "stride_time_right_s"]].stack(), errors="coerce").dropna().between(0.5, 3).mean() > 0.98)
    ck.published("participants in the study (ten stroke patients, two visits)", len(subs), 10, 0, "Zenodo record description / paper abstract")
    ck.published("foot-visit aggregate tables reproduced from strides (20 visits x 2 feet = 40), max abs difference in the six means < 1e-6",
                 n_agg - (1 if max(agg_diff) < 1e-6 else 0), 39, 0, "deposit's own aggregate_params files")
    print("max agg diff", max(agg_diff), "tables", n_agg)

    meta = dict(
        study=DS,
        paper=dict(authors="Zhou L, Rackoll T, Ekrod L, Balc MG, Klostermann F, Arnrich B, Nave AH", year=2024,
                   title="Monitoring and Visualizing Stroke Rehabilitation Progress using Wearable Sensors",
                   venue="2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC):1-6",
                   doi="10.1109/embc53108.2024.10782489"),
        lab="Arnrich (Hasso Plattner Institute) and Nave (Charite Berlin)",
        data_url="https://zenodo.org/records/10534055", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Combined the deposit's per-patient, per-visit, per-foot stride tables into one table with participant codes; "
                 "stride_ratio is shown as a percentage; the raw and interim IMU signals, foot trajectories, plots, aggregates and "
                 "patient-reported improvement ratings are not included."),
        how=["One row is one stride of one foot (left or right) of one patient at one visit (session 1 or 2, about a week apart), in order of "
             "stride start time; the other foot's columns are empty. Strides come from the deposit's processed stride tables, produced by "
             "the authors' foot-IMU pipeline during a walking test; no belt or perturbation was applied.",
             "excluded_by_authors marks the strides the deposit's own aggregate tables leave out: flagged outliers, strides inside turning "
             "intervals (and interrupted strides, none occur). Averaging the remaining strides reproduces the authors' aggregate means exactly.",
             "stride_start_s is the deposit's timestamp in seconds from the start of the recording. Stride length and speed are per stride "
             "from the IMU pipeline; stance_pct is the deposit's stance ratio times 100.",
             "participants.csv has height, weight, Functional Ambulation Category and days since stroke for each visit."],
        unit_note="stride of one foot",
        differences=["The paper (IEEE, not open access) could not be read, so only the number of patients and the deposit's own aggregate tables were checked; "
                     "the deposit page mentions four walking conditions but the processed files carry no condition label."],
    )
    stdlib.release(DS, MODULE, df, meta, ck, participants=persons, participant_columns=PCOLS)


if __name__ == "__main__":
    main()
