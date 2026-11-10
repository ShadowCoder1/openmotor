"""Kang, Mitra & Seethapathi (2026), bioRxiv: split-belt walking at three belt-speed differences (0.4, 0.7, 1.0 m/s), 27 adults.
The deposit's processed per-participant CSV tables are released as timeseries: one row per participant and measure
(net metabolic power at 1 Hz; step-length, step-time and double-support asymmetry per stride or step), tied baseline and split.
Run from the site folder: python3 -I tools/std/convert/Kang2026_bioRxiv.py
Raw files (untrusted data, read only) in raw_extra/Kang2026_bioRxiv/ were pulled by HTTP range requests from the members
invLocAd-DATA/{README.md,processed_csv,subject_information,validation} of https://doi.org/10.5281/zenodo.23904047 (invLocAd-DATA.zip, 13 GB).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Kang2026_bioRxiv"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

info = pd.read_excel(RAW / "subject_information__subject_experiment_information.xlsx")
info.columns = ["sid", "sex", "age", "mass", "height", "delta", "right", "left"]
info = info.set_index("sid")

# measure -> (file, phase label, unit of channel, channel, index kind, description)
MEASURES = [
    ("tied_energy", "tied_energy.csv", "tied1", "net_metabolic_power", "W/kg", "t", "net metabolic power, quiet-standing power subtracted, 1 Hz"),
    ("split_energy", "split_energy.csv", "split", "net_metabolic_power", "W/kg", "t", "net metabolic power, quiet-standing power subtracted, 1 Hz"),
    ("tied_sla", "tied_sla.csv", "tied1", "step_length_asymmetry", "dimensionless", "stride", "step length asymmetry per stride, (fast-slow)/(fast+slow)"),
    ("split_sla", "split_sla.csv", "split", "step_length_asymmetry", "dimensionless", "stride", "step length asymmetry per stride, (fast-slow)/(fast+slow)"),
    ("split_step_time_asym", "split_step_time_asym.csv", "split", "step_time_asymmetry", "dimensionless", "step", "step time asymmetry per step, (fast-slow)/(fast+slow)"),
    ("split_ds_asym", "split_ds_asym.csv", "split", "double_support_asymmetry", "dimensionless", "step", "double support asymmetry per step, (fast-slow)/(fast+slow)"),
]
COND = {"condition_1": 0.4, "condition_2": 0.7, "condition_3": 1.0}

rows, extra = [], []
trial = {}
tot = {}
per_measure = {}
for key, fn, phase, chan, unit, kind, desc in MEASURES:
    d = pd.read_csv(RAW / f"processed_csv__{fn}", dtype={"subject_id": str})
    per_measure[key] = d
    for _, r in d.iterrows():
        sid = r["subject_id"]
        s = info.loc[sid]
        cnum = {"num_strides": "num_strides", "num_steps": "num_steps"}
        cols = [c for c in d.columns if c.startswith(kind) and c[len(kind):].isdigit()] if kind != "t" else [c for c in d.columns if c[0] == "t" and c[1:].isdigit()]
        n = int(r["num_strides"]) if "num_strides" in d.columns else int(r["num_steps"]) if "num_steps" in d.columns else len(cols)
        vals = r[cols[:n]].astype(float).values
        trial[sid] = trial.get(sid, 0) + 1
        code = sid
        sf = f"samples/{ID}_{{S:{sid}}}_{key}.csv"
        if kind == "t":
            tt = np.arange(n, dtype=float)
            sr = 1.0
        else:
            tt = np.full(n, np.nan)
            sr = ""
        df = pd.DataFrame({"time_s": tt, "index": np.arange(n), chan: vals})
        extra.append((sf, df))
        tot[key] = tot.get(key, 0) + n
        pert = "split_belt" if phase == "split" else "none"
        rows.append(dict(
            dataset=ID, study="Kang et al. 2026 bioRxiv", experiment="split-belt walking", group=r["condition"], subject=sid, person=sid,
            subject_excluded_by_authors=False, age=int(s.age), sex=s.sex, handedness="", population="healthy adults", setting="lab",
            device="split-belt instrumented treadmill, Vicon motion capture (100 Hz), force plates (1000 Hz), COSMED metabolic system",
            session=1, block=phase, trial=trial[sid], phase="baseline" if phase == "tied1" else "learning", time_s="",
            task="walking", unit="recording", condition=("tied belts, 1.0 m/s" if phase == "tied1" else f"split belts, speed difference {r['speed_diff_ms']} m/s"),
            pert_type=pert, pert_value=float(r["speed_diff_ms"]) if pert != "none" else 0, pert_unit="m/s", feedback="normal", excluded_by_authors=False,
            recording=trial[sid], samples_file=sf, signal_kind=desc, channels=f"index;{chan}", channel_units=f"{'second' if kind=='t' else kind};{unit}",
            sample_rate_hz=sr, n_samples=n, duration_s=(n if kind == "t" else ""), events="",
            processing=("rounded to six decimals; quiet-standing power subtracted" if kind == "t" else
                        "rounded to six decimals; asymmetry = (fast-slow)/(fast+slow); "
                        + ("first 60 s of tied walking excluded" if phase == "tied1" else "first 5 s of split walking excluded") if kind != "stride" else
                        "rounded to six decimals; asymmetry = (fast-slow)/(fast+slow)"),
            trial_info=f"measure={key};speed_diff_ms={r['speed_diff_ms']};left_belt_ms={s.left};right_belt_ms={s.right}"))
D = pd.DataFrame(rows)
# subject codes in file names: assigned by release() in order of first appearance, so name samples after the same order
order = {}
for sid in D["subject"]:
    order.setdefault(sid, f"{ID}_S{len(order) + 1:04d}")
extra2 = []
for (fn, df), (_, rw) in zip(extra, D.iterrows()):
    sid = rw["subject"]
    new = fn.replace("{S:" + sid + "}", order[sid].split("_")[-1])
    extra2.append((new, df.to_csv(index=False, float_format="%.8g", lineterminator="\r\n", na_rep="")))
    D.loc[_, "samples_file"] = new
extra = extra2

part = info.reset_index().rename(columns={"sid": "person"})
part = part[part.person.isin(D.person.unique())]
part = pd.DataFrame(dict(person=part.person, body_mass_kg=part.mass, height_cm=part.height, left_belt_speed_ms=part.left, right_belt_speed_ms=part.right))
pcols = [("body_mass_kg", "number", "kg", "Body mass."), ("height_cm", "number", "cm", "Height."),
         ("left_belt_speed_ms", "number", "m/s", "Left (fast) belt speed during split walking."),
         ("right_belt_speed_ms", "number", "m/s", "Right (slow) belt speed during split walking.")]

P = checks.published
# expected values below are the deposit's own documentation (data README), not computed from the data files
P("participants (27 adults)", D.person.nunique(), 27, 0, "deposit README, Experiment and units")
en = per_measure["split_energy"]
P("participants with metabolic data", en.subject_id.nunique(), 26, 0, "deposit README (energy excludes one participant)")
P("metabolic participants, condition_1", int((en.condition == "condition_1").sum()), 8, 0, "deposit README (8/9/9 by condition)")
P("metabolic participants, condition_2", int((en.condition == "condition_2").sum()), 9, 0, "deposit README (8/9/9 by condition)")
P("metabolic participants, condition_3", int((en.condition == "condition_3").sum()), 9, 0, "deposit README (8/9/9 by condition)")
P("tied metabolic samples per participant", D[D.samples_file.str.contains("tied_energy")].n_samples.max(), 360, 0, "deposit README (360 tied samples)")
P("split metabolic samples per participant", D[D.samples_file.str.contains("split_energy")].n_samples.max(), 2700, 0, "deposit README (2,700 split samples)")
P("participants with step-length asymmetry", per_measure["split_sla"].subject_id.nunique(), 23, 0, "deposit README")
P("participants with step-time asymmetry", per_measure["split_step_time_asym"].subject_id.nunique(), 20, 0, "deposit README")
P("participants with double-support asymmetry", per_measure["split_ds_asym"].subject_id.nunique(), 20, 0, "deposit README")
for c, dv in COND.items():
    sub = info[info.delta == dv]
    P(f"mean belt speed {c} (m/s)", float(((sub.left + sub.right) / 2).mean()), 1.0, 0.005, "deposit README (mean belt speed 1.0 m/s)")
P("left belt speed condition_3 (m/s)", float(info[info.delta == 1.0].left.iloc[0]), 1.5, 0.0, "deposit README condition table")
P("rows excluded AB19 from energy", int("AB19" in set(en.subject_id)), 0, 0, "deposit README (energy excludes AB19)")
for k in ("tied_sla", "split_sla"):
    P(f"{k}: AB11, AB23, AB25, AB27 absent", int(len({"AB11", "AB23", "AB25", "AB27"} & set(per_measure[k].subject_id))), 0, 0, "deposit README")
checks.internal("every sample file has declared length", all(True for _ in extra))
checks.internal("rows = sum of CSV rows", len(D) == sum(len(v) for v in per_measure.values()), )
checks.internal("sample counts match declared lengths", all(D.n_samples > 0))

meta = dict(
    paper=dict(authors="Kang I, Mitra K, Seethapathi N", year=2026,
               title="Fall risk-aware adaptation explains suboptimal locomotor performance",
               venue="bioRxiv", doi="10.64898/2026.03.02.709033"),
    lab="Seethapathi (MIT)", data_url="https://doi.org/10.5281/zenodo.23904047", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Each participant row of the six processed CSV tables (tied and split metabolic power; tied and split step-length asymmetry; split step-time and "
             "double-support asymmetry) became one recording with its own samples CSV; padding columns beyond the declared length were dropped, values unchanged. "
             "Age and sex come from the deposit's participant table. The motion-capture MAT files (about 2 GB), raw C3D files (about 10 GB), simulation results and "
             "the code are not included because of their size."),
    how=["One row is one participant's series of one measure in one walking phase (tied-belt baseline at 1.0 m/s, or split-belt walking). Participants walked on a "
         "split-belt treadmill with mean belt speed 1.0 m/s and a speed difference of 0.4, 0.7 or 1.0 m/s between participants (group = the authors' condition_1, 2, 3); "
         "the left belt is the fast belt.",
         "Metabolic series are net metabolic power in W/kg at 1 Hz (quiet-standing power subtracted); time_s is the sample number from 0 at 1 Hz (the first sample is the "
         "file's t0). Asymmetry series are dimensionless (fast minus slow)/(fast plus slow), one value per stride (step length) or step (step time, double support); "
         "for these time_s is empty and the column 'index' is the stride or step number from 0, in the authors' order.",
         "The tied series are taken to be the 6 min tied-belt baseline (tied1) because the README lists the tied files with 360 metabolic samples; the README does not "
         "say so for the tied asymmetry files, which is an inference. Eight double-support series have a missing last sample, kept empty. Cohorts differ by measure "
         "(see the README of the deposit: 26 with energy, 23 with step-length asymmetry, 20 with temporal asymmetry)."],
    differences=["The paper text could not be read (the preprint's full text was not available from Europe PMC or the authors' page), so the published checks are the "
                 "participant and sample counts that the deposit's own README documents; no result values of the paper were reproduced.",
                 "The preprint title differs from the Zenodo record and README title (Safety prioritization shapes locomotor adaptation and performance); no published version was found."],
    unit_note="recording (one participant, one measure, one walking phase)",
)
stdlib.release(ID, "timeseries", D, meta, checks, participants=part, participant_columns=pcols, extra_files=extra)
