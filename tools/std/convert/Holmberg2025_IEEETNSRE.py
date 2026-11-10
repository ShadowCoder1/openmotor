"""Holmberg et al. (2025), IEEE TNSRE 33:1939-1950: pose-estimated 3D skeletons of infants' spontaneous movements (Helsinki and Pisa),
with corrected age and neurodevelopmental outcome. One row per 20 s video segment (600 time steps of 18 joints x 3 coordinates).

Run from the site folder: python3 -I tools/std/convert/Holmberg2025_IEEETNSRE.py
Raw files (untrusted data, read only) in raw_extra/Holmberg2025_IEEETNSRE/ from https://zenodo.org/records/14269867:
  data.zip (data/streams/combined/<test_id>_<n>_<k>.feather, data/features.csv), metadata.zip (metadata/combined.csv)
Each feather holds four streams stacked on 600 steps: j (joint coordinates), b (bone vectors), v (velocity), a (acceleration).
v is the first difference of j (r = 0.999 on a checked file) and a the first difference of v (r = 1.000), so only j is released.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Holmberg2025_IEEETNSRE"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
OUTCOME = {0: "typical outcome", 1: "weak (minor) neurological impairment", 2: "major neurological impairment (MNI)"}

zd = zipfile.ZipFile(RAW / "data.zip")
meta_df = pd.read_csv(io.BytesIO(zipfile.ZipFile(RAW / "metadata.zip").read("metadata/combined.csv")), dtype={"test_id": str})
feats = pd.read_csv(io.BytesIO(zd.read("data/features.csv")), dtype={"test_id": str}, usecols=["segment", "test_id", "outcome", "corrected_age"])
info = meta_df.set_index("test_id")
files = [n for n in zd.namelist() if n.startswith("data/streams/combined/") and n.endswith(".feather")]


def key(n):
    t, a, b = n.rsplit("/", 1)[1][:-8].split("_")
    return (t, int(a), int(b))


files.sort(key=key)
tests = sorted({key(n)[0] for n in files})
POS = {t: i + 1 for i, t in enumerate(tests)}
SAMPLES = RAW / "_samples"
SAMPLES.mkdir(exist_ok=True)
fe = feats.set_index("segment")

rows, extra = [], []
count = {}
CH = None
age_mismatch = 0
for n in files:
    t, a, b = key(n)
    seg = f"{t}_{a}_{b}"
    df = pf.read_table(io.BytesIO(zd.read(n))).to_pandas()
    J = df.xs("j", level=1)
    assert J.shape == (600, 54) and not J.isna().any().any()
    if CH is None:
        CH = list(J.columns)
    assert list(J.columns) == CH
    count[t] = count.get(t, 0) + 1
    code = f"{ID}_S{POS[t]:04d}"
    fn = f"samples/{code}_seg{count[t]:03d}.csv"
    out = pd.DataFrame(J.values, columns=CH)
    out.insert(0, "time_s", np.arange(600) / 30.0)
    p = SAMPLES / f"{code}_seg{count[t]:03d}.csv"
    out.to_csv(p, index=False, float_format="%.8g", lineterminator="\r\n")
    extra.append((fn, p))
    oc = int(info.loc[t, "outcome"])
    age = float(info.loc[t, "corrected_age"])
    if seg in fe.index and (int(fe.loc[seg, "outcome"]) != oc or float(fe.loc[seg, "corrected_age"]) != age):
        age_mismatch += 1
    rows.append(dict(
        dataset=ID, study=ID, experiment=ID, group=OUTCOME[oc], subject=t, person=t, subject_excluded_by_authors=False,
        age="", sex="", handedness="", population="infants (supine, spontaneous movement)", setting="clinic",
        device="3D video camera (Neuro Event Labs), pose-estimation model", session=a, block=f"recording {a}", trial=count[t],
        phase="test", time_s="", task="infant spontaneous movements", unit="recording", condition="supine spontaneous movement",
        pert_type="none", pert_value=0, pert_unit="", feedback="none", excluded_by_authors=False,
        recording=count[t], samples_file=fn, signal_kind="3D joint coordinates of 18 body landmarks from pose estimation (neck-centred, rotated)",
        channels=";".join(CH), channel_units=";".join(["pose-model units (not stated by the authors)"] * len(CH)),
        sample_rate_hz=30, n_samples=600, duration_s=20.0, events="",
        processing="pose estimation, centring on the neck and three rotations (paper II-C), segment set to 600 steps by repeating shorter segments",
        trial_info=f"corrected_age_days={age:g};outcome_code={oc};file_name_recording_field={a};segment_field={b}"))
df = pd.DataFrame(rows)
df["age"] = [float(x.split("corrected_age_days=")[1].split(";")[0]) / 365.25 for x in df.trial_info]

per = df.drop_duplicates("person")
oc = per.trial_info.str.extract(r"outcome_code=(\d)")[0].astype(int)
ages = per.trial_info.str.extract(r"corrected_age_days=([\d.]+)")[0].astype(float)
seg_oc = df.trial_info.str.extract(r"outcome_code=(\d)")[0].astype(int)
mni, typ = ages[oc == 2], ages[oc < 2]
P = checks.published
P("infants", len(per), 173, 0, "II-B Subject Selection and Video Segmentation (173 infants)")
P("video segments", len(df), 1843, 0, "II-B (1843 video segments)")
P("segments of typically developing infants (outcome 0 or 1)", int((seg_oc < 2).sum()), 1692, 0, "II-B (1692 segments from typically developing infants)")
P("segments of infants with MNI", int((seg_oc == 2).sum()), 151, 0, "II-B (151 segments from the infants with MNI)")
P("infants with MNI", int((oc == 2).sum()), 14, 0, "II-B (14 infants with MNI)")
P("typically developing infants", int((oc < 2).sum()), 159, 0, "II-B (159 infants)")
P("typically developing infants, mean corrected age (days)", float(typ.mean()), 91.5, 0.05, "II-B (mean 91.5 days)")
P("typically developing infants, SD of corrected age (days)", float(typ.std()), 21.0, 0.5, "II-B (SD 21 days)")
P("MNI infants, mean corrected age (days)", float(mni.mean()), 86.6, 0.05, "II-B (mean 86.6 days)")
P("MNI infants, SD of corrected age (days)", float(mni.std()), 30.4, 0.05, "II-B (SD 30.4 days)")
P("typically developing infants, oldest corrected age (days)", float(typ.max()), 158, 0, "II-B (range up to 158 days)")
P("MNI infants, youngest and oldest corrected age (days)", float(mni.min() + mni.max()), 34 + 143, 0, "II-B (range 34-143 days)")
P("samples per segment (600 steps = 20 s)", int(df.n_samples.iloc[0]), 600, 0, "II-B (600 time steps)")
P("landmarks per skeleton", len(CH) // 3, 18, 0, "II-A (18 anatomical landmarks)")
checks.internal("every segment has 600 x 54 finite values", True)
checks.internal("segment file ages and outcomes agree with metadata/combined.csv", age_mismatch == 0, str(age_mismatch))
checks.internal("every infant with files is in the metadata; 3 metadata infants have no segment", len(set(meta_df.test_id) - set(tests)) == 3 and set(tests) <= set(meta_df.test_id))

paper = dict(authors="Holmberg D, Airaksinen M, Marchi V, Guzzetta A, Tuiskula A, Haataja L, Vanhatalo S, Roos T", year=2025,
             title="Learning Developmental Age From 3D Infant Kinetics Using Adaptive Graph Neural Networks",
             venue="IEEE Trans Neural Syst Rehabil Eng 33:1939-1950", doi="10.1109/tnsre.2025.3568269")
meta = dict(
    paper=paper, lab="Vanhatalo / Roos (University of Helsinki) with Guzzetta (University of Pisa)", data_url="https://zenodo.org/records/14269867",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("data/streams/combined/*.feather (joint stream only) were written as one samples CSV per segment, with a time_s column (30 Hz) added; "
             "corrected age (days) and outcome from metadata/combined.csv are in the table (age in years = days / 365.25; days are in trial_info). "
             "Values were written with 8 significant digits. Not included: the bone, velocity and acceleration streams (derived from the joint stream), "
             "the 2D proxy streams, data/features.csv, trained models and predictions."),
    how=["One row is one 20 s video segment of one infant lying supine, selected by pediatricians as an example of active spontaneous movement. "
         "Samples are 600 steps (20 s, 30 Hz as stated in the paper) with the x, y, z coordinates of 18 landmarks (neck, shoulders, hips, elbows, wrists, "
         "knees, ankles, nose, eyes, ears). Coordinates are those of the authors' preprocessing: neck-centred and rotated so the spine and body line align "
         "with the axes (paper II-C), so the neck is zero; their unit is not stated.",
         "Segments shorter than 600 steps were repeated by the authors to fill 600 steps, so a segment can contain repeated stretches. The paper "
         "says the segments come from different recordings of the same infants; the middle number of the authors' file name (1 to 5) is given as "
         "session and block and is probably the recording or visit, which the deposit does not say.",
         "group is the neurodevelopmental outcome (0 typical, 1 weak impairment, 2 major neurological impairment, MNI). Age is corrected age, "
         "constant per infant in the deposit. Sex is not in the deposit. Three infants listed in the metadata have no segment and are not in the table."],
    differences=["metadata/combined.csv lists 176 infants (122 typical, 40 weak impairment, 14 MNI); the 173 with segments are the paper's cohort. "
                 "The paper's age statistics (typical 91.5 +/- 21 days, MNI 86.6 +/- 30.4 days) reproduce from the 173 infants with segments."],
    unit_note="video segment (20 s)",
)
stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)
