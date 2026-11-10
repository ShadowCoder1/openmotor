"""Kulvicius et al. (2025, Commun Med 5:16): 5-second snippets of infant spontaneous movement, labelled fidgety movements
present (FM+) or absent (FM-): ViTPose skeleton key points from video (50 Hz) and six body-worn IMUs (60 Hz), 45 infants.

Raw file, raw_extra/Kulvicius2025_CommunMed/sensor_data.zip (Zenodo 14046527), read as text with plain readers:
  UMGDatensatz_slicesselected_vitpose_fm/  1784 files, 250 rows x 34 columns (17 key points x, y), no header
  UMGDatensatz_slicesselected_imu_fm/      1683 files, 300 rows x 72 columns (6 IMUs x 12 channels), no header
  UMGDatensatz_slicesselected_mat_fm/      1776 pressure-mat files: not carried (500 x 1024 values each, about 1.8 GB as text)
File names carry infant number, age in days, video frames and the label (see the deposit README).
Run from the site folder: python3 -I tools/std/convert/Kulvicius2025_CommunMed.py
"""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Kulvicius2025_CommunMed"
SRC = "Kulvicius et al. 2025, Commun Med 5:16"
KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
      "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
SK_CH = [f"{k}_{a}" for k in KP for a in "xy"]
IMU_CH = [f"{q}_{i}_{a}" for i in range(1, 7) for q in ("acc", "gyr", "mag") for a in "xyz"]
IMU_CH = [c for i in range(1, 7) for c in (f"acc_{i}_x", f"acc_{i}_y", f"acc_{i}_z", f"gyr_{i}_x", f"gyr_{i}_y", f"gyr_{i}_z",
                                          f"mag_{i}_x", f"mag_{i}_y", f"mag_{i}_z", f"roll_{i}", f"pitch_{i}", f"yaw_{i}")]
IMU_U = [{"acc": "m/s^2", "gyr": "rad/s", "mag": "a.u.", "rol": "deg", "pit": "deg", "yaw": "deg"}[c[:3]] for c in IMU_CH]
SK_RE = re.compile(r"idxp(\d+)_(\d+)d_camera1_framestart-(\d+)_frameend-(\d+)_label_([01])\.txt")
IMU_RE = re.compile(r"idxp(\d+)_(\d+)d_camera1_frames_(\d+)-(\d+)_mtw_samples_(\d+)-(\d+)_label_([01])\.txt")


def main():
    checks = stdlib.Checks()
    z = zipfile.ZipFile(stdlib.raw_dir(ID) / "sensor_data.zip")
    recs = []
    for info in z.infolist():
        nm = info.filename.rsplit("/", 1)[-1]
        if "vitpose_fm/" in info.filename and nm.endswith(".txt"):
            m = SK_RE.fullmatch(nm); assert m, nm
            recs.append(dict(kind=0, inf=int(m[1]), age=int(m[2]), f0=int(m[3]), f1=int(m[4]), label=int(m[5]), path=info.filename))
        elif "imu_fm/" in info.filename and nm.endswith(".txt"):
            m = IMU_RE.fullmatch(nm); assert m, nm
            recs.append(dict(kind=1, inf=int(m[1]), age=int(m[2]), f0=int(m[3]), f1=int(m[4]), label=int(m[7]), path=info.filename))
    recs.sort(key=lambda r: (r["inf"], r["kind"], r["age"], r["f0"]))
    n_sk = sum(r["kind"] == 0 for r in recs); n_imu = sum(r["kind"] == 1 for r in recs)
    sub_no, count, rows, extra = {}, {}, [], []
    tmp_dir = stdlib.raw_dir(ID) / "samples_tmp"
    tmp_dir.mkdir(exist_ok=True)
    group = {}  # (infant code, modality) -> list of DataFrames, written as one CSV per infant and modality
    for r in recs:
        s = sub_no.setdefault(r["inf"], len(sub_no) + 1)
        count[s] = count.get(s, 0) + 1
        n = count[s]
        code = f"{ID}_S{s:04d}"
        A = np.loadtxt(io.BytesIO(z.read(r["path"])))
        if r["kind"] == 0:
            hz, ch, un, kind, tag = 50, SK_CH, ["pixel"] * 34, "2D body key points (17, ViTPose) from video, image coordinates", "skeleton"
            dev = "HD camcorder 1920 x 1080 at 50 fps, ViTPose key points"
            proc = "ViTPose pose estimates as shared, no further processing"
            assert A.shape == (250, 34), A.shape
        else:
            hz, ch, un, kind, tag = 60, IMU_CH, IMU_U, "six IMUs (shoulders, hips, feet): acceleration, angular velocity, magnetic field, orientation", "imu"
            dev = "six Xsens MTw Awinda IMUs at 60 Hz"
            proc = "none stated beyond the sensor output"
            assert A.shape[0] == 300 and A.shape[1] in (48, 60, 72), A.shape
            ch, un = IMU_CH[:A.shape[1]], IMU_U[:A.shape[1]]  # 29 files hold only 4 or 5 IMUs: the first n blocks, which sensors are missing is not stated
        fn = f"samples/{code}_{tag}.csv"
        d = pd.DataFrame(A, columns=ch)
        d.insert(0, "time_s", np.arange(len(A)) / hz)
        d.insert(0, "recording", n)
        group.setdefault((fn, tuple(IMU_CH if r["kind"] else SK_CH)), []).append(d)
        fm = "present" if r["label"] else "absent"
        rows.append(dict(
            dataset=ID, study=SRC, experiment="Fidgety movement classification", group="", subject=f"I{r['inf']}", person=f"I{r['inf']}",
            subject_excluded_by_authors=False, age=r["age"] / 365.25, sex="", handedness="", population="healthy (typically developing infants)",
            setting="lab", device=dev, session="", block=f"age {r['age']} days", trial=n, phase="test", time_s="",
            task="infant spontaneous movements (supine)", unit="recording",
            condition="fidgety movements present" if r["label"] else "fidgety movements absent",
            pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False, recording=n, samples_file=fn,
            signal_kind=kind, channels=";".join(ch), channel_units=";".join(un), sample_rate_hz=hz, n_samples=len(A), duration_s=len(A) / hz,
            events="", processing=proc, trial_info=f"modality={tag};fidgety_movements={fm};age_days={r['age']};video_frames={r['f0']}-{r['f1']}"))
    for (fn, allch), ds in group.items():  # IMU snippets with fewer sensors leave the missing columns empty
        big = pd.concat(ds, ignore_index=True).reindex(columns=["recording", "time_s"] + list(allch))
        pth = tmp_dir / fn.split("/")[-1]
        big.to_csv(pth, index=False, float_format="%.9g", lineterminator="\r\n")
        extra.append((fn, pth))
    df = pd.DataFrame(rows)
    checks.internal("skeleton and IMU snippets all 5 s long", bool((df.duration_s == 5).all()))
    checks.internal("ages within 20 to 130 days", 20 <= df.age.min() * 365.25 and df.age.max() * 365.25 <= 130,
                    f"{df.age.min()*365.25:.0f}-{df.age.max()*365.25:.0f}")
    pub = "deposit README"
    checks.published("infants", df.person.nunique(), 45, 0, f"{pub}, Skeleton data (45 infants)")
    checks.published("skeleton snippets", n_sk, 1784, 0, f"{pub}, Skeleton data (1784 files)")
    checks.published("IMU snippets", n_imu, 1683, 0, f"{pub}, IMU data (1683 files)")
    checks.published("skeleton rows per snippet (5 s at 50 Hz)", 250, int(df[df.sample_rate_hz == 50].n_samples.iloc[0]), 0, f"{pub}, Skeleton data")
    checks.published("IMU rows per snippet (5 s at 60 Hz)", 300, int(df[df.sample_rate_hz == 60].n_samples.iloc[0]), 0, f"{pub}, IMU data")
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    meta = dict(
        study=SRC, paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Each text file of skeleton key points and of IMU data became a block of rows in one samples CSV per infant and modality (recording column = the row's recording number) with a time_s column added "
                 "(row index / sampling rate within the snippet) and column names from the README; age in days from the file name converted to years. "
                 "The 1776 pressure-mat files and the authors' training/test feature matrices are not included."),
        how=["One row is one 5 s snippet of one infant (its samples are the rows of the infant's samples file with the matching recording number), recorded in either of two ways (see trial_info modality, signal_kind and channels). "
             "Skeleton snippets: 250 frames at 50 Hz of 17 ViTPose key points (x, y in image pixels of the 1920 x 1080 video). IMU snippets: 300 samples at 60 Hz "
             "from six IMUs (acc, gyr, mag in x, y, z, and roll, pitch, yaw per IMU, in this order for IMUs 1 to 6).",
             "29 IMU snippets (15 with 4 IMUs, 14 with 5) have fewer columns than the other 1654 (6 IMUs): their blocks are named imu 1..n in file order and the deposit does not say which sensors are missing. IMU units are not stated in the deposit; m/s^2 for acceleration, rad/s for angular velocity, deg for orientation and arbitrary units "
             "for the magnetic field are inferred from the Xsens MTw output and the value ranges. The 6 IMU positions (shoulders and hips, feet) "
             "are given by the README without a per-number mapping.",
             "Each snippet was labelled by assessors as fidgety movements present (FM+) or absent (FM-); this is the condition. Infants were recorded in laboratory "
             "sessions during the first months of life; age is the age at the recording. Sex is not shared per infant. The two modalities of the same "
             "time window are separate rows (the deposit's file counts differ slightly between modalities). The listing gives 51 infants; the deposit reports 45."],
        unit_note="5-second snippet (skeleton key points or IMU signals)",
        differences=["The pressure-mat recordings of the same deposit (500 frames x 32 x 32 values per snippet) were left out for size."])
    return stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)


if __name__ == "__main__":
    main()
