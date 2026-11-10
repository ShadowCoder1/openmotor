"""Kappert, van Alphen, Smeele, Balm & van der Heijden (2019), PLoS ONE 14(8):e0221593: tongue mobility measured with
stereo optical tracking in patients after surgery or chemoradiation and in healthy controls. One row per tracked recording
(one .mat file), module timeseries.

Run from the site folder: python3 -I tools/std/convert/Kappert2019_PLoSOne.py
Raw files (untrusted data, read only, loaded with scipy.io.loadmat; nothing run): raw_extra/Kappert2019_PLoSOne/Data/ from
https://osf.io/ye35d/ (Data_Manual.pdf, read as text: markerL/M/R are raw 2D tracks, markers3D raw 3D marker positions,
markers3D_corr positions after transformation to the reference landmarks, markers3D_corr_JAW with jaw compensation, fnlabel
the frame numbers of each movement; markers 1-6 as in its figure 1: 1 and 3 eye caruncles, 2 glabella, 4 nose apex, 5 tongue tip,
6 mental region). Folder Surgery / Chemoradiation / Healthy (Observer 1, Observer 1 intraobserver, Observer 2, Reproducibility).
markers3D_corr columns 1-3 are the tongue tip and 4-6 the jaw (mental) marker in the reference frame (read from the authors'
ROM_GUI.m and TongueToReferencePoint.m, as text); markers3D_corr_JAW differs only in columns 1-3 (checked below)."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Kappert2019_PLoSOne"
RAW = stdlib.raw_dir(ID) / "Data"
ck = stdlib.Checks()
CH = [f"marker{k}_{a}" for k in range(1, 7) for a in "xyz"] + ["tongue_tip_ref_x", "tongue_tip_ref_y", "tongue_tip_ref_z",
      "jaw_ref_x", "jaw_ref_y", "jaw_ref_z", "tongue_tip_jawcomp_x", "tongue_tip_jawcomp_y", "tongue_tip_jawcomp_z"]
GROUPS = {"Surgery": ("Surgery", "partial glossectomy with primary closure, at least 6 months after treatment"),
          "Chemoradiation": ("Chemoradiation", "chemoradiation for head and neck cancer, at least 6 months after treatment"),
          "Healthy": ("Healthy", "healthy")}
KINDS = [("Healthy/Observer 1", "main measurement, tracked by observer 1", "Main measurements"),
         ("Healthy/Observer 1 intraobserver", "same recording tracked again by observer 1 (intra-observer reliability)", "Reliability"),
         ("Healthy/Observer 2", "same recording tracked by observer 2 (inter-observer reliability)", "Reliability"),
         ("Healthy/Reproducibility", "repeated measurement of one healthy person (repeatability)", "Repeatability"),
         ("Surgery", "main measurement, tracked by observer 1", "Main measurements"),
         ("Chemoradiation", "main measurement, tracked by observer 1", "Main measurements")]
tmp = RAW.parent / "_samples"
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir()
rows, extra, seq = [], [], {}
hashes = {}
mismatch = []
for folder, cond, exper in KINDS:
    grp = folder.split("/")[0]
    for f in sorted((RAW / folder).glob("*.mat"), key=lambda p: p.name):
        m = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        M = m["markers3D"]
        raw = np.vstack([M.x, M.y, M.z])  # 18 x N, order x(6), y(6), z(6)
        n = raw.shape[1]
        cols = [r for k in range(6) for r in (M.x[k], M.y[k], M.z[k])]
        c, j = m["markers3D_corr"], m["markers3D_corr_JAW"]
        if c.shape[0] != n:  # corrected arrays of a different length than the raw tracks (only R19): left empty
            mismatch.append((f.name, n, c.shape[0]))
            corr = np.full((n, 9), np.nan)
        else:
            assert c.shape == (n, 6) and j.shape == (n, 6) and np.allclose(c[:, 3:], j[:, 3:]), f.name
            corr = np.column_stack([c[:, i] for i in range(6)] + [j[:, i] for i in range(3)])
        data = np.column_stack(cols + [corr[:, i] for i in range(9)])
        assert data.shape == (n, 27) and np.isfinite(data[:, :18]).all()
        code = f.stem.replace("ROM_", "")
        person = {"H024": "H24"}.get(code, "REPEAT" if code.startswith("repro") else code)
        hashes.setdefault(hashlib.md5(c.tobytes()).hexdigest(), []).append(f"{folder}/{f.name}")
        seq[person] = seq.get(person, 0) + 1
        df = pd.DataFrame(data, columns=CH)
        df.insert(0, "time_s", np.round(np.arange(n) * 0.01, 2))
        arc = f"samples/{folder.replace('/', '_').replace(' ', '')}_{code}.csv"
        p = tmp / Path(arc).name
        df.to_csv(p, index=False, float_format="%.8g", lineterminator="\r\n")
        extra.append((arc, str(p)))
        fn = m["fnlabel"]
        ev = ";".join(f"{nm}={(int(getattr(fn, a)) - 1) / 100:.2f}" for nm, a in
                      [("left_right_start", "start_leftright"), ("left_right_end", "end_leftright"),
                       ("down_up_start", "start_updown"), ("down_up_end", "end_updown")])
        gname, pop = GROUPS[grp]
        rows.append(dict(
            dataset=ID, study=ID, experiment=exper, group=gname, subject=person, person=person, subject_excluded_by_authors=False,
            population=pop, setting="clinic", device="three Basler av1000-100gc cameras (100 fps), paper cube marker on the tongue tip",
            session=1, block=exper, trial=seq[person], phase="test", task="tongue range of motion (protrusion left-right, down-up)",
            unit="recording", condition=cond, pert_type="none", pert_value=0, pert_unit="", feedback="normal",
            excluded_by_authors=False, recording=seq[person], samples_file=arc,
            signal_kind="3D positions of six facial and tongue markers (raw, in the camera frame), the tongue tip and jaw marker transformed to reference landmarks, and the jaw-compensated tongue tip",
            channels=";".join(CH), channel_units=";".join(["mm"] * 27), sample_rate_hz=100, n_samples=n, duration_s=round(n * 0.01, 2),
            events=ev, processing="markers3D raw tracking; markers3D_corr resampled and transformed to the reference frame by the authors; "
            "markers3D_corr_JAW with jaw compensation; frames where a marker was not tracked are 0 in the authors' arrays and kept as 0"))
out = pd.DataFrame(rows)
for cname in [c[1] for c in stdlib.columns_for("timeseries")]:
    if cname not in out.columns:
        out[cname] = np.nan
dup = [v for v in hashes.values() if len(v) > 1]
by = out.groupby("group").size().to_dict()
ck.internal("82 recordings: 19 surgery, 19 chemoradiation, 21+9+9+5 healthy; corr columns 4-6 of the jaw-compensated array equal the plain array in every file",
            len(out) == 82 and by == {"Chemoradiation": 19, "Healthy": 44, "Surgery": 19}, str(by))
ck.internal("the only file whose corrected arrays have another length than its raw tracks is R19 (2081 raw frames, 3647 corrected, the length of R16)", mismatch == [("ROM_R19.mat", 2081, 3647)], str(mismatch))
print("identical corrected arrays:", dup)
ck.published("participants after surgery enrolled (files in the Surgery folder)", (out.group == "Surgery").sum(), 19, 0, "Methods, Patients (19 post-surgery patients enrolled)")
ck.published("participants after chemoradiation enrolled", (out.group == "Chemoradiation").sum(), 19, 0, "Methods, Patients (19 enrolled)")
ck.published("healthy recordings re-tracked by observer 2 (inter-rater reliability)", (out.condition.str.contains("observer 2")).sum(), 9, 0, "Methods, reliability (nine healthy participants)")
ck.published("repeated measurements of one healthy participant", (out.experiment == "Repeatability").sum(), 5, 0, "Methods, repeatability (measured five times)")
ck.published("tracking rate (fps)", out.sample_rate_hz.median(), 100, 0, "Methods, optical tracking (100 fps)")


def rtotal(g, folder_filter):
    v = []
    for r in rows:
        if r["group"] == g and folder_filter(r):
            d = pd.read_csv(tmp / Path(r["samples_file"]).name)
            s, e = [float(x.split("=")[1]) for x in r["events"].split(";")[:2]]
            w = d[(d.time_s >= s) & (d.time_s <= e + 1e-9)]
            w = w[(w[["tongue_tip_jawcomp_x", "tongue_tip_jawcomp_y", "tongue_tip_jawcomp_z"]].abs().sum(axis=1) > 0)]
            v.append(w.tongue_tip_jawcomp_x.max() - w.tongue_tip_jawcomp_x.min())
    return float(np.mean(v))


main = lambda r: r["experiment"] == "Main measurements"  # noqa: E731
ck.published("mean left-right range of the tongue tip (mm), healthy, jaw-compensated (paper: peak of the distribution near 85 mm)",
             rtotal("Healthy", main), 85, 6, "Results, Fig 6 text (R_total of healthy peaks near 85 mm)")
ck.published("mean left-right range (mm), after surgery (paper: peak near 60 mm)", rtotal("Surgery", main), 60, 6,
             "Results (post-surgery R_total peaks near 60 mm)")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="van der Heijden (University of Twente / Netherlands Cancer Institute)", data_url="https://osf.io/ye35d/",
    license="none stated", license_url="",
    changes="converted to the OpenMotor format: each .mat file's arrays written as one samples CSV with a time_s column (sample number x 0.01 s) and the values unchanged; participant codes replaced.",
    unit_note="recording (one .mat file: one participant's tongue-protrusion session, 100 frames per second)",
    how=["One row is one recording: a person (patient after surgery, after chemoradiation, or healthy control) protruding the tongue as far as possible left-right, then down-up (the paper also names a forward protrusion), tracked with three cameras. The samples file has time_s and 27 channels in mm: the raw 3D positions of the six tracked markers (marker1-6: two eye caruncles, glabella, nose apex, tongue tip, mental region, as in the deposit's manual), the tongue tip and jaw marker after transformation to the reference landmarks, and the tongue tip with jaw compensation.",
         "events gives the authors' frame labels of each movement (left_right_start/end, down_up_start/end) as seconds from the first sample. Frames where a marker was not tracked are 0 in the authors' arrays and stay 0 here. The deposit does not give age, sex, tumour stage, or which patients were excluded from the paper's analysis, and the videos are not shared (participant privacy, per the deposit's manual).",
         "Each healthy participant H17-H25 appears up to three times: tracked by observer 1, tracked again by observer 1, and tracked by observer 2 (same video, different tracking, for the reliability analysis); REPEAT is the single healthy participant measured five times. Rows keep this order within each participant (trial counts them)."],
    differences=["The paper analyses 19 healthy, 17 post-surgery and 18 post-chemoradiation participants after exclusions; the deposit has 19 surgery and 19 chemoradiation recordings and does not say which were excluded, so none is flagged (subject_excluded_by_authors is False for all).",
                 f"Healthy folder 'Observer 1' has 21 files with IDs H01-H11, H17-H25 and H024; H024 holds exactly the same corrected arrays as H24 and is coded as the same person. In Chemoradiation, R19's corrected arrays (markers3D_corr and markers3D_corr_JAW) are identical to R16's and have R16's length (3647 frames) while R19's raw tracks have 2081 frames, so they look like a copy of R16's; the corrected channels of R19 are left empty here and its raw channels are kept.",
                 "The paper reports repeatability SD of about 3.5 mm without and 2.5 mm with jaw compensation for range; the left-right range from the five repeated recordings computed here from the authors' arrays and frame labels is about 4.2 mm and 3.4 mm, so that value is not reproduced exactly (the authors' own range definition is not given in the deposit)."],
)
stdlib.release(ID, "timeseries", out, meta, ck, extra_files=extra)
shutil.rmtree(tmp, ignore_errors=True)
