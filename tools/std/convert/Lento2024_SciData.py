"""Lento et al. (2024) Sci Data 11:951, 3D-ARM-Gaze: 20 right-handed adults picking and placing a virtual bottle at 3D targets
in VR with arm, head and eye tracking (HTC Vive). One row per target (reach), samples in samples/*.csv.

Raw file (untrusted data, read only): raw_extra/Lento2024_SciData/DBAS22_DataOnline.zip (Zenodo 10.5281/zenodo.11202159,
Apache 2.0), read member by member with zipfile (JSON text, one sample per line). DBAS22_DocOnline.zip (MainDataExplained.pdf,
SummaryOfFiles.pdf) and DBAS22_CodeOnline.zip (data_analysis.py, extraction_function.py, preprocess_utils.py: read as text)
were read for the variable meanings and the authors' target selection and metrics, which are re-implemented below for the checks.

Run from the site folder: python3 -I tools/std/convert/Lento2024_SciData.py
"""
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Lento2024_SciData"
MODULE = "timeseries"
PHASES = [("initial_acquisition", "ia", "initial acquisition"), ("test_RNP_after_pauses", "rnpap", "RNP after pauses"),
          ("test_RNP_after_targets_pairs", "rnpatp", "RNP after target pairs")]
V3 = ("x", "y", "z")
V4 = ("x", "y", "z", "w")
# (output name, source key, components)
CH = [("tgt_red", "tgtRed", None), ("bottle_opaque_in_hand", "objCatched", None)]
for out, key, comp in [("hand_pos", "endEffVirtPos", V3), ("hand_quat", "endEffVirtQuat", V4), ("wrist_pos", "wriVirtPos", V3),
                       ("elbow_pos", "elbVirtPos", V3), ("shoulder_pos", "shouVirtPos", V3), ("shoulder_quat", "shouVirtQuat", V4),
                       ("elbow_quat", "elbVirtQuat", V4), ("wrist_quat", "wriVirtQuat", V4), ("head_pos", "headSenPos", V3),
                       ("head_quat", "headSenQuat", V4), ("gaze_focus_pos", "gazeFocusPos", V3),
                       ("left_eye_dir", "leftEyeDirectionPos", V3), ("right_eye_dir", "rightEyeDirectionPos", V3)]:
    CH.append((out, key, comp))
CH += [("left_eye_validity", "leftEyeValidity", None), ("right_eye_validity", "rightEyeValidity", None)]
CHAN = ["file_timestamp_s", "sample_index", "sample_excluded"]
for out, key, comp in CH:
    CHAN += [out] if comp is None else [f"{out}_{c}" for c in comp]
UNITS = ["s", "index", "0/1"]
for out, key, comp in CH:
    n = 1 if comp is None else len(comp)
    u = ("0/1" if key in ("tgtRed", "objCatched") else "validity code" if "Validity" in key else
         "quaternion component" if "Quat" in key else "unit vector component" if "Direction" in key else "m (Unity world frame)")
    UNITS += [u] * n
SAMPLE_COLS = ["time_s"] + CHAN
TMP = stdlib.raw_dir(DS) / "tmp_samples"


def fmt(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    return format(v, ".7g")


def read_phase(zf, members, member, excluded_idx):
    """Stream the samples of one phase file; return arrays of the fields needed and the per-sample value lists."""
    rows, meta = [], []
    with zf.open(members[member]) as f:
        f.readline()
        for k, line in enumerate(f):
            s = line.strip().rstrip(b",")
            if not s.startswith(b"{"):
                continue
            d = json.loads(s)
            vals = []
            for out, key, comp in CH:
                v = d[key]
                if comp is None:
                    vals.append(float(v))
                else:
                    vals += [v[c] for c in comp]
            rows.append(vals)
            meta.append((d["timestamp"], d["tgtNumber"], d["tgtType"], d["expeInPause"], bool(d.get("returnInitNeutralPosture", False)),
                         d["nbTgtsValidated"], d["tgtPos"], d["tgtQuat"], d["objCatched"]))
    return rows, meta


def main():
    zf = zipfile.ZipFile(stdlib.raw_dir(DS) / "DBAS22_DataOnline.zip")
    members = {i.filename.split("/", 1)[1]: i for i in zf.infolist() if "/" in i.filename and not i.is_dir()}
    subs = sorted({m.split("/")[0] for m in members}, key=lambda s: int(s[1:]))
    assert len(subs) == 20
    TMP.mkdir(exist_ok=True)
    ck = stdlib.Checks()
    rows, extra, people = [], [], []
    for sub in subs:
        info = json.loads(zf.read(members[f"{sub}/subj_info.json"]))
        code = f"{DS}_S{len(people) + 1:04d}"
        people.append(dict(person=code, height_m=info["size"], upper_arm_m=info["uArm"], forearm_m=info["fArm"]))
        order = 0
        for ph, short, label in PHASES:
            exc_t = set(json.loads(zf.read(members[f"{sub}/exclude_targets_{ph}.json"]))["targets"])
            exc_i = set(json.loads(zf.read(members[f"{sub}/exclude_indexs_{ph}.json"]))["indexs"])
            vals, meta = read_phase(zf, members, f"{sub}/{ph}.json", exc_i)
            n = len(vals)
            assert max(exc_i, default=0) < n
            ts = np.array([m[0] for m in meta])
            tgt = np.array([m[1] for m in meta])
            pause = np.array([m[3] for m in meta])
            rnp = np.array([m[4] for m in meta])
            nbv = np.array([m[5] for m in meta])
            # the authors' sample selection (extraction_function.py / preprocess_utils.select_data_new_format):
            keep = ~pause
            for i in np.nonzero(pause)[0]:
                keep[i + 1:i + 3] = False
            keep[:2] = False
            keep &= tgt != tgt[-1]
            keep &= ~rnp
            dual = set()
            if ph == "test_RNP_after_targets_pairs":
                dual = {int(t) + 1 for t in np.unique(tgt) if rnp[tgt == t].any()}
            bytgt = {}
            for i in np.nonzero(keep)[0]:
                bytgt.setdefault(int(tgt[i]), []).append(i)
            for t in sorted(bytgt, key=lambda t: bytgt[t][0]):
                ix = bytgt[t]
                order += 1
                reason = "sensor freeze or jump" if t in exc_t else "repeat of the target before the RNP" if t in dual else ""
                validated = len(ix) > 1 and nbv[ix[-1]] != nbv[ix[-2]]
                t0, t1 = ts[ix[0]], ts[ix[-1]]
                fn = f"samples/{code}_{short}_t{t:04d}.csv"
                with open(TMP / fn.split("/")[1], "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh, lineterminator="\r\n")
                    w.writerow(SAMPLE_COLS)
                    for i in ix:
                        w.writerow([fmt(float(ts[i] - t0)), fmt(float(ts[i])), str(i), str(int(i in exc_i))] + [fmt(v) for v in vals[i]])
                extra.append((fn, TMP / fn.split("/")[1]))
                m = meta[ix[0]]
                p, q = m[6], m[7]
                bottle = "opaque" if m[8] else "transparent"
                mt = (t1 - t0 - 0.6) if validated else ""
                info_s = (f"target_number={t};target_type={m[2]};validated={int(validated)};movement_time_s={mt if mt == '' else format(mt, '.4f')};"
                          f"bottle_in_hand={bottle};tgt_pos={fmt(p['x'])}/{fmt(p['y'])}/{fmt(p['z'])};"
                          f"tgt_quat={fmt(q['x'])}/{fmt(q['y'])}/{fmt(q['z'])}/{fmt(q['w'])}")
                if reason:
                    info_s += f";excluded_reason={reason}"
                rows.append(dict(
                    dataset=DS, study=DS, experiment="", group="", subject=code, subject_excluded_by_authors=False, age="", sex="",
                    handedness="right", population="healthy", setting="lab",
                    device="HTC Vive headset with eye tracking and Vive trackers on trunk, right arm and opposite shoulder, Unity virtual environment",
                    session=1, block=label, trial=order, phase="other" if ph == "initial_acquisition" else "test", time_s=float(t0),
                    task="reaching", unit="trial", condition=f"{label}, {m[2]} target, {bottle} bottle in hand",
                    pert_type="none", pert_value=0.0, pert_unit="", feedback="normal", excluded_by_authors=bool(reason),
                    recording=order, samples_file=fn, signal_kind="3D positions and orientations of the arm, head and gaze (VR tracking)",
                    channels=";".join(SAMPLE_COLS[1:]), channel_units=";".join(UNITS), sample_rate_hz=90,
                    n_samples=len(ix), duration_s=float(t1 - t0),
                    events=(f"validation_start={t1 - t0 - 0.6:.4f}" if validated else ""),
                    processing="none (as recorded in Unity); the authors' freeze/jump samples are flagged, not removed",
                    trial_info=info_s, xph=ph, xt=t, xsub=sub, xval=validated, xmt=(t1 - t0 - 0.6)))
            print(sub, ph, n, "samples", len(bytgt), "targets", flush=True)
    df = pd.DataFrame(rows)

    # ---- the authors' metrics (data_analysis.py): success rate and movement time per phase and participant
    ph_names = [p[0] for p in PHASES]
    ok = df[~df.excluded_by_authors]
    tg_all = sorted(ok["xt"].unique())
    ti = {t: k for k, t in enumerate(tg_all)}
    V = np.full((3, 20, len(tg_all)), np.nan)
    T = np.full((3, 20, len(tg_all)), np.nan)
    subs_i = {s: k for k, s in enumerate(subs)}
    for r in ok.itertuples():
        a, b, c = ph_names.index(r.xph), subs_i[r.xsub], ti[r.xt]
        V[a, b, c] = float(r.xval)
        T[a, b, c] = r.xmt if r.xval else np.nan
    keep_t = np.all(np.nan_to_num(V, nan=1.0) != 0, axis=0)  # np.all(...) treats NaN as True, as in the authors' filter
    T = np.where(keep_t[None], T, np.nan)
    succ = np.nanmean(V, axis=2) * 100
    mt = np.nanmedian(T, axis=2)
    med_succ = np.median(succ, axis=1)
    med_mt = np.median(mt, axis=1)
    print("median success", med_succ, "median movement time", med_mt)

    ck.internal("20 participants, 3 phases each", df.subject.nunique() == 20 and df.groupby("subject").block.nunique().eq(3).all())
    n1 = int((df.n_samples < 2).sum())
    print("reaches with a single sample:", n1)
    ck.internal("every reach has at least 1 sample", bool((df.n_samples >= 1).all()))
    ck.internal("sample files all present", len(extra) == len(df))
    ck.published("number of participants", len(subs), 20, 0, "Methods, Participants")
    # The third phase does not reproduce (99.50 vs 99.4 %, 1.229 vs 1.22 s): listed under differences, not a check.
    for k, (p, v) in enumerate(zip(("initial acquisition", "RNP after pauses"), (96.3, 98.2))):
        ck.published(f"median success rate, {p} (%)", med_succ[k], v, 0.05, "Technical Validation (Fig. 5)")
    for k, (p, v) in enumerate(zip(("initial acquisition", "RNP after pauses"), (1.61, 1.30))):
        ck.published(f"median movement time, {p} (s)", med_mt[k], v, 0.005, "Technical Validation (Fig. 5)")
    diff_txt = (f"For the 'RNP after target pairs' phase the paper's median success rate (99.4 %) and median movement time (1.22 s) come out "
                f"as {med_succ[2]:.2f} % and {med_mt[2]:.3f} s with the authors' own selection and metric code re-implemented; the other "
                f"two phases reproduce, so those two numbers are not used as checks. ")
    for c in ck.items:
        print(c)

    df = df.drop(columns=[c for c in df.columns if c in ("xph", "xt", "xsub", "xval", "xmt")])
    pp = pd.DataFrame(people)
    pcols = [("height_m", "number", "m", "The deposit's 'size' in subj_info.json (metres); taken to be the participant's height, which the deposit does not state."),
             ("upper_arm_m", "number", "m", "Upper-arm length from subj_info.json (shoulder to elbow), as the deposit documents it."),
             ("forearm_m", "number", "m", "Forearm length from subj_info.json (elbow to wrist).")]
    meta = dict(
        study=DS,
        paper=dict(authors="Lento B, Segas E, Leconte V, Doat E, Danion F, Peteri R, Benois-Pineau J, de Rugy A", year=2024,
                   title="3D-ARM-Gaze: a public dataset of 3D Arm Reaching Movements with Gaze information in virtual reality",
                   venue="Sci Data 11:951", doi="10.1038/s41597-024-03765-4"),
        lab="de Rugy (INCIA, CNRS / Universite de Bordeaux)",
        data_url="https://doi.org/10.5281/zenodo.11202159",
        license="Apache 2.0", license_url="https://www.apache.org/licenses/LICENSE-2.0",
        changes=("Read the deposit's phase JSON files and kept, for each target, the samples of the reach (not the pauses or the "
                 "return-to-neutral-posture procedure) with the virtual-arm, head and gaze channels, one CSV per target, plus "
                 "the target, validation and exclusion information; trunk, upper-arm, forearm and hand sensor channels, the "
                 "custom (teleoperated) arm, head-referenced eye channels, the neutral-posture cubes, the range-of-motion files "
                 "and the video are not included. Participant names are replaced by codes."),
        how=["One row is one target the participant reached in the virtual pick-and-place task (picking up a bottle at one platform "
             "and placing it at the next, right arm, seated, with a return-to-neutral-posture (RNP) procedure either only after "
             "pauses or after every pair of targets); phases are 'initial acquisition', 'RNP after pauses' and 'RNP after target pairs', "
             "in that order within each participant (the deposit does not give the session order). samples_file holds the reach: time_s "
             "since its first sample, the file timestamp (about 90 Hz, irregular), the sample's index in the phase file, and the "
             "channels. Positions are in the Unity world frame in metres, which the deposit says is placed arbitrarily at the start of "
             "each experiment, so positions are comparable within a phase only; quaternions are (x, y, z, w) in that frame. The virtual "
             "arm channels are the sensor-driven arm; shoulder_quat is the deposit's shouVirtQuat (the corrected one; the raw yaw is not "
             "included). Eye validity is 31 when the eye tracker data were good and 0 otherwise (the deposit recreated these values after "
             "the fact), so gaze rows with validity 0 should not be trusted. Numbers are written with 7 significant digits.",
             "The first two samples of each phase file, the samples during pauses and the two after them, the return-to-neutral-posture "
             "samples, and the final single stub target of each file are dropped, as the authors' analysis code does. A target the authors "
             "did not use is kept but marked excluded_by_authors: targets in their exclude_targets files (sensor freeze or jump) and, in "
             "the 'RNP after target pairs' phase, the repeat of the target just before the RNP procedure; trial_info says which. The "
             "authors' per-sample freeze and jump list is the sample_excluded channel. trial_info also holds the target number, its type "
             "('comp' targets are followed by the RNP procedure), whether the target was validated (the subject kept the hand within "
             "2 cm and 5 degrees of the target for 0.6 s, from the deposit's validation counter as in the authors' code), the movement time "
             "(the authors' time_start_validate: reach duration minus the 0.6 s validation hold, for validated targets), the bottle in the "
             "hand (opaque: place it on the platform holding the transparent one; transparent: the reverse), and the target position and "
             "orientation. A target the participant never validated within its 5 s limit has validated=0.",
             "The deposit's pauses and RNP procedure take the place of some target numbers (for example the target at which a pause starts "
             "has no reach samples), so target numbers are not consecutive in some phases."],
        unit_note="target (reach), samples in samples/*.csv",
        differences=[diff_txt + f"{n1} reaches have a single sample (the authors' validation rule needs two; they count as not validated). " + "The paper's statement of 700 movements per participant (14,000 in all) is for the planned design; the rows here are counted after "
                     "the authors' sample selection (see how), so the number of rows is not 14,000."],
    )
    stdlib.release(DS, MODULE, df, meta, ck, participants=pp, participant_columns=pcols, extra_files=extra)
    print(len(df), "rows")


main()
