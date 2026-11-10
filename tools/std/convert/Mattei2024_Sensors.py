"""Mattei et al. (2024), Sensors 24(16):5207 (MOVING dataset): hand tracking of the Virtual Glove, two orthogonal Leap Motion
controllers (horizontal and vertical) recording the right hand of healthy volunteers during 10 minutes of open/close,
wrist rotation and finger tapping movements (with rest and motor imagery) cued on a screen. One row per sensor recording.

Raw file (untrusted data, read only): raw_extra/Mattei2024_Sensors/virtual_glove.rar (Zenodo 17152403, CC BY 4.0), a non-solid
RAR5 of 20 JSON files (Subj_NN_bci_32_gesture.{Horizontal,Vertical}Leap_undefined.json, about 925 MB each). One member at a
time is unpacked with 7-Zip, stream-parsed (Mattei2024_Sensors_leap.py) for the timestamp, the palm position and the five
fingertip positions of the first hand, written as a samples CSV, and the JSON is deleted before the next member.
edf_with_trigger.zip (EEG) is not used.
Run from the site folder: python3 -I tools/std/convert/Mattei2024_Sensors.py
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Mattei2024_Sensors_leap import frames  # noqa: E402

ID = "Mattei2024_Sensors"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
SEVENZ = r"C:\Program Files\7-Zip\7z.exe"
RAR = RAW / "virtual_glove.rar"
TMP = RAW / "_x"
SAMPLES = RAW / "_samples"
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
CH = [f"{p}_{a}" for p in ["palm"] + [f"{f}_tip" for f in FINGERS] for a in "xyz"] + ["confidence", "n_hands", "hand_is_left"]
CU = ";".join(["m (inferred)"] * 18 + ["0 to 1", "count", "0 right, 1 left"])
MEMBER = re.compile(r"virtual_glove\\Subj_(\d+)_bci_32_gesture\.(Horizontal|Vertical)Leap_undefined\.json")

lst = subprocess.run([SEVENZ, "l", "-slt", str(RAR)], capture_output=True, text=True, check=True).stdout
members = []
for blk in lst.split("\n\n"):
    d = dict(l.split(" = ", 1) for l in blk.splitlines() if " = " in l)
    m = MEMBER.fullmatch(d.get("Path", ""))
    if m:
        members.append((int(m.group(1)), 0 if m.group(2) == "Horizontal" else 1, d["Path"]))
members.sort()
assert len(members) == 20, len(members)
authors_subjects = sorted({m[0] for m in members})
POS = {s: i + 1 for i, s in enumerate(authors_subjects)}
SAMPLES.mkdir(exist_ok=True)

rows, extra = [], []
tot_frames = 0
durations, right_share, have_share = [], [], []
for s, orient, path in members:
    code = f"{ID}_S{POS[s]:04d}"
    name = "horizontal" if orient == 0 else "vertical"
    fn = f"samples/{code}_{name}.csv"
    p = SAMPLES / f"{code}_{name}.csv"
    if p.exists() and (SAMPLES / (p.name + ".done")).exists():
        # parsed and written by an earlier run of this converter: read the finished samples file back
        import pandas as pd
        dfc = pd.read_csv(p)
        t = dfc["time_s"].to_numpy(float)
        V = dfc[CH].to_numpy(float)
        ts = t * 1e6
    else:
        TMP.mkdir(exist_ok=True)
        r = subprocess.run([SEVENZ, "x", "-y", f"-o{TMP}", str(RAR), path], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-300:]
        jp = TMP / path
        ts, vals = [], []
        try:
            for fr in frames(jp):
                ts.append(int(fr["Timestamp"]["$numberLong"]))
                hs = fr["Hands"]
                if not hs:
                    vals.append([np.nan] * 18 + [np.nan, 0, np.nan])
                    continue
                h = hs[0]
                v = [h["PalmPosition"][a] for a in "xyz"]
                tips = {f["Type"]: f["TipPosition"] for f in h["Fingers"]}
                for k in range(5):
                    v += [tips[k][a] for a in "xyz"] if k in tips else [np.nan] * 3
                vals.append(v + [h["Confidence"], len(hs), 1 if h["IsLeft"] else 0])
        finally:
            import gc
            import time
            gc.collect()
            for _try in range(30):
                try:
                    shutil.rmtree(TMP)
                    break
                except OSError:
                    time.sleep(2)
            assert not TMP.exists(), "could not delete the unpacked JSON"
        ts = np.array(ts, float)
        V = np.array(vals, float)
        t = (ts - ts[0]) / 1e6
        with open(p, "w", newline="") as fh:
            fh.write("time_s," + ",".join(CH) + "\r\n")
            for ti, row in zip(t, V):
                fh.write(f"{ti:.6f}," + ",".join("" if np.isnan(x) else f"{x:.7g}" for x in row) + "\r\n")
        (SAMPLES / (p.name + ".done")).write_text("ok")
    extra.append((fn, p))
    dur = float(t[-1] - t[0])
    durations.append(dur)
    has = ~np.isnan(V[:, 0])
    have_share.append(float(has.mean()))
    right_share.append(float((V[has, 20] == 0).mean()) if has.any() else float("nan"))
    tot_frames += len(t)
    n_back = int((np.diff(ts) < 0).sum())
    rows.append(dict(
        dataset=ID, study=ID, experiment="MOVING virtual glove", group="", subject=str(s), person=str(s),
        subject_excluded_by_authors=False, age="", sex="", handedness="right", population="healthy", setting="lab",
        device=f"Virtual Glove: two orthogonal Leap Motion controllers, {name} one, under the right arm",
        session=1, block=f"{name} sensor", trial=orient + 1, phase="test", time_s="", task="hand gestures (open/close, wrist rotation, finger tapping)",
        unit="recording", condition=f"{name} Leap Motion", pert_type="none", pert_value=0, pert_unit="", feedback="none",
        excluded_by_authors=False, recording=orient + 1, samples_file=fn,
        signal_kind="Leap Motion hand tracking: palm and fingertip 3D positions", channels=";".join(CH), channel_units=CU,
        sample_rate_hz=round((len(t) - 1) / dur, 3), n_samples=len(t), duration_s=round(dur, 3), events="",
        processing="none (positions as exported; only timestamp, palm and fingertip positions of the first hand kept)",
        trial_info=f"sensor_orientation={name};frames_with_timestamp_going_backwards={n_back};frames_without_hand={int((~has).sum())}"))
shutil.rmtree(TMP, ignore_errors=True)
import pandas as pd  # noqa: E402
D = pd.DataFrame(rows)

P = checks.published
P("Leap Motion controllers (orthogonal sensors) per subject", D.groupby("person").size().max(), 2, 0, "Abstract / Section 2 (two orthogonal Leap Motion Controllers)")
P("recording length per sensor (min)", float(np.mean(durations)) / 60, 10, 0.5, "Section 2 (each recording about 10 min)")
checks.internal("20 recordings, 10 subjects", len(D) == 20 and D.person.nunique() == 10)
checks.internal("every recording has frames and a tracked hand in most of them", bool(D.n_samples.min() > 1000 and np.min(have_share) > 0.5),
                f"frames {D.n_samples.min()} to {D.n_samples.max()}, hand present {np.min(have_share):.3f} to {np.max(have_share):.3f}")

meta = dict(
    paper=dict(authors="Mattei E, Lozzi D, Di Matteo A, Cipriani A, Manes C, Placidi G", year=2024,
               title="MOVING: A Multi-Modal Dataset of EEG Signals and Virtual Glove Hand Tracking",
               venue="Sensors 24(16):5207", doi="10.3390/s24165207"),
    lab="Placidi (University of L'Aquila)", data_url="https://doi.org/10.5281/zenodo.17152403",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Each of the 20 raw Leap Motion JSON dumps (one per subject and sensor) became one samples CSV. Per frame only the timestamp, the "
             "first hand's palm position and its five fingertip positions (TipPosition), the hand's tracking confidence, the number of hands "
             "seen and whether the hand is flagged left were kept; time_s is the frame timestamp minus the first frame's, divided by 1e6 "
             "(the timestamps are in microseconds, inferred from a median frame interval of about 13.4 ms, 75 Hz). Positions are written with "
             "7 significant digits. Not carried: bone joints, directions, rotations, velocities, wrist and arm positions (about 99% of the "
             "JSON). The EEG file edf_with_trigger.zip is not part of this release."),
    how=["One row is one Leap Motion sensor recording (horizontal or vertical controller) of one volunteer's right hand during the whole "
         "10-minute session, in which cued blocks of rest, motor imagery and motor execution of three movements (hand open/close, wrist rotation, finger tapping) "
         "alternate (paper, Section 2: fixation cross 2 s, instruction 6 s, each triplet repeated 8 times). The deposit has no cue or event file, so events is empty and "
         "the movements cannot be separated here without the EEG trigger file.",
         "Samples (samples/<subject>_horizontal.csv and _vertical.csv) are the Leap frames in the order recorded: time_s, then x, y, z of the palm and of the thumb, index, middle, "
         "ring and pinky fingertips in each sensor's own frame (not registered between the two sensors), then confidence, number of hands and hand_is_left. "
         "Positions are in the Leap SDK frame; the unit is inferred to be metres (palm-to-fingertip distances in the first frame are about 6 cm, a human hand size in metres), the deposit and paper do not state it. "
         f"The frame rate varies; sample_rate_hz is the mean over the recording ({D.sample_rate_hz.min():.1f} to {D.sample_rate_hz.max():.1f} Hz across recordings, "
         f"{D.duration_s.min() / 60:.1f} to {D.duration_s.max() / 60:.1f} min long, {D.n_samples.min()} to {D.n_samples.max()} frames). Some timestamps go backwards or jump; frames are left in file order and the count of "
         f"backward steps is in trial_info. A frame with no hand has empty position cells ({100 * (1 - np.max(have_share)):.1f} to {100 * (1 - np.min(have_share)):.1f}% of frames per recording). "
         f"The hand tracked is flagged right in {100 * np.nanmin(right_share):.1f} to {100 * np.nanmax(right_share):.1f}% of the frames with a hand; the rest carry the sensor's own left-hand flag (hand_is_left = 1) and are kept."],
    differences=["The paper (and the Zenodo description) states eleven subjects, but virtual_glove.rar holds 20 files = 10 subjects x 2 sensors: the authors' "
                 "subject numbers run 1 to 8, 10 and 11, subject 9 has no Leap Motion files, so this release has 10 persons, not 11. The deposit does not say why.",
                 "The paper does not state the sensor frame rate, units or axis definitions, so none of them could be checked against it."],
    unit_note="recording (one sensor, one subject, whole session)",
)
stdlib.release(ID, "timeseries", D, meta, checks, extra_files=extra)
