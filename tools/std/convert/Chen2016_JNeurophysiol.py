"""Chen, Valsecchi & Gegenfurtner (2016) J Neurophysiol 116(6):2586-2593: role of motor execution in the ocular tracking of
self-generated movements.

Raw file: raw_extra/Chen2016_JNeurophysiol/data.zip (Zenodo 154406, 'data_JNP 2015b.zip'). Used: 'Matlab data/Sub<n>/Trial<k>.mat'
(one MATLAB file per trial with the Zebris marker recording of the target that moved with the participant's finger, plus the trial's
direction and congruency) and 'data description.txt'. NOT used: 'Eyelink data/S<n>.edf' (EyeLink binary files that cannot be read
without the manufacturer's software), the authors' MATLAB experiment programs (read as text only, never run) and the calibration
files. Participant numbers are the folder numbers, replaced by codes. Module: timeseries (one row per trial, marker samples as CSV).

Run from the site folder: python3 -I tools/std/convert/Chen2016_JNeurophysiol.py
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Chen2016_JNeurophysiol"
RAW = stdlib.raw_dir(DS)
TMP = RAW / "_samples"
TMP.mkdir(exist_ok=True)


def main():
    ck = stdlib.Checks()
    z = zipfile.ZipFile(RAW / "data.zip")
    pat = re.compile(r"Matlab data/Sub(\d+)/Trial(\d+)\.mat$")
    items = sorted(((int(m[1]), int(m[2]), n) for n in z.namelist() if (m := pat.search(n))))
    subs = sorted({s for s, _, _ in items})
    ck.published("participants", len(subs), 12, 0, "Methods, Participants (12 observers)")
    per = {s: sorted((t, n) for ss, t, n in items if ss == s) for s in subs}
    ck.internal("trial numbers run 1..N without gaps for every participant",
                all([t for t, _ in per[s]] == list(range(1, len(per[s]) + 1)) for s in subs))
    ck.published("trials in total (11 participants x 320 plus participant 1 x 480)", len(items), 11 * 320 + 480, 0,
                 "Methods, Procedure (320 trials, 80 per condition; participant 1 completed 480, 120 per condition)")
    ck.published("trials per participant, participants with 320 trials", sum(len(per[s]) == 320 for s in subs), 11, 0,
                 "Methods, Procedure (320 trials per participant, except participant 1)")
    ck.published("trials of the participant with extra trials", max(len(per[s]) for s in subs), 480, 0,
                 "Methods, Procedure (participant 1 completed 480 trials, 120 per condition)")

    rows, files = [], []
    dts = []
    cell = {}
    n_nan_first = 0
    for s in subs:
        prev = None
        blk = 0
        for t, n in per[s]:
            m = sio.loadmat(io.BytesIO(z.read(n)))
            cong = int(m["which_condition"].ravel()[0])
            dr = int(m["which_dir"].ravel()[0])
            assert cong in (1, 2) and dr in (1, 2)
            mk = np.asarray(m["markers"], dtype=float)
            assert mk.shape[0] == 5
            ts = mk[3]
            n_nan_first += bool(np.isnan(mk[0, 0]))
            dts.extend(np.diff(ts).tolist())
            trig = float(m["trig_time"].ravel()[0])
            t0 = ts[0]
            key = (cong, dr)
            if key != prev:
                blk += 1
                prev = key
            cell.setdefault((s, key), 0)
            cell[(s, key)] += 1
            fname = f"p{s:02d}_t{t:03d}.csv"
            with open(TMP / fname, "w", newline="") as fh:
                fh.write("time_s,target_x_mm,target_y_mm,target_z_mm,digital_input\n")
                for i in range(mk.shape[1]):
                    vals = [("" if np.isnan(v) else f"{v:.8g}") for v in mk[:3, i]]
                    fh.write(f"{ts[i] - t0:.4f},{vals[0]},{vals[1]},{vals[2]},{int(mk[4, i])}\n")
            files.append((f"samples/{fname}", str(TMP / fname)))
            cname = "congruent" if cong == 1 else "incongruent"
            dname = "left" if dr == 1 else "right"
            rows.append(dict(
                dataset=DS, study=DS, experiment="finger-driven target tracking", group="", subject=f"P{s:02d}", person=f"P{s:02d}",
                subject_excluded_by_authors=False, age="", sex="", handedness="right", population="healthy", setting="lab",
                device="Zebris ultrasound marker (100 Hz) on the target; right index finger moved the target on a belt; EyeLink 1000 gaze not shared in readable form",
                session=1, block=f"block {blk}", trial=t, phase="test", time_s="",
                task="eye movements", unit="recording", condition=f"{cname} mapping, finger moves {dname}",
                pert_type="target_mapping", pert_value=0 if cong == 1 else 1,
                pert_unit="0 = target moves with the finger, 1 = target moves against the finger",
                feedback="normal", excluded_by_authors=False,
                recording=t, samples_file=f"samples/{fname}", signal_kind="3D position of the Zebris marker that was the visual target (it moved with or against the right index finger)",
                channels="target_x_mm;target_y_mm;target_z_mm;digital_input", channel_units="mm;mm;mm;0/1 trigger line",
                sample_rate_hz=100, n_samples=mk.shape[1], duration_s=float(ts[-1] - t0),
                events=f"sync_trigger_to_eyetracker={trig - t0:.4f}",
                processing="none (raw Zebris samples as saved by the authors; the first sample of each recording is empty in the authors' files)",
                trial_info=f"congruency={cname};finger_direction={dname}"))
    ck.internal("median sampling interval is close to 10 ms", 0.0095 < float(np.median(dts)) < 0.0105, f"{float(np.median(dts)):.5f} s")
    ck.internal("every participant has equal trials in all four congruency x direction cells",
                all(len({cell[(s, k)] for k in [(1, 1), (1, 2), (2, 1), (2, 2)]}) == 1 for s in subs))
    ck.published("trials per condition (incongruent) for a 320-trial participant",
                 sum(cell[(2, k)] for k in [(1, 1), (1, 2)]) / 2, 80, 0, "Methods, Procedure (80 trials per condition)")
    ck.internal("one row and one sample file per trial", len(rows) == len(files) == 4000)

    meta = dict(
        study=DS,
        paper=dict(authors="Chen J, Valsecchi M, Gegenfurtner KR", year=2016,
                   title="Role of motor execution in the ocular tracking of self-generated movements", venue="J Neurophysiol 116(6):2586-2593",
                   doi="10.1152/jn.00574.2016"),
        lab="Gegenfurtner (University of Giessen)",
        data_url="https://doi.org/10.5281/zenodo.154406",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Only the Zebris marker recordings were converted: one CSV per trial (time from the first sample, x y z in mm, digital input). "
                 "The eye recordings (EyeLink .edf files) are in the authors' deposit in a binary format that cannot be read without the "
                 "manufacturer's software and are not part of this release, so the gaze results of the paper cannot be rebuilt from it. "
                 "Participant folder numbers were replaced by codes."),
        unit_note="trial (recording) of one participant",
        how=["One row is one trial: the participant pressed a key, heard a cue and moved the right index finger horizontally to the left or to the "
             "right and back while a visible target (a Zebris marker on a belt, the finger itself hidden) moved either with the finger "
             "(congruent mapping) or against it (incongruent mapping). Each participant did four blocks, one for each combination of mapping and "
             "finger direction, in a counterbalanced order (80 trials per block, 120 for participant 1). The condition column and trial_info give "
             "mapping and direction; block numbers the runs of identical condition.",
             "The samples file holds the 3D position of the target marker in mm (the authors' file gives no unit; mm is inferred from the "
             "range of about 10 cm along the movement and from the authors' statement that the movement was limited to 17 cm), sampled at about 100 Hz, "
             "and a digital input line that the program raised 0.5 s after the cue to synchronize with the eye tracker. time_s counts from the "
             "first sample of the recording; the sync_trigger time in events is on the same scale. The authors' axes are as saved; the movement "
             "is along the y and z axes.",
             "Direction codes come from the authors' data description (which_dir 1 = left, 2 = right; which_condition 1 = congruent, "
             "2 = incongruent). Age, sex and handedness are per participant in the paper only as group statistics (12 right-handed, 7 women, mean "
             "age 26.6), so they are not given per row."],
        differences=["The paper rejected 6.6% of trials after visual inspection; the deposit has no per-trial exclusion flags, so every trial is a row "
                     "and excluded_by_authors is False.",
                     "The paper gives a 100 Hz Zebris rate; the median interval between the saved time stamps is 9.9 ms (about 101 Hz), because the "
                     "stamps are the computer's read times, so n_samples and the time stamps are as saved and sample_rate_hz states the nominal 100 Hz.",
                     "Eye movement results (latency, pursuit gain, saccade rate) cannot be reproduced: the gaze data are only in EyeLink .edf files "
                     "that were not converted."],
    )
    stdlib.release(DS, "timeseries", rows, meta, ck, extra_files=files)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
