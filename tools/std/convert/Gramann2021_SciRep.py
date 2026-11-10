"""Gramann et al. (2021), Sci Rep 11:18186: EEG and motion capture during full-body heading changes (OpenNeuro ds004460, CC0).

Raw files, raw_extra/Gramann2021_SciRep/ (all flattened from the BIDS tree, names kept):
  participants.tsv, README
  sub-NN_ses-body_task-Rotation_tracksys-HTCVive_motion.tsv   head pose in the physical rotation session (no header; 9 columns, see
                       the channels.tsv: quaternion w x y z, position x y z (m), PhaseSpace-aligned yaw, LSL timestamp in seconds)
  sub-NN_ses-joy_task-Rotation_tracksys-VIRPos_motion.tsv     virtual head pose in the joystick session (8 columns: quaternion,
                       position, timestamp)
  sub-NN_ses-*_task-Rotation_events.tsv                       experiment markers (onset in seconds from the EEG start)
  sub-NN_ses-*_scans.tsv                                      acquisition offsets of the motion files relative to the EEG
Not converted: the EEG, and the PhaseSpace full-body motion capture (four rigid bodies, 100-170 MB per participant; the deposit does not
say which rigid body is which body part).
Run from the site folder: python3 -I tools/std/convert/Gramann2021_SciRep.py
"""
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Gramann2021_SciRep"
SRC = "Gramann et al. 2021, Sci Rep"
raw = stdlib.raw_dir(ID)
N_SUB = 20

parts = pd.read_csv(raw / "participants.tsv", sep="\t")
assert len(parts) == N_SUB
SESS = [  # (BIDS session, tracking system, label, session number, channel names, units)
    ("body", "HTCVive", "physical rotation",
     ["quat_w", "quat_x", "quat_y", "quat_z", "pos_x", "pos_y", "pos_z", "ps_vive_yaw_offset_deg"],
     ["quaternion", "quaternion", "quaternion", "quaternion", "m", "m", "m", "deg"], 1),
    ("joy", "VIRPos", "joystick rotation",
     ["quat_w", "quat_x", "quat_y", "quat_z", "pos_x", "pos_y", "pos_z"],
     ["quaternion", "quaternion", "quaternion", "quaternion", "virtual m", "virtual m", "virtual m"], 2),
]
DEVICE = {"body": "HTC Vive Pro head-mounted display with Lighthouse head tracking (physical whole-body rotation on the spot)",
          "joy": "40-inch TV monitor at 1.5 m and a joystick (virtual rotation); the head pose is the virtual viewpoint"}


def acq_seconds(txt):
    m = re.search(r"T(\d+):(\d+):(\d+(?:\.\d+)?)$", txt)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def parse_events(path):
    """Returns (trials, raw marker rows kept, counts). Markers are text, one line per marker, onset in seconds from the EEG start."""
    ev = [l.split("\t") for l in path.read_text(encoding="utf-8").splitlines()[1:]]
    kept, trials, cur, pending_ground = [], [], None, None
    c = dict(breaks=0, learn=0, test=0)
    for e in ev:
        t, v = float(e[0]), e[4]
        if v.startswith("('/"):
            continue  # the authors' HED annotation strings repeated at each event; not kept
        kept.append((t, v))
        if v == "break start" or v.startswith("break start,"):
            c["breaks"] += 1
        m = re.match(r"start_outward_rotation:[^,]*,(learn|test),(?:learn:)?(\d+),(clock|anticlock),(fast|slow),(\d+),", v)
        if m:
            cur = dict(phase=m.group(1), trial_nr=int(m.group(2)), direction=m.group(3), speed=m.group(4), angle_label_deg=int(m.group(5)),
                       ground_rotate_angle_deg=pending_ground, start_outward_s=t, end_outward_s=None, start_back_s=None, end_back_s=None,
                       duration_outward_s=None, duration_back_s=None)
            pending_ground = None
            trials.append(cur)
            c[m.group(1)] += 1
            continue
        m = re.match(r"ground_rotate_angle,(-?[\d.]+)$", v)
        if m:
            pending_ground = float(m.group(1))
            continue
        if cur is None:
            continue
        for key, pat in (("end_outward_s", "end_outward_rotation,"), ("start_back_s", "start_back_rotation,"), ("end_back_s", "end_back_rotation,")):
            if v.startswith(pat) and cur[key] is None:
                cur[key] = t
        m = re.match(r"duration_(outward|back)_rotation,([\d.]+),", v)
        if m:
            cur["duration_%s_s" % m.group(1)] = float(m.group(2))
    return trials, kept, c


def f(v):
    return "" if v is None else format(v, ".8g")


rows, extra = [], []
checks_data = dict(test_counts={}, learn_counts={}, breaks={}, labels=[], align=[], rates=[], n_per_rec=[])
for i in range(1, N_SUB + 1):
    sub = f"sub-{i:02d}"
    prow = parts[parts.participant_id == sub].iloc[0]
    code = f"S{i:04d}"
    for ses, tracksys, label, chans, units, ses_no in SESS:
        mfile = raw / f"{sub}_ses-{ses}_task-Rotation_tracksys-{tracksys}_motion.tsv"
        efile = raw / f"{sub}_ses-{ses}_task-Rotation_events.tsv"
        sfile = raw / f"{sub}_ses-{ses}_scans.tsv"
        df = pd.read_csv(mfile, sep="\t", header=None, na_values=["n/a"])
        assert df.shape[1] == len(chans) + 1, (mfile.name, df.shape)
        ts = df.iloc[:, -1].to_numpy(float)
        assert np.all(np.diff(ts) > 0)
        scans = pd.read_csv(sfile, sep="\t")
        eeg_t = acq_seconds(scans[scans.filename.str.contains("_eeg.vhdr")].acq_time.iloc[0])
        mot_t = acq_seconds(scans[scans.filename.str.contains("_motion.tsv")].acq_time.iloc[0])
        t0 = mot_t - eeg_t
        time_s = ts - ts[0] + t0
        out = pd.DataFrame(df.iloc[:, :-1].to_numpy(float), columns=chans)
        out.insert(0, "time_s", np.round(time_s, 6))
        buf = io.StringIO()
        out.to_csv(buf, index=False, float_format="%.8g", lineterminator="\r\n")
        arc = f"samples/{code}_{ses}_head.csv"
        extra.append((arc, buf.getvalue().encode("utf-8")))
        trials, kept, c = parse_events(efile)
        tb = pd.DataFrame(trials)
        tb.insert(0, "trial", range(1, len(tb) + 1))
        buf = io.StringIO()
        tb.to_csv(buf, index=False, float_format="%.8g", lineterminator="\r\n")
        tarc = f"trials/{code}_{ses}_trials.csv"
        extra.append((tarc, buf.getvalue().encode("utf-8")))
        buf = io.StringIO()
        evdf = pd.DataFrame(kept, columns=["onset_s", "marker"])
        evdf.to_csv(buf, index=False, float_format="%.8f", lineterminator="\r\n")
        earc = f"events/{code}_{ses}_markers.csv"
        extra.append((earc, buf.getvalue().encode("utf-8")))
        # alignment check: the head turns between the start and the end of the outward rotation of each test trial
        turns = []
        tt, ww = out["time_s"].to_numpy(), out
        qw, qx, qy, qz = (out[k].to_numpy() for k in ("quat_w", "quat_x", "quat_y", "quat_z"))
        yaw = np.degrees(np.unwrap(2 * np.arctan2(qz, qw)))  # rotation about the axis that carries the turning (quat_z; x and y stay near 0 in the sessions probed)
        for tr in trials:
            if tr["phase"] == "test" and tr["end_outward_s"] is not None:
                a, b = np.searchsorted(tt, [tr["start_outward_s"], tr["end_outward_s"]])
                if b > a + 2:
                    turns.append(np.ptp(yaw[a:b]))
        checks_data["align"].append(float(np.median(turns)) if turns else np.nan)
        checks_data["test_counts"][(i, ses)] = c["test"]
        checks_data["learn_counts"][(i, ses)] = c["learn"]
        checks_data["breaks"][(i, ses)] = c["breaks"]
        checks_data["labels"] += [tr["angle_label_deg"] for tr in trials if tr["phase"] == "test"]
        rate = (len(ts) - 1) / (ts[-1] - ts[0])
        checks_data["rates"].append(rate)
        rows.append(dict(
            dataset=ID, study=SRC, experiment="", group="", subject=sub, person=sub, subject_excluded_by_authors=False,
            age=None, sex=None, handedness=None, population="healthy", setting="lab",
            device=DEVICE[ses], session=ses_no, block=label, trial=ses_no, phase="test", time_s=None,
            task="whole-body heading change (spot rotation)", unit="recording", condition=label, pert_type="none", pert_value=0,
            pert_unit="", feedback="normal", excluded_by_authors=False,
            recording=ses_no, samples_file=arc, signal_kind="head pose: orientation quaternion and position of the head" +
            (" (HTC Vive) and the PhaseSpace-Vive yaw offset" if ses == "body" else " (virtual viewpoint moved by the joystick)"),
            channels=";".join(chans), channel_units=";".join(units), sample_rate_hz=rate, n_samples=len(ts),
            duration_s=float(ts[-1] - ts[0]), events="", processing="none; time_s computed from the authors' timestamps (see README)",
            trial_info=f"trials_file={tarc};markers_file={earc};learning_trials={c['learn']};test_trials={c['test']};breaks={c['breaks']}"))
frame = pd.DataFrame(rows)
tc = checks_data["test_counts"]
# the participant who left after the first block (motion sickness): fewer than the 140 test trials in a session
short = sorted({i for (i, s), n in tc.items() if n < 140})
print("participants with fewer than 140 test trials:", short, {k: v for k, v in tc.items() if v != 140})
frame["subject_excluded_by_authors"] = frame["person"].isin([f"sub-{i:02d}" for i in short])
frame["excluded_by_authors"] = frame["subject_excluded_by_authors"]

ck = stdlib.Checks()
an = frame[~frame.subject_excluded_by_authors].drop_duplicates("person")
allp = frame.drop_duplicates("person")
print("DIAG align", [round(x,1) for x in checks_data["align"]])
print("DIAG rates", [round(x,2) for x in checks_data["rates"]])
print("DIAG labels not x3", sorted(set(int(x) for x in checks_data["labels"] if x % 3)))
ck.published("participants recruited", allp.person.nunique(), 20, 0, "Methods, Participants (20 healthy adults)")
ck.published("participants analysed (one dropped out after the first block)", an.person.nunique(), 19, 0, "Methods, Participants (19 remaining)")
full = [(i, s) for (i, s), n in tc.items() if i not in short]
ck.published("experimental (test) trials per participant and condition (min over analysed)", min(tc[k] for k in full), 140, 0,
             "Methods, Procedure (140 experimental trials per rotation condition)")
ck.published("experimental (test) trials per participant and condition (max over analysed)", max(tc[k] for k in full), 140, 0,
             "Methods, Procedure (140 experimental trials per rotation condition)")
ck.published("learning trials per condition (max over analysed)", max(checks_data["learn_counts"][k] for k in full), 3, 0,
             "Methods, Procedure (three learning trials before the 140)")
ck.published("breaks between the five blocks of 28 trials (min over analysed sessions)", min(checks_data["breaks"][k] for k in full), 4, 0,
             "Methods, Procedure (five blocks of 28 trials)")
lab = np.array(checks_data["labels"])
ck.published("smallest stimulus angle in the trial labels (deg)", lab.min(), 30, 0, "Methods (sphere stopped at 30 to 150 deg eccentricity)")
ck.published("largest stimulus angle in the trial labels (deg)", lab.max(), 150, 0, "Methods (sphere stopped at 30 to 150 deg eccentricity)")
ck.internal("40 recordings (20 participants x 2 sessions)", len(frame) == 40, str(len(frame)))
ck.internal("the head turns during the outward rotation (median range of heading from the quaternion per session >= 25 deg: marker times and motion are aligned)",
            np.nanmin(checks_data["align"]) >= 25, f"lowest session median {np.nanmin(checks_data['align']):.1f}")
ck.internal("effective sampling rates do not exceed the nominal 90 (head tracking) and 60 Hz (joystick)",
            all(r <= 90 for r in checks_data["rates"][0::2]) and all(r <= 60.1 for r in checks_data["rates"][1::2]),
            f"{min(checks_data['rates']):.2f}-{max(checks_data['rates']):.2f}")
odd = sorted(set(int(x) for x in lab if x % 3))
ck.published("number of test-trial angle labels that are not 3-deg steps (the two additional eccentricities)", len(odd), 2, 0,
             "Methods (additional eccentricities of 67 and 112 deg besides the 3-deg-step clusters)")
ck.published("smaller additional eccentricity (deg)", odd[0], 67, 0, "Methods (additional eccentricities of 67 and 112 deg)")
ck.published("larger additional eccentricity (deg)", odd[-1], 112, 0, "Methods (additional eccentricities of 67 and 112 deg)")
for c in ck.items:
    if not c["ok"]:
        print("FAILING:", c)
meta = dict(
    paper=dict(authors="Gramann K, Hohlefeld FU, Gehrke L, Klug M", year=2021, title="Human cortical dynamics during full-body heading changes",
               venue="Sci Rep 11(1):18186", doi="10.1038/s41598-021-97749-8"),
    lab="Gramann (TU Berlin, BeMoBIL)", data_url="https://openneuro.org/datasets/ds004460",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Only the head tracking (HTC Vive in the physical rotation session, the virtual viewpoint in the joystick session) and the "
             "experiment markers were converted: one row is one session of one participant, with the head samples in samples/, a table of "
             "trials built from the markers in trials/ and the markers themselves in events/. The EEG and the PhaseSpace full-body "
             "motion capture were not converted (size; the deposit does not say which of its four rigid bodies is which body part). "
             "Participant labels are replaced by codes; the authors' HED annotation strings at each marker (long repeated text) were dropped. "
             "The authors' timestamps (seconds on the recording clock) became time_s from the start of the EEG recording."),
    how=["One row is one session of one participant: whole-body rotation on the spot with a head-mounted display (physical rotation) or "
         "joystick rotation in front of a monitor. In each trial a sphere circled the participant at 30 m, clockwise or anticlockwise, "
         "fast or slow, and stopped at an angle between 30 and 150 deg; the participant followed it, then turned back to the initial heading "
         "and confirmed it with a button press. 3 learning and 140 experimental trials per session, in 5 blocks of 28.",
         "samples_file has the head pose: orientation as a unit quaternion (w, x, y, z), position in metres (virtual metres for the "
         "joystick session) and, for the physical session, the authors' channel Rigid_headRigid_PS_orientation_yaw (ps_vive_yaw_offset_deg: a yaw offset between the PhaseSpace and HTC Vive "
         "tracking that the README of the deposit mentions; its unit is not stated, its values lie in -180 to 180 and it is read as degrees; it is not the head heading, "
         "which is carried by the quaternion: in the sessions inspected the turning shows in quat_w and quat_z while quat_x and quat_y stay near 0, so the axis of the authors' quaternion convention is inferred, not stated). Samples are as shared (about 89.3 Hz and 60 Hz). time_s is "
         "the authors' sample timestamp minus the first one, plus the offset of the motion file from the EEG start given in scans.tsv, so "
         "that it shares a clock with the markers; the offsets (0.03-0.2 s) are the only alignment the deposit provides.",
         "trials/ holds one line per trial built from the authors' markers: phase (learn or test), trial number, direction, speed, "
         "angle_label_deg (the number the authors put in the trial label; their trials were designed at 30 to 150 deg, but the deposit does not "
         "say that this label is the final sphere eccentricity), ground_rotate_angle_deg (a separate marker of the authors, left as shared "
         "and not explained in the deposit), and the marker times of the start and end of the outward and backward rotation "
         "and the durations the authors logged. Markers for the sphere's angle in each frame are in events/ as lines 'k,angle'.",
         "Signed turning direction and heading error are not provided by the authors per trial; they can be computed from the head samples. "
         "The deposit does not record which session came first (the paper says order was balanced), so session 1 is the physical "
         "rotation session and session 2 the joystick session by label, not by time. The participant who left after the first block "
         "(motion sickness) is kept and marked excluded; their rows have fewer test trials."],
    unit_note="session (one participant, one rotation condition, head pose recording)",
    differences=["Age, sex and handedness are left empty: the deposit's participants.tsv repeats the values of participants 1-10 for participants 11-20 and "
                 "its summary (mean age 26.0, SD 3.9, range 21-34, 12 women) does not match the paper (30.25, SD 7.68, range 20-46, 11 women).",
                 "Effective sampling rates of the physical-rotation head tracking are 74-90 Hz in many sessions, below the nominal 90 Hz (timestamps are as shared).",
                 "The paper's Methods text for the ANOVA mentions eccentricities of 15, 30 and 45 deg while the Results use 45, 90 and 135 deg; "
                 "not used here.",
                 "The heading-error means in Fig. 1D and the ERSP epoch counts need the authors' preprocessing (heading error from yaw at the "
                 "response, outlier removal, EEG) and were not reproduced."],
)
if not ck.passed():
    sys.exit("checks failed")
rec = stdlib.release(ID, "timeseries", frame, meta, ck, extra_files=extra)
