"""Voudouris, Goettker, Mueller & Fiehler (2016) Vision Res 122:73-80: kinesthetic information facilitates saccades towards
proprioceptive-tactile targets.

Raw file: raw_extra/Voudouris2016_VisRes/Data.zip (Zenodo 48220). Layout: Exp_<1-3>/S<participant>_<l|c|r><s|mov>/trial-<k>.mat (one
EyeLink recording per trial, converted by the authors with Edf2Mat: FSAMPLE samples at 500 Hz, FEVENT with the tracker's messages)
and S<participant>_<cm|cs|lm|ls|rm|rs>.txt (one row of event times and codes per trial). Instructions.rtf (read as text) documents
the text file columns. Nothing in the deposit is run. Module: timeseries (one row per trial recording, gaze samples as CSV).

Run from the site folder: python3 -I tools/std/convert/Voudouris2016_VisRes.py
"""
import io
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Voudouris2016_VisRes"
RAW = stdlib.raw_dir(DS)
TMP = RAW / "_samples"
TMP.mkdir(exist_ok=True)
PAT = re.compile(r"Exp_(\d)/S(\d+)_([lcr])(s|mov)/trial-(\d+)\.mat$")
TPAT = re.compile(r"Exp_(\d)/S(\d+)_([lcr])(s|mov)/S\d+_[lcr][sm]+\.txt$")
LOC = {"l": "left", "c": "centre", "r": "right"}
LOCCODE = {"l": 1, "c": 2, "r": 3}
SKIP = ("DISPLAY_COORDS", "RECCFG", "GAZE_COORDS", "!MODE")


def main():
    ck = stdlib.Checks()
    z = zipfile.ZipFile(RAW / "Data.zip")
    mats, txts = {}, {}
    for n in z.namelist():
        if n.startswith("__MACOSX"):
            continue
        m = PAT.match(n)
        if m:
            mats[(int(m[1]), int(m[2]), m[3], m[4], int(m[5]))] = n
        t = TPAT.match(n)
        if t:
            txts[(int(t[1]), int(t[2]), t[3], t[4])] = n
    folders = sorted({k[:4] for k in mats})
    assert set(folders) == set(txts), "every recording folder has one text file"
    parts = defaultdict(set)
    for e, s, l, c in folders:
        parts[e].add(s)
    conds = defaultdict(set)
    for e, s, l, c in folders:
        conds[(e, s)].add((l, c))

    rows, files = [], []
    codes_ok = True
    back_ok = back_n = 0
    dts = []
    nlog = nextra = 0
    one_eye = True
    for (e, s, l, c) in folders:
        tx = [r.split() for r in z.read(txts[(e, s, l, c)]).decode("latin1").strip().splitlines()]  # leading and trailing tabs vary by line
        tn = sorted(k[4] for k in mats if k[:4] == (e, s, l, c))
        assert tn == list(range(1, len(tn) + 1)), (e, s, l, c)
        for k in tn:
            m = sio.loadmat(io.BytesIO(z.read(mats[(e, s, l, c, k)])), squeeze_me=True, struct_as_record=False)["Edf2Mat"]
            fs = m.FSAMPLE
            tt = np.asarray(fs.time, dtype=float)
            t0 = tt[0]
            dts.append(float(np.median(np.diff(tt))))
            gx = np.asarray(fs.gx, dtype=float)
            gy = np.asarray(fs.gy, dtype=float)
            pa = np.asarray(fs.pa, dtype=float)
            valid = [(gx[i] > -30000).any() for i in range(2)]
            one_eye &= sum(valid) == 1
            ei = 1 if valid[1] else 0
            gxe, gye, pae = gx[ei].copy(), gy[ei].copy(), pa[ei].copy()
            for a in (gxe, gye, pae):
                a[a <= -30000] = np.nan
            inp = np.asarray(fs.input)
            fname = f"e{e}_s{s:02d}_{l}{c}_t{k:03d}.csv"
            with open(TMP / fname, "w", newline="") as fh:
                fh.write("time_s,gaze_x_px,gaze_y_px,pupil_area,digital_input\n")
                for i in range(len(tt)):
                    fh.write(f"{(tt[i] - t0) / 1000:.3f},{'' if np.isnan(gxe[i]) else format(gxe[i], '.6g')},"
                             f"{'' if np.isnan(gye[i]) else format(gye[i], '.6g')},{'' if np.isnan(pae[i]) else format(pae[i], '.6g')},{int(inp[i])}\n")
            files.append((f"samples/{fname}", str(TMP / fname)))
            msgs = [(int(x.sttime), str(x.message).strip()) for x in np.atleast_1d(m.FEVENT) if str(x.codestring) == "MESSAGEEVENT"]
            msgs = [(t, g) for t, g in msgs if not g.startswith(SKIP) and not g.startswith("TRIALID")]
            ev = []
            for t, g in msgs:
                name = "drift_correction" if g.startswith("OL_DCORR") else g
                ev.append(f"{name}={(t - t0) / 1000:.3f}")
            md = {g: t for t, g in msgs}
            has_row = k <= len(tx)
            info = ""
            if has_row:
                r = tx[k - 1]
                nlog += 1
                cond_code = int(r[0])
                want = LOCCODE[l] * 10 + (1 if c == "mov" else 2)
                codes_ok &= cond_code == want
                info = f"condition_code={r[0]};stimulated_digit={r[1]};" + ";".join(f"txt_col{j + 1}={x}" for j, x in enumerate(r) if j >= 2 and x != "")
                if c == "mov" and len(r) >= 10 and "hand_to_back" in md and "hand_back" in md:
                    back_n += 1
                    back_ok += abs((md["hand_back"] - md["hand_to_back"]) - (int(r[9]) - int(r[8]))) <= 1
            else:
                nextra += 1
                info = "recording_without_text_file_row=1"
            rows.append(dict(
                dataset=DS, study=DS, experiment=f"Experiment {e}", group="", subject=f"E{e}_S{s}_{l}{c}", person=f"E{e}_S{s}",
                subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab",
                device="EyeLink II (500 Hz); fixation cross on a 1024x768 px display; tactile stimulation of the left index or middle fingertip",
                session=1, block=f"{LOC[l]} location, {'moving' if c == 'mov' else 'static'} hand", trial=k, phase="test", time_s="",
                task="eye movements", unit="recording", condition=f"target {LOC[l]}; hand {'moving' if c == 'mov' else 'static'}",
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                recording=k, samples_file=f"samples/{fname}",
                signal_kind="gaze position of the recorded eye (right eye) and pupil area while fixating and then looking to a tactile stimulus",
                channels="gaze_x_px;gaze_y_px;pupil_area;digital_input", channel_units="px;px;EyeLink pupil area units;port value",
                sample_rate_hz=500, n_samples=len(tt), duration_s=float((tt[-1] - t0) / 1000), events=";".join(ev),
                processing="none (EyeLink samples as converted by the authors; missing samples empty)", trial_info=info))

    ck.published("conditions per participant in Experiment 1 (3 target locations x static or moving, per Instructions.rtf)",
                 float(np.mean([len(conds[(1, s)]) for s in parts[1]])), 6, 0, "Deposit documentation, Instructions.rtf (LS, LMOV, CS, CMOV, RS, RMOV)")
    ck.published("conditions per participant in Experiment 2 (central location, static and moving)",
                 float(np.mean([len(conds[(2, s)]) for s in parts[2]])), 2, 0, "Deposit documentation, Instructions.rtf (CS, CMOV) and abstract (static and moving condition)")
    ck.published("conditions per participant in Experiment 3 (static central location)",
                 float(np.mean([len(conds[(3, s)]) for s in parts[3]])), 1, 0, "Deposit documentation, Instructions.rtf (CS)")
    ck.published("sampling interval (ms)", float(np.median(dts)), 2.0, 0.01, "EyeLink II at 500 Hz (RECCFG message of the recordings)")
    ck.internal("condition code in the text file matches the folder name (location digit, 1 moving / 2 static) in every row", codes_ok)
    ck.internal("hand-back duration in the recording messages equals the text file's hand-back start/end difference",
                back_n > 0 and back_ok / back_n > 0.99, f"{back_ok}/{back_n}")
    ck.internal("exactly one eye has valid gaze in every recording", one_eye)
    ck.internal("one row and one sample file per recording", len(rows) == len(files) == len(mats), f"{len(rows)} rows")

    meta = dict(
        study=DS,
        paper=dict(authors="Voudouris D, Goettker A, Mueller S, Fiehler K", year=2016,
                   title="Kinesthetic information facilitates saccades towards proprioceptive-tactile targets", venue="Vision Research 122:73-80",
                   doi="10.1016/j.visres.2016.03.008"),
        lab="Fiehler (University of Giessen)",
        data_url="https://doi.org/10.5281/zenodo.48220",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Each trial's EyeLink recording (the authors' .mat conversion) became one CSV of the recorded eye's gaze position, pupil area and "
                 "the digital input, with time in seconds from the first sample; the tracker's messages (drift correction, hand start, hand at target, tap, "
                 "tap1, tap2, hand to back, hand back) are in events on the same time scale. Values of the authors' text files are in trial_info. "
                 "Participant folder numbers were replaced by codes (the same number in different experiments is not assumed to be the same person). "
                 "Saccades were not extracted: the deposit holds raw samples and the tracker's own event parse, not the authors' saccade measures."),
        unit_note="recording (one trial) of one participant",
        how=["One row is one trial. Participants fixated a central cross and made a saccade as soon as a tactile stimulus touched the index or middle "
             "fingertip of the unseen left hand. In the static condition the hand rested at the target location; in the moving condition it first moved "
             "to the location (messages hand_start and hand_target), so that kinesthetic information was also available. Experiment 1 has three "
             "locations (left, centre, right) each static and moving; Experiments 2 and 3 use the central location (Experiment 2 static and moving with "
             "varying delays between tone or hand arrival and the stimulus, Experiment 3 static only). subject is a participant in one experiment and "
             "one folder (location and hand condition).",
             "Samples: gaze_x_px and gaze_y_px of the recorded (right) eye in display pixels (1024 x 768, as in the recording's coordinates; the viewing "
             "distance is not in the deposit), pupil_area in EyeLink units, digital_input the tracker's port value. Missing samples are empty.",
             "trial_info holds condition_code (first digit location: 1 left, 2 centre, 3 right; second digit 1 moving, 2 static), stimulated_digit "
             "(the authors' codes 1 and 2 for the two fingers; which is index and which middle is not stated), and the text file's other columns as "
             "txt_col<N> exactly as saved. For moving-hand rows Instructions.rtf lists the columns: 3 drift correction time, 4 trial onset, 5 time of "
             "sound, 6 hand movement onset, 7 hand movement end, 8 time of stimulation, 9 hand back start, 10 hand back end, 11 delay. Static-hand files have "
             "fewer columns and their meaning is not listed; by comparison with the moving files columns 3-5 are the same and the last column is "
             "the delay (inferred). The text file times are on a different clock than the recordings (only the hand-back duration was checked against the "
             "messages). The last three recordings of every folder (trial numbers after the last text row) have no text row and no messages; they are included "
             "with recording_without_text_file_row=1 (their purpose is not documented)."],
        differences=["The paper could not be read when this release was made (publisher access refused), so no number from the paper was reproduced: "
                     "the published checks are the structure given in the deposit's own Instructions.rtf (conditions per participant) and the "
                     "tracker's sampling rate. The paper's latency, accuracy and precision values are not rebuilt here.",
                     "The deposit has no exclusion flags; excluded_by_authors is False in every row."],
    )
    stdlib.release(DS, "timeseries", rows, meta, ck, extra_files=files)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
    print("rows with log:", nlog, "extra:", nextra, "participants per exp:", {e: len(v) for e, v in parts.items()})


if __name__ == "__main__":
    main()
