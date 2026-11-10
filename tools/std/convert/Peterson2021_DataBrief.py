"""Peterson & Ferris (2021), Data Brief 39:107635: perturbed beam walking and standing (OpenNeuro ds003739, CC0).
30 young adults did four 10-minute sessions on a treadmill-mounted balance beam (pull or visual-rotation perturbations, standing or
walking at 0.22 m/s), with 128-channel EEG, neck and leg EMG, EOG, head and sacrum motion capture and pull load cells synchronised at
256 Hz in EEGLAB .set/.fdt files. Only the non-EEG channels (21 of 149: neck EMG, EOG, leg EMG, motion capture, load cells) and the
perturbation onsets (events.tsv) are released; the 128 EEG channels are never read out. One session at a time: fetch, reduce, delete raw.

Run on the PC: python3 -I tools/std/pc.py run Peterson2021_DataBrief   (an ONLY.txt file in the raw folder = first participant only, no release)
"""
import csv
import io
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402
import fetch  # noqa: E402

ID = "Peterson2021_DataBrief"
RAW = stdlib.raw_dir(ID)
TMP = stdlib.HERE / "tmp_samples" / ID
checks = stdlib.Checks()
ONLY = os.environ.get("PET_ONLY") or ((RAW / "ONLY.txt").exists() or None)
BASE = "https://s3.amazonaws.com/openneuro.org/ds003739"
KEEP_FREE_GB = 4.0
SR = 256.0
# (participant, session label, task, .fdt bytes, .set bytes, events.tsv bytes)
FILES = [
    (1, 1, 'pullstand', 94666256, 3696264, 9881),
    (1, 2, 'pullwalk', 98158220, 3743120, 9905),
    (1, 3, 'rotatestand', 100593476, 3776768, 10010),
    (1, 4, 'rotatewalk', 98567672, 3749576, 9998),
    (2, 1, 'pullstand', 97169456, 3757504, 9895),
    (2, 2, 'pullwalk', 96407768, 3747512, 9947),
    (2, 3, 'rotatestand', 97296404, 3760144, 9995),
    (2, 4, 'rotatewalk', 99780532, 3793488, 10003),
    (3, 1, 'pullstand', 95379668, 3693792, 9873),
    (3, 2, 'pullwalk', 94656720, 3684080, 9881),
    (3, 3, 'rotatestand', 95752168, 3699744, 9988),
    (3, 4, 'rotatewalk', 96240292, 3706296, 9980),
    (4, 1, 'pullstand', 93324660, 3639544, 9867),
    (4, 2, 'pullwalk', 93227512, 3637464, 9675),
    (4, 3, 'rotatestand', 94647184, 3658248, 9978),
    (4, 4, 'rotatewalk', 95438076, 3668864, 9982),
    (5, 1, 'pullstand', 94666852, 3681224, 9874),
    (5, 2, 'pullwalk', 98021140, 3722632, 8963),
    (5, 3, 'rotatestand', 100612548, 3761968, 10012),
    (5, 4, 'rotatewalk', 94871876, 3684912, 9983),
    (6, 1, 'pullstand', 94147736, 3672176, 9881),
    (6, 2, 'pullwalk', 94694864, 3679496, 9886),
    (6, 3, 'rotatestand', 94626324, 3679536, 9985),
    (6, 4, 'rotatewalk', 95046504, 3685176, 9990),
    (7, 1, 'pullstand', 96857152, 3746976, 9886),
    (7, 2, 'pullwalk', 95183584, 3724504, 9888),
    (7, 3, 'rotatestand', 95575752, 3730728, 9981),
    (7, 4, 'rotatewalk', 97512156, 3756720, 9985),
    (8, 1, 'pullstand', 95242588, 3696712, 9874),
    (8, 2, 'pullwalk', 95419600, 3699064, 9884),
    (8, 3, 'rotatestand', 97091380, 3722464, 9990),
    (8, 4, 'rotatewalk', 94840884, 3692256, 9988),
    (9, 1, 'pullstand', 94920152, 3690832, 9878),
    (9, 2, 'pullwalk', 93649480, 3673760, 9879),
    (9, 3, 'rotatestand', 97091380, 3720920, 10004),
    (9, 4, 'rotatewalk', 94489244, 3685992, 9981),
    (10, 1, 'pullstand', 93590476, 3656272, 9878),
    (10, 2, 'pullwalk', 95761704, 3685392, 9901),
    (10, 3, 'rotatestand', 95281328, 3679904, 9985),
    (10, 4, 'rotatewalk', 95007168, 3676224, 9988),
    (11, 1, 'pullstand', 95556680, 3712080, 9874),
    (11, 2, 'pullwalk', 93522532, 3684496, 9812),
    (11, 3, 'rotatestand', 94725856, 3701864, 9988),
    (11, 4, 'rotatewalk', 100455800, 3778776, 10002),
    (12, 1, 'pullstand', 96885760, 3742768, 9890),
    (12, 2, 'pullwalk', 97493680, 3750912, 9902),
    (12, 3, 'rotatestand', 95018492, 3718648, 9989),
    (12, 4, 'rotatewalk', 95380264, 3723504, 9987),
    (13, 1, 'pullstand', 94089924, 3667632, 9808),
    (13, 2, 'pullwalk', 96631864, 3701992, 9889),
    (13, 3, 'rotatestand', 96221220, 3697448, 9990),
    (13, 4, 'rotatewalk', 94401632, 3673024, 9978),
    (14, 1, 'pullstand', 92945604, 3627352, 9863),
    (14, 2, 'pullwalk', 93424192, 3633752, 9879),
    (14, 3, 'rotatestand', 94970216, 3655472, 9984),
    (14, 4, 'rotatewalk', 94168000, 3644704, 9978),
    (15, 1, 'pullstand', 93423596, 3649608, 9861),
    (15, 2, 'pullwalk', 97071712, 3698568, 9896),
    (15, 3, 'rotatestand', 94255016, 3661720, 9983),
    (15, 4, 'rotatewalk', 94489244, 3664864, 9982),
    (16, 1, 'pullstand', 92896732, 3635440, 9870),
    (16, 2, 'pullwalk', 93012356, 3636208, 9678),
    (16, 3, 'rotatestand', 94216872, 3654104, 9988),
    (16, 4, 'rotatewalk', 95213980, 3667488, 9985),
    (17, 1, 'pullstand', 93101756, 3643560, 9877),
    (17, 2, 'pullwalk', 93111888, 3642640, 9623),
    (17, 3, 'rotatestand', 95106104, 3671400, 9987),
    (17, 4, 'rotatewalk', 93961188, 3656032, 9983),
    (18, 1, 'pullstand', 93257908, 3652488, 9880),
    (18, 2, 'pullwalk', 104916264, 3809200, 9979),
    (18, 3, 'rotatestand', 94558380, 3670888, 9986),
    (18, 4, 'rotatewalk', 94225216, 3666416, 9980),
    (19, 1, 'pullstand', 92769188, 3628032, 9808),
    (19, 2, 'pullwalk', 93189964, 3633912, 9878),
    (19, 3, 'rotatestand', 93766296, 3642608, 9986),
    (19, 4, 'rotatewalk', 97510964, 3692872, 9994),
    (20, 1, 'pullstand', 93140496, 3637688, 9753),
    (20, 2, 'pullwalk', 93003416, 3633768, 9237),
    (20, 3, 'rotatestand', 93903376, 3649376, 9975),
    (20, 4, 'rotatewalk', 93765700, 3647528, 9982),
    (21, 1, 'pullstand', 93111888, 3645832, 9870),
    (21, 2, 'pullwalk', 94176940, 3659840, 9806),
    (21, 3, 'rotatestand', 94970216, 3671712, 9983),
    (21, 4, 'rotatewalk', 97540168, 3706208, 9982),
    (22, 1, 'pullstand', 92885408, 3636744, 9873),
    (22, 2, 'pullwalk', 93120828, 3639880, 9866),
    (22, 3, 'rotatestand', 93961784, 3652128, 9985),
    (22, 4, 'rotatewalk', 94157868, 3654760, 9980),
    (23, 1, 'pullstand', 93306780, 3647824, 9861),
    (23, 2, 'pullwalk', 95017896, 3670512, 9814),
    (23, 3, 'rotatestand', 95644888, 3680152, 9983),
    (23, 4, 'rotatewalk', 94412360, 3663608, 9982),
    (24, 1, 'pullstand', 93012952, 3638328, 9873),
    (24, 2, 'pullwalk', 93658420, 3646968, 9884),
    (24, 3, 'rotatestand', 96944764, 3692040, 9990),
    (24, 4, 'rotatewalk', 94324748, 3656872, 9982),
    (25, 1, 'pullstand', 93053480, 3596496, 9817),
    (25, 2, 'pullwalk', 98050940, 3663544, 9837),
    (25, 3, 'rotatestand', 70860824, 3289768, 7427),
    (25, 4, 'rotatewalk', 94529176, 3617496, 9979),
    (26, 1, 'pullstand', 93796096, 3660720, 9928),
    (26, 2, 'pullwalk', 92993880, 3649664, 9869),
    (26, 3, 'rotatestand', 94070256, 3665088, 9979),
    (26, 4, 'rotatewalk', 94322960, 3668480, 9975),
    (27, 1, 'pullstand', 94246672, 3671112, 9869),
    (27, 2, 'pullwalk', 93052884, 3654816, 9812),
    (27, 3, 'rotatestand', 94011252, 3668904, 9985),
    (27, 4, 'rotatewalk', 93990988, 3668632, 9988),
    (28, 1, 'pullstand', 92964676, 3525016, 9791),
    (28, 2, 'pullwalk', 21253360, 2531976, 1989),
    (28, 3, 'rotatestand', 93383664, 3531856, 9981),
    (28, 4, 'rotatewalk', 97325012, 3584760, 9996),
    (29, 1, 'pullstand', 93033812, 3638736, 9858),
    (29, 2, 'pullwalk', 93002224, 3638288, 9867),
    (29, 3, 'rotatestand', 94558380, 3660144, 9978),
    (29, 4, 'rotatewalk', 94460040, 3658824, 9974),
    (30, 1, 'pullstand', 93052884, 3636728, 9807),
    (30, 2, 'pullwalk', 92955140, 3635664, 9873),
    (30, 3, 'rotatestand', 94548844, 3658016, 9988),
    (30, 4, 'rotatewalk', 93961784, 3650136, 9981),
]

NONEEG = ["L_neck_emg", "R_neck_emg", "R_horizontal_eog", "L_horizontal_eog", "L_vertical_eog",
          "L_leg_emg_tibialis_anterior", "L_leg_emg_soleus", "L_leg_emg_medial_gastrocnemius", "L_leg_emg_peroneus_longus",
          "R_leg_emg_tibialis_anterior", "R_leg_emg_soleus", "R_leg_emg_medial_gastrocnemius", "R_leg_emg_peroneus_longus",
          "Mocap_sacrum_mediolateral", "Mocap_sacrum_superiorinferior", "Mocap_sacrum_anteriorposterior",
          "Mocap_head_mediolateral", "Mocap_head_superiorinferior", "Mocap_head_anteriorposterior", "L_load_cell", "R_load_cell"]
UNITS = ["uV"] * 13 + ["mm"] * 6 + ["N"] * 2
TASKS = {"pullstand": ("pull", "standing"), "pullwalk": ("pull", "walking"), "rotatestand": ("rotation", "standing"),
         "rotatewalk": ("rotation", "walking")}
DEVICE = ("BioSemi ActiveTwo (EEG, EOG, neck EMG; not released: EEG), Biometrics wired surface EMG (leg), Vicon motion capture (head and "
          "sacrum markers), Omega LCM703 load cells on the pull cables, treadmill-mounted balance beam; all streams aligned to 256 Hz")
PROCESSING = ("as stored in the deposit: EOG and neck EMG downsampled to 256 Hz, high-pass filtered at 1 Hz and referenced to the common "
              "median of all electrodes (Cleanline line-noise removal); leg EMG, motion capture and load cells low-pass filtered "
              "(4th order Butterworth, 250 Hz) and interpolated to the 256 Hz time base; then rounded to 3 significant digits here")


def free_gb():
    return shutil.disk_usage(str(stdlib.HERE)).free / 1e9


def get(name, url, nbytes):
    t0 = time.time()
    while free_gb() < KEEP_FREE_GB + nbytes / 1e9:
        if time.time() - t0 > 7200:
            raise RuntimeError("PC disk too full for 2 h")
        time.sleep(60)
    for attempt in range(5):
        try:
            fetch.main(ID, url, name)
            return
        except Exception as ex:  # network hiccup: retry
            print("fetch retry", name, repr(ex), flush=True)
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("fetch failed " + name)


def text(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "OpenMotor-curation/1.0"}), timeout=120).read().decode()


TMP.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
part = {}
for r in csv.DictReader(io.StringIO(text(BASE + "/participants.tsv")), delimiter="\t"):
    part[int(r["participant_id"].split("-")[1])] = r
rows, extra = [], []
n_events, n_dir, chan_counts, durs, seen_ch = [], [], [], [], set()
try:
    for (pi, ses, task, nfdt, nset, nev) in FILES:
        if ONLY and pi != 1:
            continue
        sub = f"sub-{pi:03d}"
        stem = f"{BASE}/{sub}/ses-{ses:02d}/eeg/{sub}_ses-{ses:02d}_task-{task}"
        tag = f"p{pi:03d}_s{ses}"
        get(tag + ".set", stem + "_eeg.set", nset)
        get(tag + "_events.tsv", stem + "_events.tsv", nev)
        get(tag + ".fdt", stem + "_eeg.fdt", nfdt)
        m = loadmat(RAW / (tag + ".set"), variable_names=["EEG"], squeeze_me=True, struct_as_record=False)["EEG"]
        nb, n = int(m.nbchan), int(m.pnts)
        assert float(m.srate) == SR, m.srate
        labs = [str(c.labels) for c in m.chanlocs]
        assert len(labs) == nb and labs[128:] == NONEEG, (pi, ses, labs[120:])
        assert os.path.getsize(RAW / (tag + ".fdt")) == n * nb * 4
        x = np.memmap(RAW / (tag + ".fdt"), dtype="<f4", mode="r", shape=(n, nb))
        arr = np.array(x[:, 128:], dtype=np.float32)  # only the 21 non-EEG channels are ever copied out of the memmap
        del x
        df = pd.DataFrame(arr, columns=NONEEG)
        df.insert(0, "time_s", np.char.mod("%.4f", np.arange(n) / SR))
        fn = f"samples/{ID}_S{pi:04d}_ses{ses:02d}.csv"
        df.to_csv(TMP / f"{ID}_S{pi:04d}_ses{ses:02d}.csv", index=False, float_format="%.3g", lineterminator="\r\n")
        extra.append((fn, str(TMP / f"{ID}_S{pi:04d}_ses{ses:02d}.csv")))
        ev = list(csv.DictReader(open(RAW / (tag + "_events.tsv")), delimiter="\t"))
        assert all(abs(float(e["sample"]) / SR - float(e["onset"])) < 0.01 for e in ev)
        vals = [e["value"] for e in ev]
        durs_ev = sorted({round(float(e["duration"]), 3) for e in ev})
        n_events.append(len(ev))
        n_dir.append(sorted((v, vals.count(v)) for v in set(vals)))
        chan_counts.append(nb)
        pert, posture = TASKS[task]
        p = part[pi]
        rows.append(dict(
            dataset=ID, study="Peterson & Ferris 2021, Data Brief", experiment="perturbed beam walking and standing", group="",
            subject=f"p{pi:03d}", person=f"p{pi:03d}", subject_excluded_by_authors=False, age=float(p["age"]), sex=p["gender"],
            handedness={"R": "right", "L": "left"}.get(p["handedness"], ""), population="healthy", setting="lab", device=DEVICE,
            session=ses, block=f"{pert} perturbation, {posture}", trial=ses, phase="test", time_s=0, task="walking",
            unit="recording", condition=f"{posture}, {pert} perturbations",
            pert_type="waist_pull" if pert == "pull" else "visual_field_rotation", pert_value=15 if pert == "pull" else 20,
            pert_unit="N" if pert == "pull" else "deg", feedback="normal", excluded_by_authors=False,
            recording=ses, samples_file=fn, signal_kind="neck and leg EMG, EOG, head and sacrum motion capture positions, pull load cells",
            channels=";".join(NONEEG), channel_units=";".join(UNITS), sample_rate_hz=SR, n_samples=n, duration_s=round(n / SR, 3),
            events=";".join(f"{e['value']}={float(e['onset']):.3f}" for e in ev), processing=PROCESSING,
            trial_info=f"session_label={task};n_perturbations={len(ev)};perturbation_duration_s={'/'.join(str(d) for d in durs_ev)}"))
        durs.append(n / SR)
        for f in RAW.glob(tag + "*"):
            f.unlink()
        print(sub, ses, n, f"free {free_gb():.1f} GB", flush=True)
    if ONLY:
        print(rows[0]["events"][:200], n_events, n_dir)
        tot = sum(os.path.getsize(f) for _, f in extra)
        print("csv bytes", tot)
        sys.exit(0)
    ages = [float(part[i]["age"]) for i in sorted(part)]
    sexes = [part[i]["gender"] for i in sorted(part)]
    checks.published("participants", len({r["person"] for r in rows}), 30, 0, "Abstract/Sec. 2: 30 healthy young adults")
    checks.published("females", sum(1 for s in sexes if s == "F"), 15, 0, "Sec. 2: 15 females, 15 males")
    checks.published("males", sum(1 for s in sexes if s == "M"), 15, 0, "Sec. 2: 15 females, 15 males")
    checks.published("mean age (years)", float(np.mean(ages)), 22.5, 0.05, "Sec. 2: 22.5 +/- 4.8 years")
    checks.published("SD of age (years)", float(np.std(ages, ddof=1)), 4.8, 0.05, "Sec. 2: 22.5 +/- 4.8 years")
    checks.published("sessions per participant", len(rows) / 30, 4, 0, "Sec. 2.1: four 10 min sessions")
    checks.internal("every session has at least one perturbation event", min(n_events) > 0, f"min {min(n_events)}, max {max(n_events)}")
    short = [k for k in n_events if k != 150]
    checks.published("total channels in the recordings (EEG + non-EEG)", chan_counts[0], 149, 0, "Specifications table: 149 total channels")
    checks.published("non-EEG channels released", len(NONEEG), 21, 0, "Sec. 1: neck EMG 2 + EOG 3 + leg EMG 8 + motion capture 6 + pull force 2")
    checks.published("sampling rate (Hz)", SR, 256, 0, "Sec. 2.1 and Fig. 1: synchronised at 256 Hz")
    checks.internal("all recordings 21 channels with the expected labels and a samples file", all(r["samples_file"] for r in rows))
    checks.internal("session durations positive and under 12 minutes", all(60 < d < 720 for d in durs), f"min {min(durs):.0f} s, max {max(durs):.0f} s")
    differences = [
        f"The paper says every session had 150 perturbations (75 per direction); {len(short)} of the {len(rows)} recordings have a different number in the events files (or more: counts range from {min(n_events)} to {max(n_events)}; the deposit does not say why), and the direction split is therefore not always 75/75. All events in the deposit are kept.",
        f"The paper says each session lasted 10 min; recordings run {min(durs) / 60:.1f} to {max(durs) / 60:.1f} min (median {float(np.median(durs)) / 60:.1f} min), the shortest probably stopped early; durations are therefore not used as a check.",
        "The EEG (128 channels) was not released, and neither were the 3 'good channel' lists, ICA matrices or dipole fits in the .set files.",
        "The deposit labels sessions ses-01 to ses-04 by perturbation type and posture (pull stand, pull walk, rotate stand, rotate walk); the paper says the standing session always preceded the walking session of the same perturbation and half of the participants started with rotations, so session numbers are labels, not the order performed.",
        "Values were rounded to 3 significant digits and time_s to 0.1 ms to keep the ZIP under 1 GB; sampling rate was not reduced.",
    ]
    meta = dict(
        paper=dict(authors="Peterson SM, Ferris DP", year=2021, title="Human electrocortical, electromyographical, ocular, and kinematic data during perturbed walking and standing",
                   venue="Data Brief 39:107635", doi="10.1016/j.dib.2021.107635"),
        lab="Ferris (University of Florida; data collected at the University of Michigan)", data_url="https://openneuro.org/datasets/ds003739",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes="Only the 21 non-EEG channels (neck EMG, EOG, leg EMG, head and sacrum motion capture, pull load cells) and the perturbation onsets were kept; the 128 EEG channels were not released. Samples are the deposit's 256 Hz values rounded to 3 significant digits; participant ids were replaced by codes.",
        how=["One row is one session of about 10 minutes (4 per participant), a continuous recording during which about 150 brief balance perturbations occurred: waist pulls of about 15 N lasting 1 s (pull sessions, left or right) or 20 degree visual field rotations lasting 0.5 s (rotation sessions, clockwise or counterclockwise), while the participant stood or walked heel-to-toe at 0.22 m/s on a treadmill-mounted balance beam.",
             "The samples are in samples/<file>.csv: time_s (sample index divided by 256) then the 21 channels in the order of the channels column, units in channel_units: EMG, EOG in microvolts; head and sacrum positions in mm along mediolateral, superior-inferior and anterior-posterior axes (signs not stated by the authors; the values are centred, as stored); load cells in N (only pull sessions carry a pull force).",
             "events lists the perturbation onsets (name=seconds from the start of the recording): pull_left, pull_right (the side the participant was pulled toward) or rotate_cw, rotate_ccw. The paper says 150 per session, 75 per direction; some recordings have fewer events (see Known differences).",
             "pert_value is the nominal size in the session (15 N pull, 20 deg rotation) and block names the perturbation and posture. session is the deposit's session label (1 pull standing, 2 pull walking, 3 rotation standing, 4 rotation walking), not the order performed."],
        unit_note="recording session (about 10 minutes)",
        differences=differences,
    )
    stdlib.release(ID, "timeseries", rows, meta, checks, extra_files=extra)
    shutil.rmtree(TMP, ignore_errors=True)
finally:
    for f in RAW.glob("p*"):
        f.unlink()
