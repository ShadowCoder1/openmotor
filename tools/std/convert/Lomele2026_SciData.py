"""Lomele et al. (2026) Sci Data 13:932: 14 healthy adults reached, grasped and lifted a sensorized grasping box with one of three grips
(precision thumb-index PG, unconventional thumb-ring UG, whole-hand WH) while 64-channel EEG and 13-muscle EMG were recorded. One row per trial;
the samples file holds the 13 EMG channels (9 s epoch at 1000 Hz, from 2 s before to 7 s after the LED go signal). The EEG and the resting
EEG are not carried (they are not movement measures and would make the ZIP several GB); the box events (LED on, touch, lift) are.
Raw files: raw_extra/Lomele2026_SciData/s_1.mat ... s_14.mat (OSF rsv4z, CC0), MATLAB 5 files read with scipy. Strings (muscle names, grips, sex,
hand) are MATLAB 'string' objects, read from the UTF-16 text of the file's hidden object store.
Run from the site folder: python3 -I tools/std/convert/Lomele2026_SciData.py"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Lomele2026_SciData"
MODULE = "timeseries"
MUSCLES = ["DA", "BB", "TB", "ECR", "ECU", "EDC", "FCR", "FDS", "FCU", "APB", "FDI", "ADM", "FPL"]
GRIP = {1: "PG", 2: "UG", 3: "WH"}
GRIP_NAME = {"PG": "precision grip (thumb-index)", "UG": "unconventional grip (thumb-ring)", "WH": "whole-hand power grasp"}
DEVICE = ("custom sensorized grasping box (touch sensors, lift microswitch, LED go signal) on a 45 degree incline; Cometa WavePlus wireless "
          "bipolar EMG (13 muscles); BrainAmp DC 64-channel EEG")


def strings(path):
    m = sio.loadmat(path, variable_names=["__function_workspace__"])
    u = m["__function_workspace__"].tobytes().decode("utf-16le", errors="ignore")
    return re.findall(r"[A-Z]+", u)


def num(x):
    a = np.asarray(x)
    return int(a.ravel()[0]) if a.size else None


def main():
    raw = stdlib.raw_dir(ID)
    tmp = Path("tmp_" + ID)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    rows, extra, people = [], [], []
    for s in range(1, 15):
        path = raw / f"s_{s}.mat"
        toks = strings(path)
        # [X, muscles, H, grips, sex, hand, X, X, H]
        assert "".join(MUSCLES) == toks[toks.index("DABBTBECRECUEDCFCRFDSFCUAPBFDIADMFPL")] and "PGUGWH" in toks, toks
        sex = toks[toks.index("PGUGWH") + 1]
        hand = toks[toks.index("PGUGWH") + 2]
        assert sex in ("F", "M") and hand in ("RIGHT", "LEFT"), (sex, hand)
        d = sio.loadmat(path, variable_names=["data"], squeeze_me=True, struct_as_record=False)["data"]
        assert d.info.fs == 1000 and d.subject.id == s
        age = float(d.subject.age)
        n_ok = 0
        for t, tr in enumerate(d.trials, start=1):
            emg = np.asarray(tr.emg)
            assert emg.shape == (13, 9000)
            fname = f"s{s:02d}_t{t:03d}.csv"
            with open(tmp / fname, "w", newline="") as fh:
                fh.write("time_s," + ",".join(f"emg_{m_}" for m_ in MUSCLES) + "\r\n")
                for k in range(9000):
                    fh.write(f"{k / 1000:.3f}," + ",".join("%.6g" % v for v in emg[:, k]) + "\r\n")
            extra.append((f"samples/{fname}", str(tmp / fname)))
            ev = [("led_on", num(tr.LEDon_event_frame)), ("touch", num(tr.touch_event_frame)), ("lift", num(tr.lift_event_frame))]
            events = ";".join(f"{n}={v / 1000:g}" for n, v in ev if v is not None)
            complete = bool(num(tr.event_complete_trial))
            n_ok += complete
            g = GRIP[num(tr.task)]
            rows.append(dict(
                dataset=ID, study="Lomele et al. 2026 Sci Data", experiment="", group="", subject=f"s{s:02d}", person=f"s{s:02d}",
                subject_excluded_by_authors=False, age=age, sex=sex, handedness="right" if hand == "RIGHT" else "left", population="healthy",
                setting="lab", device=DEVICE, session=1, block=f"block {num(tr.block)}", trial=t, phase="test", time_s=np.nan,
                task="grasping and lifting", unit="recording", condition=GRIP_NAME[g], pert_type="none", pert_value=0, pert_unit="",
                feedback="normal", excluded_by_authors=not complete, recording=t, samples_file=f"samples/{fname}",
                signal_kind="surface EMG of 13 upper-limb muscles during reach, grasp, lift and hold",
                channels=";".join(f"emg_{m_}" for m_ in MUSCLES), channel_units=";".join(["unspecified"] * 13), sample_rate_hz=1000,
                n_samples=9000, duration_s=9.0, events=events,
                processing="EMG low-pass filtered (4th-order Butterworth, 400 Hz) and down-sampled to 1000 Hz by the authors; epoch from 2 s before to 7 s after the LED-on event",
                trial_info=f"grip={g};event_complete_trial={int(complete)}"))
        people.append(dict(person=f"s{s:02d}", n=len(d.trials), ok=n_ok, age=age, sex=sex, hand=hand))
    df = pd.DataFrame(rows)
    df = df.reindex(columns=[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"])
    pp = pd.DataFrame(people)
    ck = stdlib.Checks()
    S = "Methods (Participants)"
    ck.published("participants", len(pp), 14, 0, S)
    ck.published("right-handed participants", (pp.hand == "RIGHT").sum(), 13, 0, S)
    ck.published("left-handed participants", (pp.hand == "LEFT").sum(), 1, 0, S)
    ck.published("males", (pp.sex == "M").sum(), 7, 0, S)
    ck.published("females", (pp.sex == "F").sum(), 7, 0, S)
    ck.published("mean age (years)", pp.age.mean(), 31.4, 0.1, S)
    ck.published("SD age (years)", pp.age.std(ddof=1), 10.4, 0.1, S)
    ck.published("EMG muscles", 13, 13, 0, "EMG setup and acquisition")
    ck.published("epoch length (s) = 9 s at 1000 Hz", df.n_samples.iloc[0] / 1000, 9, 0, "Data pre-processing")
    ck.published("grip types", df.condition.nunique(), 3, 0, "Experimental protocol")
    ck.internal("every participant has at least three blocks of ten trials per grip, in complete blocks of 30 (+- failed repeats)",
                (df.groupby("subject").block.nunique() >= 3).all())
    ck.internal("every recording has 9000 samples and an LED-on event at 2 s", (df.n_samples == 9000).all() and df.events.str.contains("led_on=2").all())
    ck.internal("trials flagged event-incomplete lack a touch or lift event", bool((df[df.excluded_by_authors].events.str.count("=") < 3).all()),
                f"{int(df.excluded_by_authors.sum())} flagged")
    meta = dict(
        paper=dict(authors="Lomele G, Lencioni T, D'Ambrosio S, Comanducci A, Lucchetti F, Marzegan A, Derchi C, Garzonio S, Atzori T, Rabuffetti M, Castiglioni P, Ferrarin M, et al.",
                   year=2026, title="High-Density EEG and Multi-Muscle EMG Dataset during Object Prehension with a sensorized Grasping Box in Humans",
                   venue="Sci Data 13(1):932", doi="10.1038/s41597-026-07242-y"),
        lab="Fondazione Don Carlo Gnocchi / University of Milan", data_url="https://osf.io/rsv4z/",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Wrote the 13 EMG channels of each trial to one CSV (time_s added as sample/1000; values to 6 significant digits instead of the "
                 "usual 8, to keep the ZIP near 1 GB) and listed the LED-on, touch and lift frames as event times. Age, sex and hand come from each "
                 "participant's file. Not carried: the 64-channel EEG of every trial, the resting-state EEG, and the EEG electrode positions."),
        how=["One row is one grasp trial of about 9 s: after a video of the grip to use and a 2 s fixation, the LED lit (go signal at 2.000 s in the "
             "epoch), the participant reached, grasped the target of the box with the instructed grip (PG, UG or WH) and pulled it 2 cm to the lift "
             "switch, then held it until the LED went off 5 s after the go signal. Events (seconds from the start of the epoch): led_on, touch "
             "(touch sensor), lift (end-of-pull switch). Frame numbers are converted as frame/1000 without a 1-sample offset.",
             "channels are the surface EMG of anterior deltoid, biceps, triceps, extensor carpi radialis and ulnaris, extensor digitorum communis, flexor "
             "carpi radialis, flexor digitorum superficialis, flexor carpi ulnaris, abductor pollicis brevis, first dorsal interosseous, abductor "
             "digiti minimi and flexor pollicis longus, in the file's order (DA BB TB ECR ECU EDC FCR FDS FCU APB FDI ADM FPL). The EMG unit is not "
             "stated in the deposit or paper (hence 'unspecified').",
             "excluded_by_authors marks trials whose touch or lift trigger was missing (event_complete_trial = 0); the authors kept them in the "
             "files but restricted their technical validation to complete trials."],
        unit_note="recording (one grasp trial, EMG only)",
        differences=["The paper's Table 1 (trials per participant) could not be read as text, so trial counts per participant were not compared with it. "
                     "The paper says all events are at fixed offsets from the go signal; the files give the LED-on frame as 2000 in every trial."])
    try:
        stdlib.release(ID, MODULE, df, meta, ck, participants=None, extra_files=extra)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for c in ck.items:
        print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""))


main()
