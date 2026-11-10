"""Schiller, Kruger, Weede, Sopha & Schmidt (2026), Journal of Voice: "Enhancing Acoustic Voice Quality Through Real-Time
Auditory Feedback Modulation". Study 2: healthy adults sustained the vowel /a:/ for 4 s, 140 times, while their auditory
feedback was either unaltered (control group) or, in the AFM group, made hoarse in real time by VQ-Synth (noise added between
the pitch-related amplitude peaks) in a baseline - ramp - hold - after design. One row is one sustained vowel with its smoothed
cepstral peak prominence (CPPS).

Raw files, raw_extra/Schiller2026_JVoice/ (OSF jp2yk, CC BY 4.0, fetched with tools/std/fetch.py):
  Study2_OSF.csv   ID (group code + number), Group (AFM, control), Age, Gender (female, male), AVQI (per participant), Trial
                   (1-140), Phase (baseline 1-20, ramp 21-70, hold 71-120, after 121-140), SNR and SDO (feedback modulation, or
                   'control' when unaltered), CPPS (dB), CPPS_z (an exact linear transform of CPPS within group: baseline-phase
                   mean 0 and SD 1), F0_input (Hz), RMS_input
  Study1_OSF.csv   972 listener ratings (vocal health, naturalness, hoarseness, ...) of resynthesised recordings of the
                   participants' own voices: perception, not motor behaviour, so it is not converted.
The deposit has no readme; the meaning of the columns comes from the authors' PsyArXiv preprint (3jgn6, version 1, Study 2:
Participants, Task, Technical set-up, Analysis). The OSF project note says the published article reflects a corrected CPPS
extraction script (the shared file is the corrected one); the journal text was behind a paywall, so participant counts, design
and settings are checked against the preprint.
Run from the site folder: python3 -I tools/std/convert/Schiller2026_JVoice.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Schiller2026_JVoice"
SRC = "Schiller et al. 2026, preprint v1 (PsyArXiv 3jgn6)"
DEVICE = ("DPA 4066 headset microphone (2.5 cm from the mouth), RME Fireface UC sound card, Sennheiser HD 280 Pro closed "
          "headphones, KiRAT real-time system with VQ-Synth")
PHASE = {"baseline": "baseline", "ramp": "learning", "hold": "learning", "after": "aftereffect"}
WIN_UNALT = "2 s steady-state mid-vowel portion"
WIN_MOD = "300 ms window, 100-400 ms after modulation onset"


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def num(s):
    return float(str(s).split()[0])


def main():
    checks = stdlib.Checks()
    raw = stdlib.raw_dir(ID)
    d = pd.read_csv(raw / "Study2_OSF.csv")
    d = d.assign(_o=np.arange(len(d))).sort_values(["ID", "Trial", "_o"], kind="stable")
    checks.internal("7619 rows of 56 participants (34 AFM, 22 control); group, age, gender and AVQI constant within participant; "
                    "one row per participant and trial",
                    bool(len(d) == 7619 and d.ID.nunique() == 56 and d.groupby("Group").ID.nunique().to_dict() == {"AFM": 34, "control": 22}
                         and (d.groupby("ID")[["Group", "Age", "Gender", "AVQI"]].nunique() == 1).all().all()
                         and not d.duplicated(["ID", "Trial"]).any()))
    mod = d.SNR != "control"
    checks.internal("feedback is modulated only in the AFM group's ramp and hold phases (SNR and SDO given together there, 'control' "
                    "everywhere else)",
                    bool(((d.SNR != "control") == (d.SDO != "control")).all()
                         and (mod == ((d.Group == "AFM") & d.Phase.isin(["ramp", "hold"]))).all()))
    checks.internal("phase follows the trial number: baseline 1-20, ramp 21-70, hold 71-120, after 121-140, in both groups",
                    bool((pd.cut(d.Trial, [0, 20, 70, 120, 140], labels=["baseline", "ramp", "hold", "after"]).astype(str) == d.Phase).all()))
    base = d[d.Phase == "baseline"].groupby("Group").CPPS_z.agg(["mean", "std"])
    checks.internal("CPPS_z is standardised against each group's baseline phase: mean 0 and SD 1 over the baseline rows",
                    bool((base["mean"].abs() < 0.002).all() and (base["std"].sub(1).abs() < 0.002).all()),
                    f"{base['mean'].abs().max():.2g}, SD {base['std'].min():.4f}-{base['std'].max():.4f}")
    checks.internal("value ranges: CPPS 9-24 dB, F0 70-340 Hz, RMS 54-91, AVQI below the 3.05 inclusion cut-off",
                    bool(d.CPPS.between(9, 24).all() and d.F0_input.between(70, 340).all() and d.RMS_input.between(54, 91).all()
                         and (d.AVQI < 3.05).all()))
    rows = []
    for pid, g in d.groupby("ID", sort=False):
        for k, r in enumerate(g.itertuples(index=False), 1):
            modulated = r.SNR != "control"
            rows.append({
                "dataset": ID, "study": ID, "experiment": "Study 2", "group": r.Group, "subject": pid, "person": pid,
                "subject_excluded_by_authors": False, "age": int(r.Age), "sex": {"female": "F", "male": "M"}[r.Gender],
                "handedness": "", "population": "healthy", "setting": "lab", "device": DEVICE, "session": 1, "block": r.Phase,
                "trial": k, "phase": PHASE[r.Phase], "time_s": "", "task": "speech", "unit": "utterance",
                "condition": "hoarseness-induced feedback" if modulated else "unaltered feedback",
                "pert_type": "voice_quality_noise" if modulated else "none", "pert_value": num(r.SNR) if modulated else 0,
                "pert_unit": "dB SNR (lower = more noise)" if modulated else "",
                "feedback": "altered" if modulated else "normal", "excluded_by_authors": False,
                "stimulus": "sustained vowel /a:/ (4 s)", "segment": "sustained vowel /a:/",
                "measure_window": WIN_MOD if modulated else WIN_UNALT, "f0_hz": float(r.F0_input),
                "masking_noise": "none", "feedback_latency_ms": 16 if r.Group == "AFM" else "",
                "authors_trial_number": int(r.Trial), "cpps_db": float(r.CPPS), "cpps_z": float(r.CPPS_z),
                "rms_input": float(r.RMS_input), "sdo_db_per_octave": num(r.SDO) if modulated else "",
            })
    a = pd.DataFrame(rows)
    ppl = d.drop_duplicates("ID")
    S = f"{SRC}, Study 2, Participants"
    for grp, n, nf, nm, lo, hi, m, sd in (("AFM", 34, 22, 12, 19, 35, 24, 3), ("control", 22, 16, 6, 19, 57, 24, 8)):
        p = ppl[ppl.Group == grp]
        checks.published(f"{grp} group: participants", len(p), n, 0, S)
        checks.published(f"{grp} group: female", (p.Gender == "female").sum(), nf, 0, S)
        checks.published(f"{grp} group: male", (p.Gender == "male").sum(), nm, 0, S)
        checks.published(f"{grp} group: youngest (years)", p.Age.min(), lo, 0, S)
        checks.published(f"{grp} group: oldest (years)", p.Age.max(), hi, 0, S)
        checks.published(f"{grp} group: mean age (years, printed as a whole number)", p.Age.mean(), m, 0.5, S)
        checks.published(f"{grp} group: SD of age (years, printed as a whole number)", p.Age.std(), sd, 0.5, S)
    T = f"{SRC}, Study 2, Task"
    checks.published("trials per participant (140 sustained vowels)", d.Trial.max(), 140, 0, T)
    checks.published("baseline phase: last trial", d[d.Phase == "baseline"].Trial.max(), 20, 0, T)
    checks.published("ramp phase: trials (21-70, 50 trials)", d[d.Phase == "ramp"].Trial.nunique(), 50, 0, T)
    checks.published("hold phase: trials (71-120, 50 trials)", d[d.Phase == "hold"].Trial.nunique(), 50, 0, T)
    checks.published("after phase: trials (121-140, 20 trials)", d[d.Phase == "after"].Trial.nunique(), 20, 0, T)
    afm = d[d.Group == "AFM"]
    ramp = afm[afm.Phase == "ramp"].assign(snr=lambda x: x.SNR.map(num))
    checks.published("ramp phase: SNR of the first step (dB)", ramp.snr.max(), 17, 0, f"{SRC}, Study 2, Technical set-up")
    checks.published("ramp phase: SNR of the last step (dB)", ramp.snr.min(), 13, 0, f"{SRC}, Study 2, Technical set-up")
    checks.published("hold phase: SNR (dB)", afm[afm.Phase == "hold"].SNR.map(num).max(), 12, 0, f"{SRC}, Study 2, Technical set-up")
    checks.published("hold phase: only one SNR value", afm[afm.Phase == "hold"].SNR.nunique(), 1, 0, f"{SRC}, Study 2, Technical set-up")
    checks.published("SDO in the ramp and hold phases (dB)", afm[afm.Phase.isin(["ramp", "hold"])].SDO.map(num).max(), 2, 0,
                     f"{SRC}, Study 2, Technical set-up")
    checks.published("ramp phase: steps of the stepwise SNR decrease", ramp.snr.nunique(), 5, 0, f"{SRC}, Study 2, Technical set-up (+17 to +13 dB)")
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s, paper = paper_meta()
    ppl_rows = [dict(person=r.ID, avqi=float(r.AVQI)) for r in ppl.itertuples()]
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/legalcode",
        changes=("Took the authors' Study 2 table (sustained-vowel trials with CPPS, F0, RMS and the feedback settings), kept every "
                 "row, ordered each participant's rows by trial number, mapped the authors' columns to the speech format (CPPS, "
                 "z-scored CPPS, F0, input RMS, SNR as pert_value, SDO), added the vowel, the measurement window, the apparatus and "
                 "the system delay from the preprint, moved the per-participant AVQI to participants.csv and replaced participant "
                 "identifiers with codes. Study 1 (listener ratings of resynthesised own-voice recordings) is not included: "
                 "it is perception, not motor behaviour."),
        how=[
            "One row is one sustained vowel /a:/ (4 s) of one participant, in trial order (trial is the order among the rows "
            "shared; authors_trial_number is the authors' trial 1-140; 221 of 7840 possible trials are not in the file, and one "
            "participant has 38 rows). Group AFM heard hoarseness-inducing noise added between the pitch-related amplitude peaks "
            "of their own voice (VQ-Synth) in the ramp phase (trials 21-70, SNR stepping down from +17 to +13 dB in five steps of "
            "10 trials) and hold phase (trials 71-120, +12 dB); baseline (1-20) and after (121-140) were unaltered. The control "
            "group heard unaltered feedback on all 140 trials.",
            "pert_value is the SNR of the added noise in dB (lower means more noise), pert_type none and pert_value 0 on unaltered "
            "rows; sdo_db_per_octave is the noise's spectral decay per octave (2 dB per octave from 1000 Hz). phase maps "
            "baseline to baseline, ramp and hold to learning and after to aftereffect for both groups by trial position; the "
            "authors' phase label is in block. feedback_latency_ms (16) is the system delay reported for VQ-Synth, given for the "
            "AFM group only.",
            "cpps_db is the smoothed cepstral peak prominence the authors extracted with Praat, from a 2 s mid-vowel portion on "
            "unaltered trials and from a 300 ms window 100-400 ms after modulation onset on modulated trials (measure_window "
            "says which); cpps_z is their standardised version, relative to each group's baseline phase. f0_hz and rms_input are "
            "the authors' F0 and RMS of the voice input.",
        ],
        unit_note="sustained vowel (utterance) of one participant",
        differences=[
            "The preprint (v1, before the authors corrected their CPPS extraction script) gives AFM-group mean CPPS by phase of 15.39, "
            "15.71, 15.93 and 15.91 dB; the shared, corrected file gives 15.34, 15.63, 15.82 and 15.91 dB. The published article "
            "was behind a paywall, so no CPPS statistic of the corrected analysis could be checked; the checks are the participant "
            "counts, ages, trial design and feedback settings.",
            "The preprint describes z-scoring CPPS within each group over all data; the shared CPPS_z is standardised against each "
            "group's baseline phase (an exact linear transform of CPPS, baseline mean 0 and SD 1).",
        ],
    )
    return stdlib.release(ID, "speech", a.to_dict("records"), meta, checks, participants=ppl_rows,
                          participant_columns=[("avqi", "number", "index",
                                                "Acoustic Voice Quality Index of the participant's recorded sustained vowel and text "
                                                "before the task, as the authors shared it (inclusion required a value below 3.05, "
                                                "the German cut-off for a healthy voice).")])


if __name__ == "__main__":
    main()
