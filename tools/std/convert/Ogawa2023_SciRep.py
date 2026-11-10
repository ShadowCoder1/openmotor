"""Ogawa et al. (2023), Sci Rep 13:1909: split-belt treadmill walking with a pulling force field (aiding = forward, impeding = backward)
at the centre of mass; 16 healthy adults each did two of four experiments. Ground reaction forces of both belts at 1 kHz.
One row per participant and experiment (32 recordings of about 22 min).

Run from the site folder: python3 -I tools/std/convert/Ogawa2023_SciRep.py
Raw files (untrusted data, read only) in raw_extra/Ogawa2023_SciRep/ from https://doi.org/10.5061/dryad.pvmcvdnp9 (Zenodo copy
https://zenodo.org/records/6612540): <experiment>_subj<participant>.txt and README.docx. Each text file is converted to a samples CSV
and the raw text file is deleted right after (the PC disk is small); a rerun skips files that are already converted.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Ogawa2023_SciRep"
RAW = stdlib.raw_dir(ID)
TMP = stdlib.HERE / "tmp_samples" / ID
TMP.mkdir(parents=True, exist_ok=True)
checks = stdlib.Checks()

CH = ["grf_right_mediolateral", "grf_right_anteroposterior", "grf_right_vertical",
      "grf_left_mediolateral", "grf_left_anteroposterior", "grf_left_vertical"]
HEADER = "time_s," + ",".join(CH) + "\r\n"
FORCE = {1: "aiding", 2: "aiding", 3: "impeding", 4: "impeding"}
TRIALS = {}
rows, extra = [], []
exp_persons = {1: set(), 2: set(), 3: set(), 4: set()}
dts_ok = True
durations = []
for e in (1, 2, 3, 4):
    for s in (range(1, 9) if e in (1, 3) else range(9, 17)):
        raw = RAW / f"{e}_subj{s}.txt"
        out = TMP / f"{ID}_P{s:02d}_exp{e}.csv"
        fn = f"samples/{ID}_P{s:02d}_exp{e}.csv"
        if not out.exists():
            n = 0
            first = last = prev = None
            with open(raw, "r", encoding="utf-8") as f, open(out, "w", newline="", encoding="utf-8") as g:
                g.write(HEADER)
                for line in f:
                    p = line.rstrip("\r\n").split("\t")
                    assert len(p) == 7, (raw.name, n, len(p))
                    g.write(",".join(p) + "\r\n")
                    t = float(p[0])
                    if n == 0:
                        first = t
                    elif abs(t - prev - 0.001) > 1e-6:
                        dts_ok = False
                    prev = t
                    n += 1
            raw.unlink()
            meta_n = (n, prev)
        else:
            d = pd.read_csv(out, usecols=["time_s"])
            meta_n = (len(d), float(d.time_s.iloc[-1]))
            dts_ok = dts_ok and bool(abs(d.time_s.diff().dropna() - 0.001).max() < 1e-6)
        n, tend = meta_n
        durations.append(tend)
        exp_persons[e].add(s)
        extra.append((fn, str(out)))
        TRIALS[s] = TRIALS.get(s, 0) + 1
        rows.append(dict(
            dataset=ID, study=ID, experiment=f"Experiment {e}: split-belt adaptation with the {FORCE[e]} force field",
            group="", subject=f"{s}_{e}", person=str(s), subject_excluded_by_authors=False,
            age="", sex="", handedness="", population="healthy adults", setting="lab",
            device="split-belt treadmill (Bertec) with force plates under each belt, 2 kg counterweight pulling at the torso",
            session=1, block=f"Experiment {e}", trial=1, phase="other", time_s="", task="walking", unit="recording",
            condition=f"{FORCE[e]} force field throughout; baseline, adaptation, re-adaptation, washouts",
            pert_type="split_belt", pert_value=2, pert_unit="ratio (right belt 1.0 m/s, left belt 0.5 m/s)",
            feedback="normal", excluded_by_authors=False,
            recording=1, samples_file=fn, signal_kind="ground reaction forces of the right and left belts (mediolateral, anteroposterior, vertical)",
            channels=";".join(CH), channel_units=";".join(["kN (inferred)"] * 6), sample_rate_hz=1000, n_samples=n,
            duration_s=round(n / 1000.0, 3), events="",
            processing="low-pass filtered at 8 Hz by the authors' recording system (Methods)",
            trial_info=f"experiment={e};force_field={FORCE[e]}"))
D = pd.DataFrame(rows)

P = checks.published
P("participants", D.person.nunique(), 16, 0, "Methods, Participant")
P("recordings (each participant did two of four experiments)", len(D), 32, 0, "Methods, Participant")
P("participants in Experiments 1 and 3", len(exp_persons[1] & exp_persons[3]), 8, 0, "Methods, Participant")
P("participants in Experiments 2 and 4", len(exp_persons[2] & exp_persons[4]), 8, 0, "Methods, Participant")
P("participants per experiment (min)", min(len(v) for v in exp_persons.values()), 8, 0, "Methods, Participant")
checks.internal("sampling interval is 1 ms in every file (paper: 1 kHz)", dts_ok)
checks.internal("each recording is at least as long as the protocol (about 20 min)", min(durations) > 1200, f"min {min(durations):.0f} s")
checks.internal("each participant has two recordings", all(v == 2 for v in TRIALS.values()))

meta = dict(
    paper=dict(authors="Ogawa T, Obata H, Yokoyama H, Kawashima N, Nakazawa K", year=2023,
               title="Different functional networks underlying human walking with pulling force fields acting in forward or backward directions",
               venue="Sci Rep 13(1):1909", doi="10.1038/s41598-023-29231-6"),
    lab="Nakazawa (University of Tokyo)",
    data_url="https://doi.org/10.5061/dryad.pvmcvdnp9", license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("The 32 tab-separated text files (one per participant and experiment) became 32 samples CSVs: tab became comma, the seven columns were named "
             "(time_s and six force channels) and no value was changed. Dryad needs a token for downloads, so the files came from the deposit's Zenodo copy (record 6612540). "
             "Age and sex are empty because the deposit has no per-participant demographics (the paper gives group means only)."),
    how=["One row is one participant's whole recording in one experiment. Each participant did two of the four experiments (1 and 3, or 2 and 4); person is the participant and subject is the participant-experiment pair. "
         "Experiments 1 and 2 used an aiding (forward-pulling) force field and Experiments 3 and 4 an impeding (backward-pulling) one, throughout the session. The protocol (Methods) is a 1 min tied-belt baseline with each force field, "
         "10 min split-belt adaptation (left belt 0.5 m/s, right 1.0 m/s), 1 min tied washout, 5 min re-adaptation, 1 min washout and an extra 1 min washout, with 15 s intervals between periods and tied belts at 0.75 m/s. "
         "The deposit has no markers for these periods, so events is empty.",
         "Samples (samples/*.csv): time_s from 0 at 1 kHz, then right-belt and left-belt ground reaction force in the mediolateral, anteroposterior and vertical directions, in the README's row order. "
         "Units and sign conventions are not stated in the deposit. Kilonewtons are inferred: the vertical force of the two belts averages about 0.7 in the one file we checked, close to a participant's body weight (66 kg is 0.65 kN), "
         "and peaks at 0.86, about 1.3 body weights. The paper's figures show the same signals in newtons.",
         "The trailing seconds of a file can be zeros (recording continued after the last period)."],
    differences=["The paper reports statistics of stride-wise peak braking and propulsive forces, which need stride detection from the vertical force; those numbers were not recomputed, so the checks are counts of participants and recordings."],
    unit_note="recording (one participant in one experiment)",
)
stdlib.release(ID, "timeseries", D, meta, checks, extra_files=extra)
shutil.rmtree(TMP, ignore_errors=True)
