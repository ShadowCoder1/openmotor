"""Merk et al. (2022) eLife 11:e75126: ECoG versus subthalamic LFP for grip-force decoding in Parkinson's disease.

Raw file: raw_extra/Merk2022_eLife/bundle.zip (Harvard Dataverse doi:10.7910/DVN/IO2FLM, original files in a Dataverse bundle: participants.tsv
and, per run, sub-XXX_ses-<left|right>_task-force_run-N_ieeg.eeg + .vhdr (BrainVision, float32, 1 kHz, multiplexed)). Only the four movement
channels (the last four of every run: MOV_RIGHT, MOV_LEFT, MOV_RIGHT_CLEAN, MOV_LEFT_CLEAN) are released; the ECoG and subthalamic LFP channels
are brain recordings, not movement, and are left out. Module: timeseries (one row per run, samples as CSV).

Run from the site folder: python3 -I tools/std/convert/Merk2022_eLife.py
"""
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Merk2022_eLife"
RAW = stdlib.raw_dir(DS)
TMP = RAW / "_samples"
TMP.mkdir(exist_ok=True)
MOV = ["MOV_RIGHT", "MOV_LEFT", "MOV_RIGHT_CLEAN", "MOV_LEFT_CLEAN"]
CH = ["grip_force_right", "grip_force_left", "grip_force_right_clean", "grip_force_left_clean"]
# Table 1 of the paper: UPDRS-III (off), disease duration (years), hemispheres implanted
TABLE1 = [(28, 10.7, "R"), (27, 14, "LR"), (33, 7.2, "LR"), (31, 10.1, "LR"), (32, 13.1, "LR"), (52, 5.9, "L"), (55, 1.4, "L"),
          (50, 8.7, "L"), (62, 9.8, "LR"), (48, 17.1, "L"), (31, 10.4, "R")]


def num(x):
    x = x.strip()
    return None if x in ("n/a", "") else float(x)


def main():
    ck = stdlib.Checks()
    z = zipfile.ZipFile(RAW / "bundle.zip")
    pt = list(csv.DictReader(io.StringIO(z.read("participants.tsv").decode("utf8")), delimiter="\t"))
    part = {r["participant_id"]: r for r in pt}
    runs = []
    for n in z.namelist():
        m = re.match(r"(sub-\d+)_ses-(left|right)_task-force_run-(\d+)_ieeg\.vhdr$", n)
        if m:
            runs.append((m[1], 0 if m[2] == "left" else 1, int(m[3]), n))
    runs.sort()
    subj_order = []
    for s, h, r, n in runs:
        if (s, h) not in subj_order:
            subj_order.append((s, h))
    sidx = {k: i + 1 for i, k in enumerate(subj_order)}

    rows, files = [], []
    seq = {}
    sr_ok, names_ok, nan_ok, rng_ok = True, True, True, True
    srs = set()
    for s, h, r, n in runs:
        hv = z.read(n).decode("utf8", "replace")
        nc = int(re.search(r"NumberOfChannels=(\d+)", hv)[1])
        si = float(re.search(r"SamplingInterval=(\d+)", hv)[1])
        assert re.search(r"BinaryFormat=IEEE_FLOAT_32", hv) and "MULTIPLEXED" in hv
        names = re.findall(r"Ch\d+=([^,]*),", hv)
        names_ok &= len(names) == nc and names[-4:] == MOV
        srs.add(1e6 / si)
        buf = z.read(n.replace("_ieeg.vhdr", "_ieeg.eeg"))
        a = np.frombuffer(buf, dtype="<f4").reshape(-1, nc)[:, -4:]
        nan_ok &= not np.isnan(a).any()
        rng_ok &= float(a[:, 2:].min()) >= 0
        ns = a.shape[0]
        k = seq[(s, h)] = seq.get((s, h), 0) + 1
        fname = f"{DS}_S{sidx[(s, h)]:04d}_trial{k:03d}.csv"
        with open(TMP / fname, "w", newline="") as fh:
            fh.write("time_s," + ",".join(CH) + "\n")
            t = np.arange(ns) * (si / 1e6)
            for i0 in range(0, ns, 50000):
                blk = a[i0:i0 + 50000]
                fh.write("".join(f"{t[i0 + j]:.3f}," + ",".join(f"{v:.9g}" for v in blk[j]) + "\n" for j in range(len(blk))))
        files.append((f"samples/{fname}", str(TMP / fname)))
        p = part[s]
        hem = "left" if h == 0 else "right"
        rows.append(dict(
            dataset=DS, study=DS, experiment="cued grip-force task (intraoperative)", group="", subject=f"{s}_{hem}", person=s,
            subject_excluded_by_authors=False, age="", sex=p["sex"] if p["sex"] in ("F", "M") else "", handedness="",
            population="Parkinson's disease", setting="clinic",
            device="hand-held grip force sensor (type and units not stated), recorded with a BrainProducts amplifier together with "
                   "ECoG and subthalamic LFP; off dopaminergic medication, awake during DBS implantation",
            session=1 + h, block=f"{hem} hemisphere recording session", trial=k, phase="test", time_s="", task="grasping and lifting",
            unit="recording", condition=f"{hem} hemisphere implanted and recorded", pert_type="none", pert_value=0, pert_unit="",
            feedback="normal", excluded_by_authors=False,
            recording=k, samples_file=f"samples/{fname}",
            signal_kind="grip force of the right and left hand during a cued go/no-go squeezing task (raw and cleaned versions)",
            channels=";".join(CH), channel_units=";".join(["arbitrary units (stored as uV in the deposit)"] * 4),
            sample_rate_hz=1e6 / si, n_samples=ns, duration_s=(ns - 1) * si / 1e6, events="",
            processing="right and left channels as shared (baseline removed: mean near zero); *_clean channels are the authors' cleaned "
                       "versions (non-negative, capped at 10000000), processing not documented in the deposit",
            trial_info=f"run={r}"))

    per_p = {}
    for r in rows:
        per_p.setdefault(r["person"], set()).add(r["session"])
    found = hem_ok = 0
    for upd, dur, hm in TABLE1:
        cand = [pid for pid, p in part.items() if num(p["UPDRS_total"]) == upd and num(p["Length_of_diagnosis_prior_to_surgery"]) == dur]
        if len(cand) == 1:
            found += 1
            want = {1 if "L" in hm else None, 2 if "R" in hm else None} - {None}
            hem_ok += per_p.get(cand[0], set()) == want
    ck.published("patients of Table 1 found in the deposit by UPDRS-III (off) and disease duration", found, 11, 0, "Table 1")
    ck.published("patients of Table 1 whose recorded hemispheres (L, R or both) match the sessions in the deposit", hem_ok, 11, 0, "Table 1, Hemisphere column")
    ck.published("sampling rate of the released signals (Hz)", srs.pop() if len(srs) == 1 else -1, 1000, 0,
                 "Methods (signals resampled to 1 kHz)")
    ck.internal("the last four channels of every run are the movement channels in the same order", names_ok)
    ck.internal("no missing samples in the released channels", nan_ok)
    ck.internal("cleaned channels are non-negative", rng_ok)
    ck.internal("one row and one sample file per run", len(rows) == len(files) == len(runs) == 73, f"{len(rows)} rows")

    parts = []
    for pid in sorted(per_p):
        p = part[pid]
        parts.append(dict(person=pid, updrs_iii_total_off=num(p["UPDRS_total"]), updrs_iii_left=num(p["UPDRS_left"]),
                          updrs_iii_right=num(p["UPDRS_right"]), disease_duration_years=num(p["Length_of_diagnosis_prior_to_surgery"])))
    pcols = [("updrs_iii_total_off", "number", "points", "UPDRS part III total, off medication, as in the deposit's UPDRS_total (matches Table 1)."),
             ("updrs_iii_left", "number", "points", "Left-side UPDRS III sub-score (deposit's UPDRS_left; meaning inferred)."),
             ("updrs_iii_right", "number", "points", "Right-side UPDRS III sub-score (deposit's UPDRS_right; meaning inferred)."),
             ("disease_duration_years", "number", "years", "Time from diagnosis to surgery (deposit's Length_of_diagnosis_prior_to_surgery).")]
    meta = dict(
        study=DS,
        paper=dict(authors="Merk T, Peterson V, Lipski WJ, Blankertz B, Turner RS, Li N, Horn A, Richardson RM, Neumann WJ", year=2022,
                   title="Electrocorticography is superior to subthalamic local field potentials for movement decoding in Parkinson's disease",
                   venue="eLife 11:e75126", doi="10.7554/eLife.75126"),
        lab="Neumann (Charite Berlin) / Richardson (Pittsburgh)",
        data_url="https://doi.org/10.7910/DVN/IO2FLM", license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("From each BrainVision recording (ECoG, subthalamic LFP and movement channels, 1 kHz) only the four movement channels were "
                 "copied, as CSV with time in seconds; the brain-signal channels, electrode tables and imaging files were not included. "
                 "Participant numbers became codes; the dataset's centre identifiers and the empty age and handedness columns were left out."),
        unit_note="recording (one run of the cued grip-force task) of one participant and hemisphere session",
        how=["One row is one run: while an electrode strip on sensorimotor cortex and a subthalamic lead were recorded during DBS surgery, the "
             "patient squeezed a hand grip in a cued go/no-go task (a side cue showed which hand to use). subject is one participant in one "
             "hemisphere session (left or right hemisphere implanted and recorded; the hand opposite to the hemisphere is the contralateral hand); "
             "person is the participant. The deposit has no event markers, so cues and individual squeezes are not in the data.",
             "Samples: grip_force_right and grip_force_left are the two hands' force channels (MOV_RIGHT, MOV_LEFT), the *_clean channels are the "
             "authors' cleaned versions (MOV_RIGHT_CLEAN, MOV_LEFT_CLEAN). The deposit states the unit as uV only because BIDS requires a unit; "
             "the force device and unit are not described, so the values are arbitrary units. Observed: the raw channels have mean about zero "
             "(baseline removed) with spikes up to about 8e7; the cleaned channels are non-negative and capped at 1e7 and correlate about 0.99 with "
             "the raw channels. What the cleaning did is not documented (the paper says force was cleaned of noise and baseline drift and z-scored "
             "per session before decoding); the cleaned channels are not z-scored.",
             "participants.csv has the clinical scores the deposit shares (UPDRS III off medication and disease duration). Sex is in the main table; "
             "age and handedness are n/a in the deposit."],
        differences=["The deposit has 16 participants (73 runs); the paper analyses 11 (Table 1) and does not say why five were left out, nor which "
                     "sessions or movements were used. All 16 are released and no row is marked excluded; the 11 of Table 1 were matched by UPDRS-III "
                     "and disease duration only to check the data.",
                     "The paper's count of 2,685 movements and its decoding results need movement detection and the brain signals, so they were not "
                     "rebuilt; the published checks compare Table 1 and the sampling rate."],
    )
    stdlib.release(DS, "timeseries", rows, meta, ck, extra_files=files, participants=parts, participant_columns=pcols)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
