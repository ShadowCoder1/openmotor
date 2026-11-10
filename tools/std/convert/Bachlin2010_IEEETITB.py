"""Bachlin et al. (2010), IEEE Trans Inform Technol Biomed 14(2):436-446: Daphnet Freezing of Gait dataset, three body-worn
accelerometers (shank, thigh, lower back) on 10 people with Parkinson's disease and a history of freezing of gait (FoG)
walking in the lab, with the physiotherapists' video-based annotation. One row per recording (file S<subject>R<run>.txt).

Run from the site folder: python3 -I tools/std/convert/Bachlin2010_IEEETITB.py
Raw file (untrusted data, read only) in raw_extra/Bachlin2010_IEEETITB/ from https://doi.org/10.24432/C56K78
(UCI Machine Learning Repository, dataset 245): daphnet.zip. Table I of the paper (ages, sex, disease duration, Hoehn and Yahr)
was typed in below because the deposit has no clinical file.
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Bachlin2010_IEEETITB"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
FS = 64.0

# Table I of the paper: subject id -> sex, age (years), disease duration (years), Hoehn and Yahr in ON, state tested in
T1 = {1: ("M", 66, 16, 3.0, "OFF"), 2: ("M", 67, 7, 2.0, "ON"), 3: ("M", 59, 30, 2.5, "OFF"), 4: ("M", 62, 3, 3.0, "OFF"),
       5: ("M", 75, 6, 2.0, "OFF"), 6: ("F", 63, 22, 2.0, "OFF"), 7: ("M", 66, 2, 2.5, "OFF"), 8: ("F", 68, 18, 4.0, "ON"),
       9: ("M", 73, 9, 2.0, "OFF"), 10: ("F", 65, 24, 3.0, "OFF")}

CH = ["shank_forward", "shank_vertical", "shank_lateral", "thigh_forward", "thigh_vertical", "thigh_lateral",
      "trunk_forward", "trunk_vertical", "trunk_lateral", "annotation"]
CU = ";".join(["mg"] * 9 + ["0 outside the experiment, 1 experiment without freeze, 2 freeze"])

z = zipfile.ZipFile(RAW / "daphnet.zip")
names = sorted(n for n in z.namelist() if re.search(r"dataset/S\d\dR\d\d\.txt$", n))
assert len(names) == 17

rows, extra = [], []
tr = {}
ep_dur, ep_per_subj = [], {}
n_samples_total = 0
n_ann = {0: 0, 1: 0, 2: 0}
for n in names:
    m = re.search(r"S(\d\d)R(\d\d)\.txt$", n)
    s, r = int(m.group(1)), int(m.group(2))
    a = np.loadtxt(io.BytesIO(z.read(n)))
    assert a.shape[1] == 11
    t = a[:, 0] / 1000.0
    ann = a[:, 10].astype(int)
    for k in n_ann:
        n_ann[k] += int((ann == k).sum())
    tr[s] = tr.get(s, 0) + 1
    code = f"{ID}_S{s:04d}"
    fn = f"samples/{ID}_{'S%04d' % s}_run{r}.csv"
    df = pd.DataFrame(a[:, 1:], columns=CH)
    df.insert(0, "time_s", t)
    extra.append((fn, df.to_csv(index=False, float_format="%.8g", lineterminator="\r\n")))
    ev, k, i = [], 0, 0
    f = (ann == 2)
    while i < len(f):
        if f[i]:
            j = i
            while j < len(f) and f[j]:
                j += 1
            k += 1
            ev += [f"freeze_start_{k}={t[i]:.4f}", f"freeze_end_{k}={t[j - 1] + 1 / FS:.4f}"]
            ep_dur.append((j - i) / FS)
            ep_per_subj[s] = ep_per_subj.get(s, 0) + 1
            i = j
        else:
            i += 1
    n_samples_total += len(a)
    sex, age, dd, hy, state = T1[s]
    rows.append(dict(
        dataset=ID, study=ID, experiment=ID, group="", subject=str(s), person=str(s), subject_excluded_by_authors=False,
        age=age, sex=sex, handedness="", population="Parkinson's disease with a history of freezing of gait",
        setting="lab", device="three wireless 3D accelerometers (ankle/shank, thigh above the knee, lower back), 64 Hz",
        session=1, block=f"run {r}", trial=tr[s], phase="test", time_s="", task="walking", unit="recording",
        condition=f"tested {state} medication", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
        excluded_by_authors=False,
        recording=tr[s], samples_file=fn, signal_kind="3D acceleration of shank, thigh and lower back, with freezing-of-gait annotation",
        channels=";".join(CH), channel_units=CU, sample_rate_hz=FS, n_samples=len(a), duration_s=len(a) / FS,
        events=";".join(ev), processing="none (raw samples as shared; annotation made after the session from video by physiotherapists)",
        trial_info=f"run={r};medication_state={state}"))
D = pd.DataFrame(rows)

part = pd.DataFrame([dict(person=str(s), disease_duration=v[2], hoehn_yahr=v[3], medication_state=v[4]) for s, v in T1.items()])
pcols = [("disease_duration", "number", "years", "Years since PD diagnosis (Table I of the paper)."),
         ("hoehn_yahr", "number", "stage", "Hoehn and Yahr stage in the ON state (Table I of the paper)."),
         ("medication_state", "category", "OFF, ON", "Whether the participant was tested OFF medication (more than 12 h after the last dose) or ON.")]

P = checks.published
ep = np.array(ep_dur)
cnt = np.array([ep_per_subj.get(s, 0) for s in range(1, 11)])
P("participants", D.person.nunique(), 10, 0, "Abstract / III.A Participants")
P("male participants", sum(1 for v in T1.values() if v[0] == "M"), 7, 0, "III.A Participants")
P("participants with FoG", int((cnt > 0).sum()), 8, 0, "IV.A Study statistics")
P("FoG events", len(ep), 237, 0, "Abstract, IV.A")
P("FoG events per patient: mean", cnt.mean(), 23.7, 0.05, "IV.A")
P("FoG events per patient: SD", cnt.std(ddof=1), 20.7, 0.05, "IV.A")
P("FoG events per patient: max", cnt.max(), 66, 0, "IV.A")
P("FoG duration: mean (s)", ep.mean(), 7.3, 0.05, "IV.A")
P("FoG duration: SD (s)", ep.std(ddof=1), 6.7, 0.05, "IV.A")
P("FoG duration: max (s)", ep.max(), 40.5, 0.05, "IV.A")
P("share of FoG episodes under 20 s (%)", 100 * (ep < 20).mean(), 93.2, 0.05, "IV.A")
P("age: mean", np.mean([v[1] for v in T1.values()]), 66.4, 0.05, "Table I")
P("age: SD", np.std([v[1] for v in T1.values()], ddof=1), 4.8, 0.05, "Table I")
P("disease duration: mean", np.mean([v[2] for v in T1.values()]), 13.7, 0.05, "Table I")
P("Hoehn and Yahr: mean", np.mean([v[3] for v in T1.values()]), 2.6, 0.05, "Table I")
P("total recorded data (min)", n_samples_total / FS / 60, 8 * 60 + 20, 3, "IV.A (8 h 20 min recorded)")
checks.internal("17 recordings, all 10 subjects present", len(D) == 17 and D.person.nunique() == 10)
checks.internal("no sample lost", n_samples_total == D.n_samples.sum())
checks.internal("annotation codes only 0/1/2", sum(n_ann.values()) == n_samples_total)
checks.internal("subjects 4 and 10 have no freeze (documentation)", ep_per_subj.get(4, 0) == 0 and ep_per_subj.get(10, 0) == 0)

meta = dict(
    paper=dict(authors="Bächlin M, Plotnik M, Roggen D, Maidan I, Hausdorff JM, Giladi N, Tröster G",
               year=2010, title="Wearable assistant for Parkinson's disease patients with the freezing of gait symptom",
               venue="IEEE Trans Inform Technol Biomed 14(2):436-446", doi="10.1109/TITB.2009.2036165"),
    lab="Troster (ETH Zurich) / Hausdorff (Tel Aviv Sourasky Medical Center)",
    data_url="https://doi.org/10.24432/C56K78", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Each of the 17 text files (one per subject and run) became one samples CSV; time_s is the file's millisecond column divided by 1000, "
             "and the 9 acceleration channels and the annotation are unchanged. Age, sex, disease duration and Hoehn and Yahr stage were typed in from "
             "Table I of the paper (the deposit has no clinical file) and assume that subject number n in the files is subject ID n in the table. "
             "The deposit's Matlab scripts and documentation are not used."),
    how=["One row is one recording of one participant walking in the lab (straight-line walking, walking with many turns, and a daily-living task "
         "of fetching a drink) wearing three wireless accelerometers. The deposit splits some participants' sessions into several runs because of "
         "recording breaks; recording numbers the runs in order.",
         "Samples (samples/<subject>_run<n>.csv) are at 64 Hz: time_s, then 3D acceleration in mg of the shank (just above the ankle), the thigh (just above the knee) "
         "and the lower back, each as forward, vertical and lateral, then the annotation (0 not part of the experiment, 1 experiment without freeze, 2 freezing of gait). "
         "Annotations were made afterwards from video by physiotherapists and may be up to a couple of hundred ms off.",
         "events lists each freezing episode, a run of samples annotated 2, as freeze_start_k and freeze_end_k in seconds from the file's time origin "
         "(end = time of the last annotated sample plus one sample interval)."],
    differences=["The paper's 'two patients' tested ON medication are subjects 2 and 8 in Table I; whether the deposit's file numbers follow the table's IDs is assumed, "
                 "not stated."],
    unit_note="recording (one participant and run)",
)
stdlib.release(ID, "timeseries", D, meta, checks, participants=part, participant_columns=pcols, extra_files=extra)
