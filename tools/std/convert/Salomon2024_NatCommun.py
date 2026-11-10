"""Salomon et al. (2024), Nat Commun 15:4853: Kaggle freezing-of-gait contest data, lower-back accelerometer series of people with
Parkinson's disease and freezing of gait doing FOG-provoking tests (tDCS-FOG, 128 Hz, m/s^2; DeFOG, 100 Hz, g), with expert video labels
(StartHesitation, Turn, Walking). One row per labelled training series.

Raw file (untrusted data, read only): raw_extra/Salomon2024_NatCommun/Competition Dataset.zip from https://zenodo.org/records/12565292 (CC BY 4.0).
Used: train/tdcsfog, train/defog, the three metadata CSVs, subjects.csv, events.csv, tasks.csv. Not used: the public and private test series
(no labels), train/notype (unspecified-type events; the Event column is not documented in the deposit and does not match the paper's Table 6),
the unlabelled daily-living series (separate 31 GB zips), sample_submission.csv.
Run from the site folder: python3 -I tools/std/convert/Salomon2024_NatCommun.py
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Salomon2024_NatCommun"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
z = zipfile.ZipFile(RAW / "Competition Dataset.zip")
P = "Competition Dataset/"


def R(n, **kw):
    return pd.read_csv(io.BytesIO(z.read(P + n)), dtype={"Subject": str, "Visit": str, "Id": str}, **kw)


tm, dm, su, ev, tk = R("tdcsfog_metadata.csv"), R("defog_metadata.csv"), R("subjects.csv"), R("events.csv"), R("tasks.csv")
ids = {k: sorted(n.split("/")[-1][:-4] for n in z.namelist() if n.startswith(f"{P}train/{k}/") and n.endswith(".csv")) for k in ("tdcsfog", "defog")}
assert len(ids["tdcsfog"]) == 833 and len(ids["defog"]) == 91

# person demographics: first row of subjects.csv for the person (clinical values can differ between visits; see README)
su["_v"] = pd.to_numeric(su.Visit, errors="coerce").fillna(0)
su1 = su.sort_values(["Subject", "_v"]).groupby("Subject", as_index=False).first()
demo = su1.set_index("Subject")

# rows in metadata order
order = []
for k, meta in (("tdcsfog", tm), ("defog", dm)):
    m = meta.set_index("Id")
    for i in ids[k]:
        order.append((k, i, m.loc[i]))
first = {}
for k, i, r in order:
    first.setdefault(r.Subject, len(first) + 1)
code = lambda s: f"{ID}_S{first[s]:04d}"

evs = {i: g for i, g in ev.groupby("Id")}
tks = {i: g for i, g in tk.groupby("Id")}
FS = {"tdcsfog": 128.0, "defog": 100.0}
rows, extra, tr = [], [], {}
nsamp = {"tdcsfog": 0, "defog": 0}
ev_n, ev_dur = {}, {}
for k, i, r in order:
    d = pd.read_csv(io.BytesIO(z.read(f"{P}train/{k}/{i}.csv")))
    fs = FS[k]
    assert (np.diff(d.Time.values) == 1).all() and d.Time.values[0] == 0
    out = pd.DataFrame({"time_s": d.Time.values / fs})
    lab_ok = globals().get("lab_ok", True) and bool(d[["StartHesitation", "Turn", "Walking"]].isin([0, 1]).all().all())
    chans = ["AccV", "AccML", "AccAP", "StartHesitation", "Turn", "Walking"]
    for c in chans:
        out[c] = d[c].values
    if k == "defog":
        for c in ("Valid", "Task"):
            out[c] = d[c].astype(int).values
        chans += ["Valid", "Task"]
    tr[r.Subject] = tr.get(r.Subject, 0) + 1
    fn = f"samples/{code(r.Subject)}_{k}_rec{tr[r.Subject]:03d}.csv"
    extra.append((fn, out.to_csv(index=False, float_format="%.8g", lineterminator="\r\n")))
    nsamp[k] += len(d)
    e, nk = [], {}
    g = evs.get(i)
    if g is not None:
        for _, x in g.iterrows():
            kin = "kinetic" if x.Kinetic == 1 else ("akinetic" if x.Kinetic == 0 else "kineticunknown")
            n = nk[x.Type] = nk.get(x.Type, 0) + 1
            e += [f"{x.Type}_{kin}_{n}_start={x.Init:.6g}", f"{x.Type}_{kin}_{n}_end={x.Completion:.6g}"]
            ev_n[x.Type] = ev_n.get(x.Type, 0) + 1
            ev_dur[x.Type] = ev_dur.get(x.Type, 0.0) + (x.Completion - x.Init)
    tg = tks.get(i)
    if tg is not None:
        cnt = {}
        for _, x in tg.iterrows():
            n = cnt[x.Task] = cnt.get(x.Task, 0) + 1
            nm = x.Task.replace("-", "").replace(" ", "")
            e += [f"task_{nm}_{n}_start={x.Begin:.6g}", f"task_{nm}_{n}_end={x.End:.6g}"]
    dd = demo.loc[r.Subject]
    tdcs = k == "tdcsfog"
    units = ";".join(["m/s^2" if tdcs else "g"] * 3 + ["0/1"] * 3 + ([] if tdcs else ["0/1", "0/1"]))
    rows.append(dict(
        dataset=ID, study=ID, experiment="tDCS-FOG" if tdcs else "DeFOG", group="", subject=r.Subject, person=r.Subject,
        subject_excluded_by_authors=False, age=dd.Age, sex=dd.Sex, handedness="",
        population="Parkinson's disease with freezing of gait", setting="clinic" if tdcs else "home",
        device=("APDM Opal accelerometer on the lower back, 128 Hz" if tdcs else "Axivity AX3/AX6 accelerometer on the lower back, 100 Hz"),
        session=int(r.Visit), block=(f"test {r.Test}" if tdcs else "FOG-provoking protocol"), trial=tr[r.Subject], phase="test",
        time_s="", task="walking", unit="recording", condition=f"{r.Medication} medication", pert_type="none", pert_value=0,
        pert_unit="", feedback="normal", excluded_by_authors=False, recording=tr[r.Subject], samples_file=fn,
        signal_kind="3D lower-back acceleration (vertical, mediolateral, anteroposterior) with expert freezing-of-gait labels",
        channels=";".join(chans), channel_units=units, sample_rate_hz=fs, n_samples=len(d), duration_s=len(d) / fs,
        events=";".join(e),
        processing="none stated for the accelerations; labels made from video by expert reviewers",
        trial_info=f"source_id_visit={r.Visit};medication={r.Medication}" + (f";test={r.Test}" if tdcs else "")))
D = pd.DataFrame(rows)

part = []
for s in first:
    dd = demo.loc[s]
    part.append(dict(person=s, disease_duration=dd.YearsSinceDx, updrs3_on=dd.UPDRSIII_On, updrs3_off=dd.UPDRSIII_Off, nfogq=dd.NFOGQ))
part = pd.DataFrame(part)
pcols = [("disease_duration", "number", "years", "Years since PD diagnosis (YearsSinceDx in subjects.csv; from the participant's first visit row)."),
         ("updrs3_on", "number", "points", "MDS-UPDRS part III ON medication (first visit row; empty when not tested)."),
         ("updrs3_off", "number", "points", "MDS-UPDRS part III OFF medication (first visit row; empty when not tested)."),
         ("nfogq", "number", "points", "New Freezing of Gait Questionnaire score (first visit row; DeFOG participants were screened with another questionnaire, so values may be 0 or empty).")]

C = checks.published
pt, pd_ = D[D.experiment == "tDCS-FOG"], D[D.experiment == "DeFOG"]
C("tDCS-FOG training series", len(pt), 833, 0, "Methods, Datasets (tDCS FOG)")
C("tDCS-FOG training patients", pt.person.nunique(), 62, 0, "Methods, Datasets (tDCS FOG)")
C("DeFOG training series with three class labels", len(pd_), 91, 0, "Methods, Datasets (DeFOG)")
C("DeFOG training patients", pd_.person.nunique(), 38, 0, "Methods, Datasets (DeFOG)")
C("tDCS-FOG sampling rate (Hz)", pt.sample_rate_hz.iloc[0], 128, 0, "Methods, Datasets (tDCS FOG)")
C("DeFOG sampling rate (Hz)", pd_.sample_rate_hz.iloc[0], 100, 0, "Methods, Datasets (DeFOG)")
C("training Start Hesitation events", ev_n["StartHesitation"], 107, 0, "Table 6")
C("training Start Hesitation time (min)", ev_dur["StartHesitation"] / 60, 39.8, 0.05, "Table 6")
C("training Walking time (min)", ev_dur["Walking"] / 60, 43.5, 0.05, "Table 6")
checks.internal("no person in both studies", set(pt.person).isdisjoint(set(pd_.person)))
checks.internal("all samples kept", D.n_samples.sum() == sum(nsamp.values()))
checks.internal("every person has demographics", part.shape[0] == D.person.nunique() and D.age.notna().all())
checks.internal("labels are 0/1 in the shared series", lab_ok)

meta = dict(
    paper=dict(authors="Salomon A, Gazit E, Ginis P, Urazalinov B, Takoi H, Yamaguchi T, Goda S, Lander D, Lacombe J, Sinha AK, Nieuwboer A, Kirsch LC, Holbrook R, Manor B, Hausdorff JM",
               year=2024, title="A machine learning contest enhances automated freezing of gait detection and reveals time-of-day effects",
               venue="Nat Commun 15:4853", doi="10.1038/s41467-024-49027-0"),
    lab="Hausdorff (Tel Aviv Sourasky Medical Center) with Nieuwboer (KU Leuven)",
    data_url="https://zenodo.org/records/12565292", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Each labelled training series (833 tDCS-FOG and 91 DeFOG) became one samples CSV. Time (a sample counter in the files) was divided by the sampling rate "
             "(128 or 100 Hz) to give time_s; the three acceleration channels and the three label channels are unchanged. In DeFOG files the Valid and Task columns "
             "(true/false) were written as 1/0. Participant codes replaced the authors' subject codes. Age and sex come from subjects.csv. Event rows of events.csv "
             "became the events column, and DeFOG rows of tasks.csv became task_* entries in events. Not included: the public and private test series (no labels), "
             "the 46 'notype' training series (their Event column is not documented in the deposit), the unlabelled daily-living recordings (two 31 GB files) and the deposit's code."),
    how=["One row is one recording: a participant wearing an accelerometer on the lower back during a freezing-of-gait (FOG) provoking test, either in the clinic "
         "(tDCS-FOG, APDM Opal, 128 Hz, acceleration in m/s^2) or at home (DeFOG, Axivity, 100 Hz, acceleration in g), on or off medication. Only the labelled "
         "training series are included.",
         "Samples (samples/<subject>_<study>_rec<nnn>.csv): time_s, then AccV, AccML, AccAP (vertical, mediolateral, anteroposterior acceleration; the deposit does not state axis signs), "
         "then StartHesitation, Turn, Walking, each 0/1 per sample, the expert video labels of FOG type. DeFOG files also have Valid and Task (1/0): the paper says only samples that "
         "were both Valid and Task were used in the contest scoring.",
         "events lists the authors' FOG events (events.csv) as <Type>_<kinetic|akinetic|kineticunknown>_<k>_start and _end in seconds from the start of the recording; kinetic/akinetic "
         "is the authors' Kinetic flag (empty in the source for some events). For DeFOG, the protocol task periods (tasks.csv: Rest, 4MW, MB, TUG, Turning, Hotspot ...) are listed as task_<name>_<k>_start/_end.",
         "Demographics and clinical scores are per participant from subjects.csv (first visit row; scores can differ between visits). Participants are anonymous codes."],
    differences=["The paper's Table 6 gives 1948 Turn and 395 Walking events in training; events.csv gives %d Turn and %d Walking events for the same series "
                 "(overlapping events are listed separately in the file), so those counts were not used as checks. Training Turn time is %.1f min here and 316.4 min in the paper." % (
                     ev_n["Turn"], ev_n["Walking"], ev_dur["Turn"] / 60),
                 "The 46 'notype' training series (Table 6: 1032 events of unspecified type, 83.2 min) are not included: their Event column has only %s of positive time in the shared files." % "under one minute",
                 "Table 4B demographics cover all 128 people including test-set participants and cannot be checked on the training rows alone."],
    unit_note="recording (one participant, one FOG-provoking test)",
)
stdlib.release(ID, "timeseries", D, meta, checks, participants=part, participant_columns=pcols, extra_files=extra)
