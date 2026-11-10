"""Viglialoro et al. (2025), Sci Data 12:1937: gait analysis of people with Parkinson's disease before and after Nordic walking or adapted physical activity.

Run from the site folder: python3 -I tools/std/convert/Viglialoro2025_SciData.py
Raw file (untrusted data, read only) in raw_extra/Viglialoro2025_SciData/, from https://doi.org/10.6084/m9.figshare.29371769.v1 :
  data.zip   (OneDrive_2025-06-20.zip) Dataset/Group_*/ID??/ID??_{Pre,Post}/ID??_{Pre,Post}_{S,T1,T2,T3}/ with per-trial
             _TR.trc (55 markers, 100 Hz), _JA.csv (joint angles), _GRF.mot (2 force plates), _EMG.csv (16 muscles), _STP.csv (spatio-temporal
             parameters). The _c3d, _vsk, _Antr.mp files and Dataset.rar duplicate or only support these and are not used.
Several deposit files have extra sections or mis-matched contents; this converter reads only the first section of each file, and skips a file
whose first section is not the expected kind (counted in the README). Participant names found in some files are never copied.
Participant sex, age, mass and height come from Table 2 of the paper (the deposit's Partecipants.xls is not in the Figshare files).
"""
import io
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Viglialoro2025_SciData"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

# Table 2 of the paper: group (S = study/Nordic walking, C = control), age, sex, body mass (kg), height (m)
TAB = """01 S 67 M 85 1.72|02 S 78 M 79 1.76|03 S 78 M 75 1.69|04 S 63 F 91 1.65|05 S 67 F 64 1.63|06 S 71 M 58 1.72|07 S 78 M 66 1.67
08 S 85 M 69 1.66|09 S 73 M 86 1.79|10 S 69 M 86 1.75|11 S 72 M 73 1.61|12 S 78 M 78 1.68|13 S 74 M 79 1.66|14 S 78 M 74 1.75
15 C 82 M 82 1.67|16 C 63 M 73 1.62|17 C 83 M 68 1.60|18 C 66 M 76 1.60|19 C 80 F 47 1.51|20 C 75 M 58 1.60|21 C 85 F 58 1.58
22 C 78 M 76 1.70|23 C 80 F 82 1.82|24 C 75 M 104 1.75"""
PART = {}
for rec in TAB.replace("\n", "|").split("|"):
    i, g, a, s, m, h = rec.split()
    PART[f"ID{i}"] = dict(group=g, age=int(a), sex=s, mass=float(m), height=float(h))
GROUP = {"S": "Nordic walking", "C": "adapted physical activity (control)"}

z = zipfile.ZipFile(RAW / "data.zip")
members = {}
for n in z.namelist():
    m = re.search(r"/Dataset/Group_\w+/(ID\d\d)/ID\d\d_(Pre|Post)/ID\d\d_\w+?_(S|T[123])/ID\d\d_\w+?_\w+?_(JA|EMG|STP|GRF|TR)\.(csv|mot|trc)$", n)
    if m:
        members[(m.group(1), m.group(2), m.group(3), m.group(4))] = n
checks.internal("48 trial folders (24 participants x pre/post x S,T1,T2,T3) with markers", sum(1 for k in members if k[3] == "TR") == 192)
skipped = []
TMP = RAW / "_tmp_samples"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir()


def text(n):
    return z.read(n).decode("utf-8", "replace").splitlines()


def first_section(lines):
    out = []
    for l in lines:
        if not l.strip():
            break
        out.append(l)
    return out


def clean_name(s):
    s = s.strip()
    return s.split(":")[-1] if ":" in s else s


def parse_trc(n):
    t = first_section(text(n))
    hdr = t[1].split("\t")
    rate, nfr, nmk, unit, start = float(hdr[0]), int(hdr[2]), int(hdr[3]), hdr[4], int(hdr[6])
    names = [x.strip() for x in t[2].split("\t")[2:] if x.strip()]
    assert len(names) == nmk, (len(names), nmk)
    body = pd.read_csv(io.StringIO("\n".join(t[4:])), sep="\t", header=None)
    body = body.iloc[:, : 2 + 3 * nmk]
    cols = ["frame", "time_s"] + [f"{nm}_{a}" for nm in names for a in "xyz"]
    body.columns = cols
    assert len(body) == nfr, (len(body), nfr)
    return body, rate, nmk, unit, start


def parse_ja(n):
    t = first_section(text(n))
    h0 = t[0].split(",")
    assert h0[0] == "" and t[1].startswith("Frame,Sub Frame"), "unexpected JA layout"
    names = [clean_name(x) for x in h0[2:] if x.strip()]
    ncol = len(h0)
    body = pd.read_csv(io.StringIO("\n".join(t[3:])), header=None)
    assert body.shape[1] >= 2 + 3 * len(names), (body.shape, len(names))
    body = body.iloc[:, : 2 + 3 * len(names)]
    body.columns = ["frame", "sub"] + [f"{nm}_{a}" for nm in names for a in "xyz"]
    return body, names


def parse_emg(n):
    t = first_section(text(n))
    if not t[0].startswith("Frame, Cub Frame") or len(t) < 4:
        return None
    names = [x.strip() for x in t[0].split(",")[2:]]
    try:
        body = pd.read_csv(io.StringIO("\n".join(t[2:])), header=None)
    except Exception:
        return None
    if body.shape[1] < 2 + len(names) or not np.issubdtype(body.iloc[:, 0].dtype, np.number):
        return None
    body = body.iloc[:, : 2 + len(names)]
    body.columns = ["frame", "sub"] + names
    return body, names


def parse_grf(n):
    t = text(n)
    cols = t[1].split("\t")
    cols = [c.strip() for c in cols if c.strip()]
    body = pd.read_csv(io.StringIO("\n".join(t[2:])), sep="\t", header=None)
    body = body.iloc[:, : len(cols)]
    body.columns = cols
    return body


def parse_stp(n):
    t = first_section(text(n))
    if not t[0].startswith("Context,Name,Value,Units"):
        return None
    out = []
    for l in t[1:]:
        p = l.split(",")
        out.append((p[0].strip().lower(), re.sub(r"\W+", "_", p[1].strip().lower()), p[2].strip(), p[3].strip()))
    return out


rows, extra = [], []
emg_cov, emg_missing, stp_missing = [], 0, 0
ntr = {}
mk_count, rates, rate_list = set(), set(), []
trial_no = {}
for pid in sorted(PART):
    P = PART[pid]
    for sess, sname in ((1, "Pre"), (2, "Post")):
        for tr in ("S", "T1", "T2", "T3"):
            if (pid, sname, tr, "TR") not in members:
                continue
            trial_no.setdefault(pid, 0)
            static = tr == "S"
            tag = f"{ID}_{pid}_{sname}_{tr}"
            base = dict(dataset=ID, study=ID, experiment="Nordic walking vs adapted physical activity, pre/post", group=GROUP[P["group"]], subject=pid, person=pid,
                        subject_excluded_by_authors=False, age=P["age"], sex=P["sex"], handedness="", population="Parkinson's disease", setting="lab",
                        device="8-camera Vicon Vero motion capture (100 Hz), 2 AMTI OR6-7-1000 force plates, 16-channel Cometa wireless EMG",
                        session=sess, block=sname, trial=0, phase=("other" if static else ("baseline" if sess == 1 else "test")), time_s="",
                        task="walking", unit="recording", pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False)
            cond = (f"{sname}-training static standing trial" if static else f"{sname}-training, 10 m straight-path walking, trial {tr[1]}")
            ntr[(pid, sess)] = ntr.get((pid, sess), 0) + (0 if static else 1)
            # markers
            body, rate, nmk, unit, start = parse_trc(members[(pid, sname, tr, "TR")])
            mk_count.add(nmk)
            rates.add(rate)
            rate_list.append(rate)
            ch = list(body.columns[2:])
            fn = f"samples/{tag}_markers.csv"
            body.drop(columns=["frame"]).to_csv(TMP / f"{tag}_markers.csv", index=False, float_format="%.8g", lineterminator="\r\n")
            extra.append((fn, TMP / f"{tag}_markers.csv"))
            trial_no[pid] += 1
            rows.append(dict(base, trial=trial_no[pid], recording=trial_no[pid], condition=cond, samples_file=fn, signal_kind="3D marker trajectories (55 markers, PyCGM2 full-body marker set)",
                             channels=";".join(ch), channel_units=";".join([unit] * len(ch)), sample_rate_hz=rate, n_samples=len(body),
                             duration_s=round(len(body) / rate, 6), events="", processing="reconstructed (rigid body fill, cyclic fill) in Vicon Nexus; time_s is the authors' Time, from the start of the capture",
                             trial_info=f"first_frame={start}"))
            # joint angles (+ STP)
            ja = parse_ja(members[(pid, sname, tr, "JA")])
            jb, jn = ja
            jb.insert(0, "time_s", (jb["frame"] - 1) / rate)
            jch = list(jb.columns[3:])
            fn = f"samples/{tag}_angles.csv"
            jb.drop(columns=["frame", "sub"]).to_csv(TMP / f"{tag}_angles.csv", index=False, float_format="%.8g", lineterminator="\r\n")
            extra.append((fn, TMP / f"{tag}_angles.csv"))
            tinfo = ""
            if not static:
                st = parse_stp(members[(pid, sname, tr, "STP")]) if (pid, sname, tr, "STP") in members else None
                if st is None:
                    stp_missing += 1
                else:
                    tinfo = ";".join(f"{c}_{nme}[{u}]={v}" for c, nme, v, u in st if v != "")
            trial_no[pid] += 1
            rows.append(dict(base, trial=trial_no[pid], recording=trial_no[pid], condition=cond, samples_file=fn, signal_kind="joint angles from the full-body PyCGM2 model (x, y, z components per angle)",
                             channels=";".join(jch), channel_units=";".join(["deg"] * len(jch)), sample_rate_hz=rate, n_samples=len(jb), duration_s=round(len(jb) / rate, 6),
                             events="", processing="computed in Vicon Nexus / PyCGM2 from the reconstructed markers",
                             trial_info=tinfo))
            # force plates
            g = parse_grf(members[(pid, sname, tr, "GRF")])
            fn = f"samples/{tag}_forceplates.csv"
            g.rename(columns={"time": "time_s"}).to_csv(TMP / f"{tag}_forceplates.csv", index=False, float_format="%.8g", lineterminator="\r\n")
            extra.append((fn, TMP / f"{tag}_forceplates.csv"))
            gch = [c for c in g.columns if c != "time"]
            gu = ["N" if "force" in c and ("_v" in c) else ("m (header says mm; values ~0.5 are plate coordinates in m)" if "_p" in c else "header: N/mm") for c in gch]
            dt = float(g["time"].iloc[1] - g["time"].iloc[0])
            trial_no[pid] += 1
            rows.append(dict(base, trial=trial_no[pid], recording=trial_no[pid], condition=cond, samples_file=fn,
                             signal_kind="ground reaction force (v), centre of pressure (p) and free moment (torque) of 2 force plates",
                             channels=";".join(gch), channel_units=";".join(gu), sample_rate_hz=round(1 / dt, 3), n_samples=len(g), duration_s=round(len(g) * dt, 6),
                             events="", processing="none stated (the paper: no filtering or interpolation); time_s is the authors' time", trial_info=""))
            # EMG
            if not static:
                e = parse_emg(members[(pid, sname, tr, "EMG")]) if (pid, sname, tr, "EMG") in members else None
                if e is None:
                    emg_missing += 1
                else:
                    eb, en = e
                    nsub = int(eb["sub"].max()) + 1
                    eb.insert(0, "time_s", (eb["frame"] - 1) / rate + eb["sub"] / (rate * nsub))
                    fn = f"samples/{tag}_emg.csv"
                    eb.drop(columns=["frame", "sub"]).to_csv(TMP / f"{tag}_emg.csv", index=False, float_format="%.8g", lineterminator="\r\n")
                    extra.append((fn, TMP / f"{tag}_emg.csv"))
                    emg_cov.append(float((eb[en].to_numpy(float) > -3000).mean()))
                    trial_no[pid] += 1
                    rows.append(dict(base, trial=trial_no[pid], recording=trial_no[pid], condition=cond, samples_file=fn,
                                     signal_kind="surface EMG, 16 muscles (L/R lumbar erector spinae, external oblique, gluteus medius, rectus femoris, biceps femoris, tibialis anterior, lateral gastrocnemius, soleus)",
                                     channels=";".join(en), channel_units=";".join(["microvolt (file label 'micronV')"] * len(en)), sample_rate_hz=rate * nsub,
                                     n_samples=len(eb), duration_s=round(len(eb) / (rate * nsub), 6), events="",
                                     processing="raw as exported; the value -3299.9 is a recurring dropout code in some channels, left as is",
                                     trial_info=""))
df = pd.DataFrame(rows)
print("emg files unusable:", emg_missing, "stp unusable:", stp_missing, "n emg", len(emg_cov))
cv = np.array(emg_cov)
print("emg coverage mean/sd/min/max", cv.mean(), cv.std(ddof=1), cv.min(), cv.max())

part = pd.DataFrame([dict(person=pid, body_mass_kg=P["mass"], height_m=P["height"], bmi=round(P["mass"] / P["height"] ** 2, 1)) for pid, P in PART.items()])
pcols = [("body_mass_kg", "number", "kg", "Body mass, from Table 2 of the paper."), ("height_m", "number", "m", "Height, from Table 2 of the paper."),
         ("bmi", "number", "kg/m2", "Body mass index as listed in Table 2 of the paper (recomputed from mass and height, one decimal).")]

pp = df.drop_duplicates("person")
checks.published("participants", len(pp), 24, 0, "Methods, Participants (final sample of 24)")
checks.published("participants in the Nordic walking group", (pp.group == GROUP["S"]).sum(), 14, 0, "Methods, Group allocation (NW n = 14)")
checks.published("participants in the control group", (pp.group == GROUP["C"]).sum(), 10, 0, "Methods, Group allocation (APA n = 10)")
checks.published("men", (pp.sex == "M").sum(), 19, 0, "Methods, Participants (19 men and 5 women)")
checks.published("women", (pp.sex == "F").sum(), 5, 0, "Methods, Participants")
checks.published("mean age (years)", pp.age.mean(), 74.92, 0.005, "Methods, Participants")
checks.published("SD of age (years)", pp.age.std(ddof=1), 6.53, 0.005, "Methods, Participants")
checks.published("SD of body mass (kg)", part.body_mass_kg.std(ddof=1), 12.37, 0.005, "Methods, Participants")
checks.published("markers per recording", list(mk_count)[0], 55, 0, "Methods, Procedure (55 reflective skin markers)")
checks.published("walking trials per participant and session (minimum)", min(ntr.values()), 3, 0, "Technical Validation (three repeated trials, pre and post)")
checks.published("walking trials per participant and session (maximum)", max(ntr.values()), 3, 0, "Technical Validation")
checks.published("motion capture frame rate, most common (Hz)", pd.Series(rate_list).mode()[0], 100, 0, "Methods, Experimental set-up (100 fps)")
checks.internal("every participant has pre and post sessions", all(len({k[1] for k in ntr if k[0] == p}) == 2 for p in PART))
checks.internal("192 marker recordings and 192 joint-angle and force-plate recordings", (df.signal_kind.str.startswith("3D marker")).sum() == 192
                and df.signal_kind.str.startswith("joint angles").sum() == 192 and df.signal_kind.str.startswith("ground reaction").sum() == 192)

meta = dict(
    paper=dict(authors="Viglialoro RM, Centrone A, Mattei L, Dalise S, Ferrari NA, Azzollini V, Chisari C, Di Puccio F", year=2025,
               title="Dataset on Gait Analysis of Parkinsonian Subjects: Effect of Nordic Walking", venue="Sci Data 12:1937", doi="10.1038/s41597-025-06209-9"),
    lab="Di Puccio (University of Pisa)", data_url="https://doi.org/10.6084/m9.figshare.29371769.v1", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted the per-trial marker (TRC), joint angle (CSV), force plate (MOT) and EMG (CSV) text files to one CSV per recording, with spatio-temporal "
             "parameters in trial_info. Only the first section of each file is read. Age, sex, body mass and height come from Table 2 of the paper. "
             "The deposit's C3D, VSK and anthropometric files are not used."),
    how=["One row is one recording of one trial of a participant with Parkinson's disease (14 in the Nordic walking group, 10 in the adapted physical activity control group), "
         "before (Pre) or after (Post) a 12-week program: for each of the three 10 m straight-path walking trials (T1 to T3) a marker recording (55 markers, mm, 100 Hz), a joint-angle recording "
         "(PyCGM2 model angles, deg, x/y/z components), a force-plate recording (two plates: force, centre of pressure and torque columns as in the .mot file) and, when present, a 16-channel surface EMG "
         "recording; the static standing trial (S) has markers, joint angles and force plates. time_s is the authors' time from the start of the capture (frame 1 = 0 s).",
         "Force plate and EMG rates were measured from the files: both are 2000 Hz (the deposit's readme says 1000 Hz; the paper gives 1000 Hz for the plates and 2000 Hz for the EMG). "
         "The .mot header states distances in mm and torques in N/mm, but the centre-of-pressure values are of the order of 0.5, which is plate size in metres, so they are labelled metres here; "
         "torque units are as the header states. Spatio-temporal parameters (left/right cadence, walking speed, stride and step time, support times, stride and step length, step width, limp index) "
         "are in trial_info as <side>_<name>[unit]=value, from the _STP.csv files. In the EMG, -3299.9 recurs as a dropout code and was not changed.",
         f"Some deposit files hold other content than their name says: {emg_missing} EMG files contain no EMG section, {stp_missing} STP files are not a parameter table; these recordings are simply absent. "
         "Some EMG files append extra sections (joint angles, a gait deviation index with a participant name); only the first section is read and no names are copied."],
    differences=["Table 2 gives a mean body mass of 74.46 kg (SD 12.37) for the 24 participants, while the text says 74.60 kg; the table values are used and the mean was not used as a check.",
                 "One participant's four pre-training recordings (S, T1, T2, T3) were captured at 120 Hz instead of 100 Hz (TRC header); their sample rates (markers, angles, EMG, force plates) follow that rate and are stated per row.",
                 "The deposit's readme and metadata give 1000 Hz for the force plates and EMG; the files are sampled at 2000 Hz.",
                 "The deposit's Partecipants.xls is cited by the paper but is not among the Figshare files; demographics come from the paper's Table 2."],
    unit_note="recording (3D markers, joint angles, force plates or EMG of one trial; samples in a CSV inside the ZIP)",
)
try:
    stdlib.release(ID, "timeseries", df, meta, checks, participants=part, participant_columns=pcols, extra_files=extra)
finally:
    shutil.rmtree(TMP, ignore_errors=True)
