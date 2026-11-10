"""Gait in Parkinson's Disease (PhysioNet gaitpdb 1.0.0, ODC-By 1.0; Hausdorff lab, Tel Aviv Sourasky Medical Center): vertical
ground reaction force under both feet while walking about 2 minutes at a self-selected pace, from three studies (Ga = Yogev et
al. 2005 dual tasking; Ju = Hausdorff et al. 2007 rhythmic auditory stimulation; Si = Frenkel-Toledo et al. 2005 treadmill
walking). One row per recording (306), samples in samples/<id>_S<code>_walk<NN>.csv.

Raw files (raw_extra/Yogev2005_EurJNeurosci/): the 306 recordings <study><Co|Pt><nn>_<walk>.txt (19 tab-separated columns, 100 Hz),
demographics.txt and format.txt (read as text).
Run from the site folder: python3 -I tools/std/convert/Yogev2005_EurJNeurosci.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Yogev2005_EurJNeurosci"
raw = stdlib.raw_dir(DS)
OUTS = raw / "_samples"
OUTS.mkdir(exist_ok=True)
ck = stdlib.Checks()

# demographics.txt: tab separated with ragged trailing tabs
lines = [l.rstrip("\t\r\n").split("\t") for l in (raw / "demographics.txt").read_text().splitlines() if l.strip()]
H = lines[0]
demo = pd.DataFrame([r[:len(H)] for r in lines[1:]], columns=H)
for c in H:
    if c not in ("ID", "Study"):
        demo[c] = pd.to_numeric(demo[c], errors="coerce")
demo = demo.set_index("ID")

CH = [f"L{i}" for i in range(1, 9)] + [f"R{i}" for i in range(1, 9)] + ["L_total", "R_total"]
EXP = {"Ga": "Ga: dual tasking (Yogev et al. 2005)", "Ju": "Ju: rhythmic auditory stimulation (Hausdorff et al. 2007)",
       "Si": "Si: treadmill walking (Frenkel-Toledo et al. 2005)"}
DEVICE = "Ultraflex Computer Dyno Graphy insole system (Infotronic): 8 force sensors under each foot, 100 Hz"

files = {}
for f in sorted(raw.glob("*_*.txt")):
    pid, walk = f.stem.split("_")
    files.setdefault(pid, []).append((int(walk), f))
people = sorted(files)
code = {pid: f"{DS}_S{i + 1:04d}" for i, pid in enumerate(people)}  # same order stdlib uses (first appearance)
rows, extra, n_bad = [], [], 0
for pid in people:
    d = demo.loc[pid]
    study, pd_group = d.Study, int(d.Group) == 1
    for k, (walk, f) in enumerate(sorted(files[pid]), start=1):
        L = [l.split("\t") for l in f.read_text().splitlines() if l.strip()]
        if any(len(r) != 19 for r in L):
            n_bad += 1
        t = [float(r[0]) for r in L]
        name = f"samples/{code[pid]}_walk{walk:02d}.csv"
        (OUTS / f"{code[pid]}_walk{walk:02d}.csv").write_text(
            "time_s," + ",".join(CH) + "\n" + "\n".join(",".join(r) for r in L) + "\n")
        extra.append((name, str(OUTS / f"{code[pid]}_walk{walk:02d}.csv")))
        dual = study == "Ga" and walk == 10
        spd = d.get(f"Speed_{walk:02d}")
        info = f"walk={walk:02d}" + (f";walking_speed_mps={spd:g}" if pd.notna(spd) else "")
        rows.append(dict(
            dataset=DS, study="Yogev et al. 2005 Eur J Neurosci (with Hausdorff 2007 and Frenkel-Toledo 2005 recordings)",
            experiment=EXP[study], group="Parkinson's disease" if pd_group else "control", subject=pid,
            subject_excluded_by_authors=False, age=d.Age, sex="M" if int(d.Gender) == 1 else "F",
            population="Parkinson's disease" if pd_group else "healthy", setting="lab", device=DEVICE, session=1,
            block=f"walk {walk:02d}", trial=k, phase="test", task="walking", unit="recording",
            condition=("dual task: serial 7 subtraction while walking" if dual else "usual walking" if walk == 1 else f"walk {walk:02d}"),
            pert_type="dual_task" if dual else "none", pert_value=1 if dual else 0,
            pert_unit="serial 7 subtraction" if dual else "", feedback="normal", excluded_by_authors=False,
            recording=k, samples_file=name, signal_kind="vertical ground reaction force under each foot (8 insole sensors per foot) and the sum per foot",
            channels=";".join(CH), channel_units=";".join(["N"] * 18), sample_rate_hz=100, n_samples=len(L), duration_s=round(t[-1], 4),
            processing="none (as shared)", trial_info=info))
df = pd.DataFrame(rows)
for c in ("handedness", "time_s", "events"):
    df[c] = ""

# participants.csv: persons with recordings
have = demo.loc[people].copy()
height = have.Height.where(have.Height < 3, have.Height / 100.0)  # Ju study gives centimetres
participants = pd.DataFrame(dict(
    person=have.index, group=["Parkinson's disease" if g == 1 else "control" for g in have.Group], study=have.Study,
    height_m=height.values, weight_kg=have.Weight.values, hoehn_yahr=have.HoehnYahr.values, updrs_total=have.UPDRS.values,
    updrs_motor=have.UPDRSM.values, tug_s=have.TUAG.values))
pcols = [("group", "text", "", "Parkinson's disease or control."),
         ("study", "text", "Ga, Ju, Si", "Study the participant took part in (Ga Yogev et al. 2005, Ju Hausdorff et al. 2007, Si Frenkel-Toledo et al. 2005). A person who took part in more than one study may appear under two codes: the deposit gives no link."),
         ("height_m", "number", "m", "Height. The deposit gives metres for the Ga and Si studies and centimetres for the Ju study; Ju values were divided by 100."),
         ("weight_kg", "number", "kg", "Body mass; empty when not shared."),
         ("hoehn_yahr", "number", "stage", "Hoehn and Yahr stage (Parkinson's disease group); empty for controls."),
         ("updrs_total", "number", "points", "Unified Parkinson's Disease Rating Scale total score as the deposit gives it (0 for controls)."),
         ("updrs_motor", "number", "points", "UPDRS motor score as the deposit gives it (UPDRSM; 0 for controls)."),
         ("tug_s", "number", "seconds", "Value of the deposit's TUAG column, taken to be the Timed Up and Go time in seconds (the deposit does not define the abbreviation or its unit).")]

# --- checks
pdn = lambda s, g: int(((have.Study == s) & (have.Group == g)).sum())  # noqa: E731
ck.internal("306 recordings read, each with 19 columns and 100 Hz sampling", len(df) == 306 and n_bad == 0 and (df.sample_rate_hz == 100).all(), f"{len(df)} rows, {n_bad} malformed")
ck.internal("recording durations 30 s to 5 minutes", df.duration_s.between(30, 300).all(), f"{df.duration_s.min()}-{df.duration_s.max()} s")
ck.internal("every person in demographics.txt except one row without recordings has a participants row", len(participants) == len(people) == 165, f"{len(people)} persons; demographics rows {len(demo)}")
ck.published("Hausdorff 2007 (study Ju): patients with idiopathic PD", pdn("Ju", 1), 29, 0, "Abstract of Hausdorff et al. 2007")
ck.published("patients with idiopathic PD in the whole database", (have.Group == 1).sum(), 93, 0, "PhysioNet database description")
ck.published("mean age of the PD patients, years (66.3)", have[have.Group == 1].Age.mean(), 66.3, 0.05, "PhysioNet database description")
print("persons by study/group:", {(s, g): pdn(s, g) for s in ("Ga", "Ju", "Si") for g in (1, 2)})
print("PD men %", 100 * (have[have.Group == 1].Gender == 1).mean(), "controls n", (have.Group == 2).sum(), "controls mean age", have[have.Group == 2].Age.mean(),
      "Ga PD age", have[(have.Study == "Ga") & (have.Group == 1)].Age.mean())

meta = dict(
    paper=dict(authors="Yogev G, Giladi N, Peretz C, Springer S, Simon ES, Hausdorff JM", year=2005,
               title="Dual tasking, gait rhythmicity, and Parkinson's disease: Which aspects of gait are attention demanding?",
               venue="Eur J Neurosci 22(5):1248-1256", doi="10.1111/j.1460-9568.2005.04298.x"),
    lab="Hausdorff (Laboratory for Gait & Neurodynamics, Tel Aviv Sourasky Medical Center)",
    data_url="https://physionet.org/content/gaitpdb/1.0.0/", license="ODC-By 1.0",
    license_url="https://opendatacommons.org/licenses/by/1-0/",
    changes=("Each recording's 19 columns are written unchanged as a CSV (tab separators replaced by commas, a header row added with "
             "the sensor names from the deposit's format.txt). Participant IDs were replaced by codes. Heights of the Ju study "
             "(centimetres) were converted to metres in participants.csv. Gender codes 1 and 2 are written as M and F."),
    how=["One row is one walking recording (about 2 minutes of walking at a self-selected pace on level ground, or a 100 m walk for the Ju study), "
         "with its samples in the CSV named in samples_file: time_s, then the vertical ground reaction force in newtons of the 8 insole "
         "sensors under the left foot (L1-L8), the 8 under the right foot (R1-R8), and the total force under the left (L_total) and right "
         "(R_total) foot, sampled at 100 Hz. The sensors' relative positions in the insole are in the deposit's format.txt.",
         "The deposit combines three studies. Walk 01 is a usual walk in all of them. In the Ga study walk 10 was walked while "
         "subtracting sevens (dual task), as format.txt states; the walks numbered 02 to 07 of the Ga and Ju studies and 02 of the Si study "
         "are not described in the deposit (the papers describe usual walking and rhythmic auditory stimulation conditions, but the deposit "
         "does not say which number is which), so they are labelled by number only. The order of walks within a person is taken to be the walk "
         "number (trial). walking_speed_mps in trial_info is the deposit's Speed_NN value for that walk when it is given.",
         "Groups: 1 = Parkinson's disease, 2 = control (the deposit's Group column; the deposit's demographics page confirms the labels). "
         "Gender 1 = male and 2 = female, checked against the deposit's demographics.html page."],
    unit_note="walking recording",
    differences=["The Ga (Yogev et al. 2005) part of the deposit has 29 patients and 18 controls with recordings, whereas the paper's abstract reports "
                 "30 patients (mean age 71.8) and 28 controls; the deposit gives a mean age of 71.1 years for those patients. The deposit holds only "
                 "part of that study's sample, so these paper numbers are not used as checks.",
                 "The Ju (Hausdorff et al. 2007) abstract reports 26 controls; demographics.txt lists 26 control rows but one of them (typed with a "
                 "different ID spelling) has no recording, so 25 controls have data. The 29 patients match.",
                 "The PhysioNet description gives 73 controls and 'mean age 66.3 years' for both groups and 63% men among the patients; with the 72 "
                 "controls who have recordings the mean age is 63.7 years, and 62.4% of the patients (the description says 63%) and 55.6% of the controls (it says 55%) are men, so those numbers are not used as checks."],
)
stdlib.release(DS, "timeseries", df, meta, ck, participants=participants, participant_columns=pcols, extra_files=extra)
for c in ck.items:
    print(c.get("what"), c["ok"], c.get("computed"), c.get("reported"))
