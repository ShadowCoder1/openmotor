"""Seethapathi & Srinivasan (2015), Biol Lett 11(9):20150486: metabolic cost of oscillating-speed treadmill walking.

Raw files, raw_extra/<id>/ (Zenodo 4959123 = copy of Dryad doi 10.5061/dryad.15v26, CC0):
  <Constant|Oscillating>SpeedTrial<k>_SacralMotion.csv  header, then for each subject a line 'Subject n' and rows time,X,Y,Z (100 Hz)
  MetabolicData_AllSubjects.txt    subject properties, resting costs, steady metabolic rates at 5 speeds, delta metabolic rates of
                                   the 6 oscillating trials (W/kg)
  Dat_PreferredWalkingSpeeds.txt   Expt 2: preferred speeds by distance (10 other subjects, numbered separately)
Subjects with no data in a trial have a 'Subject n' line followed by 1 or 2 non-data lines; they are skipped.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Seethapathi2015_BiolLett"
RAW = stdlib.raw_dir(ID)
TMP = RAW / "_samples"
TMP.mkdir(exist_ok=True)
NUM = r"(-?[0-9.]+|NaN)"


def f(x):
    return np.nan if x == "NaN" else float(x)


# ---------- metabolic file
M = (RAW / "MetabolicData_AllSubjects.txt").read_text(errors="replace").splitlines()
demo, rest, steady, delta = {}, {}, {}, {}
sec = None
for l in M:
    if "Resting costs" in l:
        sec = "rest"
    elif "Metabolic Rate at Steady" in l:
        sec = "steady"
    elif "Metabolic Rate of Oscillating Speed Walking" in l:
        sec = "delta"
    if sec is None and (m := re.match(r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([MF])\s*$", l)):
        demo[int(m[1])] = dict(mass=float(m[2]), height=float(m[3]), leg=float(m[4]), age=int(m[5]), sex=m[6])
    elif sec == "rest" and (m := re.match(rf"^(\d+)\s+{NUM}\s+{NUM}\s*$", l)):
        rest[int(m[1])] = (f(m[2]), f(m[3]))
    elif sec == "steady" and (m := re.match(rf"^(\d+)\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s*$", l)):
        steady[int(m[1])] = [f(m[i]) for i in range(2, 7)]
    elif sec == "delta" and (m := re.match(rf"^(\d+)\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s*$", l)):
        delta[int(m[1])] = [f(m[i]) for i in range(2, 8)]
assert len(demo) == len(rest) == len(steady) == len(delta) == 16, (len(demo), len(rest), len(steady), len(delta))
SPEEDS = [0.89, 1.12, 1.34, 1.56, 1.78]

# ---------- sacral motion
TRIALS = [("Constant", 1), ("Constant", 2)] + [("Oscillating", k) for k in range(1, 7)]
rec = {s: [] for s in range(1, 17)}  # subject -> list of (kind, k, header, lines)
for kind, k in TRIALS:
    L = (RAW / f"{kind}SpeedTrial{k}_SacralMotion.csv").read_text(errors="replace").splitlines()
    head = L[:12]
    v = float(re.search(r"v_avg = ([0-9.]+)", " ".join(head))[1])
    tf = re.search(r"T_fwd = ([0-9.]+) s, T_bck = ([0-9.]+)", " ".join(head))
    hdr_name = L[1].strip()
    cur, blk = None, {}
    for l in L:
        if (m := re.match(r"Subject (\d+)\s*$", l)):
            cur = int(m[1])
            blk[cur] = []
        elif cur and re.match(r"-?[0-9]", l):
            blk[cur].append(l)
    for s, rows in blk.items():
        if len(rows) > 10:
            rec[s].append(dict(kind=kind, k=k, v=v, tf=float(tf[1]) if tf else None, tb=float(tf[2]) if tf else None, name=hdr_name, rows=rows))

rows_out, extra = [], []
n_samp = 0
for s in range(1, 17):
    d = demo[s]
    for tr, r in enumerate(rec[s], 1):
        code = f"{ID}_S{s:04d}"
        fn = f"samples/{code}_trial{tr:03d}.csv"
        lines = ["time_s,X_mm,Y_mm,Z_mm"]
        for l in r["rows"]:
            p = l.split(",")
            assert len(p) == 4, l
            lines.append(",".join("" if x == "NaN" else x for x in p))
        path = TMP / f"{code}_trial{tr:03d}.csv"
        path.write_text("\n".join(lines) + "\n")
        extra.append((fn, str(path)))
        t_last = float(r["rows"][-1].split(",")[0])
        n_samp += len(r["rows"])
        osc = r["kind"] == "Oscillating"
        iv = SPEEDS.index(r["v"])
        info = f"treadmill_speed_mps={r['v']}"
        if osc:
            dm = delta[s][r["k"] - 1]
            info += f";T_fwd_s={r['tf']:g};T_bck_s={r['tb']:g};delta_metabolic_rate_wkg={'' if np.isnan(dm) else dm}"
            info += f";steady_metabolic_rate_same_speed_wkg={'' if np.isnan(steady[s][iv]) else steady[s][iv]}"
        else:
            sm = steady[s][iv]
            info += f";steady_metabolic_rate_wkg={'' if np.isnan(sm) else sm}"
        rows_out.append(dict(
            dataset=ID, study="Seethapathi 2015 Biol Lett", experiment="Expt-1 metabolic cost of oscillating-speed walking", group="",
            subject=str(s), person=str(s), subject_excluded_by_authors=False, age=d["age"], sex=d["sex"], handedness="",
            population="healthy", setting="lab", device="treadmill; motion capture (sacral marker); Oxycon Mobile respirometry",
            session=1, block=(f"oscillating trial {r['k']}" if osc else f"constant speed trial {r['k']} (file header: {r['name']})"),
            trial=tr, phase="test" if osc else "baseline", time_s="", task="walking", unit="recording",
            condition=(f"oscillating speed, mean {r['v']} m/s, T_fwd {r['tf']:g} s, T_bck {r['tb']:g} s" if osc else f"constant speed {r['v']} m/s"),
            pert_type="oscillating_speed" if osc else "none", pert_value=(r["tf"] + r["tb"]) if osc else 0, pert_unit="s" if osc else "",
            feedback="normal", excluded_by_authors=False,
            recording=tr, samples_file=fn, signal_kind="3D position of a sacral marker",
            channels="X;Y;Z", channel_units="mm;mm;mm", sample_rate_hz=100, n_samples=len(r["rows"]), duration_s=t_last, events="",
            processing="none stated; NaN (missing) samples are left empty",
            trial_info=info))
D = pd.DataFrame(rows_out)

part = pd.DataFrame([dict(person=str(s), height_cm=demo[s]["height"], mass_kg=demo[s]["mass"], leg_length_m=demo[s]["leg"],
                          sitting_metabolic_rate_wkg=rest[s][0], standing_metabolic_rate_wkg=rest[s][1],
                          **{f"steady_walking_metabolic_rate_{str(v).replace('.', 'p')}_mps_wkg": steady[s][i] for i, v in enumerate(SPEEDS)})
                     for s in range(1, 17)])
pcols = [("height_cm", "number", "cm", "Height."), ("mass_kg", "number", "kg", "Body mass."),
         ("leg_length_m", "number", "m", "Leg length."),
         ("sitting_metabolic_rate_wkg", "number", "W/kg", "Resting metabolic rate sitting."),
         ("standing_metabolic_rate_wkg", "number", "W/kg", "Resting metabolic rate standing.")] + [
    (f"steady_walking_metabolic_rate_{str(v).replace('.', 'p')}_mps_wkg", "number", "W/kg",
     f"Metabolic rate of steady treadmill walking at {v} m/s; empty when the participant did not walk at that speed.") for v in SPEEDS]

# ---------- Expt 2 preferred speeds (extra file)
P = (RAW / "Dat_PreferredWalkingSpeeds.txt").read_text(errors="replace").splitlines()
sect, sub = "average_speed", None
pref = []
props = {}
for i, l in enumerate(P):
    if l.startswith("Steady state speeds for 2 m"):
        sect = "steady_speed_middle_1.4m"
    elif l.startswith("Steady state walking speed obtained"):
        sect = "steady_speed_middle_0.75m"
    elif l.startswith("Subject properties"):
        sect = "props"
    if sect != "props":
        if (m := re.match(r"Subject (\d+)\s*$", l)):
            sub = int(m[1])
        elif sub and (m := re.match(r"^([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\s*$", l)):
            for t in (1, 2, 3):
                pref.append((sub, sect, float(m[1]), t, float(m[1 + t])))
    elif (m := re.match(r"^(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", l)):
        props[int(m[1])] = (int(m[2]), float(m[3]), float(m[4]))
assert len(props) == 10 and len({p[0] for p in pref}) == 10
pl = ["preferred_speed_participant,age_years,height_cm,mass_kg,measure,distance_m,trial,speed_mps"]
for sub, sect, dist, t, v in pref:
    a = props[sub]
    pl.append(f"PS{sub:02d},{a[0]},{a[1]:g},{a[2]:g},{sect},{dist:g},{t},{v:g}")
extra.append(("preferred_speeds/preferred_speeds.csv", "\n".join(pl) + "\n"))
extra.append(("preferred_speeds/README.txt",
              "Expt 2 of the paper (preferred walking speeds for ten distances). One row per participant, measure, distance and trial.\n"
              "measure: average_speed = mean speed over the whole bout from start to stop (distances 0.5-89 m); steady_speed_middle_1.4m and "
              "steady_speed_middle_0.75m = mean speed over the middle 1.4 m or 0.75 m of the trial (2-14 m). Speeds are m/s as the authors give them.\n"
              "The 10 participants (PS01-PS10) are numbered as in the authors' file; the deposit does not say how they relate to the 16 participants of Expt 1.\n"))

# ---------- checks
c = stdlib.Checks()
ages = np.array([demo[s]["age"] for s in range(1, 17)], float)
hts = np.array([demo[s]["height"] for s in range(1, 17)])
c.published("participants (Expt 1)", D.person.nunique(), 16, 0, "Methods (a)")
c.published("males", sum(1 for s in demo if demo[s]["sex"] == "M"), 12, 0, "Methods (a)")
c.published("females", sum(1 for s in demo if demo[s]["sex"] == "F"), 4, 0, "Methods (a)")
c.published("age mean (years)", ages.mean(), 23.25, 0.005, "Methods (a)")
c.published("age SD (years)", ages.std(ddof=1), 2.1, 0.05, "Methods (a)")
c.published("height mean (cm)", hts.mean(), 177.08, 0.005, "Methods (a)")
c.published("height SD (cm)", hts.std(ddof=1), 7.4, 0.05, "Methods (a)")
mass = np.array([demo[s]["mass"] for s in range(1, 17)])
c.published("mass SD (kg)", mass.std(ddof=1), 12.94, 0.005, "Methods (a)")
o12 = D[(D.pert_type == "oscillating_speed") & D.trial_info.str.contains("treadmill_speed_mps=1.12;")].person.nunique() if False else None
osc = D[D.pert_type == "oscillating_speed"]
s112 = set(osc[osc.trial_info.str.startswith("treadmill_speed_mps=1.12;")].subject)
s156 = set(osc[osc.trial_info.str.startswith("treadmill_speed_mps=1.56;")].subject)
c.published("subjects with oscillating trials at both speeds", len(s112 & s156), 10, 0, "Methods (a)")
c.published("subjects with oscillating trials at 1.12 m/s only", len(s112 - s156), 4, 0, "Methods (a)")
c.published("subjects with oscillating trials at 1.56 m/s only", len(s156 - s112), 2, 0, "Methods (a)")
c.published("participants in preferred-speed experiment", len({p[0] for p in pref}), 10, 0, "Methods (b)")
c.published("distances in preferred-speed experiment", len({p[2] for p in pref if p[1] == "average_speed"}), 10, 0, "Methods (b)")
bad = [(r.subject, r.trial, r.n_samples, r.duration_s) for r in D.itertuples() if abs(r.duration_s - r.n_samples * 0.01) > 1.0]
print("time gaps:", bad)
c.internal("time runs at 10 ms per sample (exceptions listed in README)", len(bad) <= 3, str(bad))
c.internal("rows = recordings with data", len(D) == sum(len(v) for v in rec.values()), str(len(D)))
c.internal("all samples kept", n_samp == D.n_samples.sum())

meta = dict(
    paper=dict(authors="Seethapathi N, Srinivasan M", year=2015,
               title="The metabolic cost of changing walking speeds is significant, implies lower optimal speeds for shorter distances, and increases daily energy estimates",
               venue="Biol Lett 11(9):20150486", doi="10.1098/rsbl.2015.0486"),
    lab="Srinivasan (Ohio State University)", data_url="https://doi.org/10.5061/dryad.15v26",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Read from the Zenodo copy of the Dryad deposit. Each subject's block in a trial file became one samples CSV (NaN samples left empty); "
             "the subject properties, resting and steady-walking metabolic rates went to participants.csv, the oscillating-trial metabolic increments and "
             "the trial's treadmill speed to trial_info, and the preferred-speed experiment to preferred_speeds/preferred_speeds.csv. "
             "Subjects who did not take part in a trial have no recording for it."),
    how=["One row is one subject's sacral-marker recording in one trial of Expt 1 (treadmill walking at a constant speed, or oscillating forwards and backwards on "
         "the belt to the tones). Samples are 100 Hz: time_s, then X, Y, Z of the sacral marker in mm (the deposit gives no units or axes; mm and the axis roles are "
         "inferred from the values: Y changes by about half a metre in oscillating trials, so it is probably the fore-aft axis, Z varies by about 3 cm and is probably vertical, and X is probably lateral).",
         "Trial order: trial numbers follow the file listing (constant trials, then oscillating trials 1-6), not the run order, which the deposit does not give; the paper says the speed order was random. "
         "The two constant-speed files are named Trial1 (1.12 m/s) and Trial2 (1.56 m/s) but their headers say 'Constant Speed Trial 7' and 'Trial 8'; the block column keeps the header text.",
         "trial_info holds the treadmill speed (m/s), for oscillating trials T_fwd and T_bck (s) and the authors' delta metabolic rate (oscillating minus steady walking at the same mean speed, W/kg), "
         "and the steady-walking metabolic rate at the same speed (W/kg). pert_value is T_fwd + T_bck in seconds.",
         "Expt 2 (preferred speeds by distance, 10 participants) is in preferred_speeds/preferred_speeds.csv."],
    differences=["The deposit's mass column averages 76.01 kg, the paper prints 75.99 kg; the paper's value is not checked.",
                 "One recording (constant 1.12 m/s trial of the participant coded S0006) has a time column that runs 2 s past its sample count x 10 ms, and about 6900 samples with missing positions; both are kept as shared."],
    unit_note="sacral-marker recording of one subject in one trial",
)
stdlib.release(ID, "timeseries", D, meta, c, participants=part, participant_columns=pcols, extra_files=extra)
import shutil
shutil.rmtree(TMP)
