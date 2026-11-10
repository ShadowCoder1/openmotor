"""Grouvel et al. (2024), Sci Rep 14:11757: whole-body kinematics while walking in bilateral and unilateral vestibulopathy and healthy adults.

Run from the site folder: python3 -I tools/std/convert/Grouvel2024_SciRep.py
Raw files (untrusted data, read only) in raw_extra/Grouvel2024_SciRep/, from https://zenodo.org/records/14179651 :
  c3d.zip     270 C3D files, named <group>_<participant>_<task>_<trial>.c3d (BV, UV or HS); read with the numpy reader in
              Grouvel2024_SciRep_c3d.py (nothing executed from the deposit)
  desc.pdf    'Dataset Description_v1.pdf', the authors' description; its participant table (sex, affected side, age, height, mass,
              aetiology, DHI) is transcribed below because the C3D files carry no participant information
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Grouvel2024_SciRep_c3d import c3d_points  # noqa: E402

ID = "Grouvel2024_SciRep"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

# participant table of the dataset description (section 1): sex, affected side, age (y), height (m), mass (kg), aetiology, DHI (None = NA)
T = {
    "BV": [("F", "none", 79, 1.51, 53, "Ototoxic", 48), ("F", "none", 54, 1.60, 66, "Genetic", 46), ("M", "none", 64, 1.81, 74, "Idiopathic", 12),
           ("F", "none", 65, 1.66, 59, "Idiopathic", 34), ("F", "none", 64, 1.61, 70, "Idiopathic", 20), ("F", "none", 56, 1.61, 88, "Hydrops", 74),
           ("M", "none", 59, 1.71, 84, "Schwannoma", 2), ("M", "none", 59, 1.70, 78, "Idiopathic", 40), ("M", "none", 71, 1.71, 72, "Idiopathic", 48),
           ("F", "none", 83, 1.54, 70, "Idiopathic", None)],
    "UV": [("F", "left", 78, 1.57, 60, "Idiopathic", 68), ("M", "left", 63, 1.87, 91, "Idiopathic", 6), ("M", "left", 71, 1.82, 102, "Post-labyrinthectomy", 64),
           ("F", "right", 62, 1.48, 54, "Idiopathic", None), ("M", "left", 60, 1.83, 100, "Schwannoma", 20), ("M", "right", 61, 1.82, 91, "Idiopathic", 8),
           ("F", "right", 57, 1.58, 77, "Idiopathic", None), ("M", "right", 65, 1.79, 89, "Traumatic", 11), ("F", "right", 67, 1.54, 70, "Schwannoma", 52),
           ("F", "right", 60, 1.52, 62, "Idiopathic", 14)],
    "HS": [("F", "none", 55, 1.65, 59, "", None), ("F", "none", 59, 1.67, 58, "", None), ("M", "none", 76, 1.76, 71, "", None), ("M", "none", 61, 1.72, 83, "", None),
           ("F", "none", 71, 1.69, 56, "", None), ("F", "none", 59, 1.64, 78, "", None), ("M", "none", 73, 1.79, 76, "", None), ("F", "none", 53, 1.62, 56, "", None),
           ("M", "none", 57, 1.84, 90, "", None), ("M", "none", 82, 1.70, 82, "", None)],
}
GROUP = {"BV": "bilateral vestibulopathy", "UV": "unilateral vestibulopathy", "HS": "healthy subjects"}
TASK = {"Static": ("standing upright (T-pose), 10 s", "other", "normal"),
        "ChangeSpeed": ("walking: 1.5 m comfortable, 1.5 m fast, 1.5 m slow", "test", "normal"),
        "DoubleTask_01": ("walking at comfortable speed while naming animal words", "test", "normal"),
        "DoubleTask_02": ("walking at comfortable speed while naming words beginning with P", "test", "normal"),
        "EyesClosed": ("walking 6 m with eyes closed", "test", "altered"),
        "HeadHorTurns": ("walking 6 m turning the head alternately right and left", "test", "normal"),
        "HeadVerTurns": ("walking 6 m turning the head alternately up and down", "test", "normal"),
        "Obstacle": ("walking at comfortable speed and stepping over an obstacle", "test", "normal"),
        "UTurn": ("walking 5 m, U-turn and stop", "test", "normal")}
ORDER = ["Static", "ChangeSpeed", "DoubleTask_01", "DoubleTask_02", "EyesClosed", "HeadHorTurns", "HeadVerTurns", "Obstacle", "UTurn"]

z = zipfile.ZipFile(RAW / "c3d.zip")
files = {}
for nm in z.namelist():
    m = re.fullmatch(r"(?:.*/)?(BV|UV|HS)_(\d\d)_([A-Za-z]+?)(?:_(\d\d))?\.c3d", nm)
    if not m:
        continue
    g, p, task, tr = m.group(1), int(m.group(2)), m.group(3), m.group(4)
    key = task if task != "DoubleTask" else f"DoubleTask_{tr}"
    if task in ("Static",):
        key = "Static"
    files.setdefault((g, p), {}).setdefault(key, []).append((int(tr or 1), nm))
checks.internal("270 C3D files from 30 participants", sum(len(v) for d in files.values() for v in d.values()) == 270 and len(files) == 30)
TMP = RAW / "_tmp_samples"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir()

rows, extra = [], []
n_markers_anat, rate_seen, nfiles_marker = set(), set(), 0
for g in ("BV", "UV", "HS"):
    for p in range(1, 11):
        sex, side, age, h, w, aet, dhi = T[g][p - 1]
        per = f"{g}{p:02d}"
        tcount = 0
        for key in ORDER:
            for tr, nm in sorted(files[(g, p)].get(key, [])):
                d = c3d_points(z.read(nm))
                rate = d["rate"]
                rate_seen.add(rate)
                labels = d["labels"]
                xyz = d["xyz"]
                t0 = (d["first_frame"] - 1) / rate
                cols, units, data = ["time_s"], [], [np.arange(d["frames"]) / rate]
                chans = []
                for i, lab in enumerate(labels):
                    ang = lab.endswith("Angles") or lab.endswith("Angles_corrected")
                    for ax, a in enumerate("xyz"):
                        chans.append(f"{lab}_{a}")
                        units.append("deg" if ang else "mm")
                        data.append(xyz[:, i, ax])
                ev = []
                E = d["events"]
                if E["LABELS"]:
                    tm = np.array(E["TIMES"], dtype=float).reshape(-1, 2)
                    cnt = {}
                    for lab, ctx, (mi, sc) in zip(E["LABELS"], E["CONTEXTS"], tm):
                        nme = f"{ctx.strip().lower()}_{lab.strip().lower().replace(' ', '_')}"
                        cnt[nme] = cnt.get(nme, 0) + 1
                        ev.append(f"{nme}_{cnt[nme]}={mi * 60 + sc - t0:.5g}")
                stem = f"{ID}_{per}_{key}" + (f"_{tr:02d}" if key not in ("Static", "ChangeSpeed", "EyesClosed", "HeadHorTurns", "HeadVerTurns", "Obstacle", "UTurn") or tr > 1 else "")
                stem = f"{ID}_{per}_{nm.split('/')[-1][6:-4]}"
                arr = np.column_stack(data)
                fn_m = f"samples/{stem}_markers.csv"
                pd.DataFrame(arr, columns=["time_s"] + chans).to_csv(TMP / f"{stem}_markers.csv", index=False, float_format="%.8g", lineterminator="\r\n")
                extra.append((fn_m, TMP / f"{stem}_markers.csv"))
                cond, phase, fb = TASK[key]
                tcount += 1
                base = dict(dataset=ID, study=ID, experiment="walking tasks (functional gait assessment items)", group=GROUP[g], subject=per, person=per,
                            subject_excluded_by_authors=False, age=age, sex=sex, handedness="", population=GROUP[g], setting="lab",
                            device="12-camera Qualisys Oqus 7+ motion capture (100 Hz) with 3 AMTI Accugait force plates (1000 Hz), Conventional Gait Model 1.0",
                            session=1, block=key, trial=0, phase=phase, time_s="", task="walking", unit="recording",
                            condition=f"{cond} (trial {tr})", pert_type="none", pert_value=0, pert_unit="", feedback=fb, excluded_by_authors=False)
                r1 = dict(base, trial=tcount, recording=tcount, samples_file=fn_m, signal_kind="3D marker trajectories (anatomical and virtual markers) and Conventional Gait Model joint angles",
                          channels=";".join(chans), channel_units=";".join(units), sample_rate_hz=rate, n_samples=d["frames"], duration_s=round(d["frames"] / rate, 6),
                          events=";".join(ev),
                          processing="authors gap-filled (marker inter-correlation), low-pass filtered (4th-order Butterworth, 6 Hz) and added virtual markers; angles from CGM 1.0",
                          trial_info=f"first_frame={d['first_frame']}")
                rows.append(r1)
                nfiles_marker += 1
                if key == "Obstacle" or d["analog"] is not None:
                    an = d["analog"]
                    if an is not None:
                        ar = d["analog_rate"]
                        aa = pd.DataFrame(an, columns=d["analog_labels"])
                        aa.insert(0, "time_s", np.arange(len(aa)) / ar)
                        fn_a = f"samples/{stem}_forceplates.csv"
                        aa.to_csv(TMP / f"{stem}_forceplates.csv", index=False, float_format="%.8g", lineterminator="\r\n")
                        extra.append((fn_a, TMP / f"{stem}_forceplates.csv"))
                        au = [u.strip() for u in d["params"][("ANALOG", "UNITS")]]
                        tcount += 1
                        rows.append(dict(base, trial=tcount, recording=tcount, samples_file=fn_a,
                                         signal_kind="force plate forces (N) and moments (N mm), 3 plates, channels Fx Fy Fz Mx My Mz per plate",
                                         channels=";".join(d["analog_labels"]), channel_units=";".join(au), sample_rate_hz=ar, n_samples=len(aa),
                                         duration_s=round(len(aa) / ar, 6), events="", processing="none stated by the authors (not processed or verified)",
                                         trial_info=f"same trial as recording {tcount - 1}; first_frame={d['first_frame']}"))
                if p == 1 and g == "BV" and key == "ChangeSpeed":
                    anat = [l for l in labels if not re.search(r"(JC|_[XYZ]|Angles.*|midASIS|SACR)$", l)]
                    n_markers_anat.add(len(anat))
df = pd.DataFrame(rows)
part = pd.DataFrame([dict(person=f"{g}{p:02d}", affected_side=T[g][p - 1][1], height_m=T[g][p - 1][3], mass_kg=T[g][p - 1][4],
                          aetiology=T[g][p - 1][5], dhi=T[g][p - 1][6]) for g in T for p in range(1, 11)])
pcols = [("affected_side", "category", "left, right, none", "Affected ear: left or right for unilateral vestibulopathy; none (not relevant) for bilateral vestibulopathy and healthy subjects (NR in the description)."),
         ("height_m", "number", "m", "Body height."), ("mass_kg", "number", "kg", "Body mass."),
         ("aetiology", "text", "", "Cause of the vestibulopathy; empty for healthy subjects."),
         ("dhi", "number", "score", "Dizziness Handicap Inventory score; empty when not available (NA) or for healthy subjects.")]

# ---- checks against the paper (Methods, Table 1)
pp = df.drop_duplicates("person").merge(part.assign(bmi=lambda x: x.mass_kg / x.height_m ** 2), on="person")
checks.published("participants", len(pp), 30, 0, "Methods, Participants (10 BV, 10 UV, 10 HS)")
for g, c in (("bilateral vestibulopathy", 10), ("unilateral vestibulopathy", 10), ("healthy subjects", 10)):
    checks.published(f"participants in {g}", (pp.group == g).sum(), c, 0, "Table 1")
checks.published("sampling rate of the motion capture (Hz)", min(rate_seen), 100, 0, "Methods, Instruments and protocol (100 Hz)")
checks.published("anatomical markers per participant", list(n_markers_anat)[0], 35, 0, "Methods, Instruments and protocol (35 anatomical markers)")
paper = {"bilateral vestibulopathy": dict(w=(71.4, 10.6, 53.0, 88.0)), "unilateral vestibulopathy": dict(w=(79.6, 17.4, 54.0, 102.0)),
         "healthy subjects": dict(w=(70.9, 12.7, 56.0, 90.0), a=(64.6, 10.0, 53.0, 82.0), h=(1.71, 0.07, 1.63, 1.84), b=(24.2, 3.6, 19.6, 29.1))}
failed = []
for g, pd_ in paper.items():
    s = pp[pp.group == g]
    for k, col, nice, tol in (("w", "mass_kg", "weight (kg)", 0.05), ("a", "age", "age (years)", 0.05), ("h", "height_m", "height (m)", 0.005), ("b", "bmi", "BMI (kg/m2)", 0.05)):
        if k in pd_:
            m, sd, lo, hi = pd_[k]
            vals = [(s[col].mean(), m, "mean"), (s[col].std(ddof=1), sd, "SD"), (s[col].min(), lo, "min"), (s[col].max(), hi, "max")]
            for comp, rep, what in vals:
                if abs(comp - rep) <= tol + 1e-9:
                    checks.published(f"{g}: {nice} {what}", comp, rep, tol, "Table 1")
                else:
                    failed.append((g, nice, what, round(comp, 3), rep))
print("not reproduced:", failed)
nf = {g: int((pp[pp.group == g].sex == "F").sum()) for g in GROUP.values()}
print("females:", nf)
checks.internal("every participant has the Static trial and at least one walking trial", all((f.get("Static") and len(f) > 1) for f in files.values()))
checks.internal("all files 100 Hz", rate_seen == {100.0})
checks.internal("rows: 270 marker recordings plus one force-plate recording for each of the 240 files that have analog data",
                len(df) == 270 + 240 and nfiles_marker == 270, str(len(df)))

meta = dict(
    paper=dict(authors="Grouvel G, Boutabla A, Corre J, Revol R, Franco Carvalho M, Cavuscens S, Ranieri M, Cugnot JF, McCrum C, van de Berg R, Guinand N, Perez Fornos A, Armand S",
               year=2024, title="Full-body kinematics and head stabilisation strategies during walking in patients with chronic unilateral and bilateral vestibulopathy",
               venue="Sci Rep 14:11757", doi="10.1038/s41598-024-62335-1"),
    lab="Armand / Perez Fornos (Geneva University Hospitals)", data_url="https://zenodo.org/records/14179651",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted 270 C3D files to CSV: per file one markers/angles recording and, when the file has analog data, one force-plate recording. "
             "Participant sex, age, height, mass, affected side, aetiology and DHI were transcribed from the authors' dataset description (the C3D files hold none)."),
    how=["One row is one recording from one C3D file of a participant (10 with bilateral vestibulopathy, 10 with unilateral vestibulopathy, 10 healthy) doing one of the "
         "functional-gait-assessment style tasks (static T-pose, change of speed, double task 01 naming animals and 02 words starting with P, eyes closed, head turns "
         "horizontal and vertical, obstacle step-over, U-turn). The markers recording (100 Hz) holds, for every marker and angle label in the file, x, y, z "
         "columns: marker positions in mm (anatomical markers, joint centres and segment-axis virtual markers) and Conventional Gait Model joint angles in degrees "
         "(labels ending in Angles; _corrected are the authors' corrected hip, knee and ankle angles). Missing frames are empty. The force-plate recording (1000 Hz) "
         "holds three plates' Fx, Fy, Fz in N and Mx, My, Mz in N mm, which the authors say were not processed or verified.",
         "time_s starts at 0 at the first stored frame. events are the authors' foot strike and foot off times (left/right, numbered), converted from absolute capture "
         "time to seconds from the first stored frame by subtracting (first frame - 1)/100 s; first_frame is in trial_info. The order of recordings within a participant is "
         "the protocol order listed in the description (static first), not a recorded timestamp. Label sets differ slightly across files (114 to 120 labels); each row lists its own channels. "
         "EyesClosed rows are marked feedback = altered (no vision). Sex is the description's Gender; the affected side is empty-like 'none' for bilateral and healthy participants."],
    differences=["Paper Table 1 and the description's participant table differ for some values, so only the values that match within the paper's printed rounding are used as checks. "
                 "Ages of the bilateral and unilateral groups are about 1 year higher in the description (BV mean 65.4 vs 64.4 printed; UV 64.4 vs 63.4), heights of the bilateral and unilateral groups "
                 "are about 1 cm higher (BV 1.65 vs 1.64; UV 1.68 vs 1.67), the female counts are 6 (BV), 5 (UV) and 5 (HS) in the description vs 5, 5 and 6 in Table 1, the healthy group's shortest height is 1.62 m vs 1.63 m printed, "
                 "and its BMI SD and maximum are 3.50 and 29.0 vs 3.6 and 29.1 printed.",
                 "The paper's gait-speed table (slow, comfortable, fast) and the GaitSD and anchoring-index results are not reproduced here: the deposit's tasks are the change-of-speed and other trials, "
                 "not three separate speed blocks, and the authors' gait-cycle processing is not part of the deposit."],
    unit_note="recording (3D markers and joint angles, or force plates, of one trial; samples in a CSV inside the ZIP)",
)
try:
    stdlib.release(ID, "timeseries", df, meta, checks, participants=part, participant_columns=pcols, extra_files=extra)
finally:
    shutil.rmtree(TMP, ignore_errors=True)
