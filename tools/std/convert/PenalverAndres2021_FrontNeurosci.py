"""Penalver-Andres, Buetler, Koenig, Muri & Marchal-Crespo (2021), Front Neurosci 15:755721: 36 healthy adults steered a virtual boat
with a joystick to align it with incoming waves (Horizon Task, HT, and Obstacle Task, OT), in three groups that got no, implicit or
explicit task instructions. One row per game recording (baseline and retention runs); the 50-60 Hz game log is a CSV in the ZIP.
Raw files (Zenodo 10.5281/zenodo.5111830, CC BY 4.0): HT_Kinematic.zip, OT_Kinematic.zip, DemoGraphicsZenodo.xlsx; 'Dataset description.docx'
was read as text. The EEG zips (49 GB) are not used.
Run from the site folder: python3 -I tools/std/convert/PenalverAndres2021_FrontNeurosci.py"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "PenalverAndres2021_FrontNeurosci"
raw = stdlib.raw_dir(DS)
TMP = raw.parent / (DS + "_samples_tmp")
shutil.rmtree(TMP, ignore_errors=True)
(TMP / "samples").mkdir(parents=True)
ck = stdlib.Checks()

demo = pd.read_excel(raw / "DemoGraphicsZenodo.xlsx", sheet_name=0)
demo["n"] = demo.ID.str[1:].astype(int)
demo = demo.set_index("n")
GROUP = {"i": "implicit instructions", "e": "explicit instructions", "ei": "implicit and explicit instructions"}
TOKEN = {"imp": "i", "exp": "e", "impExp": "ei"}
BLOCK = {0: ("HT", "baseline"), 1: ("HT", "baseline"), 2: ("OT", "baseline"), 3: ("OT", "baseline"),
         7: ("HT", "retention"), 8: ("HT", "retention"), 9: ("OT", "retention"), 10: ("OT", "retention")}
TASKNAME = {"HT": "Horizon Task", "OT": "Obstacle Task"}

# (file column, unit) for every channel kept; the first column Time.time becomes time_s
CH = [("time[ms].Ellapsed", "s"), ("time[ms].Time.realTimeSinceStartup", "s"),
      ("tiny (1).pos.x", "m.u."), ("tiny (1).pos.y", "m.u."), ("tiny (1).pos.z", "m.u."),
      ("tiny (1).rot.w", "as in file"), ("tiny (1).rot.x", "as in file"), ("tiny (1).rot.y", "as in file"), ("tiny (1).rot.z", "as in file"),
      ("transformCamera.pos.x", "m.u."), ("transformCamera.pos.y", "m.u."), ("transformCamera.pos.z", "m.u."),
      ("transformCamera.rot.w", "as in file"), ("transformCamera.rot.x", "as in file"), ("transformCamera.rot.y", "as in file"),
      ("transformCamera.rot.z", "as in file"), ("WaveAngle", "deg"), ("BoatAngle", "deg"), ("DifferenceAngle", "as in file"),
      ("WaveIn0or1", "as in file"), ("ParallelPortStatusMark", "as in file"), ("infoEEG", "as in file"),
      ("buoyAppeared.pos.x", "m.u."), ("buoyAppeared.pos.y", "m.u."), ("buoyAppeared.pos.z", "as in file"),
      ("AngleJoystick", "as in file"), ("AngleFrontJoystick", "as in file"), ("AngleRightJoystick", "as in file"),
      ("typeBuoy", "as in file"), ("buoyAppeared", "as in file"), ("buoyCorrect", "as in file")]

recs = {}
for zf in ("HT_Kinematic.zip", "OT_Kinematic.zip"):
    z = zipfile.ZipFile(raw / zf)
    for i in z.infolist():
        m = re.fullmatch(r".*/participant_(\d+)_-?\d+/movements_\d+_(\w+?)EEG_boat_(Horizon|Obstacles)_wave\w+?_[ge]_(imp|exp|impExp)_(\d+)\.csv", i.filename)
        if not m:
            assert i.filename.endswith("/"), i.filename
            continue
        pn, _, task, tok, blk = int(m[1]), m[2], m[3], m[4], int(m[5])
        recs[(pn, blk)] = (z, i, tok, "HT" if task == "Horizon" else "OT")
ck.internal("288 recordings = 36 participants x 8 blocks", len(recs) == 288 and len({k[0] for k in recs}) == 36)
ck.internal("participant numbers match the demographics sheet", {k[0] for k in recs} == set(demo.index))
ck.internal("task of each block follows the deposit's block scheme",
            all(BLOCK[b][0] == v[3] for (p, b), v in recs.items()))
tok_by_p = {}
for (p, b), v in recs.items():
    tok_by_p.setdefault(p, set()).add(v[2])
mism = sorted(p for p, t in tok_by_p.items() if len(t) != 1 or TOKEN[next(iter(t))] != demo.loc[p, "group"])
ck.internal("group in file names is the group of the demographics sheet for all but one participant (whose files say impExp, sheet says i; the sheet, which the deposit names as the group source, is used)",
            len(mism) == 1, str(len(mism)))

rows = []
dts = []
for pn in sorted({k[0] for k in recs}):
    for n, blk in enumerate(sorted(b for (p, b) in recs if p == pn)):
        z, info, tok, task = recs[(pn, blk)]
        d = pd.read_csv(z.open(info), index_col=0)           # the deposit's rows have one more field than its header: the first is a row index
        assert list(d.columns[:3]) == ["time[ms].Ellapsed", "time[ms].Time.time", "time[ms].Time.realTimeSinceStartup"] and d.columns[-1] == "signalStable"
        assert d.signalStable.isna().all() and d.iloc[:, :-1].notna().all().all(), (pn, blk)
        t = d["time[ms].Time.time"].to_numpy()
        assert (np.diff(t) > 0).all()
        dts.append(np.diff(t))
        code = f"S{len(rows) // 8 + 1:02d}"
        fn = f"samples/{code}_block{blk:02d}.csv"
        out = pd.DataFrame({"time_s": t})
        for c, _ in CH:
            out[c] = d[c].to_numpy()
        out.to_csv(TMP / fn, index=False, float_format="%.8g", lineterminator="\r\n")
        rows.append(dict(
            dataset=DS, study=DS, experiment=DS + "_Exp1", group=GROUP[demo.loc[pn, "group"]], subject=f"S{pn:02d}",
            subject_excluded_by_authors=False, age=int(demo.loc[pn, "age"]), sex={"W": "F", "M": "M"}[demo.loc[pn, "gender"]],
            population="healthy", setting="lab",
            device="Joystick (rotation about its vertical axis) steering a boat in Unity, 256-channel EEG cap, head-mounted display",
            session=1, block=str(blk), trial=n + 1, phase="baseline" if BLOCK[blk][1] == "baseline" else "test",
            time_s=0, task="tracking", unit="recording", condition=f"{TASKNAME[task]}, {BLOCK[blk][1]}",
            pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
            recording=n + 1, samples_file=fn, signal_kind="boat, camera and wave state of the sailing game and joystick angles",
            channels=";".join(["time_s"] + [c for c, _ in CH][:0] + [c for c, _ in CH]) if False else ";".join(c for c, _ in CH),
            channel_units=";".join(u for _, u in CH), sample_rate_hz="", n_samples=len(t), duration_s=float(t[-1] - t[0]),
            events="", processing="none beyond the authors' own logging (game log; the paper states 50 Hz after linear interpolation)",
            trial_info=f"{TASKNAME[task]}; block {blk} of the authors' scheme ({BLOCK[blk][1]})"))
t = pd.DataFrame(rows)
for r in rows:
    r["time_s"] = ""
t = pd.DataFrame(rows)
per = t.drop_duplicates("subject")
gm = np.concatenate(dts)
ck.published("participants", len(per), 36, 0, "Methods, Participants (36 healthy volunteers)")
ck.published("participants per instruction group", per.group.value_counts().min(), 12, 0, "Limitations (12 participants per group)")
ck.published("women", int((per.sex == "F").sum()), 14, 0, "Methods, Participants (14 women)")
ck.published("youngest age (years)", per.age.min(), 20, 0, "Methods, Participants (age 20-59 years)")
ck.published("oldest age (years)", per.age.max(), 59, 0, "Methods, Participants (age 20-59 years)")
ck.internal("each participant has 4 Horizon and 4 Obstacle recordings", bool(t.groupby("subject").size().eq(8).all()))
ck.internal("typical sampling interval 15-20 ms", 0.015 < float(np.median(gm)) < 0.02, f"median {np.median(gm) * 1000:.1f} ms, max {gm.max():.2f} s")
meta = dict(
    paper=dict(authors="Penalver-Andres J, Buetler KA, Koenig T, Muri RM, Marchal-Crespo L", year=2021,
               title="Providing Task Instructions During Motor Training Enhances Performance and Modulates Attentional Brain Networks",
               venue="Front Neurosci 15:755721", doi="10.3389/fnins.2021.755721"),
    lab="Marchal-Crespo (University of Bern)", data_url="https://zenodo.org/records/5111830",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Kept the 288 behavioural game logs (HT_Kinematic and OT_Kinematic zips) as CSV files with all 31 non-empty columns, writing "
             "the game clock Time.time as time_s; dropped the empty column signalStable and the row index; added age, sex and instruction "
             "group from DemoGraphicsZenodo.xlsx; participant numbers replaced by codes. The EEG recordings, handedness questionnaire and "
             "date stamps in folder names are not included."),
    how=["One row is one game recording: a baseline run (blocks 0 and 1 Horizon Task, 2 and 3 Obstacle Task) or a retention run (blocks 7 "
         "and 8 Horizon, 9 and 10 Obstacle) of one participant, sorted by block number; the training phase between them is not in the "
         "deposit. In the sailing game the participant rotated a joystick to steer a boat to align it with incoming waves while moving to a "
         "finish line (Horizon Task) or also avoiding obstacles (Obstacle Task). Each CSV in samples/ holds the game log with columns named as "
         "in the deposit; time_s is the game clock in seconds, positions are in the game's maritime units (m.u.) and angles in degrees.",
         "The sampling is irregular (median interval about 17 ms), so sample_rate_hz is empty. Several columns do not match the deposit's "
         "description: DifferenceAngle is described as the alignment error in degrees but only takes values 0 and 1, WaveIn0or1 is described "
         "as binary but holds values from about 20 to 200, and the 'rot' columns of the boat are not unit-quaternion components. They are "
         "released exactly as the authors shared them and should not be interpreted beyond the deposit's names. WaveAngle and BoatAngle "
         "are mirrored in sign when the boat is aligned (the first sample of a file has equal and opposite values), so the alignment error "
         "is not their difference as the paper's formula reads.",
         "Instruction groups: implicit, explicit, or both (deposit labels i, e, ei), taken from the demographics sheet. Age is the age the deposit lists at participation."],
    unit_note="game recording (one block of the sailing task)",
    differences=["For one participant the file names carry the both-instructions label (impExp) while the demographics sheet says implicit; the sheet is used (it gives 12 participants per group, as the paper states; the file names would give 13 and 11). The age in the deposit has a mean of 27.6 (SD 6.60), the paper reports 27.9 (SD 6.64); the paper's mean and SD are therefore not "
                 "used as checks. Sex 22 men and 14 women, and ages 20-59, match the paper. The paper reports kinematics at 50 Hz after "
                 "linear interpolation, the deposit's logs come at about 59 Hz (irregular)."])
shutil.rmtree(TMP / "none", ignore_errors=True)
extra = [(str(p.relative_to(TMP)).replace("\\", "/"), str(p)) for p in sorted((TMP / "samples").glob("*.csv"))]
stdlib.release(DS, "timeseries", t.reindex(columns=[c[1] for c in stdlib.columns_for("timeseries") if c[1] != "person"]), meta, ck, extra_files=extra)
shutil.rmtree(TMP, ignore_errors=True)
for c in ck.items:
    print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""))
