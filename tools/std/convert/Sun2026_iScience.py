"""Sun (2026), iScience 29(11):117915: basketball shooting by 156 players over nine sessions (baseline, six training sessions,
post-test, 14-day retention), one row per shot (168,480), in the 'game' module.

Run from the site folder: python3 -I tools/std/convert/Sun2026_iScience.py
Raw files (untrusted data, read only) in raw_extra/Sun2026_iScience/, from https://doi.org/10.5281/zenodo.22828547 (CC BY 4.0):
  Basketball.xlsx            sheets Participants_156, Sessions_1404, Blocks_16848, Shots_168480 (read with openpyxl, read-only)
  MSEEM_analysis_code.zip    the authors' code, listed and read as text only, never run (its inputs are not the workbook)
Not carried over: the per-shot pose keypoints (pixels) and confidences, IMU and ball-trajectory values, the video and IMU file
names, the release frame numbers, and the Sessions-sheet calibration/drift columns; the block rubric totals of the two raters
are repeated on the shots of their block.
"""
import json
import sys
from pathlib import Path

import numpy as np
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Sun2026_iScience"
RAW = stdlib.raw_dir(ID)
SRC = "Method details and Table 3 (iScience 29(11):117915)"
checks = stdlib.Checks()
differences = []
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
checks.internal("Basketball.xlsx matches the downloaded hash", stdlib.sha256(RAW / "Basketball.xlsx") == src["Basketball.xlsx"]["sha256"])

wb = openpyxl.load_workbook(RAW / "Basketball.xlsx", read_only=True, data_only=True)


def table(prefix):
    ws = next(w for w in wb.worksheets if w.title.startswith(prefix))
    it = ws.iter_rows(values_only=True)
    hdr = [str(h) for h in next(it)]
    rows = [r for r in it if r and r[0] is not None]
    return hdr, rows


def col(hdr, name):
    if name in hdr:
        return hdr.index(name)
    hit = [i for i, h in enumerate(hdr) if h.startswith(name)]
    assert len(hit) == 1, (name, hit)
    return hit[0]


ph, prow = table("Participants")
sh, srow = table("Sessions")
bh, brow = table("Blocks")
th, trow = table("Shots")
print("sheets:", [w.title for w in wb.worksheets], "| rows", len(prow), len(srow), len(brow), len(trow))

# participants
pc = {k: col(ph, k) for k in ("Participant_ID", "Study_Condition", "Age_Years", "Sex", "Playing_Level", "Basketball_Experience_",
                              "Dominant_Shooting_Hand", "Baseline_Familiarizati")}
P = {r[pc["Participant_ID"]]: {k: r[i] for k, i in pc.items()} for r in prow}
# sessions
sc = {k: col(sh, k) for k in ("Session_ID", "Session_Number", "Frontal_Camera", "Camera_FPS", "Hand_IMU", "IMU_Sample_Rate_Hz")}
S = {r[sc["Session_ID"]]: r[sc["Session_Number"]] for r in srow}
uniq = {k: {r[sc[k]] for r in srow} for k in ("Frontal_Camera", "Camera_FPS", "Hand_IMU", "IMU_Sample_Rate_Hz")}
checks.internal("one camera model, frame rate, IMU model and IMU rate in all sessions", all(len(v) == 1 for v in uniq.values()), str(uniq))
DEVICE = (f"{next(iter(uniq['Frontal_Camera']))} frontal and sagittal cameras at {next(iter(uniq['Camera_FPS']))} fps; "
          f"{next(iter(uniq['Hand_IMU']))} hand and forearm IMUs at {next(iter(uniq['IMU_Sample_Rate_Hz']))} Hz; basketball court")
# blocks
bc = {k: col(bh, k) for k in ("Block_ID", "Coach1_Total_0to100", "Coach2_Total_0to100", "Study_Stage")}
B = {r[bc["Block_ID"]]: (r[bc["Coach1_Total_0to100"]], r[bc["Coach2_Total_0to100"]], r[bc["Study_Stage"]]) for r in brow}
# shots
tc = {k: col(th, k) for k in ("Participant_ID", "Session_ID", "Block_ID", "Study_Condition", "Study_Stage", "Block_Number",
                              "Shot_Number_in_Block", "Shot_Type", "Environment", "Release_Time_From_Sess", "Release_Timestamp",
                              "Shot_Made")}
checks.internal("sheet sizes: 156 participants, 1,404 sessions, 16,848 blocks, 168,480 shots",
                (len(prow), len(srow), len(brow), len(trow)) == (156, 1404, 16848, 168480),
                f"{len(prow)}, {len(srow)}, {len(brow)}, {len(trow)}")
cond_order = list(dict.fromkeys(r[pc["Study_Condition"]] for r in prow))
checks.internal("two study conditions", len(cond_order) == 2, str(cond_order))
SEX = {"male": "M", "female": "F"}
PHASE = lambda st: ("baseline" if st.startswith("Baseline") else "learning" if st.startswith("Training") else  # noqa: E731
                    "test" if st.startswith(("Post", "Retention")) else "other")

rows = []
count = {}
for r in trow:
    pid = r[tc["Participant_ID"]]
    p = P[pid]
    count[pid] = count.get(pid, 0) + 1
    ts = str(r[tc["Release_Timestamp"]])
    b1, b2, _ = B[r[tc["Block_ID"]]]
    stage = str(r[tc["Study_Stage"]])
    made = r[tc["Shot_Made"]]
    rows.append(dict(
        dataset=ID, study=ID, experiment=f"{ID}_Main", group=r[tc["Study_Condition"]], subject=pid, subject_excluded_by_authors=False,
        age=p["Age_Years"], sex=SEX.get(str(p["Sex"]).lower(), "other"), handedness=str(p["Dominant_Shooting_Hand"]).lower(),
        population=f"healthy, {p['Playing_Level']}", setting="lab", device=DEVICE, session=S[r[tc["Session_ID"]]],
        block=f"{stage} block {r[tc['Block_Number']]}", trial=count[pid], phase=PHASE(stage), time_s=r[tc["Release_Time_From_Sess"]],
        task="sports and games", unit="trial", condition=r[tc["Shot_Type"]], pert_type="none", pert_value=0, pert_unit="",
        feedback="normal", excluded_by_authors=False, game="Basketball shooting", attempt=r[tc["Shot_Number_in_Block"]],
        success=bool(int(made)), environment=r[tc["Environment"]], date=ts[:10], hour=int(ts[11:13]),
        rubric_total_coach1=b1, rubric_total_coach2=b2))
checks.internal("every participant has 1,080 shots (9 sessions x 12 blocks x 10 shots)", set(count.values()) == {1080}, str(set(count.values())))
checks.internal("shots are 0/1 outcomes", {r["success"] for r in rows} <= {True, False})

# ---------------------------------------------------------------- published numbers
passed = []


missed, total = {}, {}
GROUPS = {"shot": "Table 3 baseline shot-made percentages (mean and SD for each of four shot types and two conditions)",
          "rubric": "Table 3 baseline rubric totals (mean and SD for two conditions)"}


def chk(what, computed, reported, tol, source, group=None):
    ok = abs(computed - reported) <= tol
    if group:
        total[group] = total.get(group, 0) + 1
    if ok:
        checks.published(what, computed, reported, tol, source)
        passed.append(what)
    elif group:
        missed.setdefault(group, []).append((computed - reported, what))
    else:
        differences.append(f"{what}: the paper reports {reported:g}, the shared rows give {computed:.4g} ({source}).")
    print(f"  {'ok ' if ok else 'DIFF'} {what}: {computed:.4g} vs {reported:g}")


g = {c: [p for p in P.values() if p["Study_Condition"] == c] for c in cond_order}
chk("participants", len(P), 156, 0, f"Participants, {SRC}")
for c, n in zip(cond_order, (78, 78)):
    chk(f"participants in condition {c}", len(g[c]), n, 0, f"Participants, {SRC}")
stats_ref = {  # (condition index) -> reported age mean, sd, males, competitive, years mean, sd, right hand, familiarization mean, sd
    0: (21.4, 2.3, 52, 33, 5.2, 2.1, 69, 18.9, 3.8), 1: (21.6, 2.4, 50, 35, 5.0, 2.2, 68, 19.1, 3.6)}
for i, c in enumerate(cond_order):
    ref, ps, tag = stats_ref[i], g[c], f"[{c}]"
    age = np.array([p["Age_Years"] for p in ps], float)
    yrs = np.array([p["Basketball_Experience_"] for p in ps], float)
    fam = np.array([p["Baseline_Familiarizati"] for p in ps], float)
    chk(f"age mean {tag}", age.mean(), ref[0], 0.06, f"Table 3, {SRC}")
    chk(f"age SD {tag}", age.std(ddof=1), ref[1], 0.06, f"Table 3, {SRC}")
    chk(f"male players {tag}", sum(str(p["Sex"]).lower() == "male" for p in ps), ref[2], 0, f"Table 3, {SRC}")
    chk(f"competitive players {tag}", sum(str(p["Playing_Level"]).lower().startswith("competitive") for p in ps), ref[3], 0, f"Table 3, {SRC}")
    chk(f"years of playing experience mean {tag}", yrs.mean(), ref[4], 0.06, f"Table 3, {SRC}")
    chk(f"years of playing experience SD {tag}", yrs.std(ddof=1), ref[5], 0.06, f"Table 3, {SRC}")
    chk(f"right-hand shooters {tag}", sum(str(p["Dominant_Shooting_Hand"]).lower() == "right" for p in ps), ref[6], 0, f"Table 3, {SRC}")
    chk(f"baseline familiarization minutes mean {tag}", fam.mean(), ref[7], 0.06, f"Table 3, {SRC}")
    chk(f"baseline familiarization minutes SD {tag}", fam.std(ddof=1), ref[8], 0.06, f"Table 3, {SRC}")
base = [r for r in rows if r["phase"] == "baseline"]
chk("scheduled shots at baseline", len(base), 18720, 0, f"Shot task protocol, {SRC}")
TYPES = ("Free throw", "Catch-and-shoot", "Off-dribble", "Lightly contested")
REF = {0: (63.5, 10.2, 51.2, 11.8, 44.7, 10.6, 38.4, 9.9), 1: (62.8, 10.5, 50.1, 12.1, 45.3, 10.4, 37.6, 10.1)}
RUB = {0: (62.8, 8.9), 1: (62.1, 9.1)}
for i, c in enumerate(cond_order):
    for j, st in enumerate(TYPES):
        per = {}
        for r in base:
            if r["group"] == c and r["condition"] == st:
                per.setdefault(r["subject"], []).append(r["success"])
        v = np.array([np.mean(x) * 100 for x in per.values()])
        chk(f"baseline shot-made % {st} mean [{c}]", v.mean(), REF[i][2 * j], 0.06, f"Results, Table 3, {SRC}", group="shot")
        chk(f"baseline shot-made % {st} SD [{c}]", v.std(ddof=1), REF[i][2 * j + 1], 0.06, f"Results, Table 3, {SRC}", group="shot")
    per = {}
    seen = set()
    for r in base:
        if r["group"] == c:
            key = (r["subject"], r["block"])
            if key not in seen:
                seen.add(key)
                per.setdefault(r["subject"], []).append((r["rubric_total_coach1"] + r["rubric_total_coach2"]) / 2)
    v = np.array([np.mean(x) for x in per.values()])
    chk(f"baseline rubric total mean [{c}]", v.mean(), RUB[i][0], 0.15, f"Results, Table 3, {SRC}", group="rubric")
    chk(f"baseline rubric total SD [{c}]", v.std(ddof=1), RUB[i][1], 0.15, f"Results, Table 3, {SRC}", group="rubric")
HOW = {"shot": "the share of made shots over each player's 30 baseline shots of the type, averaged over the players of the condition",
       "rubric": "the mean of the two raters' block totals over a player's 12 baseline blocks, averaged over the players of the condition"}
for gk, lst in missed.items():
    d = np.array([x[0] for x in lst])
    differences.append(f"{GROUPS[gk]}: {len(lst)} of {total[gk]} values were not reproduced (the shared rows differ from the reported value by "
                       f"{np.abs(d).min():.2g} to {np.abs(d).max():.2g} points, mean {d.mean():+.2g}); the text does not say how they were "
                       f"computed, and {HOW[gk]} was tried here.")
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']}")
assert passed, "no published number reproduced"

differences += [
    "Not carried over from the Shots sheet: the pose keypoints (pixels) and confidences at the release frame, the IMU and ball-trajectory "
    "values, the video and IMU file names and the release frame numbers; the OpenMotor game format has no columns for them.",
    "rubric_total_coach1 and rubric_total_coach2 are the two raters' block totals from the Blocks sheet (their four component scores "
    "are not carried over), repeated on the ten shots of each block.",
]
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Sun P (affiliation not verified)", data_url="https://doi.org/10.5281/zenodo.22828547", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format: participant ids replaced by codes, the four workbook sheets joined into one row per shot "
             "(participant age, sex, playing level and shooting hand from Participants; session number from Sessions; the two raters' "
             "block totals from Blocks), sex written as F/M and handedness in lower case. Pose, IMU and ball-trajectory columns, file "
             "names and frame numbers were not carried over."),
    how=[
        "One row is one basketball shot in the order of the workbook (participant, session, block, shot): success is Shot_Made, condition "
        "the shot type (free throw, catch-and-shoot, off-dribble, lightly contested), attempt the shot's number within its 10-shot block "
        "and trial its number within the participant's 1,080 shots; time_s is the release time from the start of the session.",
        "Group is the study condition (MSEEM feedback or standard coaching); block names follow Study_Stage (Baseline, Training 1-6, "
        "Post-test, Retention) and phase is baseline, learning (training) or test (post-test and 14-day retention).",
        "date and hour are the release time on the authors' clock; handedness is the dominant shooting hand.",
    ],
    unit_note="basketball shot", differences=differences)
stdlib.release(ID, "game", rows, meta, checks)
