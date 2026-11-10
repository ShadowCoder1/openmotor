"""Brissenden, Scerbak, Albin & Lee (2023), Movement Disorders: motivational vigor in de novo Parkinson's disease before and
after levodopa (baseline, SDR, LDR, SDR+LDR visits), incentivized joystick task, one row per trial, in the joystick_vigor module.

Run from the site folder: python3 -I tools/std/convert/Brissenden2023_MovDisord.py
Raw files (untrusted data, read only) in raw_extra/Brissenden2023_MovDisord/ from https://osf.io/tac5y/ (component Raw Data):
  t_<id><visit letter>_<date>.csv   trial table of each joystick session (42 sessions; a = baseline, b = SDR, c = LDR, d = SDR+LDR)
  t_redcap_export.csv               the REDCap clinical export (consent, demographics, visit dates, MDS-UPDRS III, finger tapping)
The 84 trajectory CSVs (about 700 MB) are not converted. Task code read as text (joystick_vigor.py): the joystick position is
in normalized screen units (display width = 2 units, 30.43 cm), timeToBound is seconds from the go signal.

Consent. The REDCap column retain_data_icf records, per participant, permission to retain data for future studies; one of the
11 participants answered 0 (and the consent-form note says "subject did not give permission to retain data for future studies").
That participant's rows are not converted. Sex coding (gender_demo 1 = male, 2 = female) is confirmed by the paper's 6 M : 5 F.
The coding of hand_domin_demo is not documented, so handedness is left empty.
"""
import csv
import datetime as dt
import glob
import json
import math
import os
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Brissenden2023_MovDisord"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
CM_PER_UNIT = 30.43 / 2.0
STATE = {"a": "baseline", "b": "sdr", "c": "ldr", "d": "sdr_ldr"}
LABEL = {"baseline": "baseline", "sdr": "SDR", "ldr": "LDR", "sdr_ldr": "SDR+LDR"}

# ---------------------------------------------------------------- REDCap export (clinical, consent, visit dates)
rc = list(csv.DictReader(open(RAW / "t_redcap_export.csv", newline="", encoding="utf-8-sig")))
vis = {(r["study_id"], int(r["redcap_event_name"].split("_")[1])): r for r in rc}
pids = sorted({r["study_id"] for r in rc})
checks.internal("REDCap export: 12 participants consented, 11 reached visit 2 (one withdrew)",
                len(pids) == 12 and sum((p, 2) in vis for p in pids) == 11)
retain = {p: vis[(p, 1)]["retain_data_icf"] for p in pids}
checks.internal("every participant with a retain-data answer: yes (1) or no (0); exactly one no", all(
    retain[p] in ("1", "0") for p in pids if (p, 2) in vis) and sum(retain[p] == "0" for p in pids) == 1)
no_retain = {p for p in pids if retain[p] == "0"}
checks.internal("the one 'no' matches the consent-form note ('did not give permission to retain data for future studies')",
                all("RETAIN DATA" in vis[(p, 1)]["notes_icf"].upper() and "NOT GIVE PERMISSION" in vis[(p, 1)]["notes_icf"].upper()
                    for p in no_retain))

# ---------------------------------------------------------------- trial tables
files = sorted(glob.glob(str(RAW / "t_1*.csv")))
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for f in files + [str(RAW / "t_redcap_export.csv")]:
    fn = os.path.basename(f)
    checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(f) == src[fn]["sha256"])
sessions = []
for f in files:
    b = os.path.basename(f)[2:-4]
    sid, date = b.split("_")
    sessions.append((sid[:3], sid[3], dt.datetime.strptime(date, "%y%m%d").date(), f))
checks.published("joystick sessions (11 participants x 4 visits, minus the 2 first sessions one participant could not do)",
                 len(sessions), 42, 0, "Methods, Participants and Results (one participant could not perform the task in the first two sessions)")
checks.published("participants who did the joystick task", len({s[0] for s in sessions}), 11, 0, "Methods, Participants (n = 11)")
# visit letters follow the visit order: the file date is within 31 days of the REDCap visit date for the matching visit number
near = sum(abs((s[2] - dt.date.fromisoformat(vis[(s[0], "abcd".index(s[1]) + 1)]["visit_date"][:10])).days) <= 31 for s in sessions)
checks.internal("session letters a-d match REDCap visits 1-4 (file date within 31 days of the visit date, all 42)", near == 42,
                f"{near} of 42")


def arr(s):
    return [float(x) for x in s.strip("[]").split()]


def fnum(s):
    return None if s in ("", "nan") else float(s)


all_rows = {}   # (pid, letter) -> list of trial dicts
for pid, letter, date, f in sessions:
    R = list(csv.DictReader(open(f, newline="", encoding="utf-8")))
    all_rows[(pid, letter)] = R
checks.internal("session tables: 13 columns each, subject id column constant and equal to the file name", all(
    len(R[0]) == 13 and {r["subid"] for r in R} == {pid + letter} for (pid, letter), R in all_rows.items()))
nblocks = {k: max(int(r["blockNum"]) for r in R) for k, R in all_rows.items()}
checks.published("sessions that stopped after block 9 of 10 (one LDR session)",
                 sum(v == 9 for v in nblocks.values()), 1, 0, "Results/Methods (task aborted after block 9 of 10 in one participant's LDR session)")
checks.internal("the stopped session has 9 blocks of 18 trials (162); all others have 10 blocks of 18 (180)", all(
    (v == 9 and len(all_rows[k]) == 162) or (v == 10 and len(all_rows[k]) == 180) for k, v in nblocks.items()))
stopped = [k for k, v in nblocks.items() if v == 9]
allr = [r for R in all_rows.values() for r in R]
radii = sorted({round(float(r["boundSize"]) / 2 * CM_PER_UNIT, 2) for r in allr})
for got, rep in zip(radii, (9.13, 11.41, 13.69)):
    checks.published(f"bounding-circle radius (cm), {rep}", got, rep, 0.005, "Methods, Joystick task (target radii 9.13, 11.41, 13.69 cm)")
checks.published("reward levels ($5, $10, $20): number of levels", len({r["reward"] for r in allr}), 3, 0, "Methods, Joystick task")
delay = [float(r["startDelay"]) for r in allr]
checks.published("shortest hold before the go signal (s)", round(min(delay)), 1, 0, "Methods, Joystick task (uniformly random 1-2 s)")
checks.published("longest hold before the go signal (s)", round(max(delay)), 2, 0, "Methods, Joystick task (uniformly random 1-2 s)")

# ---------------------------------------------------------------- numbers from the clinical export (all 11 participants)
ids11 = [p for p in pids if (p, 2) in vis]
age = {p: float(vis[(p, 1)]["age_demo"]) for p in ids11}
sex = {p: {"1": "M", "2": "F"}[vis[(p, 1)]["gender_demo"]] for p in ids11}
checks.published("mean age of the 11 participants (years)", st.mean(age.values()), 66.73, 0.005, "Results, Participants (mean 66.73)")
checks.published("SD of age", st.stdev(age.values()), 6.33, 0.005, "Results, Participants (SD 6.33)")
checks.published("men", sum(v == "M" for v in sex.values()), 6, 0, "Abstract and Results (6 male : 5 female)")
checks.published("women", sum(v == "F" for v in sex.values()), 5, 0, "Abstract and Results (6 male : 5 female)")
iv = [(dt.date.fromisoformat(vis[(p, 3)]["visit_date"][:10]) - dt.date.fromisoformat(vis[(p, 2)]["visit_date"][:10])).days
      for p in ids11]
checks.published("mean interval SDR to LDR visit (days)", st.mean(iv), 62.73, 0.005, "Results, Participants (62.73 days)")
checks.published("shortest interval (days)", min(iv), 27, 0, "Results, Participants (range 27-112)")
checks.published("longest interval (days)", max(iv), 112, 0, "Results, Participants (range 27-112)")
u3 = {p: [float(vis[(p, v)]["mds_iii_score"]) for v in (1, 2, 3, 4)] for p in ids11}
ft = {p: [float(vis[(p, v)]["fta_average"]) for v in (1, 2, 3, 4)] for p in ids11}
tot = [float(vis[(p, 1)]["part1score_i_ii"]) + float(vis[(p, 1)]["part2score_i_ii"]) + u3[p][0] for p in ids11]
for name, d, rep, tol in (("MDS-UPDRS III", u3, (22.5, 40.75, 52.98), 0.1), ("finger tapping", ft, (16.6, 25.25, 31.98), 0.05)):
    for v, lab, r in zip((1, 2, 3), ("SDR", "LDR", "SDR+LDR"), rep):
        sign = 1 if name == "finger tapping" else -1
        m0, mv = st.mean(d[p][0] for p in ids11), st.mean(d[p][v] for p in ids11)
        checks.published(f"{name}: improvement of the group mean from baseline, {lab} (%)", sign * (mv - m0) / m0 * 100, r, tol,
                         "Results, Clinical measures")
TOT_MEAN, TOT_SD = st.mean(tot), st.stdev(tot)

# ---------------------------------------------------------------- rows (consenting participants only)
names = [c[1] for c in stdlib.CORE] + [c[0] for c in stdlib.load_module("joystick_vigor")]
rows = []
for (pid, letter), R in all_rows.items():
    if pid in no_retain:
        continue
    state = STATE[letter]
    for i, r in enumerate(R):
        reached = r["boundReached"] == "1"
        pol = arr(r["boundCoordPolar"])
        t = fnum(r["timeToBound"])
        rows.append(dict(
            dataset=ID, study=ID, experiment="joystick_vigor", group="", subject=pid + letter, person=pid,
            subject_excluded_by_authors=False, age=age[pid], sex=sex[pid], handedness="",
            population="Parkinson's disease (de novo)", setting="lab", device="spring-loaded joystick",
            session="abcd".index(letter) + 1, block=f"block{int(r['blockNum'])}", trial=i + 1, phase="test", time_s="",
            task="reaching", unit="trial", condition=LABEL[state], pert_type="none", pert_value=0, pert_unit="",
            feedback="normal", excluded_by_authors=False,
            reward_usd=float(r["reward"]), bound_diameter_norm=float(r["boundSize"]),
            bound_radius_cm=round(float(r["boundSize"]) / 2 * CM_PER_UNIT, 2), bound_reached=reached,
            time_to_bound_ms=1000 * t if reached and t is not None else "",
            cross_angle_deg=pol[0] if reached else "", amount_earned_usd=fnum(r["amountEarned"]),
            path_rms_norm=fnum(r["RMS"]) if reached else "", pre_go_rms_norm=fnum(r["preRMS"]),
            start_delay_s=float(r["startDelay"]), medication_state=state))
persons = sorted({r["person"] for r in rows})
checks.internal("rows kept for every session of the 10 participants who allowed their data to be retained",
                len(rows) == sum(len(R) for (pid, _), R in all_rows.items() if pid not in no_retain)
                and len(persons) == 10 and not (set(persons) & no_retain))
checks.internal("blocks of 18 trials: each reward x radius cell twice per block", all(
    all(c == 2 for c in Counter((r["reward_usd"], r["bound_diameter_norm"]) for r in rows
                                if r["subject"] == s and r["block"] == b).values())
    and len(Counter((r["reward_usd"], r["bound_diameter_norm"]) for r in rows if r["subject"] == s and r["block"] == b)) == 9
    for s in sorted({r["subject"] for r in rows})[:6] for b in ("block1", "block5")))
checks.internal("crossing angle within [-180, 180] and time to bound within the 1.5 s window", all(
    -180 <= float(r["cross_angle_deg"]) <= 180 and 0 < float(r["time_to_bound_ms"]) <= 1500 for r in rows if r["bound_reached"]))
checks.internal("a trial without a crossing has no time, angle or path error", all(
    r["time_to_bound_ms"] == "" and r["cross_angle_deg"] == "" and r["path_rms_norm"] == "" for r in rows if not r["bound_reached"]))

participants = [dict(person=p, updrs3_baseline=u3[p][0], updrs3_sdr=u3[p][1], updrs3_ldr=u3[p][2], updrs3_sdr_ldr=u3[p][3],
                     finger_tapping_baseline=ft[p][0], finger_tapping_sdr=ft[p][1], finger_tapping_ldr=ft[p][2],
                     finger_tapping_sdr_ldr=ft[p][3]) for p in persons]
pcols = [(f"updrs3_{s}", "number", "points", f"MDS-UPDRS part III motor score at the {LABEL[s]} visit (REDCap mds_iii_score).")
         for s in STATE.values()] + [
    (f"finger_tapping_{s}", "number", "taps/min", f"Mean of the four finger-tapping trials at the {LABEL[s]} visit (REDCap fta_average).")
    for s in STATE.values()]

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Lee (University of Michigan)",
    data_url="https://osf.io/tac5y/",
    license="none stated",
    license_url="",
    changes="converted to the OpenMotor format, participant codes replaced, visit dates removed; one participant's rows are "
            "left out because that participant did not agree to data retention for future studies; trajectory files and the "
            "rest of the clinical export are not included.",
    unit_note="trial of one joystick session",
    how=[
        "One row is one trial of the incentivized joystick task: the person pushes a spring-loaded joystick from the centre out "
        "to a bounding circle, in any direction, within 1.5 s of a go signal, with $5, $10 or $20 at stake and a circle of "
        "radius 9.13, 11.41 or 13.69 cm. A full session is 180 trials (10 blocks of 18). Each participant did up to four "
        "visits: untreated baseline, after an acute levodopa dose (SDR), after 2-3 months of treatment with levodopa withheld "
        "(LDR), and on the usual dose (SDR+LDR); the deposit names the sessions a, b, c, d in that order, which matches the "
        "clinical visit dates.",
        "time_to_bound_ms is the authors' time from the go signal to the crossing; the paper's movement time also subtracts the "
        "initiation time (crossing 0.3 cm from the origin) and needs the trajectory files, which are not converted. "
        "cross_angle_deg is the direction of the crossing, counter-clockwise from the right on the screen; there is no "
        "instructed direction. Positions are in the task's screen units (display width = 2, 30.43 cm).",
        "Consent: the clinical export records, per participant, permission to retain data for future studies. One of the 11 "
        "participants answered no (the consent-form note reads 'subject did not give permission to retain data for future "
        "studies'), so that participant's four sessions are not in this file although they are in the authors' public deposit. "
        "The clinical numbers in the paper (age, sex, visit intervals, MDS-UPDRS, finger tapping) were checked on all 11 "
        "participants in the authors' export before that participant was removed.",
        "participants.csv holds each participant's MDS-UPDRS III and finger-tapping scores at the four visits. Handedness is empty "
        "because the export's coding is not documented; 2 of 4 sessions of one participant are missing because that participant "
        "could not do the task in the first two visits.",
    ],
    differences=["The paper reports a baseline MDS-UPDRS total (parts I-III) of 43.64 (SD 8.23, range 34-59); the export's parts "
                 f"I, II and III add to {TOT_MEAN:.2f} (SD {TOT_SD:.2f}, range {min(tot):.0f}-{max(tot):.0f}), so that one number "
                 "does not reproduce. The improvements of the group-mean MDS-UPDRS III are 22.54, 40.75 and 52.89 percent here "
                 "against 22.5, 40.75 and 52.98 in the paper.",
                 "The paper says the task stopped after block 9 of 10 in one participant's LDR session; in the deposit the one "
                 "session with 9 blocks is the baseline session (a) of the participant who did not consent to retention, so it "
                 "is not in this file.",
                 "The deposit lists no license (the paper itself is CC BY-NC-ND 4.0); the data are shared with credit to the authors "
                 "and removed on request.",
                 "Rows cover 10 of the paper's 11 participants (one did not agree to data retention); the paper's counts and "
                 "clinical numbers were reproduced on all 11 from the authors' clinical export."],
)
stdlib.release(ID, "joystick_vigor", rows, meta, checks, participants=participants, participant_columns=pcols)
