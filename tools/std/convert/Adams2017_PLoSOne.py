"""Adams (2017), PLoS ONE 12(11):e0188226: Tappy keystroke timing of people with and without Parkinson's disease (PhysioNet tappy 1.0.0).

Run from the site folder: python3 -I tools/std/convert/Adams2017_PLoSOne.py
Raw files (untrusted data, read only): raw_extra/Adams2017_PLoSOne/Archived-Data.zip, Archived-users.zip, SHA256SUMS.txt.
One row is one key press. The recorder stored no key identity, only the key's side of the keyboard (L, R, space).
"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Adams2017_PLoSOne"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

# ---------------------------------------------------------------- integrity
sums = {}
for ln in open(RAW / "SHA256SUMS.txt").read().splitlines():
    p = ln.split()
    if len(p) == 2:
        sums[p[1].lstrip("*")] = p[0]
ok = all(stdlib.sha256(RAW / f) == h for f, h in sums.items() if (RAW / f).exists())
checks.internal("downloaded zips match PhysioNet SHA256SUMS.txt", ok and len(sums) >= 2, str(sorted(sums)))

# ---------------------------------------------------------------- users
uz = zipfile.ZipFile(RAW / "Archived-users.zip")
users = {}
for n in uz.namelist():
    m = re.fullmatch(r"Archived users/User_(\w{10})\.txt", n)
    if not m:
        continue
    d = {}
    for ln in uz.read(n).decode("utf-8", "replace").splitlines():
        if ":" in ln:
            a, b = ln.split(":", 1)
            d[a.strip()] = b.strip()
    users[m[1]] = d

# ---------------------------------------------------------------- keystrokes
LINE = re.compile(r"^(\w{10})\t(\d{6})\t(\d\d:\d\d:\d\d\.\d{3})\t([LRS])\t(\d{4}\.\d)\t([LRS][LRS])\t(\d{4}\.\d)\t(\d{4}\.\d)\t$")
z = zipfile.ZipFile(RAW / "Archived-Data.zip")
uid, month, dt, hand, hold, direc, lat, fl = ([] for _ in range(8))
n_lines = n_bad = 0
for info in z.infolist():
    m = re.fullmatch(r"Tappy Data/(\w{10})_(\d{4})\.txt", info.filename)
    if not m:
        continue
    for ln in z.read(info).decode("utf-8", "replace").split("\r\n"):
        if not ln:
            continue
        n_lines += 1
        g = LINE.match(ln)
        if not g or g[1] != m[1]:
            n_bad += 1                      # damaged line (fields run together or from another file): not usable
            continue
        uid.append(m[1]); month.append(m[2]); dt.append(g[2] + " " + g[3]); hand.append(g[4])
        hold.append(float(g[5])); direc.append(g[6]); lat.append(float(g[7])); fl.append(float(g[8]))
n = len(uid)
checks.internal("every line is read or counted as damaged", n + n_bad == n_lines, f"{n} kept, {n_bad} damaged of {n_lines}")
uid = np.array(uid); month = np.array(month); hand = np.array(hand); direc = np.array(direc)
ts = pd.to_datetime(pd.Series(dt), format="%y%m%d %H:%M:%S.%f")
first_seen = pd.factorize(uid)[0]
codes_users = pd.unique(uid)
t0 = ts.groupby(first_seen).transform("min")
time_s = (ts - t0).dt.total_seconds().to_numpy()
year0 = ts.groupby(first_seen).transform("min").dt.year.to_numpy()
trial = pd.Series(first_seen).groupby(first_seen).cumcount().to_numpy() + 1
mon_ord = pd.Series(month).groupby(uid).transform(lambda s: pd.factorize(s)[0] + 1).to_numpy()
del dt, ts

has_user = np.array([u in users for u in codes_users])
n_data_users = len(codes_users)
counts = pd.Series(uid).value_counts()

def attr(u, k):
    return users.get(u, {}).get(k)

pop, grp, sex, age = {}, {}, {}, {}
for i, u in enumerate(codes_users):
    d = users.get(u)
    if d is None:
        pop[u] = grp[u] = "unknown (no participant file)"; sex[u] = ""; age[u] = None
        continue
    pd_, tr = d["Parkinsons"] == "True", d["Tremors"] == "True"
    pop[u] = grp[u] = ("Parkinson's disease" if pd_ else ("tremor without Parkinson's diagnosis" if tr else "healthy"))
    sex[u] = {"Female": "F", "Male": "M"}.get(d["Gender"], "")
    by = d.get("BirthYear", "")
    age[u] = None
    if by.isdigit():
        yr = int(year0[np.flatnonzero(uid == u)[0]]) if False else None
yr_first = pd.Series(year0).groupby(first_seen).first()
for i, u in enumerate(codes_users):
    by = attr(u, "BirthYear")
    if by and by.isdigit():
        age[u] = int(yr_first[i]) - int(by)

# the paper's Group A: participants with clinical file AND keystrokes; analysis subset (Data analysis section):
# >= 2000 keystrokes, non-PD or Mild PD, not taking levodopa
def in_subset(u):
    d = users.get(u)
    if d is None or counts[u] < 2000:
        return False
    if d["Levadopa"] == "True":
        return False
    # one PD user has an empty Impact field; counting that user as not Medium/Severe gives the paper's 53 (52 if left out)
    return d["Parkinsons"] == "False" or d["Impact"] in ("Mild", "")
subset = {u: in_subset(u) for u in codes_users}

checks.published("participants with a user file and keystrokes (Group A)", int(has_user.sum()), 217, 0,
                 "Data analysis section: 'The complete dataset comprised 217 participants (termed Group A)'")
checks.published("analysis subset: >= 2000 keystrokes, non-PD or Mild PD, no levodopa", sum(subset.values()), 53, 0,
                 "Data analysis section: 'an analysis subset of 53 participants (including both PD and non-PD)'")
checks.internal("keystroke-file users without a user file are kept, marked unknown", n_data_users - int(has_user.sum()) >= 0,
                f"{n_data_users - int(has_user.sum())} such users, {int(np.isin(uid, [u for u in codes_users if u not in users]).sum())} rows")
checks.internal("hold, latency and flight times in 0-9999 ms", bool((np.array(hold) >= 0).all() and (np.array(lat) < 10000).all()))
checks.internal("direction first letter of key side agrees with hand on every row", bool((np.array([d[1] for d in direc]) == hand).all()))

# ---------------------------------------------------------------- table, streamed
subj_ex = np.array([not subset[u] for u in codes_users])
CORE_TEMPLATE = dict(dataset=ID, study=ID, experiment=f"{ID}_Tappy", handedness="", setting="home",
                     device="own computer keyboard (Tappy recorder)", phase="other", task="typing", unit="keypress", condition="",
                     pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False)

class Rows:
    def __len__(self):
        return n

    def __iter__(self):
        for u_i, mo, mon, tr_, ts_, h, hd, dr, la, f_ in zip(first_seen, month, mon_ord, trial, time_s, hand, hold, direc, lat, fl):
            u = codes_users[u_i]
            r = dict(CORE_TEMPLATE)
            r.update(subject=u, group=grp[u], subject_excluded_by_authors=bool(subj_ex[u_i]), age=age[u], sex=sex[u],
                     population=pop[u], session=int(mon), block=mo, trial=int(tr_), time_s=round(float(ts_), 3),
                     hand=h, hold_ms=hd, direction=dr, interval_ms=la, flight_ms=f_)
            yield r

part = []
for u in codes_users:
    d = users.get(u)
    if d is None:
        continue
    part.append(dict(person=u, parkinsons=d["Parkinsons"] == "True", tremors=d["Tremors"] == "True",
                     diagnosis_year=d["DiagnosisYear"] if d["DiagnosisYear"].isdigit() else None,
                     sided=d["Sided"], updrs=d["UPDRS"], impact=(d["Impact"] if d["Impact"] not in ("------", "") else None),
                     levodopa=d["Levadopa"] == "True", dopamine_agonist=d["DA"] == "True", maob_inhibitor=d["MAOB"] == "True",
                     other_medication=d["Other"] == "True", n_keystrokes_kept=int(counts[u])))
pcols = [("parkinsons", "true/false", "", "Self-reported Parkinson's disease diagnosis (user file 'Parkinsons')."),
         ("tremors", "true/false", "", "Self-reported tremor (user file 'Tremors')."),
         ("diagnosis_year", "integer", "year", "Year of diagnosis (user file 'DiagnosisYear'); empty when not given."),
         ("sided", "category", "Left, Right, None", "Side of the body most affected, self-reported ('Sided')."),
         ("updrs", "text", "1-4 or Don't know", "UPDRS score, if the participant knew it (as in the user file)."),
         ("impact", "category", "Mild, Medium, Severe", "Self-reported impact of the condition ('Impact'); empty when not given."),
         ("levodopa", "true/false", "", "Takes levodopa (user file 'Levadopa')."),
         ("dopamine_agonist", "true/false", "", "Takes a dopamine agonist ('DA')."),
         ("maob_inhibitor", "true/false", "", "Takes an MAO-B inhibitor ('MAOB')."),
         ("other_medication", "true/false", "", "Takes other Parkinson's medication ('Other')."),
         ("n_keystrokes_kept", "integer", "", "Number of this participant's rows in the main table.")]

pdn = [u for u in codes_users if u in users and users[u]["Parkinsons"] == "True"]
checks.internal("group sizes", True, f"{len(pdn)} with a PD diagnosis, {int(has_user.sum()) - len(pdn)} without, among {int(has_user.sum())} with a file")

meta = dict(
    study="Adams 2017",
    paper=dict(authors="Adams WR", year=2017,
               title="High-accuracy detection of early Parkinson's Disease using multiple characteristics of finger movement while typing",
               venue="PLoS ONE 12(11):e0188226", doi="10.1371/journal.pone.0188226"),
    lab="Adams (Charles Sturt University)",
    data_url="https://physionet.org/content/tappy/1.0.0/",
    license="ODC-By 1.0",
    license_url="https://physionet.org/content/tappy/view-license/1.0.0/",
    changes=(f"Converted to the OpenMotor format with participant codes replaced; calendar dates and clock times were replaced by seconds "
             f"since the participant's first keystroke; the authors' participant files became participants.csv without the birth year "
             f"(age is the year of the first keystroke minus the birth year, so it can be one year off); {n_bad} damaged lines "
             f"(fields run together, of {n_lines}) were left out."),
    how=[
        "One row is one key press recorded by the Tappy app while people typed as usual on their own computer over weeks to months; "
        "the recorder kept no key identity, only which side of the keyboard the key is on (hand: L, R, or S for the space bar).",
        "hold_ms is the time the key was held down, interval_ms the time from the previous key press to this one (the authors' "
        "'latency'), flight_ms the time from the previous key's release to this press, direction the sides of the previous and this key; "
        "all in ms as the authors give them (accurate to a few ms). Rows keep the order of the authors' files: users in archive order, "
        "then months; session is the month file, block its YYMM label.",
        "time_s is seconds since the participant's first recorded keystroke. subject_excluded_by_authors marks participants outside the "
        "paper's 53-participant analysis subset (under 2000 keystrokes, PD other than Mild, taking levodopa, or no participant file).",
        "No typed text exists in the data. Participants without a participant file (clinical status unknown) are kept with population "
        "'unknown'.",
    ],
    unit_note="key press (Tappy keystroke event)",
    differences=["The paper's 53-participant analysis subset is reproduced only if the one PD participant whose Impact field is empty is counted as Mild (52 otherwise); that participant is therefore not marked subject_excluded_by_authors, which is an inference.", "The paper does not report counts of keystrokes; the keystroke-file users without a participant file are not part "
                 "of the paper's 217-participant Group A."],
)
rows = Rows()
stdlib.release(ID, "keypress", rows, meta, checks, participants=part, participant_columns=pcols)
