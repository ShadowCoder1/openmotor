"""Adams (2018), bioRxiv 385286: Tappy keystroke timing, larger Mendeley Data release (version 2, 18 May 2018).

Run from the site folder: python3 -I tools/std/convert/Adams2018_bioRxiv_handtremor.py
Raw files (untrusted data, read only): raw_extra/Adams2018_bioRxiv_handtremor/Archived_Data.zip.001-.013 (one zip split in 13
parts, joined here into joined.zip) and Archived_Users.zip. One row is one key press; the recorder stored no key identity.
The table is streamed (about 40 million rows), never held in memory.
"""
import re
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Adams2018_bioRxiv_handtremor"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
HASHES = {'Archived_Data.zip.001': '40c00627f73efb277b70cefe2bf847d78d123aacae73bbf2cfc9c0657821e6d1', 'Archived_Data.zip.002': 'b05076afd41d358ec41a355c13ba0e019106482871593e5cf8024e6dbf9bb7af', 'Archived_Data.zip.003': '6b160e9808342eb825c881d7d12f989c49bf5d4e4f7fa99be1004a1f22737c98', 'Archived_Data.zip.004': 'a0221c07345f5ca97ef84574e0f07da26b45c0f97bce3c24016c3bf4bab08d91', 'Archived_Data.zip.005': 'a7996c992dbcba1475fb0cbec78a4e77c97ac7677c0f6ec9ff869cc79bdd2894', 'Archived_Data.zip.006': 'bcaa3b286df8182a06392adb86e6a821efee0930e2d7fd399be1a301ca27f474', 'Archived_Data.zip.007': '309ceca067f9fa8a729b77edc828ae363e3ce08ab3f0985c2537592f46e22afe', 'Archived_Data.zip.008': 'a658cf1963f0e91c754110776c737747c77ea79a5e6158ad7ca982dd0e9c36c7', 'Archived_Data.zip.009': 'dfab62416816b608666a4df41e62bc22eca87bca2e2d05948e1a4a77c1e4be65', 'Archived_Data.zip.010': '6c74e007efc7a929cc9b58d10c7e5c5358eaf494e35ea307481d42b6f4f97d06', 'Archived_Data.zip.011': '6cacfe0d08b6b2e1dab7489c4b8a72589580e4188e66c8680d81d10df0e086cc', 'Archived_Data.zip.012': '31b8635f3ab515e9589d856868277fe59f16dcc76b85c99737c39eb263f6cdf6', 'Archived_Data.zip.013': 'd40c9511e5ca0f417ca7730d2630550d50696f0a15b1815e2e3cd79189e0cc2e', 'Archived_Users.zip': '8b1278d20a3b072a00e9aa027ad54b3eb989ff00cbc3014cf1efdd91c22a5035'}

ok = all(stdlib.sha256(RAW / f) == h for f, h in HASHES.items())
checks.internal("all 14 downloaded files match the SHA-256 values Mendeley Data lists", ok)
JOINED = RAW / "joined.zip"
if not JOINED.exists():
    with open(JOINED, "wb") as o:
        for i in range(1, 14):
            with open(RAW / ("Archived_Data.zip.%03d" % i), "rb") as f:
                shutil.copyfileobj(f, o)

# ---------------------------------------------------------------- users
uz = zipfile.ZipFile(RAW / "Archived_Users.zip")
users = {}
for nme in uz.namelist():
    m = re.fullmatch(r"User_(\w{10})\.txt", nme)
    if not m:
        continue
    d = {}
    for ln in uz.read(nme).decode("utf-8", "replace").splitlines():
        if ":" in ln:
            a, b = ln.split(":", 1)
            d[a.strip()] = b.strip()
    users[m[1]] = d

LINE = re.compile(r"^(\w{10})\t(\d{6})\t(\d\d:\d\d:\d\d\.\d{3})\t([LRS])\t(\d{4}\.\d)\t([LRS][LRS])\t(\d{4}\.\d)\t(\d{4}\.\d)\t$")
z = zipfile.ZipFile(JOINED)
entries = [(i, re.fullmatch(r"(\w{10})_(\d{4})\.txt", i.filename)) for i in z.infolist() if not i.is_dir()]
assert all(m for _, m in entries)

# ---------------------------------------------------------------- pass 1: counts and first time per user
count, tmin, order = {}, {}, []
n_lines = n_bad = 0
for info, m in entries:
    u = m[1]
    for ln in z.read(info).decode("utf-8", "replace").split("\r\n"):
        if not ln:
            continue
        n_lines += 1
        g = LINE.match(ln)
        if not g or g[1] != u:
            n_bad += 1
            continue
        if u not in count:
            count[u] = 0; tmin[u] = g[2] + g[3]; order.append(u)
        count[u] += 1
        k = g[2] + g[3]
        if k < tmin[u]:
            tmin[u] = k
n = sum(count.values())
checks.internal("every line is read or counted as damaged", n + n_bad == n_lines, f"{n} kept, {n_bad} damaged of {n_lines}")

def secs(yymmdd, hms):
    return date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:])).toordinal() * 86400 + int(hms[:2]) * 3600 + int(hms[3:5]) * 60 + float(hms[6:])

t0 = {u: secs(v[:6], v[6:]) for u, v in tmin.items()}
year0 = {u: 2000 + int(v[:2]) for u, v in tmin.items()}
has_user = {u: (u in users and users[u].get("Parkinsons") in ("True", "False")) for u in order}

pop, sex, age = {}, {}, {}
for u in order:
    d = users.get(u)
    if not has_user[u]:
        pop[u] = "unknown (no participant information)"; sex[u] = ""; age[u] = None
        continue
    pd_, tr = d["Parkinsons"] == "True", d["Tremors"] == "True"
    pop[u] = "Parkinson's disease" if pd_ else ("tremor without Parkinson's diagnosis" if tr else "healthy")
    sex[u] = {"Female": "F", "Male": "M"}.get(d["Gender"], "")
    age[u] = year0[u] - int(d["BirthYear"]) if d.get("BirthYear", "").isdigit() else None

# ---------------------------------------------------------------- published numbers
n_users_data = len(order)
n_info = sum(has_user.values())
checks.internal("participants with keystrokes and participant information", n_info > 500,
                f"{n_users_data} users with keystrokes, {n_info} of them with participant information (Mendeley page: over 500 people)")
checks.internal("no data user lost: users counted equal users with at least one valid row", len(order) == len(count))
checks.internal("every participant row set has a non-negative time from its first keystroke", all(t0[u] > 0 for u in order))
n_pd = sum(1 for u in order if has_user[u] and users[u]["Parkinsons"] == "True")
checks.internal("group sizes", True, f"{n_pd} with a PD diagnosis, {n_info - n_pd} without, {n_users_data - n_info} unknown")
NO_PUBLISHED = ("No number of the bioRxiv paper could be reproduced: it reports 465 anonymous participants aged 55 to 75 and a final sample of "
                "76 (27 with PD, 15 with tremor) selected by a rule (8 sequences of 50 continuous keystrokes per hand) that the deposit does "
                "not define closely enough to re-run; participants aged 55-75 with data and participant information number 347-364 here "
                "depending on the reference year. Internal checks only.")
DIFFERENCES = ["The paper's 465 participants aged 55-75 and its 76 selected subjects could not be reproduced from the deposit (see above)."]

# ---------------------------------------------------------------- table, streamed
CORE_TEMPLATE = dict(dataset=ID, study=ID, experiment=f"{ID}_Tappy", handedness="", setting="home",
                     device="own computer keyboard (Tappy recorder)", phase="other", task="typing", unit="keypress", condition="",
                     pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                     subject_excluded_by_authors=False)


class Rows:
    def __len__(self):
        return n

    def __iter__(self):
        tr, sess, prev_u, prev_m = {}, {}, None, None
        zz = zipfile.ZipFile(JOINED)
        for info, m in entries:
            u, mo = m[1], m[2]
            if u not in sess:
                sess[u] = 0
                tr[u] = 0
            sess[u] += 1
            for ln in zz.read(info).decode("utf-8", "replace").split("\r\n"):
                g = LINE.match(ln)
                if not g or g[1] != u:
                    continue
                tr[u] += 1
                r = dict(CORE_TEMPLATE)
                r.update(subject=u, group=pop[u], age=age[u], sex=sex[u], population=pop[u], session=sess[u], block=mo,
                         trial=tr[u], time_s=round(secs(g[2], g[3]) - t0[u], 3), hand=g[4], hold_ms=float(g[5]), direction=g[6],
                         interval_ms=float(g[7]), flight_ms=float(g[8]))
                yield r


part = []
for u in order:
    if not has_user[u]:
        continue
    d = users[u]
    part.append(dict(person=u, parkinsons=d["Parkinsons"] == "True", tremors=d["Tremors"] == "True",
                     diagnosis_year=d["DiagnosisYear"] if d.get("DiagnosisYear", "").isdigit() else None,
                     sided=d["Sided"] or None, updrs=d["UPDRS"] or None,
                     impact=(d["Impact"] if d["Impact"] not in ("------", "") else None),
                     levodopa=d["Levodopa"] == "True", dopamine_agonist=d["DA"] == "True", maob_inhibitor=d["MAOB"] == "True",
                     other_medication=d["Other"] == "True", n_keystrokes_kept=count[u]))
pcols = [("parkinsons", "true/false", "", "Self-reported Parkinson's disease diagnosis (user file 'Parkinsons')."),
         ("tremors", "true/false", "", "Self-reported tremor (user file 'Tremors')."),
         ("diagnosis_year", "integer", "year", "Year of diagnosis (user file 'DiagnosisYear'); empty when not given."),
         ("sided", "category", "Left, Right, None", "Side of the body most affected, self-reported ('Sided')."),
         ("updrs", "text", "1-4 or Don't know", "UPDRS score, if the participant knew it."),
         ("impact", "category", "Mild, Medium, Severe", "Self-reported impact of the condition ('Impact'); empty when not given."),
         ("levodopa", "true/false", "", "Takes levodopa (user file 'Levodopa')."),
         ("dopamine_agonist", "true/false", "", "Takes a dopamine agonist ('DA')."),
         ("maob_inhibitor", "true/false", "", "Takes an MAO-B inhibitor ('MAOB')."),
         ("other_medication", "true/false", "", "Takes other Parkinson's medication ('Other')."),
         ("n_keystrokes_kept", "integer", "", "Number of this participant's rows in the main table.")]

meta = dict(
    study="Adams 2018",
    paper=dict(authors="Adams WR", year=2018,
               title="The detection of hand tremor through the characteristics of finger movement while typing",
               venue="bioRxiv preprint 385286 (posted 5 August 2018)", doi="10.1101/385286"),
    lab="Adams (Charles Sturt University)",
    data_url="https://data.mendeley.com/datasets/z39mhdsynx/2",
    license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=(f"Converted to the OpenMotor format with participant codes replaced; calendar dates and clock times were replaced by seconds "
             f"since the participant's first keystroke; the participant files became participants.csv without the birth year (age is "
             f"the year of the first keystroke minus the birth year, so it can be one year off); {n_bad} damaged lines (fields run "
             f"together or taken from another file, of {n_lines}) were left out."),
    how=[
        "One row is one key press recorded by the Tappy app while people typed as usual on their own computer over weeks to months; "
        "the recorder kept no key identity, only which side of the keyboard the key is on (hand: L, R, or S for the space bar).",
        "hold_ms is the time the key was held down, interval_ms the time from the previous key press to this one (the authors' "
        "'latency'), flight_ms the time from the previous key's release to this press, direction the sides of the previous and this key; "
        "all in ms as the authors give them. Rows keep the authors' file order: users in archive order, then months (session is the "
        "month file, block its YYMM label).",
        "time_s is seconds since the participant's first recorded keystroke. This release is a larger superset of the PhysioNet Tappy "
        "release (Adams2017_PLoSOne); the paper analysed 76 selected subjects, and its selection (8 sequences of 50 continuous keystrokes "
        "with each hand, no levodopa) is not marked here, so subject_excluded_by_authors is false for everyone.",
        "No typed text exists in the data. Participants without participant information are kept with population 'unknown'.",
    ],
    unit_note="key press (Tappy keystroke event)",
    differences=DIFFERENCES,
)
stdlib.release(ID, "keypress", Rows(), meta, checks, no_published_reason=NO_PUBLISHED, participants=part, participant_columns=pcols)
