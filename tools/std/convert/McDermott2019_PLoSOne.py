"""McDermott & Himmelbach (2019), PLoS ONE: effects of arm weight and target height on hand selection in a low-cost virtual-reality task.

Run from the site folder: python3 -I tools/std/convert/McDermott2019_PLoSOne.py
Raw files (untrusted data, read only) in raw_extra/McDermott2019_PLoSOne/ from https://doi.org/10.7910/DVN/PQ2ABU (Harvard Dataverse,
CC0), dataset.zip with one log file per participant (30 .txt files) and the authors' MATLAB files. total30-1.mat and pureTotal27-1.mat
(plain numeric arrays) are read only to learn which 3 of the 30 participants the authors left out of their 27-person analysis: a
participant is analysed when his or her contacts are in pureTotal27-1.mat (matched on cube start and contact times).
The authors' C# logging code (github.com/EricJamesMcDermott/VR_HandSelection, GPL v3) was read as text: each log line is written when a
cube is touched and holds the palm position at cube spawn, the cube position, the spawn time and the contact time.
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "McDermott2019_PLoSOne"
R = stdlib.raw_dir(ID)
ck = stdlib.Checks()
z = zipfile.ZipFile(R / "dataset.zip")
PAT = re.compile(r"^(Left|Right)HandTransform= \(([^)]*)\) , CubeTransform= \(([^)]*)\) , startTime= ([\d.]+) , contactTime= ([\d.]+) , Level(\d)$")

m27 = sio.loadmat(io.BytesIO(z.read("pureTotal27-1.mat")))
m30 = sio.loadmat(io.BytesIO(z.read("total30-1.mat")))
KEEP = {(round(float(r[6]), 2), round(float(r[7]), 2)) for k in ("totalDataL", "totalDataR", "totalDataL_weighted", "totalDataR_weighted") for r in m27[k]}
ALL = {(round(float(r[6]), 2), round(float(r[7]), 2)) for k in ("totalDataL", "totalDataR", "totalDataL_weighted", "totalDataR_weighted") for r in m30[k]}

files = sorted((n for n in z.namelist() if n.endswith(".txt")), key=lambda n: int(re.search(r"(\d+)\.txt$", n).group(1)))
recs = []
for n in files:
    pnum = int(re.search(r"(\d+)\.txt$", n).group(1))   # the authors' participant number from the file name (initials are not used)
    lines = z.read(n).decode("utf-8").splitlines()
    parsed = []
    for l in lines:
        mt = PAT.match(l.strip())
        assert mt, "unreadable log line"
        parsed.append(mt)
    key = [(round(float(p.group(4)), 2), round(float(p.group(5)), 2)) for p in parsed]
    analysed = sum(k in KEEP for k, p in zip(key, parsed) if p.group(6) != "1") > 50
    in_all = sum(k in ALL for k, p in zip(key, parsed) if p.group(6) != "1")
    assert in_all == sum(p.group(6) != "1" for p in parsed), "contacts missing from the authors' 30-person table"
    t0 = float(parsed[0].group(4))
    for i, p in enumerate(parsed):
        lv = int(p.group(6))
        cube = [float(v) for v in p.group(3).split(",")]
        palm = [float(v) for v in p.group(2).split(",")]
        rnd = lv - 1
        w = rnd >= 3
        recs.append(dict(
            dataset=ID, study=ID, experiment=ID, group="", subject=f"P{pnum:02d}", subject_excluded_by_authors=not analysed,
            age="", sex="", handedness="right" if analysed else "", population="healthy", setting="lab",
            device="Leap Motion hand tracker with a virtual-reality scene on a laptop screen (3D hands, cubes to touch)",
            session=1, block=f"round_{rnd}" if rnd else "training", trial=i + 1,
            phase="practice" if rnd == 0 else ("test" if w else "baseline"), time_s=float(p.group(4)) - t0,
            task="reaching with free choice of hand", unit="trial", condition="weighted left glove" if w else "non-weighted",
            pert_type="arm_weight" if w else "none", pert_value=500 if w else 0, pert_unit="g" if w else "", feedback="normal",
            excluded_by_authors=(not analysed),
            round=rnd, level_orig=lv, weighted_glove=w, cube_x=cube[0], cube_row=int(cube[1]), cube_z=cube[2],
            hand_used=p.group(1).lower(), palm_x=palm[0], palm_y=palm[1], palm_z=palm[2],
            start_time_s=float(p.group(4)), contact_time_s=float(p.group(5))))
t = pd.DataFrame(recs)
t["person"] = t["subject"]
cols = [c[1] for c in stdlib.columns_for("hand_selection_vr")]

# ---------------------------------------------------------------- checks against the paper
persons = t.drop_duplicates("subject")
ana = t[~t.subject_excluded_by_authors]
ck.published("participants recruited and tested", len(persons), 30, 0, "Methods, Participants: 30 participants")
ck.published("participants in the analysis", ana.subject.nunique(), 27, 0, "Methods, Participants: three excluded, 27 analysed")
ck.published("excluded participants", persons.subject_excluded_by_authors.sum(), 3, 0, "Methods, Participants: three participants excluded")
rt = ana[(ana.hand_used == "right") & (ana.cube_x < 4) & (ana["round"] > 0)]
ck.published("right-hand contacts left of the midline, non-weighted rounds (27 participants)", ((rt.weighted_glove == False)).sum(), 267, 0, "Results, crossovers: 267 right-hand crossovers in the non-weighted condition")
ck.published("right-hand contacts left of the midline, weighted rounds (27 participants)", rt.weighted_glove.sum(), 327, 0, "Results, crossovers: 327 in the weighted condition")
ck.published("increase in right-hand crossovers with the weighted glove (percent)", (rt.weighted_glove.sum() / (rt.weighted_glove == False).sum() - 1) * 100, 22.5, 0.05, "Results, crossovers: rising 22.5 percent")
ck.internal("47 to 49 cubes per test round and 4 test rounds per participant", bool(t[t["round"] > 0].groupby(["subject", "round"]).size().between(47, 49).all() and (t[t["round"] > 0].groupby("subject")["round"].nunique() == 4).all()), str(t[t["round"] > 0].groupby(["subject", "round"]).size().describe().to_dict()))
ck.internal("rows kept: all 6067 log lines", len(t) == 6067, str(len(t)))
ck.internal("13 cube columns between 1 and 7, 3 rows", t.cube_x.nunique() == 13 and set(t.cube_row) == {1, 2, 3})
ck.internal("contact after appearance on every row", bool((t.contact_time_s > t.start_time_s).all()))

meta = dict(
    paper=dict(authors="McDermott EJ, Himmelbach M", year=2019, title="Effects of arm weight and target height on hand selection: A low-cost virtual reality paradigm",
               venue="PLoS ONE 14(6):e0207326", doi="10.1371/journal.pone.0207326"),
    lab="Himmelbach (University of Tubingen)", data_url="https://doi.org/10.7910/DVN/PQ2ABU", license="CC0 1.0",
    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("File names (initials and a number) replaced by participant codes; the log lines were split into columns without changing values; "
             "time_s is the cube appearance time minus the first cube appearance of the participant; the authors' MATLAB summary files were not copied."),
    how=["One row is one cube touched in a virtual-reality hand-selection task (Leap Motion tracking of both hands). Cubes appeared one at a time at one of "
         "3 rows x 13 columns; the participant touched each with the hand of their choice. After 10 training cubes (round 0) there were 4 rounds of 48 cubes: "
         "rounds 1 and 2 as normal, rounds 3 and 4 with a 500 g sandbag in the left glove (weighted_glove). cube_x, cube_row and cube_z are the authors' CubeTransform "
         "(Unity units, midline at x = 4.0, row 1 bottom to 3 top); palm_x, palm_y, palm_z are the authors' HandTransform of the hand that touched the cube; "
         "start_time_s and contact_time_s are the authors' startTime and contactTime (seconds of the program clock).",
         "The deposit has only cubes that were touched, so the few missing cubes (a round has 47 to 49 rows, one has 49) are not in the file. The coordinate axes are Unity world axes and "
         "the paper does not name them; palm y and z are therefore given only as numbers. There is no hand angle or endpoint error: the measure of interest is which hand "
         "was chosen for which cube.",
         "subject_excluded_by_authors marks the 3 participants whom the authors left out of their analysis (one below the right-handedness cutoff, two for other reasons "
         "given in the paper); they are identified here by their absence from the authors' pureTotal27-1.mat. Age and sex are not shared per participant."],
    unit_note="cube touched",
    differences=["The paper's repeated-measures ANOVA and correlation results rest on the authors' MATLAB tables of per-participant hand-selection biases and EHI and grip scores, "
                 "which are not rebuilt here; the published checks are the participant counts and the 267 and 327 crossover counts."],
)
stdlib.release(ID, "hand_selection_vr", t[cols].to_dict("records"), meta, ck)
