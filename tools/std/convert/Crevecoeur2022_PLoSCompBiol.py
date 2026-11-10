"""Crevecoeur, Mathew & Lefevre (2022), PLoS Comput Biol: separability of human motor memories during reaching adaptation with
force cues. 47 participants in three experiments (18, 14, 15), KINARM forward reaches of 15 cm with lateral background loads that
cue velocity-dependent force fields. One row per reach, the 1.2 s kinematics and forces of each reach as a CSV inside the ZIP
(module timeseries).

Run from the site folder: python3 -I tools/std/convert/Crevecoeur2022_PLoSCompBiol.py
Raw files (untrusted data, read only) in raw_extra/Crevecoeur2022_PLoSCompBiol/ (Zenodo 7223372, copy of the Dryad deposit
10.5061/dryad.b2rbnzsk3): data_experiment1/2/3.zip, one MATLAB file per participant (ex<e>s<n>.mat) with x, y, dx, dy, fx, fy
(trials x 1200 samples at 1 kHz) and extracted (trials x 8) + descriptions, and README.md.
"""
import csv
import io
import json
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Crevecoeur2022_PLoSCompBiol"
RAW = stdlib.raw_dir(ID)
TMP = RAW / "_samples"
TMP.mkdir(exist_ok=True)
checks = stdlib.Checks()
CH = ["x", "y", "dx", "dy", "fx", "fy"]
FS = 1000.0
ONSET = 200  # samples before reach onset kept by the authors (README: truncated from 200 ms prior to movement onset)

# trial types (README). value: (description, field gain N s/m or 0, field direction, background load description)
EXP1 = {
    1: ("positive background load, no force field", 0, "", "positive 4 N"),
    2: ("negative background load, no force field", 0, "", "negative 4 N"),
    3: ("positive background load, clockwise force field (negative context)", 13, "CW", "positive 4 N"),
    4: ("negative background load, counter-clockwise force field (negative context)", 13, "CCW", "negative 4 N"),
    5: ("positive background load, counter-clockwise force field (positive context)", 13, "CCW", "positive 4 N"),
    6: ("negative background load, clockwise force field (positive context)", 13, "CW", "negative 4 N"),
    7: ("no background load, counter-clockwise force field", 13, "CCW", "none"),
    8: ("no background load, clockwise force field", 13, "CW", "none"),
}
EXP23 = {
    2: ("light background force, no force field (catch trial)", 0, "", "light 2.5 N"),
    3: ("heavy background force, no force field (catch trial)", 0, "", "heavy 5 N"),
    6: ("light background force, light force field", 7.5, "", "light 2.5 N"),
    7: ("heavy background force, heavy force field", 15, "", "heavy 5 N"),
}
TYPES = {1: EXP1, 2: EXP23, 3: EXP23}
EXPN = {1: "exp1_opposing_contexts", 2: "exp2_light_heavy", 3: "exp3_heavy_then_light_heavy"}
PHASE = lambda e, st: ({1: "baseline", 5: "washout", 9: "test"}.get(st, "learning") if e == 1 else "learning")

rows, files = [], []
npers = {}
catch_angle = {}  # (exp, subject) -> {tp: [angles]}
sizes = {}
for e in (1, 2, 3):
    z = zipfile.ZipFile(RAW / f"data_experiment{e}.zip")
    names = sorted((n for n in z.namelist() if re.search(r"(?:^|/)(?:ex\d)?s(\d+)\.mat$", n) and "__MACOSX" not in n),
                   key=lambda n: int(re.search(r"s(\d+)\.mat$", n).group(1)))
    npers[e] = len(names)
    for n in names:
        sn = int(re.search(r"s(\d+)\.mat$", n).group(1))
        d = sio.loadmat(io.BytesIO(z.read(n)), squeeze_me=True, struct_as_record=False)
        m = d[[k for k in d if not k.startswith("__")][0]]
        E = np.asarray(m.extracted).astype(int)
        desc = [str(x) for x in m.descriptions]
        assert desc[:8] == ["TrialNum", "Sequence", "TPType", "TPCount", "Reaction Time", "Perturbation Onset", "Success", "Set"], desc
        arr = {c: np.asarray(getattr(m, c), dtype=float) for c in CH}
        assert all(a.shape == (len(E), 1200) for a in arr.values())
        order = np.lexsort((E[:, 1], E[:, 7]))  # execution order: Sequence (overall order in the session)
        order = np.argsort(E[:, 1], kind="stable")
        sid = f"E{e}_S{sn}"
        sizes.setdefault(e, []).append(len(E))
        for rank, k in enumerate(order, start=1):
            tn, seq, tp, tpc, rt, po, succ, st = (int(v) for v in E[k])
            desc_t, gain, fdir, bg = TYPES[e][tp]
            fld = gain > 0
            fname = f"{EXPN[e].split('_')[0]}_s{sn:02d}_t{rank:03d}.csv"
            path = TMP / fname
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh, lineterminator="\r\n")
                w.writerow(["time_s"] + CH)
                for i in range(1200):
                    w.writerow([f"{(i - ONSET) / FS:.3f}"] + [format(arr[c][k, i], ".7g") for c in CH])
            files.append((f"samples/{fname}", str(path)))
            rows.append(dict(
                dataset=ID, study="Crevecoeur et al. 2022 PLoS Comput Biol", experiment=EXPN[e], group="", subject=sid, person=sid,
                subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab",
                device="KINARM robotic manipulandum", session=1, block=f"set{st}", trial=rank, phase=PHASE(e, st), time_s="",
                task="reaching", unit="recording", condition=desc_t, pert_type="force_field" if fld else "background_load" if bg != "none" else "none",
                pert_value=gain if fld else (4 if e == 1 and bg != "none" else 2.5 if "light" in bg else 5 if "heavy" in bg else 0),
                pert_unit="N s/m" if fld else ("N" if bg != "none" else ""), feedback="normal", excluded_by_authors=False,
                recording=rank, samples_file=f"samples/{fname}", signal_kind="hand (cursor) position, velocity and robot force during a forward reach",
                channels=";".join(CH), channel_units="m;m;m/s;m/s;N;N", sample_rate_hz=FS, n_samples=1200, duration_s=1.2,
                events=f"reach_onset=0.2;perturbation_onset={(ONSET + po) / FS:.3f}".replace("reach_onset=0.2", "reach_onset=0.200"),
                processing="low-pass 50 Hz, 4th-order Butterworth, dual pass; velocity by 4th-order centred finite differences (paper, Methods)",
                trial_info=f"trial_type=TP{tp};background_load={bg};force_field_direction={fdir or 'none'};set={st};sequence_in_session={seq};"
                           f"trial_number_by_type={tn};count_of_type={tpc};reaction_time_ms={rt};success={succ}"))

print({e: (npers[e], min(v), max(v)) for e, v in sizes.items()})
n_tot = sum(npers.values())
checks.published("participants in experiment 1", npers[1], 18, 0, "Methods, Participants (18 in Experiment 1)")
checks.published("participants in experiment 2", npers[2], 14, 0, "Methods, Participants (14 in Experiment 2)")
checks.published("participants in experiment 3", npers[3], 15, 0, "Methods, Participants (15 in Experiment 3)")
checks.published("participants in total", n_tot, 47, 0, "Methods, Participants (47)")
checks.internal("one row per stored trial, 1200 samples each", len(rows) == sum(sum(v) for v in sizes.values()) and len(files) == len(rows))
e3 = [r for r in rows if r["experiment"] == EXPN[3] and r["block"] == "set1"]
checks.published("experiment 3, block 1: catch trials per participant (median)", float(np.median([sum(1 for r in e3 if r["subject"] == s and "catch" in r["condition"]) for s in {r["subject"] for r in e3}])), 10, 0,
                 "Methods, Experiment 3 (one heavy-only block of 50 force-field and 10 catch trials)")
e1 = [r for r in rows if r["experiment"] == EXPN[1]]
for st, nm, rep in ((5, "washout block", 30), (9, "final block without background force", 30)):
    per = [sum(1 for r in e1 if r["subject"] == sb and r["block"] == f"set{st}") for sb in {r["subject"] for r in e1}]
    checks.published(f"experiment 1, {nm}: trials per participant (median)", float(np.median(per)), rep, 0, "Methods, Experiment 1 (30 trials)")
paper = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Crevecoeur & Lefevre (UCLouvain)", data_url="https://doi.org/10.5061/dryad.b2rbnzsk3",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes="the authors' MATLAB matrices (trials x samples) were written as one CSV per reach, with a time column added; trials were "
            "put in the order of the authors' Sequence; values keep 7 significant digits; participant codes replaced.",
    unit_note="reach (recording) of one participant",
    how=[
        "One row is one forward reach (15 cm, KINARM) of one participant: 18 people in experiment 1 (opposing background loads "
        "cue opposing force fields), 14 in experiment 2 (light and heavy loads) and 15 in experiment 3 (a heavy-only block, then "
        "two blocks of light and heavy trials). Rows are in the order the participant performed them (the authors' Sequence). "
        "Experiment 1 blocks (set): 1 baseline with background loads only, 2-4 and 6-8 field blocks of the two contexts, 5 washout, 9 a test block without background load (phase baseline, learning, washout, test). Experiment 2 set 6 has no catch trials. "
        "Trials the authors did not include (gaps in Sequence) are not in the deposit and are not rows.",
        "samples_file holds the reach's 1200 samples at 1 kHz: x and y (hand/cursor position, m), dx and dy (velocity, m/s) and "
        "fx and fy (force on the handle, N), as stored by the authors. time_s is 0 at reach onset (the cursor leaving the start "
        "target); the stored window starts 200 ms before onset (README: structures truncated from 200 ms prior to movement "
        "onset), so time_s runs from -0.200 to 0.999. This offset is the README's statement; the hand paths leave the start "
        "position about 10 ms after sample 200. events: perturbation_onset is when the cursor crossed one third of the reach "
        "(there is no perturbation at that time; the authors' Perturbation Onset, in ms after reach onset).",
        "trial_info gives the authors' trial type (TP1-TP8 of experiment 1, TP2/3/6/7 of experiments 2 and 3), the background "
        "load (a lateral force that ramps up before the reach and cues the field; sign convention in the paper: rightward loads "
        "are negative force offsets), the force-field direction, the set (block of about 60 trials), the reaction time in ms "
        "(go cue to the cursor leaving the start target) and success (reach completed in 600-800 ms). The field is a lateral "
        "force proportional to forward velocity (13 N s/m in experiment 1; 7.5 and 15 N s/m for light and heavy trials in "
        "experiments 2 and 3). pert_type is force_field on field trials and background_load on catch trials (load only); "
        "pert_value is the field gain (N s/m) or the load size (N). The field direction of experiments 2 and 3 is not stated "
        "in the deposit, so it is not given.",
    ],
    differences=["A trial counted as a catch trial has the background load but no field, as in the paper; the deposit does not "
                 "say which trials the authors left out, and some blocks have 1-6 fewer trials than the 60 of the Methods.",
                 "The Zenodo record is a copy of the Dryad deposit (same DOI, CC0); its readme says the files are in *.m format, "
                 "but they are MATLAB .mat files. The Zenodo page lists no related paper; the paper is Crevecoeur F, Mathew J, "
                 "Lefevre P (2022), PLoS Comput Biol 18(10):e1009966 (DOI checked on Crossref)."],
)
stdlib.release(ID, "timeseries", rows, meta, checks, extra_files=files)
