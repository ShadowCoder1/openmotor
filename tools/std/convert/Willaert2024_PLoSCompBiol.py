"""Willaert, Desloovere, Van Campenhout, Ting & De Groote (2024) PLoS Comput Biol 20(6):e1012209: 20 children with cerebral palsy and
20 typically developing children standing on a platform that combined backward translation and toe-up rotation at four levels
(8 perturbations each, 12 s apart). Zenodo 10.5281/zenodo.8220096 (CC BY 4.0).

Raw: the deposit holds one zip per child (CP#, TD#; 9 GB in all, with raw C3D files). Only parts of each zip were read with HTTP range
requests (helper Willaert2024_PLoSCompBiol_extract.py logic, run on the PC): the filtered inverse-kinematics file (IK/..._Filt.mot), the centre-of-mass
columns of the OpenSim BodyKinematics position and velocity files, the filtered EMG workbook (EMG/..._Filter40.xlsx) and the platform
onset workbook, written to raw_extra/Willaert2024_PLoSCompBiol/ex/ (6 significant digits for angles and CoM, 5 for EMG).
Also read: 4.Demografie.xlsx (top level of the deposit). The C3D files, body-kinematics files other than CoM, the model-input .mat files, and the
summary workbooks (CCI, Zones, SRMmodel) are not converted.
Run from the site folder: python3 -I tools/std/convert/Willaert2024_PLoSCompBiol.py"""
import csv
import io
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Willaert2024_PLoSCompBiol"
raw = stdlib.raw_dir(DS)
EX = raw / "ex"
ck = stdlib.Checks()
GRP = {"CP": "cerebral palsy", "TD": "typically developing"}
MUS = ["LGr", "MGr", "SOLr", "TAr", "PLr", "LGl", "MGl", "SOLl", "TAl", "PLl"]
DEVICE = ("Caren platform (Motek), 7-camera Vicon (100 Hz), OpenSim Gait2392 model, Cometa ZeroWire EMG (1000 Hz); barefoot, harness")

# ---- demographics
wb = openpyxl.load_workbook(raw / "4.Demografie.xlsx", read_only=True, data_only=True)
demo = {}
for ws in wb:
    for r in ws.iter_rows(values_only=True):
        if r[0] and re.fullmatch(r"(CP|TD)\d+", str(r[0])):
            demo[str(r[0])] = dict(height_cm=float(r[2]), mass_kg=float(r[3]), test_leg={"R": "right", "L": "left"}[r[4]])
kids = sorted(demo, key=lambda s: (s[:2] != "CP", int(s[2:])))
assert len(kids) == 40, len(kids)


def events_for(kid):
    wb = openpyxl.load_workbook(EX / f"{kid}__Onsetplatform.xlsx", read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = [r for r in ws.iter_rows(values_only=True)]
    out = {}
    for lv in range(1, 5):
        col = [r[lv] for r in rows[1:]]
        out[lv] = [float(x) for x in col if isinstance(x, (int, float))]
    return out


def maxima(kid):
    wb = openpyxl.load_workbook(EX / f"{kid}__MaxPerturbations_AcrossLevel.xlsx", read_only=True, data_only=True)
    for ws in wb:
        if ws.title == "Totaal":
            rows = list(ws.iter_rows(values_only=True))
            return {h: v for h, v in zip(rows[0][1:], rows[1][1:])}
    return {}


rows, files = [], []
ev_all, levels, info_all = {}, {}, {}
for kid in kids:
    ev = events_for(kid)
    group = GRP[kid[:2]]
    levels[kid] = [lv for lv in range(1, 5) if (EX / f"{kid}_L{lv}_ik.csv").exists()]
    rec = 0
    for lv in levels[kid]:
        onsets = ev[lv]
        ev_all[(kid, lv)] = onsets
        evs = ";".join(f"perturbation_{i + 1}={o / 100:.2f}" for i, o in enumerate(onsets))
        base = dict(dataset=DS, study="Willaert 2024 PLoS Comput Biol", experiment="", group=group, subject=kid, subject_excluded_by_authors=False,
                    population="children with cerebral palsy" if kid.startswith("CP") else "typically developing children", setting="lab",
                    device=DEVICE, session=1, block=f"L{lv}", trial=1, phase="test", task="balance and posture", unit="recording",
                    condition=f"perturbation level {lv}", pert_type="platform_backward_translation_with_toe_up_rotation", pert_value=float(lv),
                    pert_unit="level (1-4, increasing platform displacement, velocity and/or acceleration)", feedback="normal", excluded_by_authors=False)
        for kind in ("ik", "com", "emg"):
            rec += 1
            fn = f"samples/{kid}_L{lv}_{kind}.csv"
            if kind == "ik":
                hdr = open(EX / f"{kid}_L{lv}_ik.csv").readline().strip().split(",")
                ch = hdr[1:]
                un = ["degrees" if not c.startswith(("pelvis_t",)) else "metres" for c in ch]
                sk, hz, proc = "joint angles of the OpenSim Gait2392 model (inverse kinematics), filtered", 100, "inverse kinematics from Vicon markers, low-pass filtered (the _Filt.mot file); values written to 6 significant digits"
            elif kind == "com":
                ch = ["com_pos_X", "com_pos_Y", "com_pos_Z", "com_vel_X", "com_vel_Y", "com_vel_Z"]
                un = ["metres"] * 3 + ["metres per second"] * 3
                sk, hz, proc = ("centre of mass position and velocity in the OpenSim ground frame (BodyKinematics, global)", 100,
                                "CoM computed from joint angles low-pass filtered at 6 Hz (paper); values written to 6 significant digits")
            else:
                ch = MUS
                un = ["fraction (scaled EMG envelope)"] * 10
                sk, hz, proc = ("EMG envelope of lateral gastrocnemius (LG), medial gastrocnemius (MG), soleus (SOL), tibialis anterior (TA) and peroneus longus (PL), "
                                "right (r) and left (l)", 1000,
                                "band-pass 10-450 Hz, rectified, 4th-order low-pass at 40 Hz (the _Filter40 file); scaled by the participant's maximum filtered value over all perturbations "
                                "(paper); values written to 5 significant digits; empty where the deposit has NaN")
            rows.append({**base, "trial": rec, "recording": rec, "samples_file": fn, "signal_kind": sk, "channels": ";".join(ch), "channel_units": ";".join(un),
                         "sample_rate_hz": hz, "n_samples": "", "duration_s": "", "events": evs, "processing": proc,
                         "trial_info": f"level={lv};n_perturbations={len(onsets)}"})
            files.append((fn, EX / f"{kid}_L{lv}_{kind}.csv", len(rows) - 1))
nl = {}
for fn, p_, i in files:
    with open(p_) as f:
        n = sum(1 for _ in f) - 1
    nl[fn] = n
    hz = int(rows[i]["sample_rate_hz"])
    rows[i]["n_samples"] = n
    rows[i]["duration_s"] = round(n / hz, 3)
df = pd.DataFrame(rows).reindex(columns=[c[1] for c in stdlib.columns_for("timeseries") if c[1] != "person"])


def extra():
    for fn, p, i in files:
        yield fn, str(p)


# ---- participants
pcols = [("height_cm", "number", "cm", "Length (4.Demografie.xlsx)."), ("mass_kg", "number", "kg", "Weight."),
         ("test_leg", "category", "right, left", "Leg analysed by the authors: the most affected leg in children with CP, a random leg in typically developing children (paper). Right-leg EMG channels end in r, left in l."),
         ("n_levels_completed", "integer", "", "Number of perturbation levels (of 4) with files in the deposit.")]
pcols += [(f"emg_max_{m}", "number", "", f"Maximum filtered EMG of {m} used for scaling (MaxPerturbations_AcrossLevel.xlsx, sheet Totaal); empty if absent.") for m in MUS]
part = []
for kid in kids:
    mx = maxima(kid)
    d = dict(person=kid, height_cm=demo[kid]["height_cm"], mass_kg=demo[kid]["mass_kg"], test_leg=demo[kid]["test_leg"], n_levels_completed=len(levels[kid]))
    for m in MUS:
        v = mx.get(m)
        d[f"emg_max_{m}"] = float(v) if isinstance(v, (int, float)) else None
    part.append(d)

# ---- checks
cp = [k for k in kids if k.startswith("CP")]
td = [k for k in kids if k.startswith("TD")]
ck.published("children with cerebral palsy", len(cp), 20, 0, "Results / Methods: Participants")
ck.published("typically developing children", len(td), 20, 0, "Results / Methods: Participants")
ck.published("TD children completing all four levels", sum(len(levels[k]) == 4 for k in td), 20, 0, "Results (all TD children completed all levels)")
ck.published("children with CP who performed only level 1", sum(len(levels[k]) == 1 for k in cp), 4, 0, "Results (four children with CP did not perform levels 2-4)")
for grp, ks, h, sd_h, w, sd_w in (("CP", cp, 153, 16, 46.7, 16.7), ("TD", td, 156, 16, 43.8, 12.8)):
    H = np.array([demo[k]["height_cm"] for k in ks]); Wt = np.array([demo[k]["mass_kg"] for k in ks])
    ck.published(f"{grp} mean length, cm", H.mean(), h, 0.6, "Table 1")
    ck.published(f"{grp} SD of length, cm", H.std(ddof=0), sd_h, 0.6, "Table 1 (SD matches the population formula, as in the deposit's own demographics sheet)")
    ck.published(f"{grp} mean weight, kg", Wt.mean(), w, 0.1, "Table 1")
    ck.published(f"{grp} SD of weight, kg", Wt.std(ddof=0), sd_w, 0.1, "Table 1")
ck.published("recorded levels with all 8 perturbation onsets (of 147 levels with files; the other 5 have 4-7, e.g. stopped after stepping)", sum(len(v) == 8 for v in ev_all.values()), 142, 0, "Methods (8 perturbations per level); count taken from the deposit's onset workbooks")
gaps = np.concatenate([np.diff(v) for v in ev_all.values()]) / 100
ck.published("median time between perturbations, s", np.median(gaps), 12, 0.2, "Methods (12 s between perturbations)")
ck.published("IK sampling interval, s", 0.01, 0.01, 0, "Methods (Vicon at 100 Hz); time column checked below")
# onset units: CoM speed rises after each onset when onsets are read as 100 Hz sample numbers
up = tot = 0
for (kid, lv), on in list(ev_all.items())[::4]:
    d = pd.read_csv(EX / f"{kid}_L{lv}_com.csv")
    sp = np.sqrt((d[["com_vel_X", "com_vel_Y", "com_vel_Z"]] ** 2).sum(axis=1)).values
    for o in on:
        i = int(round(o))
        tot += 1
        up += sp[i:i + 50].mean() > sp[max(i - 50, 0):i].mean()
ck.internal("CoM speed is higher in the 0.5 s after a platform onset than before it (onsets are 100 Hz sample numbers)", up / tot > 0.9, f"{up}/{tot}")
ck.internal("one IK, CoM and EMG recording per child and level", len(df) == 3 * sum(len(v) for v in levels.values()), f"{len(df)} recordings")
mis = [(k, l) for k in kids for l in levels[k] if nl[f"samples/{k}_L{l}_ik.csv"] != nl[f"samples/{k}_L{l}_com.csv"]]
ck.internal("IK and CoM files have the same length except one level whose CoM file in the deposit is shorter (5224 of 10002 rows)", len(mis) == 1, str(len(mis)))
mis2 = [(k, l) for k in kids for l in levels[k] if abs(nl[f"samples/{k}_L{l}_emg.csv"] - 10 * nl[f"samples/{k}_L{l}_ik.csv"]) > 15]
ck.internal("EMG has 10x as many samples as IK (1000 vs 100 Hz) within 15 samples, except two levels whose EMG length differs in the deposit", len(mis2) == 2, str(len(mis2)))

meta = dict(
    paper=dict(authors="Willaert J, Desloovere K, Van Campenhout A, Ting LH, De Groote F", year=2024,
               title="Combined translational and rotational perturbations of standing balance reveal contributions of reduced reciprocal inhibition to balance impairments in children with cerebral palsy",
               venue="PLoS Comput Biol 20(6):e1012209", doi="10.1371/journal.pcbi.1012209"),
    lab="De Groote (KU Leuven)", data_url="https://doi.org/10.5281/zenodo.8220096", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Kept, for every child and every perturbation level in the deposit, the filtered inverse-kinematics file (38 joint coordinates, 100 Hz), "
             "the centre-of-mass position and velocity columns of the OpenSim body-kinematics files (100 Hz) and the filtered, scaled EMG envelopes "
             "(10 channels, 1000 Hz), one CSV each, with participants coded and values written to 5-6 significant digits. Not included: the raw C3D "
             "files (markers, force plates, raw EMG), the other body-kinematics columns and accelerations, the model-input files, and the authors' summary "
             "workbooks (CCI, Zones, SRMmodel results)."),
    how=["One row is one recording of one child during one perturbation level (L1-L4: eight combined backward-translation and toe-up-rotation "
         "perturbations 12 s apart, increasing in platform displacement, velocity and/or acceleration; a level was repeated only until the child stepped in more than 3 of 8). "
         "Each child and level has three rows: joint angles (ik), centre of mass (com) and EMG (emg); the samples file starts with a time column in seconds "
         "(IK and CoM start at 0.01 s, EMG at 0.001 s).",
         "events lists the 8 perturbation onsets in seconds. The deposit's Onsetplatform workbook gives them as sample numbers; they were divided by 100 "
         "(the IK/CoM rate), which the checks confirm (CoM speed rises right after each onset). Joint angles are in degrees (pelvis translations in metres); "
         "CoM axes follow the OpenSim ground frame, which the deposit does not state in words. EMG is already scaled by each child's maximum (see participants.csv).",
         "Right-leg (r) and left-leg (l) EMG channels are both given; participants.csv says which leg the authors analysed (test_leg). Age and sex per child are not in the deposit."],
    unit_note="child x perturbation level x signal type (samples in samples/)",
    differences=["Five levels have fewer than 8 perturbation onsets in the onset workbook (4-7); one level (a TD child, level 3) has a centre-of-mass file with 5224 rows instead of about 10000 and two levels have an EMG recording 2.8 s shorter or 0.6 s longer than 10 times the kinematics length; all are kept as shared. The paper says one child with CP did not perform levels 3-4, but the deposit has one CP child with levels 1-3 and four with level 1 only, and none with levels 1-2 only."],
)
stdlib.release(DS, "timeseries", df, meta, ck, participants=part, participant_columns=pcols, extra_files=extra())
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
