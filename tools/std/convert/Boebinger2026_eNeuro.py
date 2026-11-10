"""Boebinger et al. (2026) eNeuro 13(4):ENEURO.0423-25.2026: 19 older adults and 17 people with Parkinson's disease (plus 2 PD participants the authors
left out) on a motorized platform, 48 forward/backward support-surface translations at three magnitudes; EEG, EMG, CoM kinematics, forces, joint angles.
Deposit: OSF 56u43 (no license stated), one ZIP: a 2.1 GB MATLAB v7.3 file of tables (trials 'data', participant x direction x magnitude averages
'dataAv', 'dataSD') and the authors' statistics table (xlsx). Released here: the 232 participant x direction x magnitude averages ('dataAv') as
timeseries, with the authors' statistics table as an extra CSV. Single-trial signals (1646 trials) are not included (size); EEG is not in the deposit's tables.
Run from the site folder: python3 -I tools/std/convert/Boebinger2026_eNeuro.py"""
import io
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Boebinger2026_eNeuro"
raw = stdlib.raw_dir(DS)
ck = stdlib.Checks()
z = zipfile.ZipFile(raw / "DataAndCode.zip")
X = pd.read_excel(io.BytesIO(z.read("Data and Code/Data/HOA_PD_SRM_Outputs__24-Apr-2024_StatsTable_wAnalysis.xlsx")))
f = h5py.File(raw / "tables.mat", "r")
R = f["#refs#"]


def strings(a):  # MATLAB string array stored as uint64 words: [1, ndims, dims..., lengths..., UTF-16 chars]
    a = np.asarray(a).ravel().astype("<u8")
    nd = int(a[1])
    n = int(np.prod([int(x) for x in a[2:2 + nd]]))
    lens = a[2 + nd:2 + nd + n].astype(int)
    u = np.frombuffer(a[2 + nd + n:].tobytes(), dtype="<u2")
    out, p = [], 0
    for L in lens:
        out.append("".join(chr(c) for c in u[p:p + L]))
        p += L
    return out


def name(o):
    return "".join(chr(c) for c in f[o][...].ravel())


names = [name(x) for x in R["Fe"][:, 0]]  # variables of dataAv, in order; their data are in the cell R['od']
cells = {n: f[R["od"][i, 0]] for i, n in enumerate(names) if n != "patient"}
patient = strings(R["Vf"][...])
N = len(patient)
assert N == 232 and names[0] == "patient"
pdir = cells["pertdir_calc_round_deg"][...].ravel().astype(int)
mag = cells["condition"][...].ravel()
gcount = cells["GroupCount"][...].ravel().astype(int)
atime = cells["atime"][...][:, 0]
mtime = cells["mtime"][...][:, 0]
assert np.allclose(cells["atime"][...], atime[:, None]) and np.allclose(cells["mtime"][...], mtime[:, None])
fs_a, fs_m = 1 / np.diff(atime).mean(), 1 / np.diff(mtime).mean()

ANALOG_UNIT = {"LVDT": "cm", "COMPos": "cm", "COMVelo": "cm/s", "COMAccel": "g (inferred)", "Fz": "N"}
analog, mocap = [], []
for n in names:
    if n in ("patient", "pertdir_calc_round_deg", "condition", "GroupCount", "atime", "mtime", "FT") or n.endswith("normcoef") or n.startswith("peak_"):
        continue
    shp = cells[n].shape
    if shp[0] == 3000:
        u = ("cm" if n.startswith(("LVDT", "COMPos")) else "cm/s" if "COMVelo" in n else "g (inferred; the deposit states no unit)" if "COMAccel" in n
             else "N" if n.endswith("Fz") else "normalized to 1 = the participant's maximum over trials (side-specific)" if n.endswith("_norm")
             else "as shared (filtered EMG epoch average; unit not stated)")
        analog.append((n, u))
    else:
        assert shp[0] == 300, (n, shp)
        mocap.append((n, "deg" if "_a" in n else "unspecified (named _m in the deposit)"))
SCAL = [n for n in names if n.endswith("normcoef") or n.startswith("peak_")]
assert sum(1 for n in names if n == "FT") == 1 and np.isnan(cells["FT"][...]).all()

Xk = X.set_index(["patient", "pertdir_calc_round_deg", "condition"])
assert Xk.index.is_unique
SEX = {1: "F", 0: "M"}
rows, files = [], []
seq = {}
pidx = {}
for i in range(N):
    p = patient[i]
    pidx.setdefault(p, len(pidx) + 1)
    key = (p, pdir[i], mag[i])
    inx = key in Xk.index
    s = Xk.loc[key] if inx else None
    seq[p] = seq.get(p, 0) + 1
    k = seq[p]
    fb = "forward" if pdir[i] == 90 else "backward"
    pdrow = X[X.patient == p]
    has = len(pdrow) > 0
    isPD = p.startswith("PD")
    group = "Parkinson's disease" if isPD else "older adults"
    scal = ";".join(f"{n}={cells[n][...].ravel()[i]:.6g}" for n in SCAL) if inx else ""
    info = f"averaged_trials={gcount[i]};magnitude_label={mag[i]:g}" + (f";stepping_trials_n={int(s['Num_step'])}" if inx else "") + (";" + scal if scal else "")
    base = dict(dataset=DS, study="Boebinger et al. 2026 eNeuro", experiment="", group=group, subject=p, person=p,
                subject_excluded_by_authors=not has,
                age=float(pdrow.age.iloc[0]) if has else None, sex=SEX[int(pdrow.gender.iloc[0])] if has else None, handedness="",
                population=("Parkinson's disease (tested OFF medication)" if isPD else "older adults without PD"), setting="lab",
                device="motorized platform with AMTI OR6-6 force plates; Vicon Nexus (100 Hz); surface EMG (1000 Hz); EEG (not in the deposit's tables)",
                session=1, block="", trial=k, time_s="", phase="test", task="balance and posture", unit="recording",
                condition=f"{fb} translation, magnitude label {mag[i]:g}", pert_type="support_surface_translation", pert_value=float(mag[i]),
                pert_unit="magnitude label of the deposit (5, 7.5, 10; 12 for two participants)", feedback="normal", excluded_by_authors=not inx,
                recording=k, events="perturbation_onset=0", trial_info=info)
    pn = f"P{pidx[p]:03d}"
    fa, fm = f"samples/{pn}_dir{pdir[i]}_mag{mag[i]:g}_analog.csv", f"samples/{pn}_dir{pdir[i]}_mag{mag[i]:g}_mocap.csv"
    rows.append({**base, "samples_file": fa,
                 "signal_kind": "average over repeated translations of one direction and magnitude: EMG of 12 leg muscles, platform position, centre-of-mass kinematics, vertical ground reaction forces",
                 "channels": ";".join(c[0] for c in analog), "channel_units": ";".join(c[1] for c in analog), "sample_rate_hz": round(fs_a),
                 "n_samples": len(atime), "duration_s": float(atime[-1] - atime[0]),
                 "processing": ("as shared: EMG high-pass 35 Hz, rectified, low-pass 40 Hz, averaged over the trials of the condition (_norm = divided by the "
                                "participant's side-specific maximum); all other channels averaged over the same trials")})
    rows.append({**base, "samples_file": fm,
                 "signal_kind": "average over repeated translations of one direction and magnitude: joint angles and moments of the hips, knees and ankles (motion capture)",
                 "channels": ";".join(c[0] for c in mocap), "channel_units": ";".join(c[1] for c in mocap), "sample_rate_hz": round(fs_m),
                 "n_samples": len(mtime), "duration_s": float(mtime[-1] - mtime[0]),
                 "processing": "as shared; empty cells are NaN in the deposit (the _a2 and _m channels are empty for most rows)"})
    files.append((fa, i, analog, atime))
    files.append((fm, i, mocap, mtime))
df = pd.DataFrame(rows)


def extra():
    for fn, i, chans, tv in files:
        d = pd.DataFrame({"time_s": np.round(tv, 4)})
        for n, _ in chans:
            d[n] = cells[n][:, i]
        yield fn, d.to_csv(index=False, float_format="%.7g", lineterminator="\n")
    pc = {p: f"{DS}_P{j:04d}" for p, j in pidx.items()}
    t = X.drop(columns=["vicon_id"]).copy()
    t["patient"] = t["patient"].map(pc)
    t = t.rename(columns={"patient": "person"})
    yield "tables/condition_measures.csv", t.to_csv(index=False, float_format="%.8g", lineterminator="\r\n")


# participants.csv
PC = [("height_cm", "height_cm", "number", "cm", "Height."), ("weight_kg", "weight_kg", "number", "kg", "Weight."),
      ("moca", "moca_total", "number", "points", "Montreal Cognitive Assessment total."), ("minibest", "minibest_total", "number", "points of 28", "Mini-BESTest total."),
      ("abc", "abc_scale", "number", "score", "Activities-specific Balance Confidence score (deposit column abc)."),
      ("pd_duration", "pd_duration_years", "number", "years", "Years since PD diagnosis (PD only)."),
      ("mds_updrs_iii_hy_video", "hoehn_yahr_stage", "number", "stage", "Hoehn and Yahr stage from video (PD only)."),
      ("mds_updrs_iii_total_video", "mds_updrs_iii_total", "number", "points", "MDS-UPDRS part III total from video (PD only)."),
      ("TrailsA", "trails_a", "number", "as shared", "Trail Making Test A (deposit column TrailsA; unit not stated)."),
      ("TrailsB", "trails_b", "number", "as shared", "Trail Making Test B (deposit column TrailsB; unit not stated)."),
      ("bdi_ii_total", "bdi_ii_total", "number", "points", "Beck Depression Inventory-II total."),
      ("stai_state_score", "stai_state", "number", "points", "State anxiety score."), ("stai_trait_score", "stai_trait", "number", "points", "Trait anxiety score.")]
part = []
for p, g in X.groupby("patient"):
    d = dict(person=p)
    for a, b, *_ in PC:
        v = g[a].iloc[0]
        d[b] = None if pd.isna(v) else float(v)
    part.append(d)
pcols = [(b, c, d, e) for a, b, c, d, e in PC]

# checks
P = X.drop_duplicates("patient")
oa, pdg = P[P.PD == 0], P[P.PD == 1]
ck.published("older adults analysed", len(oa), 19, 0, "README / Methods: Participants (nineteen older adults)")
ck.published("people with PD analysed", len(pdg), 17, 0, "README / Methods: Participants (seventeen individuals with PD)")
ck.published("female older adults", (oa.gender == 1).sum(), 6, 0, "Methods: Participants (6 female)")
ck.published("female people with PD", (pdg.gender == 1).sum(), 4, 0, "Methods: Participants (4 female)")
ck.published("age of older adults, mean years", oa.age.mean(), 71, 0.5, "Methods: Participants (71 years)")
ck.published("age of people with PD, mean years", pdg.age.mean(), 69, 0.5, "Methods: Participants (69 years)")
ck.published("height of older adults, mean cm", oa.height_cm.mean(), 175, 0.5, "Methods: Participants (175 cm)")
ck.published("height of people with PD, mean cm", pdg.height_cm.mean(), 171, 0.5, "Methods: Participants (171 cm)")
ck.published("weight of older adults, mean kg", oa.weight_kg.mean(), 79, 0.5, "Methods: Participants (79 kg)")
ck.published("weight of people with PD, mean kg", pdg.weight_kg.mean(), 84, 0.7, "Methods: Participants (84 kg)")
ck.published("EMG sampling rate, Hz", fs_a, 1000, 1, "Methods: EMG (sampled at 1000 Hz)")
ck.published("motion capture sampling rate, Hz", fs_m, 100, 1, "Methods: CoM kinematics (collected at 100 Hz)")
ck.published("conditions per analysed participant (3 magnitudes x 2 directions)", X.groupby("patient").size().mean(), 6, 0, "Methods: Balance perturbations")
ck.internal("232 averages, two sample files each", len(df) == 464)
ck.internal("38 participants in the deposit, 36 analysed", df.person.nunique() == 38 and X.patient.nunique() == 36)
ck.internal("rows kept: all 216 rows of the authors' table are matched", int((~df.excluded_by_authors).sum()) == 2 * 216)

meta = dict(
    paper=dict(authors="Boebinger SE, Payne AM, Xiao J, Martino G, Borich MR, Lucas McKay J, Ting LH", year=2026,
               title="Cortically Mediated Muscle Responses to Balance Perturbations Increase with Perturbation Magnitude in Older Adults with and without Parkinson's Disease",
               venue="eNeuro 13(4):ENEURO.0423-25.2026", doi="10.1523/eneuro.0423-25.2026"),
    lab="McKay / Ting (Emory University)", data_url="https://osf.io/56u43/", license="none stated", license_url="",
    changes=("Kept all 232 participant x direction x magnitude averages of the deposit's MATLAB table 'dataAv' (38 participants) as CSV files "
             "(values to 7 significant digits, time column added); the authors' statistics table is included as tables/condition_measures.csv with the "
             "participant code in place of the identifiers and the video-ID column dropped. The 1646 single-trial recordings, the standard-deviation table, "
             "the all-empty FT channel and the model code are not included; participants are coded."),
    how=["One row of the main table is one signal file. Each participant x direction x magnitude average has two rows: an analog file (EMG of 12 leg muscles, "
         "platform position, centre-of-mass position/velocity/acceleration, vertical forces; 3000 samples at 1000 Hz) and a motion-capture file (hip, knee and ankle "
         "angles and moments; 300 samples at 100 Hz). Time 0 is platform onset; both run from -0.5 to 2.5 s. Direction 90 is forward and 270 backward (the lab's convention, as in the "
         "companion Parkinson's study); the magnitude label 5, 7.5, 10 is the deposit's 'condition' (small, medium, large; small = 5.1 cm, 11.1 cm/s, 0.15 g). "
         "Two older adults also have a fourth label, 12, in the deposit; the authors' table does not use it.",
         "Units are those the paper implies (the deposit states none): platform and centre of mass in cm, velocity in cm/s, acceleration in g, forces in N, "
         "angles in degrees. Channels ending _norm are the EMG divided by the participant's maximum (normcoef) as the paper describes; the normcoef values and the "
         "authors' peak CoM measures are in trial_info. _a2 and _m channels are mostly empty. Channel names are the authors'; the deposit does not describe Hip_a2, Knee_a2, Ankle_a2 or the _m channels.",
         "Participants PD07 and PD17 (coded) are in the deposit's tables but not in the authors' statistics table: rows are marked excluded, as the paper analysed 17 of the 19 PD participants "
         "recruited (some excluded for a brain tumour or severe neuropathy, or leaving early). Age and sex are empty for them."],
    unit_note="signal file of one participant x direction x magnitude average (samples in samples/)",
    differences=["The paper's methods say participant-condition pairs with fewer than 3 averaged trials were removed; the authors' table still contains rows whose trial count "
                 "(averaged_trials in trial_info) is below 3, and they are kept and not marked excluded here.",
                 "The paper reports single-trial EEG-derived measures (beta, N1); the EEG signals are not in the deposit's tables, only the derived columns in condition_measures.csv."],
)
stdlib.release(DS, "timeseries", df, meta, ck, participants=part, participant_columns=pcols, extra_files=extra())
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
