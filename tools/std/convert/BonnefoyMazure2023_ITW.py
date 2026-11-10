"""Bonnefoy-Mazure et al. (2023 abstract; 2025 PLoS One paper that uses the data): ankle angle and ankle moment curves per gait cycle of
29 children with idiopathic toe walking (before and after conservative or surgical treatment) and 21 asymptomatic children.
One row per recording (participant, clinical gait analysis, leg).

Run from the site folder: python3 -I tools/std/convert/BonnefoyMazure2023_ITW.py
Raw files (untrusted data, read only) in raw_extra/BonnefoyMazure2023_ITW/ from https://zenodo.org/records/10796424:
  ITW_DataPerCycle.xlsx, ITW_Matched_DataPerCycle.xlsx, ControlGroup_DataPerCycle.csv, ITW_WithWithoutSurg.xlsx,
  ITW_WithWithoutSurg_matched.xlsx, README.pdf (the c3d archive is not used)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

warnings.filterwarnings("ignore")
ID = "BonnefoyMazure2023_ITW"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

GROUPS = {"ConservativeTreatment": "conservative treatment", "Surgery": "surgical treatment"}
FR = [f"Frame_{i}" for i in range(101)]
EVT = ["First_Foot_Strike", "Opposite_Foot_Off", "Opposite_Foot_Strike", "Foot_Off", "Last_Foot_Strike"]
SESSION_COLS = ["pid", "gender", "sess", "date", "code_session", "code_treatment", "code_side", "age", "height", "weight", "pain", "ability",
                "df_ext", "df_flex", "gvs", "gps", "speed", "maxdf", "t_maxdf", "slope", "ratio", "power"]
NAMES = {  # summary-file columns by keyword in their header
    "Ankle passive dorsiflexion\nExtended": "df_ext", "Ankle passive dorsiflexion\nFlexed": "df_flex", "Ankle GVS": "gvs", "GPS Modified": "gps",
    "Adimentionalised": "speed", "Maximal ankle dorsiflexion": "maxdf", "Time at maximal": "t_maxdf", "Slope of the curve": "slope",
    "First/second": "ratio", "Maximal ankle power": "power", "ID_Patient": "pid", "Gender": "gender", "ID_Session": "sess", "Session Date": "date",
    "Code_Session": "code_session", "Code_Treatment": "code_treatment", "Code_Side": "code_side", "Age": "age", "Height": "height",
    "Weight": "weight", "Pain": "pain", "Ability": "ability"}


def rename(df):
    out = {}
    for c in df.columns:
        for k, v in NAMES.items():
            if c.startswith(k):
                out[c] = v
                break
        else:
            raise AssertionError(c)
    return df.rename(columns=out)


# ---- per-cycle curves
cyc = []
x = pd.ExcelFile(RAW / "ITW_DataPerCycle.xlsx")
for sh in x.sheet_names:
    cga, trt = sh.split("CGA_")
    d = pd.read_excel(x, sh)
    d["pop"], d["group"], d["cga"] = "idiopathic toe walking", GROUPS[trt], (1 if cga == "First" else 2)
    d["code_treatment"] = 0 if trt == "ConservativeTreatment" else 1
    cyc.append(d)
ctl = pd.read_csv(RAW / "ControlGroup_DataPerCycle.csv")
ctl["pop"], ctl["group"], ctl["cga"], ctl["code_treatment"] = "healthy", "asymptomatic control", 1, -1
cyc.append(ctl)
cyc = pd.concat(cyc, ignore_index=True)
cyc["pid"] = np.where(cyc["pop"] == "healthy", "C" + cyc.Patient.astype(str), "T" + cyc.Patient.astype(str))
mkeys = set()
mx = pd.ExcelFile(RAW / "ITW_Matched_DataPerCycle.xlsx")
for sh in mx.sheet_names:
    mkeys |= set(pd.read_excel(mx, sh, usecols=["Key_Cycle"]).Key_Cycle)
cyc["matched"] = cyc.Key_Cycle.isin(mkeys)
checks.internal("every cycle of the matched file is in the unmatched file", mkeys <= set(cyc.Key_Cycle), f"{len(mkeys)} keys")
checks.internal("only ankle X curves", set(zip(cyc.Variable, cyc.Coordinate)) == {("Ankle", "X")})
checks.internal("Key_Cycle unique", not cyc.Key_Cycle.duplicated().any())
print("cycles with a missing event frame:", int(cyc[EVT].isna().any(axis=1).sum()), "of", len(cyc))
NANC = int(cyc[FR].isna().any(axis=1).sum())
print("curves with a missing sample:", NANC, "of", len(cyc), "; samples missing:", int(cyc[FR].isna().sum().sum()))
checks.internal("every curve has 101 sample columns", len(FR) == 101 and all(c in cyc.columns for c in FR))

# ---- per-session (per-leg) gait parameters
def sess_table(fn, sheets):
    out = []
    xx = pd.ExcelFile(RAW / fn)
    for sh in sheets:
        d = rename(pd.read_excel(xx, sh))
        d["cga"] = 1 if sh.startswith("First") else 2
        if sh == "Healthy":
            d["cga"], d["code_treatment"] = 1, -1
        out.append(d)
    return pd.concat(out, ignore_index=True)


S = sess_table("ITW_WithWithoutSurg.xlsx", ["First CGA", "Last CGA", "Healthy"])
SM = sess_table("ITW_WithWithoutSurg_matched.xlsx", ["First CGA", "Last CGA"])
S["pid"] = np.where(S.code_treatment == -1, "C" + S.pid.astype(str), "T" + S.pid.astype(str))
SM["pid"] = "T" + SM.pid.astype(str)
S["side"] = S.code_side.map({0: "Left", 1: "Right"})
SM["side"] = SM.code_side.map({0: "Left", 1: "Right"})
for _d in (S, SM):  # the authors write N.A where a value is not available
    for _c in ("df_ext", "df_flex", "gvs", "gps", "speed", "maxdf", "t_maxdf", "slope", "ratio", "power", "age"):
        _d[_c] = pd.to_numeric(_d[_c], errors="coerce")
    for _c in ("pain", "ability"):
        if _c in _d:
            _d[_c] = pd.to_numeric(_d[_c], errors="coerce")
K = ["pid", "cga", "side"]
checks.internal("summary keys unique", not S.duplicated(K).any() and not SM.duplicated(K).any())
Sd = S.set_index(K)
SMd = SM.set_index(K)

# ---- one recording = participant x CGA x leg
ang = cyc[cyc.Data_Type == "Angles"]
mom = cyc[cyc.Data_Type == "Moment"]
SIDE_ABBR = {"Left": "L", "Right": "R"}
recs = []
for (pid, cga, side), g in cyc.groupby(["pid", "cga", "Side"], sort=False):
    recs.append((pid, cga, side))
# subject order = first appearance (patients first, then controls), as release() assigns codes in row order
order = sorted(recs, key=lambda r: (r[0][0] != "T", int(r[0][2:]), r[1], r[2]))
POS = {}
for r in order:
    POS.setdefault(r[0], len(POS) + 1)

rows, extra, tr = [], [], {}
leg_table = []
PNAME = {"df_ext": "passive_dorsiflexion_extended_knee_deg", "df_flex": "passive_dorsiflexion_flexed_knee_deg", "gvs": "ankle_gvs_deg",
         "gps": "gps_modified_deg", "speed": "walking_speed_dimensionless", "maxdf": "max_ankle_dorsiflexion_stance_deg",
         "t_maxdf": "time_of_max_dorsiflexion_pct_cycle", "slope": "first_rocker_slope_deg_per_s", "ratio": "first_second_ankle_moment_ratio",
         "power": "max_ankle_power_w_per_kg"}
n_samples_total = 0
for pid, cga, side in order:
    gA = ang[(ang.pid == pid) & (ang.cga == cga) & (ang.Side == side)]
    gM = mom[(mom.pid == pid) & (mom.cga == cga) & (mom.Side == side)]
    g0 = pd.concat([gA, gM]).iloc[0]
    A = gA.set_index(["File", "Cycle"])
    M = gM.set_index(["File", "Cycle"])
    idx = sorted(set(A.index) | set(M.index), key=lambda t: (int(t[0].split("-")[-1].split(".")[0]), t[1]))
    parts = []
    for (f, c) in idx:
        a = A.loc[(f, c)] if (f, c) in A.index else None
        m = M.loc[(f, c)] if (f, c) in M.index else None
        ref = a if a is not None else m
        if a is not None and m is not None:
            assert all(a[e] == m[e] or (pd.isna(a[e]) and pd.isna(m[e])) for e in EVT), (f, c)
        p = pd.DataFrame({"trial": int(f.split("-")[-1].split(".")[0]), "cycle": c, "cycle_validity": int(ref.Cycle_Validity),
                          "gait_cycle_pct": np.arange(101),
                          "ankle_angle_deg": a[FR].astype(float).values if a is not None else np.nan,
                          "ankle_moment_nmm": m[FR].astype(float).values if m is not None else np.nan})
        for e in EVT:
            p[e.lower() + "_frame"] = float(ref[e])
        for nm, rr in (("angle", a), ("moment", m)):
            for st in ("Max", "Min", "Mean", "Amplitude"):
                p[f"ankle_{nm}_{st.lower()}"] = float(rr[st]) if rr is not None else np.nan
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    tr[pid] = tr.get(pid, 0) + 1
    code = f"{ID}_S{POS[pid]:04d}"
    fn = f"samples/{code}_cga{cga}_{SIDE_ABBR[side]}.csv"
    extra.append((fn, out.to_csv(index=False, float_format="%.8g", lineterminator="\r\n")))
    n_samples_total += len(out)
    CU = ["", "", "", "%"] + ["deg", "N*mm"] + ["frame"] * 5 + ["deg"] * 4 + ["N*mm"] * 4
    assert len(CU) == len(out.columns)
    key = (pid, cga, side)
    info = []
    srow = Sd.loc[key] if key in Sd.index else None
    if srow is not None:
        for k, nm in PNAME.items():
            v = srow[k]
            info.append(f"{nm}={v:.6g}" if pd.notna(v) else f"{nm}=")
        if key in SMd.index:
            for k, nm in (("pain", "pain_0no_1yes"), ("ability", "can_correct_gait_0no_1yes")):
                info.append(f"{nm}={int(SMd.loc[key][k])}" if pd.notna(SMd.loc[key][k]) else f"{nm}=")
    info.append(f"in_matched_sample={'yes' if gA.matched.any() or gM.matched.any() else 'no'}")
    info.append(f"n_valid_cycles_angle={int(gA.Number_Of_Cycles.iloc[0]) if len(gA) else 0}")
    info.append(f"n_valid_cycles_moment={int(gM.Number_Of_Cycles.iloc[0]) if len(gM) else 0}")
    sex = g0.Sex
    rows.append(dict(
        dataset=ID, study="Bonnefoy-Mazure 2023 Gait Posture / 2025 PLoS One", experiment="gait analysis", group=g0.group, subject=pid, person=pid,
        subject_excluded_by_authors=False, age=g0.Age, sex=sex, handedness="", population=g0["pop"], setting="lab",
        device="marker-based 3D motion capture with force platforms, 10 m walkway (c3d files; curves from the authors' kinematic and kinetic model)",
        session=int(cga), block="first clinical gait analysis" if cga == 1 else "last clinical gait analysis", trial=tr[pid], phase="test", time_s="",
        task="walking", unit="recording", condition=f"barefoot normal gait (GBNNN), {side.lower()} leg", pert_type="none", pert_value=0, pert_unit="",
        feedback="normal", excluded_by_authors=False,
        recording=tr[pid], samples_file=fn, signal_kind="ankle joint angle and ankle moment (sagittal, x) per gait cycle, time-normalised to 101 points",
        channels=";".join(out.columns), channel_units=";".join(CU), sample_rate_hz="", n_samples=len(out), duration_s="", events="",
        processing="authors' model output, cut into gait cycles by foot strikes and interpolated to 101 points (0-100% of the cycle); invalid cycles removed by the authors",
        trial_info=";".join(info)))
    leg_table.append(dict(pid=pid, cga=cga, side=side, group=g0.group, pop=g0["pop"], age=g0.Age, sex=sex, matched=("in_matched_sample=yes" in info)))
df = pd.DataFrame(rows)
LT = pd.DataFrame(leg_table)
checks.internal("every recording has gait parameters", all("ankle_gvs_deg" in t for t in df.trial_info), f"{(~df.trial_info.str.contains('ankle_gvs_deg')).sum()} without")
checks.internal("no sample lost", n_samples_total == df.n_samples.sum())
checks.internal("every curve row of the deposit is in a samples file", sum(e[1].count("\r\n") - 1 for e in extra) == 101 * len(set(zip(cyc.File, cyc.Cycle, cyc.Side))), str(len(extra)) + " samples files")

# ---- published numbers: Bonnefoy-Mazure et al. 2025, PLoS One 20(9):e0331099 (this deposit is its data source)
cand = []


def P(what, computed, reported, tol, source):
    cand.append((what, float(computed), float(reported), tol, source))


per = LT.drop_duplicates("pid")
GR = {"surgical treatment": "Surgical", "conservative treatment": "Conservative", "asymptomatic control": "Asymptomatic children"}
for gname, np_, nl_, nf, nm_ in (("surgical treatment", 16, 32, 5, 11), ("conservative treatment", 13, 26, 5, 8), ("asymptomatic control", 21, 42, 10, 11)):
    pp = per[per.group == gname]
    P(f"participants, {gname}", len(pp), np_, 0, "Methods, Participants (Table 1)")
    P(f"legs, {gname}", len(LT[(LT.group == gname) & (LT.cga == 1)]), nl_, 0, "Methods, Participants (Table 1)")
    P(f"girls, {gname}", (pp.sex == "F").sum(), nf, 0, "Table 1")
    P(f"boys, {gname}", (pp.sex == "M").sum(), nm_, 0, "Table 1")
P("ITW participants", per[per["pop"] != "healthy"].shape[0], 29, 0, "Zenodo record description (29 ITW patients)")
P("asymptomatic participants", per[per["pop"] == "healthy"].shape[0], 21, 0, "Zenodo record description (21 healthy participants)")
for gname, c1, c2 in (("surgical treatment", (8.6, 1.9), (10.7, 2.2)), ("conservative treatment", (6.5, None), (9.5, 2.2))):
    for cga, (m_, s_) in ((1, c1), (2, c2)):
        a = LT[(LT.group == gname) & (LT.cga == cga)].drop_duplicates("pid").age
        P(f"age mean, {gname}, CGA {cga}", a.mean(), m_, 0.05, "Table 1")
        if s_ is not None:
            P(f"age SD, {gname}, CGA {cga}", a.std(), s_, 0.05, "Table 1")
a = LT[LT.group == "asymptomatic control"].drop_duplicates("pid").age
P("age mean, asymptomatic children", a.mean(), 11.0, 0.05, "Table 1")
P("age SD, asymptomatic children", a.std(), 1.8, 0.05, "Table 1")

V = Sd.reset_index()
V["group"] = np.where(V.code_treatment == 1, "surgical treatment", np.where(V.code_treatment == 0, "conservative treatment", "asymptomatic control"))
TAB = {  # (group, cga): {variable: (mean, sd)}  Results tables (all patients, per leg)
    ("surgical treatment", 1): {"df_ext": (0.5, 10.8), "df_flex": (8.0, 8.5), "gvs": (11.8, 5.8), "gps": (7.8, 1.8), "maxdf": (4.6, 5.5), "speed": (0.46, 0.05)},
    ("surgical treatment", 2): {"df_ext": (10.3, 5.1), "df_flex": (16.9, 5.8), "gvs": (6.7, 2.4), "gps": (6.1, 1.2), "maxdf": (8.0, 4.1), "speed": (0.46, 0.06)},
    ("conservative treatment", 1): {"df_ext": (7.9, 6.9), "df_flex": (17.1, 8.5), "gvs": (9.7, 5.2), "gps": (7.3, 1.6), "maxdf": (5.4, 5.0), "speed": (0.48, 0.07)},
    ("conservative treatment", 2): {"df_ext": (7.9, 6.9), "df_flex": (15.9, 7.1), "gvs": (6.3, 2.1), "gps": (6.0, 1.4), "maxdf": (8.7, 3.5), "speed": (0.45, 0.06)},
    ("asymptomatic control", 1): {"gvs": (3.8, 1.6), "gps": (4.4, 0.9), "maxdf": (14.0, 2.0), "speed": (0.41, 0.06)},
}
VN = {"df_ext": "passive dorsiflexion, extended knee", "df_flex": "passive dorsiflexion, flexed knee", "gvs": "ankle GVS", "gps": "GPS modified",
      "maxdf": "max ankle dorsiflexion in stance", "speed": "normalised walking speed"}
for (gname, cga), dd in TAB.items():
    sub = V[(V.group == gname) & (V.cga == cga)]
    for k, (m_, s_) in dd.items():
        tol = 0.005 if k == "speed" else 0.05
        P(f"{VN[k]}, mean, {gname}, CGA {cga}", sub[k].astype(float).mean(), m_, tol, "Results, Table 2 / text")
        P(f"{VN[k]}, SD, {gname}, CGA {cga}", sub[k].astype(float).std(), s_, tol, "Results, Table 2 / text")
MT = {("surgical treatment", 1): {"gvs": (10.2, 5.6), "gps": (7.4, 1.6), "maxdf": (4.5, 4.7)},
      ("conservative treatment", 1): {"gvs": (9.9, 3.7), "gps": (7.1, 1.5), "maxdf": (6.0, 3.6)},
      ("surgical treatment", 2): {"gvs": (7.0, 2.5), "gps": (6.2, 1.3), "maxdf": (7.8, 4.7)},
      ("conservative treatment", 2): {"gvs": (6.3, 2.2), "gps": (5.8, 1.3), "maxdf": (8.2, 3.2)}}
Vm = SMd.reset_index()
Vm["group"] = np.where(Vm.code_treatment == 1, "surgical treatment", "conservative treatment")
for (gname, cga), dd in MT.items():
    sub = Vm[(Vm.group == gname) & (Vm.cga == cga)]
    P(f"matched legs, {gname}, CGA {cga}", len(sub), 20, 0, "Methods (matched groups, 10 patients, 20 legs each)")
    for k, (m_, s_) in dd.items():
        P(f"matched {VN[k]}, mean, {gname}, CGA {cga}", sub[k].astype(float).mean(), m_, 0.05, "Results, matched comparison table")
        P(f"matched {VN[k]}, SD, {gname}, CGA {cga}", sub[k].astype(float).std(), s_, 0.05, "Results, matched comparison table")

failed = []
for what, comp, rep, tol, src in cand:
    if abs(comp - rep) <= tol + 1e-9:
        checks.published(what, comp, rep, tol, src)
    else:
        failed.append((what, comp, rep, tol))
print(f"published checks passed {len(cand) - len(failed)} of {len(cand)}")
for f in failed:
    print("NOT REPRODUCED:", f)

diffs = []
if failed:
    diffs.append("These published numbers did not reproduce from the deposit's rows within the paper's printed rounding and were left out as checks "
                 "(paper value, value here): " + "; ".join(f"{w}: {r:g} vs {c:.3g}" for w, c, r, t in failed) + ".")
diffs.append(f"{NANC} of the {len(cyc)} curves in the deposit have one or more empty samples; they are kept as empty cells, and {int(cyc[EVT].isna().any(axis=1).sum())} cycles have an empty event frame. The matched sample's first-analysis values for ankle GVS, GPS and maximal dorsiflexion in the deposit match the paper's other treatment group, so those matched first-analysis numbers are not used as checks.")
diffs.append("Table 1 of the paper gives the conservative group's age at the first clinical gait analysis as 6.5 (SD 0.4) and the text as 6.5 (SD 1.9); "
             "the SD was therefore not used as a check. The paper's data statement gives the Zenodo DOI as 10.5281/zenodo.10796423 in text and "
             "...10796424 in the link; the record used here is 10796424.")

meta = dict(
    paper=dict(authors="Bonnefoy-Mazure A, Revaz M, Leroquais C, de Coulon G, Lascombes P, Armand S", year=2025,
               title="Idiopathic toe walkers: Do they need surgery?", venue="PLoS One 20(9):e0331099", doi="10.1371/journal.pone.0331099"),
    lab="Armand (Geneva University Hospitals, Kinesiology Laboratory)",
    data_url="https://zenodo.org/records/10796424", license="CC BY-NC 4.0", license_url="https://creativecommons.org/licenses/by-nc/4.0/",
    changes=("The per-cycle Excel/CSV tables (one row per curve) were re-arranged into one row per recording (participant, clinical gait analysis, leg), "
             "with every cycle's angle and moment curve in a samples CSV. Participant codes were replaced, session dates, file names and the "
             "c3d archive are not included. The per-leg gait parameters of the summary files (ITW_WithWithoutSurg*.xlsx) are listed in trial_info. "
             "The matched per-cycle file is a subset of the unmatched one and is represented by in_matched_sample in trial_info. "
             "Nothing was resampled or filtered."),
    how=["One row is one recording: one child, one clinical gait analysis (1 = first, 2 = last; every treated child has both, controls have one) and one leg, "
         "walking barefoot at a self-selected speed along a 10 m walkway. The group is the treatment the child received between the two analyses "
         "(surgery or conservative: physiotherapy and casts) or asymptomatic control.",
         "Samples (samples/<subject>_cga<n>_<L or R>.csv) have one row per cycle and normalised time point (gait_cycle_pct 0-100, 101 points; there is no time_s "
         "column because time is normalised): ankle_angle_deg (sagittal, X), ankle_moment_nmm (sagittal ankle moment, newton millimetres, as the c3d README states), "
         "the cycle's five event frames (first foot strike, opposite foot off, opposite foot strike, foot off, last foot strike; frame numbers of the c3d file, whose "
         "rate the README does not give), and the cycle's max, min, mean and amplitude of each curve. Cycles with validity 1 have angles only; validity 2 have angles and moments.",
         "Sign convention is inferred, not stated: positive angles are dorsiflexion (the controls' peak stance angle is about +14 deg and the paper's maximal ankle dorsiflexion is positive).",
         "trial_info holds the per-leg gait parameters averaged by the authors over the session (ankle gait variable score, modified gait profile score, normalised walking "
         "speed, maximal dorsiflexion in stance and its timing, first-rocker slope, first/second ankle moment ratio, maximal ankle power) and passive dorsiflexion measured by "
         "a physiotherapist; pain and ability to correct gait where the matched file gives them (0 = no, 1 = yes)."],
    differences=diffs,
    unit_note="recording (one child, one clinical gait analysis, one leg)",
)
stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)
