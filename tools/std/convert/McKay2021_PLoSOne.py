"""McKay et al. (2021) PLoS ONE 16(5):e0252119: support-surface translations in 12 directions (backward 240/270/300 and forward 60/90/120 deg
analysed) given to 44 people with Parkinson's disease, 18 older adults without PD and 6 young adults; per-participant averages (over repeated
translations of one direction) of platform position, centre-of-mass kinematics and EMG of tibialis anterior and medial gastrocnemius, plus the
authors' sensorimotor-response-model regressors; one table of model parameters and clinical scores per participant.
Deposit: Harvard Dataverse doi:10.7910/DVN/2XIUFP (CC BY 4.0): S1.tab (participants) and S2.mat (MATLAB 7.3 / HDF5 table 'mckay2020').
Run from the site folder: python3 -I tools/std/convert/McKay2021_PLoSOne.py"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "McKay2021_PLoSOne"
raw = stdlib.raw_dir(DS)
ck = stdlib.Checks()
S1 = pd.read_csv(raw / "S1.tab", sep="\t")
f = h5py.File(raw / "S2.mat", "r")
R = f["#refs#"]
G = R["g"]


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


names = [("".join(chr(c) for c in f[x][...].ravel())) for x in R["C"][:, 0]]
assert names == ["patient", "pertdir", "atime", "LVDT_Y", "COMPosminusLVDT_Y", "COMVelo_Y", "COMAccel_Y", "mus", "e", "side", "dir",
                 "d", "v", "a", "aPrime", "vPrime", "dPrime"], names
patient, mus, side, bf = strings(R["c"][...]), strings(R["d"][...]), strings(R["e"][...]), strings(R["f"][...])
pertdir = R["i"][0].astype(int)
N = len(patient)
assert N == len(mus) == len(side) == len(bf) == len(pertdir) == 1612
col = {nm: f[G[i, 0]][...] for i, nm in enumerate(names) if i in (2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16)}
atime = col["atime"]  # (samples, rows): HDF5 order is the reverse of MATLAB's (rows, samples)
assert atime.shape == (1620, N) and np.allclose(atime[:, 0], atime[:, -1])
tvec = atime[:, 0]
assert abs(tvec[0] + 0.5) < 1e-9 and abs(1 / np.diff(tvec).mean() - 1080) < 1
CH = [("LVDT_Y", "LVDT_Y", "cm"), ("COMPosminusLVDT_Y", "COMPosminusLVDT_Y", "cm"), ("COMVelo_Y", "COMVelo_Y", "cm/s"),
      ("COMAccel_Y", "COMAccel_Y", "g (inferred; the deposit states no unit)"), ("e", "EMG", "normalized units (maximum 80-425 ms after onset = 1)"),
      ("d", "d_regressor", "cm (as shared)"), ("v", "v_regressor", "cm/s (as shared)"), ("a", "a_regressor", "g (inferred)"),
      ("aPrime", "aPrime_regressor", "g (inferred)"), ("vPrime", "vPrime_regressor", "cm/s (as shared)"), ("dPrime", "dPrime_regressor", "cm (as shared)")]
MUS = {"TA": "tibialis anterior", "MGAS": "medial gastrocnemius"}
SIDE = {"L": "left", "R": "right"}
grp = S1.set_index("Patient")
POP = {"PD": "Parkinson's disease (tested OFF medication)", "Non-PD": "older adults without PD", "Young": "healthy young adults"}
SEX = {"Female": "F", "Male": "M"}
seq, rows, files = {}, [], []
for i in range(N):
    p = patient[i]
    g = grp.loc[p]
    seq[p] = seq.get(p, 0) + 1
    k = seq[p]
    fn = f"samples/{p}_rec{k:02d}.csv"
    fn = fn.replace(p, "P" + format(sorted(grp.index).index(p) + 1, "03d"))  # file names carry no original codes
    fb = "forward" if bf[i] == "F" else "backward"
    rows.append(dict(
        dataset=DS, study="McKay et al. 2021 PLoS ONE", experiment="", group=g["Study Group"], subject=p, person=p, subject_excluded_by_authors=False,
        age=float(g["Age"]), sex=SEX[g["Sex"]], handedness="", time_s="", population=POP[g["Study Group"]], setting="lab",
        device="translating platform with LVDT; motion capture (120 Hz); surface EMG (1080 or 1200 Hz)", session=1, block="", trial=k, phase="test",
        task="balance and posture", unit="recording", condition=f"{MUS[mus[i]]}, {SIDE[side[i]]} leg, {fb} translation {pertdir[i]} deg",
        pert_type="support_surface_translation", pert_value=float(pertdir[i]), pert_unit="deg (direction of translation)", feedback="normal",
        excluded_by_authors=False, recording=k, samples_file=fn,
        signal_kind=(f"average over repeated translations of one direction: platform position (LVDT), centre-of-mass position relative to the platform, "
                     f"velocity and acceleration (translation axis), EMG of the {SIDE[side[i]]} {MUS[mus[i]]}, and the authors' model regressor signals"),
        channels=";".join(c[1] for c in CH), channel_units=";".join(c[2] for c in CH), sample_rate_hz=1080, n_samples=len(tvec),
        duration_s=float(tvec[-1] - tvec[0]), events="perturbation_onset=0",
        processing=("as shared: trial average per direction; EMG high-pass filtered at 35 Hz, rectified and normalized (paper); "
                    "all signals on a common 1080 Hz time base; empty cells are NaN in the deposit"),
        trial_info=f"muscle={mus[i]};leg={side[i]};direction_group={fb}"))
    files.append((fn, i))
df = pd.DataFrame(rows)


def fmt(a):
    return ",".join("" if not np.isfinite(x) else format(x, ".7g") for x in a)


def extra():
    for fn, i in files:
        a = np.column_stack([col[c[0]][:, i] for c in CH])
        yield fn, "time_s," + ",".join(c[1] for c in CH) + "\n" + "".join(f"{tvec[j]:.5f}," + fmt(a[j]) + "\n" for j in range(len(tvec)))


KEEP = [("Age", "age_years", "number", "years", "Age."),
        ("PD Duration", "pd_duration_years", "number", "years", "Years since PD diagnosis (PD only)."),
        ("height_cm", "height_cm", "number", "cm", "Height."), ("weight_kg", "weight_kg", "number", "kg", "Weight."),
        ("moca_total", "moca_total", "number", "points", "Montreal Cognitive Assessment total."),
        ("num_falls_6_mo", "falls_previous_6_months", "integer", "count", "Falls in the previous 6 months (empty where not recorded)."),
        ("hy", "hoehn_yahr_stage", "number", "stage", "Hoehn and Yahr stage (PD only)."),
        ("mds_updrs_iii", "mds_updrs_iii_total", "number", "points", "MDS-UPDRS part III total (PD only, OFF medication)."),
        ("minibestest_total", "minibestest_total", "number", "points of 28", "Mini-BESTest total."),
        ("bbs_56", "berg_balance_scale", "number", "points of 56", "Berg Balance Scale."),
        ("fab_40", "fullerton_advanced_balance", "number", "points of 40", "Fullerton Advanced Balance scale."),
        ("dgi_24", "dynamic_gait_index", "number", "points of 24", "Dynamic Gait Index."),
        ("gait_speed_m_s", "gait_speed_m_s", "number", "m/s", "Gait speed (PD only)."),
        ("freezer", "freezer", "integer", "1 = freezer, 0 = not", "Freezing of gait status (PD only)."),
        ("TAka", "TA_ka", "number", "model units", "Sensorimotor response model (SRM) fit to tibialis anterior EMG: acceleration gain (as shared)."),
        ("TAkv", "TA_kv", "number", "model units", "SRM tibialis anterior velocity gain."),
        ("TAkd", "TA_kd", "number", "model units", "SRM tibialis anterior displacement gain."),
        ("TAlambda", "TA_delay_s", "number", "s", "SRM tibialis anterior delay (stabilizing pathway)."),
        ("TAkaPrime", "TA_ka_prime", "number", "model units", "SRM tibialis anterior antagonist-pathway acceleration gain (k_a')."),
        ("TAkvPrime", "TA_kv_prime", "number", "model units", "SRM tibialis anterior antagonist-pathway velocity gain."),
        ("TAkdPrime", "TA_kd_prime", "number", "model units", "SRM tibialis anterior antagonist-pathway displacement gain."),
        ("TAlambdaPrime", "TA_delay_prime_s", "number", "s", "SRM tibialis anterior antagonist-pathway delay."),
        ("TAkaAg", "TA_ka_Ag", "number", "model units", "Deposit column TAkaAg: acceleration gain of a second tibialis anterior fit (suffix Ag not defined in the deposit)."),
        ("TAkvAg", "TA_kv_Ag", "number", "model units", "Deposit column TAkvAg (see TA_ka_Ag)."),
        ("TAkdAg", "TA_kd_Ag", "number", "model units", "Deposit column TAkdAg (see TA_ka_Ag)."),
        ("TAlambdaAg", "TA_delay_Ag_s", "number", "s", "Deposit column TAlambdaAg (see TA_ka_Ag)."),
        ("MGkaAg", "MG_ka_Ag", "number", "model units", "Deposit column MGkaAg: acceleration gain of the medial gastrocnemius fit."),
        ("MGkvAg", "MG_kv_Ag", "number", "model units", "Deposit column MGkvAg (see MG_ka_Ag)."),
        ("MGkdAg", "MG_kd_Ag", "number", "model units", "Deposit column MGkdAg (see MG_ka_Ag)."),
        ("MGlambdaAg", "MG_delay_Ag_s", "number", "s", "Deposit column MGlambdaAg: delay of the medial gastrocnemius fit.")]
pcols = [(b, c, d, e) for a, b, c, d, e in KEEP]
part = []
for p, r in grp.iterrows():
    d = dict(person=p)
    for a, b, c, dd, e in KEEP:
        v = r[a]
        d[b] = None if pd.isna(v) else (int(v) if c == "integer" else float(v))
    part.append(d)

# published checks (expected values from the paper's text)
G_ = S1["Study Group"]
old = S1[G_ != "Young"]
yng = S1[G_ == "Young"]
ck.published("people with Parkinson's disease", (G_ == "PD").sum(), 44, 0, "Results: Participants (N = 44 PD)")
ck.published("older adults without PD", (G_ == "Non-PD").sum(), 18, 0, "Results: Participants (N = 18 Non-PD)")
ck.published("young adults", len(yng), 6, 0, "Results: Participants (6 young, 4 female)")
ck.published("female young adults", (yng["Sex"] == "Female").sum(), 4, 0, "Results: Participants")
ck.published("age of young adults, mean", yng["Age"].mean(), 21, 0.5, "Results: Participants (21 +/- 2 years)")
ck.published("age of young adults, SD", yng["Age"].std(), 2, 0.5, "Results: Participants")
ck.published("PD duration, mean years", S1["PD Duration"].mean(), 7.4, 0.1, "Results: Participants (7.4 +/- 4.8 years)")
ck.published("PD duration, SD years", S1["PD Duration"].std(), 4.8, 0.1, "Results: Participants")
ck.published("weight of young adults, mean kg", yng["weight_kg"].mean(), 61, 0.6, "Results: Participants (61 +/- 12 vs 77 +/- 14 kg)")
ck.published("weight of older adults, mean kg", old["weight_kg"].mean(), 77, 0.6, "Results: Participants")
ck.published("weight of young adults, SD kg", yng["weight_kg"].std(), 12, 0.6, "Results: Participants")
ck.published("TA-TA stabilizing delay, mean ms (62 older adults)", old["TAlambda"].mean() * 1000, 129, 1, "Results: delays (129 +/- 20 ms, N = 62)")
ck.published("TA-TA stabilizing delay, SD ms", old["TAlambda"].std() * 1000, 20, 1, "Results: delays")
ck.published("MG-MG stabilizing delay, mean ms", old["MGlambdaAg"].mean() * 1000, 143, 1, "Results: delays (143 +/- 19 ms)")
ck.published("MG-TA destabilizing delay, mean ms", old["TAlambdaPrime"].mean() * 1000, 183, 1, "Results: delays (183 +/- 21 ms)")
ck.published("older adults, delay lambda, mean ms", old["TAlambda"].mean() * 1000, 129, 1, "Results: age effect (129 vs 151 ms)")
ck.published("young adults, delay lambda, mean ms", yng["TAlambda"].mean() * 1000, 151, 1, "Results: age effect (129 vs 151 ms)")
ck.published("young adults, delay lambda', mean ms", yng["TAlambdaPrime"].mean() * 1000, 136, 1, "Results: age effect (183 vs 136 ms)")
ck.published("k_a' of PD relative to Non-PD, percent", (S1[G_ == "PD"]["TAkaPrime"].mean() / S1[G_ == "Non-PD"]["TAkaPrime"].mean() - 1) * 100, 95, 6,
             "Results: TA antagonist gain (+95%, model-based; group means here)")
pd_f = S1[(G_ == "PD") & S1["num_falls_6_mo"].notna()]
ck.published("people with PD who fell in the previous 6 months, percent", (pd_f["num_falls_6_mo"] > 0).mean() * 100, 47, 1.5, "Results: falls (47% vs 12%)")
ck.published("PD falls per 6 months, maximum", S1["num_falls_6_mo"].max(), 180, 0, "Results: fall frequency (maximum 180)")
ck.internal("one row per table row of the deposit", len(df) == 1612 == N)
ck.internal("all 68 participants have rows", df.person.nunique() == 68 == len(S1))
ck.internal("each sample file has 1620 samples", bool((df.n_samples == 1620).all()))
ck.internal("rows per participant at most 24 (6 directions x 2 muscles x 2 legs)", int(df.groupby("person").size().max()) <= 24)

meta = dict(
    paper=dict(authors="McKay JL, Lang KC, Bong SM, Hackney ME, Factor SA, Ting LH", year=2021,
               title="Abnormal center of mass feedback responses during balance: A potential biomarker of falls in Parkinson's disease",
               venue="PLoS ONE 16(5):e0252119", doi="10.1371/journal.pone.0252119"),
    lab="Ting (Emory University)", data_url="https://doi.org/10.7910/DVN/2XIUFP", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Kept every row of the deposit's MATLAB table (1612 participant x direction x muscle x leg averages of 68 participants) as one CSV of 1620 samples "
             "each (values to 7 significant digits, a time column added), and the participant table S1.tab reduced to demographics, clinical scores and the "
             "model parameters (medication lists, item-level scores and questionnaire items are left out); participants are coded."),
    how=["One row is the average, over the repeated translations of one direction, of one participant's responses for one muscle of one leg (tibialis "
         "anterior or medial gastrocnemius, left or right). Time 0 is the onset of a ramp-and-hold support-surface translation (peak velocity 25 cm/s, "
         "peak displacement 7.5 cm); each CSV runs from -0.5 to about 1.0 s at 1080 Hz. Directions 240, 270, 300 deg are backward and 60, 90, 120 deg forward.",
         "The channels are the platform position (LVDT_Y, cm), the centre of mass relative to the platform (cm), its velocity and acceleration, the normalized "
         "EMG, and six signals the authors call d, v, a, aPrime, vPrime, dPrime (the kinematic regressors of their sensorimotor response model; the "
         "deposit and paper text do not define how they were built, so they are shared as found). The deposit states no units; cm, cm/s and g are inferred "
         "from the paper (platform peak 7.5 cm, 25 cm/s, 0.1 g).",
         "participants.csv holds the age, clinical scores and fitted model parameters per participant (delays in seconds). People with PD were tested "
         "OFF medication. Rows with a few empty samples are NaN in the deposit (one participant-direction-muscle row per case)."],
    unit_note="participant x direction x muscle x leg average recording (samples in samples/)",
    differences=["The paper reports 12% of non-PD older adults with a previous fall; the deposit gives 3 of 18 (17%). The PD value (47%) reproduces when the one "
                 "person without a recorded fall count is left out.",
                 "The paper reports k_a' as a model-based percent difference (+95%); the ratio of group means here is within a few points of it."],
)
stdlib.release(DS, "timeseries", df, meta, ck, participants=part, participant_columns=pcols, extra_files=extra())
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")
