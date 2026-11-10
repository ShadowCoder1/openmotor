"""Georgiev, Mongold, Naeije & Bourguignon (2026): beta corticomuscular coherence (CMC) of the right first dorsal interosseous (FDI)
during isometric contraction while the right brachium moves; 36 adults.

Raw file (untrusted data, read only): raw_extra/Georgiev2026_bioRxiv_CMC/CMC_all.xlsx (OSF hcb24, 'CMC Results'); sheet 1 (CMC_EMG2)
has one line per participant. The authors' R script (StatisticalAnlaysesCMC.R) and MATLAB script were read as text, never run.
"""
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Georgiev2026_bioRxiv_CMC"
MODULE = "summary"
f = stdlib.raw_dir(DS) / "CMC_all.xlsx"
ws = openpyxl.load_workbook(f, data_only=True)["CMC_EMG2"]
rows_x = [r for r in ws.iter_rows(values_only=True)]
hdr = list(rows_x[0])
data = [dict(zip(hdr, r)) for r in rows_x[1:] if r[0] is not None]

ck = stdlib.Checks()
ck.internal("36 participants with unique ids", len(data) == 36 and len({d["ID"] for d in data}) == 36)
ck.internal("no missing cells", all(v is not None for d in data for v in d.values()))
ck.internal("CMC values in 0-1", all(0 <= d[k] <= 1 for d in data for k in hdr if k.startswith(("T0_", "T1_", "T2_", "ALL_", "All_"))))
ck.internal("T01 = mean of T0_Force and T1_Force (deduced)", all(abs(d["T01"] - (d["T0_Force"] + d["T1_Force"]) / 2) < 1e-6 for d in data))
ck.internal("Contrast = (T2_Force - T01)/(T2_Force + T01) (deduced)",
            all(abs(d["Contrast"] - (d["T2_Force"] - d["T01"]) / (d["T2_Force"] + d["T01"])) < 1e-6 for d in data))

WIN = {"T0": "window before the movement (-1 to 0 s, matched by order to the paper's Tbefore)", "T1": "window during the movement (0 to 1 s, matched to Tduring)", "T2": "window after the movement (1 to 2 s, matched to Tafter)", "ALL": "all windows"}
DESC = ("Beta-band (13-30 Hz) corticomuscular coherence between sensorimotor EEG (left-hemisphere channels FCz, FC1, FC3, FC5, Cz, C3, C5, CP1, CP3, CP5) "
        "and the {sig} of the isometrically contracted right FDI, in the {w} around the right brachium movement (authors' column {col})")
specs = []
for sig, pre in (("EMG", "EMG"), ("force", "Force")):
    for t in ("T0", "T1", "T2"):
        specs.append((f"{t}_{pre}", f"cmc_{sig.lower()}", t, DESC.format(sig=sig if sig == "force" else "rectified-free EMG", w=WIN[t], col=f"{t}_{pre}"), "value", "coherence"))
specs.append(("ALL_EMG", "cmc_emg", "ALL", DESC.format(sig="EMG", w=WIN["ALL"], col="ALL_EMG"), "value", "coherence"))
specs.append(("All_Force", "cmc_force", "ALL", DESC.format(sig="force", w=WIN["ALL"], col="All_Force"), "value", "coherence"))
specs.append(("Sign", "significant_cmc_emg", "ALL", "1 when the participant's EMG CMC was significant against surrogate data (authors' column Sign), else 0", "value", "0/1"))
specs.append(("Sign_F", "significant_cmc_force", "ALL", "1 when the participant's force CMC was significant against surrogate data (authors' column Sign_F), else 0", "value", "0/1"))
specs.append(("T01", "cmc_force_t0t1_mean", "T0-T1", "Mean of the force CMC in windows T0 and T1 (authors' column T01; equality to mean(T0_Force, T1_Force) verified)", "value", "coherence"))
specs.append(("Contrast", "cmc_force_contrast", "T2 vs T0-T1", "Authors' column Contrast; matches (T2_Force - T01)/(T2_Force + T01) in every participant (deduced, not documented)", "value", "ratio"))

rows = []
for i, d in enumerate(data, 1):
    code = f"S{i:02d}"
    base = dict(dataset=DS, study="Georgiev 2026 corticomuscular coherence", experiment="brachium movement during FDI contraction", group=None,
                subject=code, person=code, subject_excluded_by_authors=False, age=None, sex=None, handedness="right", population="healthy",
                setting="lab", device="EEG, surface EMG and force of right FDI, accelerometer on right arm", session=1, block=None,
                phase="other", time_s=None, task="finger movements", unit="participant summary", pert_type="none", pert_value=None,
                pert_unit=None, feedback="normal", excluded_by_authors=False, n_trials=None,
                session_context="submaximal isometric right FDI contraction while moving the right brachium")
    for k, (col, meas, cond, desc, stat, unit) in enumerate(specs, 1):
        rows.append({**base, "trial": k, "condition": cond, "measure": meas, "measure_description": desc, "statistic": stat,
                     "value": float(d[col]), "value_unit": unit})
ck.internal("36 x 14 rows", len(rows) == 36 * len(specs), str(len(rows)))
ck.published("participants", len(data), 36, 0, "Abstract (thirty-six participants)")
import numpy as np
from scipy import stats
emg = np.array([d["ALL_EMG"] for d in data]); frc = np.array([d["All_Force"] for d in data])
ck.internal("paired t, CMC EMG vs force, as printed in the preprint (-2.24): not reproduced", True, f"{stats.ttest_rel(emg, frc).statistic:.2f}")
ck.internal("Spearman rho EMG vs force, preprint 0.60: not reproduced", True, f"{stats.spearmanr(emg, frc).statistic:.2f}")
nS = sum(int(d["Sign"]) for d in data); nF = sum(int(d["Sign_F"]) for d in data)
meta = dict(
    study=DS,
    paper=dict(authors="Georgiev C, Mongold SJ, Naeije G, Bourguignon M", year=2026,
               title="Gaining insight into the non-focality of beta oscillation suppression along the sensorimotor cortex using corticomuscular coherence",
               venue="European Journal of Neuroscience (preprint on bioRxiv)", doi="10.1111/ejn.70600"),
    lab="Bourguignon (Universite libre de Bruxelles)", data_url="https://osf.io/hcb24/",
    license="none stated", license_url="",
    changes=("Turned the one-line-per-participant table CMC_all.xlsx (sheet CMC_EMG2) into a long table with one row per participant and "
             "measure; participant ids were re-coded. The second sheet (the significant-participants subset and window means) was not "
             "converted because it repeats these values. The preprocessed EEG mat files (about 43 MB each) and the 550 MB epoch file were not converted."),
    how=["One row is one measure of one of 36 participants: beta (13-30 Hz) corticomuscular coherence between left sensorimotor EEG and the "
         "EMG or force of the isometrically contracted right first dorsal interosseous in three time windows (T0, T1, T2) and overall, "
         "plus whether it was significant against surrogate data.",
         "Coherence is unitless (0-1). The deposit does not define T0, T1 and T2; they are matched by order to the paper's Tbefore (-1 to 0 s), "
         "Tduring (0 to 1 s) and Tafter (1 to 2 s) around brachium movement onset (an inference).",
         "T01 and Contrast were deduced to be the force CMC mean of T0 and T1 and (T2 - T01)/(T2 + T01); age and sex are not in the deposit."],
    unit_note="measure of one participant",
    differences=["The deposit does not reproduce the preprint v1 statistics (it may be a later version): paired t EMG vs force over 36 participants is -1.71 (paper -2.24), Spearman rho 0.74 (paper 0.60). The paper reports 10 participants (EMG) and 19 (force) with significant CMC (also in the authors' R script), but the Sign and Sign_F "
                 "columns of the deposit flag 15 and 22 participants; the flags are kept as given and the counts are not used as checks. The paper's "
                 "window means are only in a figure. The preprint is CC BY-ND while the OSF deposit states no license; the paper was later published "
                 "in Eur J Neurosci (10.1111/ejn.70600)."])
stdlib.release(DS, MODULE, rows, meta, ck)
