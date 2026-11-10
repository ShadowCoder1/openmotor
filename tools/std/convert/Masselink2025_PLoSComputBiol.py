"""Masselink & Lappe (2025) PLoS Comput Biol 21(6):e1013041: adaptation across the 2D population code explains the spatially
distributive nature of motor learning (saccade adaptation with pre- and post-saccadic localizations at 11 probe positions).

Raw file: raw_extra/Masselink2025_PLoSComputBiol/data.zip (Zenodo 15298976), member Masselink_Lappe_2025_data/data.mat, read with
scipy.io.loadmat (struct arrays only; the deposit's modelling .m files are not read as code and nothing is run). The deposit's
readme.pdf describes the fields; the design was read from the PLoS Comput Biol article (Methods).

Run from the site folder: python3 -I tools/std/convert/Masselink2025_PLoSComputBiol.py
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Masselink2025_PLoSComputBiol"
MODULE = "saccade"
N_SUBJ, N_TRIAL = 13, 530
TRIAL_KIND = {1: "pre-saccadic localization trial", 2: "post-saccadic localization trial", 3: "saccade trial"}


def main():
    with zipfile.ZipFile(stdlib.raw_dir(DS) / "data.zip") as z:
        m = sio.loadmat(io.BytesIO(z.read("Masselink_Lappe_2025_data/data.mat")), squeeze_me=True, struct_as_record=False)
    data = m["data"]
    tpos = np.asarray(data.targetpos, dtype=float)
    ttype = np.asarray(data.trialtype).astype(int)
    ck = stdlib.Checks()

    ck.internal("trial type vector has 530 entries", len(ttype) == N_TRIAL)
    phase_of = np.where(np.arange(1, N_TRIAL + 1) <= 165, 0, np.where(np.arange(1, N_TRIAL + 1) <= 365, 1, 2))
    recs = []
    sess_data = {}
    for s in range(1, N_SUBJ + 1):
        for ses in (1, 2):
            o = getattr(getattr(data, f"sbj{s}"), f"ses{ses}")
            tg = np.asarray(o.target).astype(int)
            Mx, V1, V2 = (np.asarray(getattr(o, k), dtype=float) for k in ("Mx", "V1", "V2px"))
            assert Mx.shape == (N_TRIAL, 2) and V1.shape == (N_TRIAL, 2) and V2.shape == (N_TRIAL, 2) and len(tg) == N_TRIAL
            sess_data[(s, ses)] = (tg, Mx, V1, V2)
    # ---- design checks (from the paper's Methods and the deposit's readme) ---------------------------------------------
    pre_counts = {k: int((ttype[:165] == k).sum()) for k in (1, 2, 3)}
    post_counts = {k: int((ttype[365:] == k).sum()) for k in (1, 2, 3)}
    ck.published("pre-adaptation trials of each kind (pre-saccadic loc., post-saccadic loc., saccade)", pre_counts[1], 55, 0,
                 "Methods, Trial structure (165 pre-adaptation trials, 55 of each kind)")
    ck.published("post-adaptation trials of each kind (post-saccadic loc.)", post_counts[2], 55, 0, "Methods, Trial structure")
    ck.published("adaptation trials (all saccade trials)", int((ttype[165:365] == 3).sum()), 200, 0, "Methods, Trial structure (200 adaptation trials)")
    ck.published("trials per session", len(ttype), 530, 0, "Methods, Trial structure (165 + 200 + 165)")
    ck.published("probe positions", len(tpos), 11, 0, "Methods, Probe positions (11)")
    ck.published("subjects", N_SUBJ, 13, 0, "Methods, Participants (N = 13)")
    ck.internal("adaptation target (probe position 1) is 12 deg right of fixation", bool(np.allclose(tpos[0], (12, 0))))
    ck.internal("every adaptation-phase trial is a saccade trial to probe position 1",
                all((sess_data[k][0][165:365] == 1).all() for k in sess_data))

    rows = []
    for (s, ses), (tg, Mx, V1, V2) in sess_data.items():
        for i in range(N_TRIAL):
            t = i + 1
            k = int(ttype[i])
            ph = int(phase_of[i])
            tx, ty = tpos[tg[i] - 1]
            has_sacc = k in (2, 3)
            loc = V1[i] if k == 1 else (V2[i] if k == 2 else np.array([np.nan, np.nan]))
            sacc = Mx[i] if has_sacc else np.array([np.nan, np.nan])
            ok = (~np.isnan(sacc).any()) if k == 3 else ((~np.isnan(loc).any()) if k == 1 else ((~np.isnan(loc).any()) and (~np.isnan(sacc).any())))
            rows.append((s, ses, t, k, ph, tg[i], tx, ty, sacc[0], sacc[1], loc[0], loc[1], not ok))
    R = pd.DataFrame(rows, columns=["sbj", "ses", "trial", "kind", "ph", "tgt", "tx", "ty", "sx", "sy", "lx", "ly", "excl"])
    ck.internal("13 subjects x 2 sessions x 530 trials", len(R) == 13 * 2 * 530)
    ck.internal("measure present exactly where the trial kind has one (no values in other columns)",
                bool(R.loc[R.kind == 1, ["sx", "sy"]].isna().all().all() and R.loc[R.kind == 3, ["lx", "ly"]].isna().all().all()))

    # median horizontal saccade amplitude per subject, pre-adaptation, to the adaptation target (post-saccadic localization trials,
    # the only saccades the paper analyses for the pre- and post-adaptation phases)
    pre = R[(R.ph == 0) & (R.kind == 2) & (R.tgt == 1)].groupby("sbj")["sx"].median()
    last = R[(R.ph == 1) & (R.trial > 365 - 20)].groupby("sbj")["sx"].median()
    ck.published("mean over subjects of the median horizontal saccade amplitude to the adaptation target before adaptation (deg)",
                 pre.mean(), 11.9, 0.1, "Results, Saccade amplitudes and localizations (started at 11.9 dva)")
    shortening = float((pre - last).mean())
    ck.internal("saccade amplitudes shorten by 1.5-3 deg by the end of adaptation", 1.5 < shortening < 3.0, f"{shortening:.2f} deg")

    pcode = lambda s: f"P{int(s):02d}"
    kinds = R.kind.to_numpy()
    out = pd.DataFrame({
        "dataset": DS, "study": DS, "experiment": "saccade adaptation with localization probes", "group": "",
        "subject": [f"{s}_{e}" for s, e in zip(R.sbj, R.ses)],
        "person": [pcode(s) for s in R.sbj],
        "subject_excluded_by_authors": False,
        "age": "", "sex": "", "handedness": "", "population": "healthy", "setting": "lab",
        "device": "EyeLink 1000 (right eye, 1000 Hz), Eizo FlexScan F930 monitor (75 Hz) at 62 cm, trackpad",
        "session": R.ses.astype(int),
        "block": np.array(["pre-adaptation", "adaptation", "post-adaptation"])[R.ph.to_numpy()],
        "trial": R.trial.astype(int),
        "phase": np.array(["baseline", "learning", "aftereffect"])[R.ph.to_numpy()],
        "time_s": "", "task": "eye movements", "unit": "saccade",
        "condition": [TRIAL_KIND[k] for k in kinds],
        "pert_type": np.where(R.ph.to_numpy() == 1, "intrasaccadic_step", "none"),
        "pert_value": np.where(R.ph.to_numpy() == 1, -3.0, 0.0),
        "pert_unit": "deg",
        "feedback": np.where(R.ph.to_numpy() == 1, "altered", "normal"),
        "excluded_by_authors": R.excl.to_numpy(),
        "target_amplitude_deg": np.hypot(R.tx, R.ty),
        "target_direction_deg": np.degrees(np.arctan2(R.ty, R.tx)) % 360,
        "step_x_deg": np.where(R.ph.to_numpy() == 1, -3.0, np.nan),
        "step_y_deg": np.where(R.ph.to_numpy() == 1, 0.0, np.nan),
        "step_delay_ms": np.where(R.ph.to_numpy() == 1, 0.0, np.nan),
        "saccade_amplitude_x_deg": R.sx,
        "saccade_amplitude_y_deg": R.sy,
        "eye": "right",
        "localization_report_deg": R.lx,
        "localization_report_y_deg": R.ly,
        "localization_kind": np.where(kinds == 1, "pre-saccadic", np.where(kinds == 2, "post-saccadic", "")),
        "trial_info": [f"probe_position={int(g)}" for g in R.tgt],
    })
    for c in stdlib.columns_for(MODULE):
        if c[1] not in out.columns:
            out[c[1]] = ""
    meta = dict(
        study=DS,
        paper=dict(authors="Masselink J, Lappe M", year=2025,
                   title="Adaptation across the 2D population code explains the spatially distributive nature of motor learning",
                   venue="PLoS Comput Biol 21(6):e1013041", doi="10.1371/journal.pcbi.1013041"),
        lab="Lappe (University of Munster)",
        data_url="https://doi.org/10.5281/zenodo.15298976",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Rearranged the authors' MATLAB struct (13 subjects x 2 sessions x 530 trials; saccade amplitudes, pre- and "
                 "post-saccadic localizations, probe position and trial type) into one trial table with participant codes. "
                 "The grand-average arrays of the deposit (means and standard errors per probe position, the adaptation curve and "
                 "the computed displacement of visual space) are not included; the modelling code is not part of this copy."),
        how=["One row is one trial of one session (530 trials: 165 pre-adaptation, 200 adaptation, 165 post-adaptation). Saccade "
             "trials and post-saccadic localization trials carry the saccade; pre-saccadic and post-saccadic localization trials carry "
             "the participant's reported position of a brief flash.",
             "Positions and amplitudes are in degrees of visual angle relative to the fixation point, x positive rightward and y "
             "positive upward; the saccade is given as the authors' 'saccade amplitude' vector (saccade_amplitude_x_deg, "
             "saccade_amplitude_y_deg), which the readme gives with respect to the fixation point. Target and probe positions come "
             "from the authors' list of 11 positions: the 12 deg rightward adaptation target, 4 positions on a 3 deg circle around it "
             "and 6 positions on a 6 deg circle.",
             "In the 200 adaptation trials the target stepped 3 deg inward (to the left) during the saccade (pert_value and step_x_deg "
             "-3). Rows the authors left empty (no valid saccade, latency outside 100-400 ms, landing outside the target window or "
             "gaze not held during pre-saccadic localization) are kept and marked excluded_by_authors."],
        unit_note="trial (one saccade and/or one localization report)",
        differences=[f"The paper reports that saccade amplitude shortened by 2.4 deg by the end of adaptation (t(12) = -6.46); the rows here give {shortening:.1f} deg (median of the last 20 adaptation trials against the median before adaptation, per subject), and 1.6 deg for all 200 adaptation trials, so that number is not reproduced; the pre-adaptation amplitude of 11.9 deg is."],
    )
    stdlib.release(DS, MODULE, out, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
