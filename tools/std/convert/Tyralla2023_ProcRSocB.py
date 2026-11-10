"""Tyralla, Pome & Zimmermann (2023) Proc R Soc B: motor recalibration of visual and saccadic maps.

Raw files: raw_extra/Tyralla2023_ProcRSocB/mymat_rawData.mat (trial-by-trial saccade landing positions and localization
clicks, rows = trials, columns = participants, one matrix per experiment and session), mymat_Exp1.mat (the authors' block
means, used only as a check), mymat_Subj1.mat (their example participant, a check). The Dryad deposit is mirrored on Zenodo
record 7626001 (CC0).
Run from the site folder: python3 -I tools/std/convert/Tyralla2023_ProcRSocB.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Tyralla2023_ProcRSocB"
MODULE = "saccade"
RAW = stdlib.raw_dir(DS)
FIX = -6.5            # fixation square 6.5 deg left of screen centre; target and flash 6.5 deg right of it (Methods)
TARGET = 6.5
DEVICE = "EyeLink 1000 Plus (left eye, 1000 Hz), CRT monitor at 57 cm in a dark room, chin rest, mouse and keyboard"
SESS = {"short": 12, "medium": 25, "long": 50}
EXPN = {1: "exp1 no retinal error", 2: "exp2 no prediction error", 3: "exp3 constant error"}


def mat(name):
    m = sio.loadmat(str(RAW / name), squeeze_me=False, struct_as_record=False)
    return m[[k for k in m if not k.startswith("__")][0]][0, 0]


def layout(N):
    """Row ranges (1-based, inclusive) of one session of 220 + 2N trials."""
    b = [("baseline localization", 1, 20, "loc", "baseline"), ("baseline saccades", 21, 40, "sac", "baseline"),
         ("adaptation", 41, 140, "sac", "learning"), ("post-adaptation localization", 141, 160, "loc", "aftereffect"),
         ("no-error trials early", 161, 160 + N, "sac", "other"), ("localization after early no-error trials", 161 + N, 180 + N, "loc", "test"),
         ("no-error trials late", 181 + N, 180 + 2 * N, "sac", "other"), ("localization after late no-error trials", 181 + 2 * N, 200 + 2 * N, "loc", "test"),
         ("final saccades", 201 + 2 * N, 220 + 2 * N, "sac", "washout")]
    return b


def main():
    raw = mat("mymat_rawData.mat")
    auth = {1: mat("mymat_Exp1.mat")}
    ck = stdlib.Checks()
    frames = []
    parts = {}
    sessions = {}
    for e in (1, 2, 3):
        E = getattr(raw, f"Exp{e}")[0, 0]
        for key in (SESS if e < 3 else ("long",)):
            N = SESS[key]
            sac = np.array(getattr(E.sacLand[0, 0], key), float)
            loc = np.array(getattr(E.Loc[0, 0], key), float)
            assert sac.shape == loc.shape and sac.shape[0] == 220 + 2 * N, (e, key, sac.shape)
            parts[e] = sac.shape[1]
            sessions[(e, key)] = (sac, loc)
            lay = layout(N)
            kind = np.empty(sac.shape[0], dtype=object)
            block = np.empty(sac.shape[0], dtype=object)
            phase = np.empty(sac.shape[0], dtype=object)
            for nm, a, z, k, ph in lay:
                kind[a - 1:z], block[a - 1:z], phase[a - 1:z] = k, nm, ph
            is_sac = kind == "sac"
            ck.internal(f"exp{e} {key}: saccade values sit only on saccade trials, localization clicks only on localization trials",
                        bool(np.isnan(sac[~is_sac]).all() and np.isnan(loc[is_sac]).all()))
            adapt = block == "adaptation"
            noerr = np.array([b.startswith("no-error") for b in block])
            for j in range(sac.shape[1]):
                n = sac.shape[0]
                v = np.where(is_sac, sac[:, j], loc[:, j])
                missing = np.isnan(v)
                pert_type = np.where(adapt, "intrasaccadic_step", np.where(noerr, {1: "error_clamp_retinal", 2: "error_clamp_prediction", 3: "step_beyond_predicted_landing"}[e], "none"))
                pert_value = np.where(adapt, 3.0, np.where(noerr, {1: 0.0, 2: np.nan, 3: 3.0}[e], 0.0))
                df = pd.DataFrame({
                    "dataset": DS, "study": DS, "experiment": EXPN[e], "group": "", "subject": f"e{e}|{key}|{j}", "person": f"e{e}|{j}",
                    "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
                    "setting": "lab", "device": DEVICE, "session": "", "block": block, "trial": np.arange(1, n + 1), "phase": phase,
                    "time_s": "", "task": "eye movements", "unit": "trial", "condition": f"{key} ({N} no-error trials per block)",
                    "pert_type": pert_type, "pert_value": pert_value, "pert_unit": "deg", "feedback": np.where(adapt, "altered", np.where(noerr, "altered", "normal")),
                    "excluded_by_authors": missing,
                    "target_amplitude_deg": TARGET - FIX, "target_direction_deg": np.where(is_sac, 0.0, np.nan),
                    "step_x_deg": np.where(adapt, 3.0, np.nan), "step_y_deg": np.where(adapt, 0.0, np.nan),
                    "step_delay_ms": np.where(adapt, 0.0, np.nan),
                    "saccade_end_x_deg": np.where(is_sac, sac[:, j] - FIX, np.nan),
                    "saccade_amplitude_x_deg": np.where(is_sac, sac[:, j] - FIX, np.nan),
                    "eye": "left",
                    "localization_report_deg": np.where(~is_sac, loc[:, j] - FIX, np.nan),
                    "localization_kind": np.where(~is_sac, "pre-saccadic", ""),
                })
                frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in out.columns:
            out[c] = ""
    out = out[[c[1] for c in stdlib.columns_for(MODULE)]]

    # ---- checks ---------------------------------------------------------------------------------------------------
    ck.published("Experiment 1 participants", parts[1], 9, 0, "Methods, Participants")
    ck.published("Experiment 2 participants", parts[2], 9, 0, "Methods, Participants")
    ck.published("Experiment 3 participants", parts[3], 7, 0, "Methods, Participants")
    for key, tot in (("short", 244), ("medium", 270), ("long", 320)):
        ck.published(f"trials per session, {key} condition", sessions[(1, key)][0].shape[0], tot, 0, "Methods, Procedure (244, 270 and 320 trials)")
    for e in (1, 2, 3):
        for key in (SESS if e < 3 else ("long",)):
            sac, loc = sessions[(e, key)]
            frac = (np.isfinite(sac) | np.isfinite(loc)).sum(0) / sac.shape[0]
            lo = layout(SESS[key])
    # example participant of the authors' Figure 2: Experiment 1, long session
    s1 = mat("mymat_Subj1.mat")
    sv, lv = np.array(s1.landposDeg, float).ravel(), np.array(s1.resploc, float).ravel()
    sac, loc = sessions[(1, "long")]
    col = [j for j in range(sac.shape[1]) if np.array_equal(np.nan_to_num(sac[:, j], nan=-1), np.nan_to_num(sv, nan=-1))]
    ck.internal("the authors' example participant (mymat_Subj1) is one column of Experiment 1, long session", len(col) == 1)
    j = col[0]
    lay = {nm: (a, z) for nm, a, z, k, ph in layout(50)}

    def gain(vec, nm):
        a, z = lay[nm]
        x = vec[a - 1:z]
        x = x[np.isfinite(x)]
        return x.mean() / TARGET, x.std(ddof=1) / TARGET

    # compute from the released rows (positions are relative to the fixation square, so subtract 13 - 6.5)
    rows = out[(out["subject"] == "e1|long|%d" % j)]
    assert len(rows) == 320
    def blockvals(name, colname):
        x = rows.loc[rows["block"] == name, colname].replace("", np.nan).astype(float).dropna()
        return x
    for nm, rep, sd in (("baseline saccades", 0.96, 0.12), ("adaptation", 1.08, 0.16), ("no-error trials early", 1.07, 0.15),
                        ("no-error trials late", 0.95, 0.16), ("final saccades", 0.95, 0.11)):
        x = blockvals(nm, "saccade_end_x_deg") + FIX
        ck.published(f"example participant, saccadic gain (landing / 6.5 deg), {nm}: mean", x.mean() / TARGET, rep, 0.0105, "Results, Example subject (Fig. 2)")
        ck.published(f"example participant, saccadic gain, {nm}: SD", x.std(ddof=1) / TARGET, sd, 0.0105, "Results, Example subject (Fig. 2)")
    for nm, rep, sd in (("baseline localization", 0.85, 0.11), ("post-adaptation localization", 0.88, 0.10),
                        ("localization after early no-error trials", 0.78, 0.07), ("localization after late no-error trials", 0.82, 0.10)):
        x = blockvals(nm, "localization_report_deg") + FIX
        ck.published(f"example participant, localization gain, {nm}: mean", x.mean() / TARGET, rep, 0.0105, "Results, Example subject (Fig. 2)")
        ck.published(f"example participant, localization gain, {nm}: SD", x.std(ddof=1) / TARGET, sd, 0.0105, "Results, Example subject (Fig. 2)")
    # the authors' block means (mymat_Exp1) against the converted rows
    LP = auth[1].LandPos[0, 0]
    S_, L_ = LP.Sac[0, 0], LP.Loc[0, 0]
    ok_s = ok_l = True
    detail = []
    for tag, key in (("12", "short"), ("25", "medium"), ("50", "long")):
        for nm, au in (("baseline saccades", "Bsl"), ("adaptation", "Adapt")):
            a = np.array(getattr(S_, f"{au}_{tag}")).ravel()
            mine = np.array([out[(out["subject"] == f"e1|{key}|{j}") & (out["block"] == nm)]["saccade_end_x_deg"].replace("", np.nan).astype(float).mean() + FIX for j in range(parts[1])])
            ok_s &= bool(np.allclose(mine, a, atol=1e-6, equal_nan=True))
    ck.internal("participant means of baseline and adaptation saccade landing positions equal the authors' LandPos values in all three sessions (mymat_Exp1)", ok_s)
    for nm, au in (("baseline localization", "Bsl"), ("post-adaptation localization", "Loc1"),
                   ("localization after early no-error trials", "Loc2"), ("localization after late no-error trials", "Loc3")):
        a = np.array(getattr(L_, f"{au}_12")).ravel()
        mine = np.array([out[(out["subject"] == f"e1|short|{j}") & (out["block"] == nm)]["localization_report_deg"].replace("", np.nan).astype(float).mean() + FIX for j in range(parts[1])])
        ok_l &= bool(np.allclose(mine, a, atol=1e-6, equal_nan=True))
    ck.internal("participant means of the four localization blocks equal the authors' values in the short session (mymat_Exp1)", ok_l)
    ck.internal("row count: 9 x (244 + 270 + 320) + 9 x (244 + 270 + 320) + 7 x 320", len(out) == 18 * 834 + 7 * 320, str(len(out)))

    meta = dict(
        study=DS,
        paper=dict(authors="Tyralla S, Pome A, Zimmermann E", year=2023, title="Motor recalibration of visual and saccadic maps",
                   venue="Proc R Soc B 290(1994):20222566", doi="10.1098/rspb.2022.2566"),
        lab="Zimmermann (Heinrich-Heine University Dusseldorf)",
        data_url="https://doi.org/10.5061/dryad.rbnzs7hft", license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Read the authors' raw trial matrices (mymat_rawData.mat; rows are trials, columns are participants), put the saccade "
                 "and localization values of each trial on one row with the trial's block labelled from the protocol described in the "
                 "paper (20 localization + 20 saccade baseline trials, 100 adaptation trials, 20 localization trials, then two no-error "
                 "blocks of 12, 25 or 50 trials each followed by 20 localization trials, and 20 final saccade trials), and shifted "
                 "positions from screen-centre to fixation-relative degrees (+6.5). The authors' block means and effect tables "
                 "(mymat_Exp1-3) are not included."),
        how=["One row is one trial of one participant in one session (the 12, 25 or 50 no-error-trial condition; Experiment 3 has only "
             "the 50 condition). Saccade trials (red fixation square) hold the primary saccade's landing position; localization trials "
             "(blue fixation square, no saccade) hold the position where the participant clicked to report a 24 ms flash shown at the "
             "target position, with gaze held at fixation.",
             "Positions are in degrees to the right of the fixation square (6.5 deg left of screen centre): the target and the flash are "
             "13 deg away and the target is stepped to 16 deg during adaptation trials. saccade_end_x_deg and saccade_amplitude_x_deg "
             "hold the same value (the saccade starts at fixation); the authors' gain is the screen-centre position divided by 6.5 deg, "
             "that is (value - 6.5) / 6.5 here.",
             "In the no-error blocks the post-saccadic target was placed at the online prediction of the landing position (experiment 1, "
             "no retinal error), at the prediction plus the participant's baseline error (experiment 2, no prediction error), or 3 deg "
             "beyond the prediction (experiment 3, constant error); the rows say so in pert_type and the sizes are not given per trial.",
             "Empty values (excluded_by_authors) are trials the authors marked missing with NaN: blinks, anticipatory saccades, landings "
             "or clicks outside 3.5-9.5 deg, fixation breaks. Participants in experiment 2 and 3 partly took part in earlier experiments; "
             "the deposit does not say which, so every experiment's participants have their own code."],
        unit_note="trial (a saccade trial or a localization trial)",
        differences=["The authors' summary values for saccades after the early and late no-error blocks (NoErrorEarly, NoErrorLate in mymat_Exp1) could not be reproduced from the raw rows with any block window that the README or paper describes, and the baseline and post-adaptation localization means of the medium and long sessions of experiment 1 differ slightly from the authors' values for a few participants; the saccade baseline and adaptation means and the short-session localization means match exactly, and the paper's example-participant gains reproduce."],
    )
    stdlib.release(DS, MODULE, out, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
