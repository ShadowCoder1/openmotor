"""Valsecchi, Cassanello, Herwig, Rolfs & Gegenfurtner (2020) J Vis: trans-saccadic perceptual recalibration and saccadic adaptation.

Raw file: raw_extra/Valsecchi2020_JVis/data.zip (Zenodo 3607596; folders ZENODOEXP1, ZENODOEXP2ADJ, ZENODOEXP2SACC). Only
the saccadic adaptation session of Experiment 2 (ZENODOEXP2SACC: TRIALDATA<observer>_<day>.txt trial logs and EVENTS<observer>_<day>.asc
EyeLink event files, whose saccades are the EyeLink parser's) is converted. The 1000-Hz sample files and the two perceptual size
adjustment folders stay in the original deposit. Nothing from the deposit is executed.
Run from the site folder: python3 -I tools/std/convert/Valsecchi2020_JVis.py
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Valsecchi2020_JVis"
MODULE = "saccade"
PXDEG = 26.5          # 424 px = 16 deg (training rows of the trial logs: -424 without a step, -424 -+ 2.4 deg * 26.5 = -487.6 / -360.4)
ECC_PX = 424.0
DEVICE = "EyeLink II (left eye, 500 Hz), Eizo CG223W LCD at 40 cm, keyboard"
MIN_TEST_SACC_PX = 100.0   # a test-trial primary saccade must cover at least 100 px (3.8 deg) horizontally
Z = zipfile.ZipFile(stdlib.raw_dir(DS) / "data.zip")
BASE = "ZENODO/ZENODOEXP2SACC/"


def parse_events(text):
    msgs, sacc = {}, []
    for l in text.splitlines():
        if l.startswith("MSG") and " TRIAL " in l:
            p = l.split()
            msgs.setdefault(int(p[3]), {})[p[4]] = int(p[1])
        elif l.startswith("ESACC"):
            p = l.split()
            if "." in p[5:11]:      # saccade lost in a blink: coordinates are '.'
                continue
            sacc.append(dict(st=int(p[2]), en=int(p[3]), dur=int(p[4]), sx=float(p[5]), sy=float(p[6]), ex=float(p[7]),
                             ey=float(p[8]), amp=float(p[9]), pv=float(p[10])))
    return msgs, sacc


def main():
    ck = stdlib.Checks()
    rows = []
    n_sessions = 0
    for obs in range(1, 13):
        for ses in (1, 2):
            a = np.loadtxt(io.BytesIO(Z.read(f"{BASE}TRIALDATA{obs}_{ses}.txt")))
            ev = Z.read(f"{BASE}EVENTS{obs}_{ses}.asc").decode(errors="replace")
            msgs, sacc = parse_events(ev)
            assert a.shape == (210, 17) and len(msgs) == 210, (obs, ses, a.shape, len(msgs))
            n_sessions += 1
            t0 = min(m["STON"] for m in msgs.values())
            train_side = np.sign(a[a[:, 13] == 2][0, 6])
            stepped_seen = False
            for k, r in enumerate(a):
                tid = int(r[16])
                m = msgs[tid]
                training = r[13] == 2
                if training:
                    side = np.sign(r[6])
                    step_deg = (abs(r[6]) - ECC_PX) / PXDEG        # outward positive
                    chosen = [s for s in sacc if m["STON"] <= s["st"] <= m.get("CHANGG", -1)]
                    s = chosen[-1] if chosen else None
                    if abs(step_deg) > 0.05:
                        stepped_seen = True
                    phase = "learning" if abs(step_deg) > 0.05 else ("washout" if stepped_seen else "baseline")
                else:
                    side = -1.0 if r[0] == 1 else 1.0               # target position 1 = left, 2 = right (saccade directions in the events)
                    step_deg = np.nan
                    chosen = [s for s in sacc if m["STON"] <= s["st"] <= m.get("STOFF", m["STON"] + 1500) and abs(s["ex"] - s["sx"]) >= MIN_TEST_SACC_PX]
                    s = chosen[0] if chosen else None
                    phase = "test"
                dx = (s["ex"] - s["sx"]) if s else np.nan
                gain = side * dx / ECC_PX if s else np.nan
                bad = (not s) or gain < 0.5 or gain > 1.5
                rows.append(dict(
                    dataset=DS, study=DS, experiment="saccadic adaptation session (Experiment 2)", group="",
                    subject=f"o{obs}|s{ses}", person=f"o{obs}", subject_excluded_by_authors=False, age="", sex="", handedness="",
                    population="healthy", setting="lab", device=DEVICE, session=ses,
                    block=f"{'training' if training else 'test'} block {int(r[2])}", trial=k + 1, phase=phase,
                    time_s=(s["st"] - t0) / 1000.0 if s else np.nan, task="eye movements", unit="trial",
                    condition=("training" if training else "test") + (", target " + ("same side as training" if (r[0] == (1 if train_side < 0 else 2)) else "opposite side") if not training else ""),
                    pert_type="intrasaccadic_step" if training and abs(step_deg) > 0.05 else "none",
                    pert_value=step_deg if training else 0.0, pert_unit="deg", feedback="normal" if training else "none",
                    excluded_by_authors=bool(bad),
                    target_amplitude_deg=16.0, target_direction_deg=180.0 if side < 0 else 0.0,
                    step_x_deg=side * step_deg if training else np.nan, step_y_deg=0.0 if training else np.nan,
                    saccade_amplitude_deg=s["amp"] if s else np.nan,
                    saccade_amplitude_x_deg=dx / PXDEG if s else np.nan,
                    saccade_amplitude_y_deg=-(s["ey"] - s["sy"]) / PXDEG if s else np.nan,
                    saccade_gain=gain, saccade_duration_ms=s["dur"] if s else np.nan, saccade_peak_velocity_deg_s=s["pv"] if s else np.nan,
                    eye="left",
                    trial_info=f"trial_id={tid};target_position={int(r[0])};peripheral_size_level={int(r[1])};peripheral_size_pix={r[4]:g}"
                               + (f";post_saccadic_target_px={r[6]:g}" if training else ""),
                ))
    out = pd.DataFrame(rows)
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in out.columns:
            out[c] = ""
    out = out[[c[1] for c in stdlib.columns_for(MODULE)]]

    # ---- checks ---------------------------------------------------------------------------------------------------
    ck.published("Experiment 2: observers", out["person"].nunique(), 12, 0, "Experiment 2 (Space), Participants")
    ck.published("Experiment 2: trials per session (120 test + 90 training)", out.groupby("subject").size().mean(), 210, 0, "Experiment 2, Design")
    ck.published("Experiment 2: sessions (two days per observer)", n_sessions, 24, 0, "Experiment 2, Design")
    tt = out["condition"].str.startswith("test")
    ck.published("Experiment 2: test trials per session (12 blocks of 10)", tt.groupby(out["subject"]).sum().mean(), 120, 0, "Experiment 2, Design")
    g = pd.to_numeric(out["saccade_gain"], errors="coerce")
    got = g.dropna()
    ck.published("Experiment 2: % of saccades with gain below 0.5 (discarded)", 100 * (got < 0.5).mean(), 2.2, 0.5, "Experiment 2, Methods (2.2% of trials)")
    ck.published("Experiment 2: % of saccades with gain above 1.5 (discarded)", 100 * (got > 1.5).mean(), 0.2, 0.2, "Experiment 2, Methods (0.2% of trials)")
    tr = out[~tt & ~out["excluded_by_authors"]]
    out_g = pd.to_numeric(tr.loc[tr["pert_type"] == "intrasaccadic_step", "saccade_gain"])
    ck.internal("training trials: gain is about 1 in no-step trials", abs(pd.to_numeric(tr.loc[tr["pert_type"] == "none", "saccade_gain"]).mean() - 1) < 0.05)
    ck.internal("210 rows per session, 24 sessions, 5040 rows", len(out) == 5040, str(len(out)))
    tp = out["trial_info"].str.extract(r"target_position=(\d)")[0].astype(int)
    ax = pd.to_numeric(out["saccade_amplitude_x_deg"], errors="coerce")
    pos1 = tt & (tp == 1) & ax.notna()
    ck.internal("target position 1 is the left target: test saccades to position 1 go left in more than 98% of trials",
                bool((ax[pos1] < 0).mean() > 0.98), f"{(ax[pos1] < 0).mean():.3f}")
    outw = pd.to_numeric(tr.loc[tr["pert_value"].astype(float) > 0.05, "saccade_gain"]).mean()
    inw = pd.to_numeric(tr.loc[tr["pert_value"].astype(float) < -0.05, "saccade_gain"]).mean()
    ck.internal("training gain is higher in outward-step trials than in inward-step trials", bool(outw > inw), f"{outw:.3f} vs {inw:.3f}")
    meta = dict(
        study=DS,
        paper=dict(authors="Valsecchi M, Cassanello C, Herwig A, Rolfs M, Gegenfurtner KR", year=2020,
                   title="A comparison of the temporal and spatial properties of trans-saccadic perceptual recalibration and saccadic adaptation",
                   venue="J Vis 20(4):2", doi="10.1167/jov.20.4.2"),
        lab="Gegenfurtner / Rolfs (Giessen / Humboldt-Universitat Berlin)",
        data_url="https://doi.org/10.5281/zenodo.3607596", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Only the saccadic adaptation session of Experiment 2 is converted: trial logs and the EyeLink saccade events were joined "
                 "by trial number, one primary saccade per trial was chosen (the saccade that triggered the step in training trials; the "
                 "first horizontal saccade of at least 3.8 deg in test trials), and pixels were converted to degrees at 26.5 px/deg. "
                 "The 1000-Hz gaze samples, the saccade traces and the two perceptual size-adjustment experiments (Experiment 1 and "
                 "the adjustment sessions of Experiment 2) stay in the original deposit."),
        how=["One row is one trial of one observer on one of two days: the saccade made to a gray circle 16 deg to the left or right of the "
             "fixation point. Training trials (90 per day, 5 or 15 before each of the 12 test blocks) show the target after the saccade at "
             "the same place or 2.4 deg farther or nearer; test trials (120 per day) show no post-saccadic target.",
             "saccade_amplitude_x_deg and saccade_amplitude_y_deg are the horizontal and vertical extent of the primary saccade from the "
             "EyeLink saccade event (screen pixels / 26.5; x positive rightward, y positive upward); saccade_amplitude_deg is the "
             "EyeLink-reported amplitude. saccade_gain is the saccade extent toward the target divided by the 424-px (16 deg) "
             "eccentricity, the authors' definition. excluded_by_authors marks no detected saccade or gain outside 0.5-1.5.",
             "pert_value is the step in degrees, positive when the target moved farther out; step_x_deg gives it in screen direction. "
             "Target position 1 is the left target and 2 the right one (the observer's training side is fixed). phase follows the step "
             "sequence of each day: no-step training before the first step is baseline, step training is learning, no-step training "
             "after it is washout, and test trials are test."],
        unit_note="trial (one primary saccade)",
        differences=["Gain discard rates are computed from the saccades chosen here (EyeLink events, 24 sessions); see the checks for how they compare with the paper's 2.2% below 0.5 and 0.2% above 1.5. Training trials in which no saccade started between trial start and the step message (about 7%) have no saccade values and are marked excluded."],
    )
    stdlib.release(DS, MODULE, out, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
