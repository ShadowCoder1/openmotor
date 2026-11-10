"""Masselink, Cheviet, Froment-Tilikete, Pelisson & Lappe (2023) PLoS Comput Biol: saccade adaptation and fatigue
compensation in cerebellar patients and controls (Zenodo 8156033).

Raw file: raw_extra/Masselink2023_PLoSComputBiol/data.zip, member Masselink_etal_2023_data/data.mat (struct 'data':
data.<in|out|no>.<C1..C8|P1..P8>.<preloc|postloc|sac|sacpv|sacdur>, 280 values per participant and condition, and
data.trialtype). The readme.pdf and the .m files (modeling code) were not used for values; nothing is executed.
The meaning of trial types and fields comes from the paper's Methods (Experimental procedure) and the data layout.

Run from the site folder: python3 -I tools/std/convert/Masselink2023_PLoSComputBiol.py
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

DS = "Masselink2023_PLoSComputBiol"
MODULE = "saccade"
COND = {"in": ("inward", -6.0), "out": ("outward", 6.0), "no": ("no step", 0.0)}
PEOPLE = [f"C{i}" for i in range(1, 9)] + [f"P{i}" for i in range(1, 9)]
FIELDS = ("preloc", "postloc", "sac", "sacpv", "sacdur")
SRC = "Methods, Experimental procedure"


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(DS) / "data.zip")
    m = sio.loadmat(io.BytesIO(z.read("Masselink_etal_2023_data/data.mat")), squeeze_me=True, struct_as_record=False)["data"]
    tt = np.asarray(m.trialtype).astype(int)
    N = np.arange(1, 281)
    ck = stdlib.Checks()

    def get(c, p, f):
        return np.asarray(getattr(getattr(getattr(m, c), p), f), dtype=float)

    ck.internal("280 trials with types 1 (pre-saccadic localization) x40, 2 (post-saccadic localization) x40, 3 (saccade) x200",
                len(tt) == 280 and (tt == 1).sum() == 40 and (tt == 2).sum() == 40 and (tt == 3).sum() == 200)
    ck.internal("pre-exposure = trials 1-60 (40 localization + 20 saccade trials), exposure = 61-180 all saccade trials",
                bool((tt[60:180] == 3).all() and (tt[40:60] == 3).all() and (tt[:40] < 3).all()))
    only = {1: ("preloc",), 2: ("postloc", "sac"), 3: ("sac", "sacpv", "sacdur")}
    ok = True
    for c in COND:
        for p in PEOPLE:
            for t, fs in only.items():
                for f in FIELDS:
                    has = np.isfinite(get(c, p, f)[tt == t]).any()
                    ok &= (has == (f in fs)) or (f in fs and not has)     # fields outside the trial type are always empty
    ck.internal("each measure is filled only on its own trial type (preloc on 1; postloc and sac on 2; sac, sacpv, sacdur on 3)", ok)

    rows = []
    for c, (cname, step) in COND.items():
        for p in PEOPLE:
            grp = "control" if p[0] == "C" else "patient"
            sac, pv, du = get(c, p, "sac"), get(c, p, "sacpv"), get(c, p, "sacdur")
            pre, post = get(c, p, "preloc"), get(c, p, "postloc")
            for i in range(280):
                t = tt[i]
                n = i + 1
                expo = 61 <= n <= 180
                stepped = step != 0.0 and expo and t == 3
                have = {1: np.isfinite(pre[i]), 2: np.isfinite(post[i]) and np.isfinite(sac[i]), 3: np.isfinite(sac[i])}[t]
                rows.append({
                    "dataset": DS, "study": DS, "experiment": "saccade adaptation", "group": grp,
                    "subject": f"{p}_{cname}", "person": p, "subject_excluded_by_authors": False,
                    "age": "", "sex": "", "handedness": "",
                    "population": "healthy" if grp == "control" else "cerebellar ataxia", "setting": "lab",
                    "device": "video eye tracker, 20 deg rightward saccade target, mouse-reported flash localization",
                    "session": "", "block": "pre-exposure" if n <= 60 else ("exposure" if n <= 180 else "post-exposure"),
                    "trial": n, "phase": "baseline" if n <= 60 else ("learning" if n <= 180 else "aftereffect"),
                    "time_s": "", "task": "eye movements", "unit": "saccade", "condition": cname,
                    "pert_type": "intrasaccadic_step" if stepped else "none", "pert_value": step if stepped else 0.0,
                    "pert_unit": "deg", "feedback": "altered" if stepped else "normal",
                    "excluded_by_authors": not have,
                    "target_amplitude_deg": 20.0, "target_direction_deg": 0.0,
                    "step_x_deg": step if stepped else (0.0 if t == 3 else ""),
                    "saccade_amplitude_deg": sac[i], "saccade_amplitude_x_deg": sac[i],
                    "saccade_peak_velocity_deg_s": pv[i], "saccade_duration_ms": du[i], "eye": "",
                    "localization_report_deg": pre[i] if t == 1 else (post[i] if t == 2 else np.nan),
                    "localization_kind": {1: "pre-saccadic", 2: "post-saccadic", 3: ""}[t],
                })
    out = pd.DataFrame(rows)
    for col in stdlib.columns_for(MODULE):
        if col[1] not in out.columns:
            out[col[1]] = ""
    ck.internal("16 participants (8 control, 8 patient) x 3 conditions x 280 trials = 13440 rows",
                len(out) == 13440 and out.person.nunique() == 16 and out.groupby("group").person.nunique().eq(8).all())

    # ---- published numbers (Results; values are group means over participants) ------------------------------------------
    sacb = out[(out.trial <= 60) & (out.trial > 40)]       # pre-exposure saccade trials (type 3)
    se = {}
    for g in ("control", "patient"):
        vals = []
        for (p, cn), q in sacb[sacb.group == g].groupby(["person", "condition"]):
            s = q.saccade_amplitude_deg.astype(float).dropna()
            vals.append((p, s.std(ddof=1) / np.sqrt(len(s))))
        se[g] = pd.DataFrame(vals).groupby(0)[1].mean().mean()
    ck.published("pre-exposure saccade variability (standard error), controls (deg)", se["control"], 0.28, 0.02, "Results, Saccade variability")
    ck.published("pre-exposure saccade variability (standard error), patients (deg)", se["patient"], 0.54, 0.05, "Results, Saccade variability")

    def change(cond, g):
        v = []
        for p, q in out[(out.group == g) & (out.condition == cond) & (out.trial.between(1, 180))].groupby("person"):
            base = pd.to_numeric(q[q.trial.between(41, 60)].saccade_amplitude_deg, errors="coerce").mean()
            last = pd.to_numeric(q[q.trial.between(161, 180)].saccade_amplitude_deg, errors="coerce").mean()
            v.append(last - base)
        return np.array(v)

    cin, cout = change("inward", "control"), change("outward", "control")
    ck.published("controls: saccade amplitude change in the last 20 exposure trials vs pre-exposure, inward step (deg)", cin.mean(), -3.9, 0.1, "Results, Motor command change")
    ck.published("controls: same, SD across participants, inward step (deg)", cin.std(ddof=1), 1.8, 0.1, "Results, Motor command change")
    ck.published("controls: saccade amplitude change in the last 20 exposure trials vs pre-exposure, outward step (deg)", cout.mean(), 3.6, 0.1, "Results, Motor command change")
    ck.published("controls: same, SD across participants, outward step (deg)", cout.std(ddof=1), 2.0, 0.1, "Results, Motor command change")
    ck.published("participants: 8 controls + 8 patients", out.person.nunique(), 16, 0, "Participants")

    meta = dict(
        study=DS,
        paper=dict(authors="Masselink J, Cheviet A, Froment-Tilikete C, Pelisson D, Lappe M", year=2023,
                   title="A triple distinction of cerebellar function for oculomotor learning and fatigue compensation",
                   venue="PLoS Comput Biol 19(8):e1011322", doi="10.1371/journal.pcbi.1011322"),
        lab="Lappe (University of Muenster) / Pelisson (Lyon)",
        data_url="https://doi.org/10.5281/zenodo.8156033",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Converted data.mat (per participant, condition and trial values) to the OpenMotor saccade format with "
                 "participant codes and one subject sequence per participant and condition; the authors' group curves "
                 "(means, standard errors, fitted parameters) and the modeling code stay only in the original deposit."),
        how=["One row is one of 280 trials of one participant in one condition (inward step, outward step, no step), as the "
             "authors store them. Saccade trials (type 3) hold the primary saccade amplitude (deg, to a 20 deg rightward target), "
             "its peak velocity (deg/s) and duration (ms); localization trials hold the reported position of a brief flash, "
             "either before the saccade (pre-saccadic, fixating) or after it (post-saccadic, trials that also hold the "
             "saccade amplitude).",
             "The target step of 6 deg (opposite to the saccade direction for inward, along it for outward) is applied only in the "
             "120 exposure trials (61-180) that are saccade trials, and is given in pert_value and step_x_deg. Trials 1-60 are "
             "pre-exposure, 181-280 post-exposure.",
             "A trial whose measurement is missing in the authors' file is kept as an empty row and marked excluded_by_authors. "
             "Peak velocity and duration exist only for about 60% of saccade trials, as in the deposit. Sessions (one per "
             "condition, at least a week apart, in random order) are not identified in the deposit."],
        unit_note="trial (primary saccade and/or flash localization report)",
        differences=["The paper's motor-command changes come from a model fit; the converted rows give controls' change in mean "
                     "saccade amplitude between the 20 pre-exposure saccade trials and the last 20 exposure trials, which "
                     "agrees (-3.9 and +3.6 deg), but the patient values and the localization-based measures were not reproduced "
                     "from the rows (they depend on the authors' model).",
                     "The deposit does not state the units of the fields; amplitudes (about 20), peak velocity (about 440) and "
                     "duration (about 65) were read as deg, deg/s and ms from the 20 deg target and the paper."],
    )
    stdlib.release(DS, MODULE, out, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}  ok={c['ok']}")


if __name__ == "__main__":
    main()
