"""Hosseini, Nguyen & Joiner (2017), PLoS Comput Biol: decay of force-field adaptation, four force-field types x 14
participants, in the timeseries module. The deposit holds, per participant, the lateral position, velocity and force of
error-clamp trials in three epochs (baseline: single trials; adaptation: averaged over windows of trials; decay: averaged
over windows of 2 trials), each as a 1501-sample trace; these are participant-level averages, not single trials, so one row
here is one trace.

Run from the site folder: python3 -I tools/std/convert/Hosseini2017_PLoSCompBiol.py
Raw files (untrusted data, read only) in raw_extra/Hosseini2017_PLoSCompBiol/, from the Zenodo copy (record 4931545, CC0) of
the Dryad deposit https://doi.org/10.5061/dryad.hv02n: Exp_Data.mat and its readme.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hosseini2017_PLoSCompBiol"
RAW = stdlib.raw_dir(ID)
MODULE = "timeseries"
checks = stdlib.Checks()
FS = 1000.0
m = sio.loadmat(RAW / "Exp_Data.mat", squeeze_me=True, struct_as_record=False)["Exp_Data"]

EXPS = [("pFF_exp", "pFF", "position-dependent field", dict(base="baseline", adapt="adapt", decay="decay"), "pFF_baseline_EC_NDX"),
        ("vFF_exp", "vFF", "velocity-dependent field", dict(base="baseline", adapt="adapt", decay="decay"), "vFF_baseline_EC_NDX"),
        ("ucFF_exp", "ucff", "unbiased combination field", dict(base="base", adapt="adapt", decay="decay"), "ucFF_baseline_EC_NDX"),
        ("pbFF_exp", "pbFF", "position-biased combination field", dict(base="baseline", adapt="adapt", decay="decay"), "pbFF_base_EC_NDX")]
EPOCH = [("baseline", "base", "baseline", 1), ("adapt", "adapt", "learning", 15), ("decay", "decay", "aftereffect", 2)]
CH = [("lateral_position_m", "Position", "m"), ("lateral_velocity_m_s", "Velocity", "m/s"), ("lateral_force_N", "Force", "N")]
t_axis = (np.arange(1501) - 750) / FS


def fmt(v):
    return "" if v != v else "%.8g" % v


rows, extra = [], []
dropped = 0
nwin = {}
coef = {}
npart = {}
for fld, tag, label, tags, ndxname in EXPS:
    e = getattr(m, fld)
    d, inf = e.data, e.info
    ns = int(inf.num_of_subjects)
    npart[tag] = ns
    coef[tag] = (float(inf.FF_K) / 45.0, float(inf.FF_B) / 15.0)
    for s in range(ns):
        sub = f"{ID}_{tag}_S{s + 1:02d}"
        k = 0
        for ep, key, phase, navg in EPOCH:
            etag = tags[key]
            cells = {v: getattr(d, f"{v}_{tag}_{etag}_sub") for v in ("Force", "Velocity", "Position")}
            names_ndx = [a for a in inf._fieldnames if a.endswith("_EC_NDX_sub") and a.split("_")[1].startswith(etag[:4])]
            ndx_sub = np.atleast_2d(getattr(inf, [a for a in names_ndx][0])) if names_ndx else None
            ndx_grp = getattr(inf, [a for a in names_ndx][0].replace("_sub", "")) if names_ndx else None
            nw = len(cells["Force"])
            nwin[(tag, ep)] = nw
            for w in range(nw):
                k += 1
                arrs = [np.asarray(cells[v][w], float)[:, s] for _, v, _ in CH]  # CH order: Position, Velocity, Force
                if all(np.isnan(a).all() for a in arrs):
                    dropped += 1
                    continue
                fname = f"{tag}_S{s + 1:02d}_{ep}_w{w + 1:02d}.csv"
                lines = ["time_s," + ",".join(c[0] for c in CH)]
                for i in range(1501):
                    lines.append(f"{t_axis[i]:.3f}," + ",".join(fmt(a[i]) for a in arrs))
                extra.append((f"samples/{fname}", ("\r\n".join(lines) + "\r\n").encode()))
                loc_s = int(ndx_sub[w, s]) if ndx_sub is not None else ""
                loc_g = int(np.atleast_1d(ndx_grp)[w]) if ndx_grp is not None else ""
                rows.append(dict(
                    dataset=ID, study=ID, experiment=f"{ID}_{tag}", group=tag, subject=sub, person=sub, subject_excluded_by_authors="",
                    handedness="right", population="healthy", setting="lab", device="KINARM End-Point Lab robotic manipulandum",
                    session=1, block=ep, trial=k, phase=phase, task="reaching", unit="recording",
                    condition=f"{label} learning; error-clamp trace, {ep} epoch, {'single trial' if ep == 'baseline' else 'average of a window of %d trials' % navg}",
                    pert_type="error_clamp", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                    recording=k, samples_file=f"samples/{fname}", signal_kind="lateral hand position, velocity and force on error-clamp trials",
                    channels=";".join(c[0] for c in CH), channel_units=";".join(c[2] for c in CH), sample_rate_hz=FS, n_samples=1501,
                    duration_s=1.5, events="peak_velocity=0.75",
                    processing=("none stated (single error-clamp trials)" if ep == "baseline" else f"average of the error-clamp trials of a window of {navg} trials, as shared"),
                    trial_info=f"epoch={ep};window={w + 1};ec_location_participant={loc_s};ec_location_group={loc_g}"))

checks.internal("rows + all-NaN traces dropped = 4 experiments x 14 participants x 53 traces", len(rows) + dropped == 4 * 14 * 53, f"{len(rows)} + {dropped}")
checks.published("participants (4 groups of 14)", len({r["subject"] for r in rows}), 56, 0, "Methods, Participants (56 healthy adults)")
checks.published("participants in the smallest force-field group", min(len({r['subject'] for r in rows if r['group'] == t}) for t in npart), 14, 0, "Methods, Participants (4 groups of 14)")
checks.published("baseline error-clamp trials per participant", nwin[("pFF", "baseline")], 12, 0, "Methods, Trial structure (12 baseline error-clamp trials)")
checks.published("unbiased combination field: position coefficient (cK)", coef["ucff"][0], 0.71, 0.005, "Methods, Force field definitions (0.71)")
checks.published("unbiased combination field: velocity coefficient (cB)", coef["ucff"][1], 0.71, 0.005, "Methods, Force field definitions (0.71)")
checks.published("position-biased combination field: position coefficient (cK)", coef["pbFF"][0], 0.85, 0.005, "Methods, Force field definitions (0.85)")
checks.published("position-biased combination field: velocity coefficient (cB)", coef["pbFF"][1], 0.53, 0.005, "Methods, Force field definitions (0.53)")
checks.published("position-dependent field: position coefficient (cK), velocity-dependent field: velocity coefficient (cB), sum", coef["pFF"][0] + coef["vFF"][1], 2.0, 0, "Methods, Force field definitions (+-1 each)")
checks.published("trace window (ms, 1501 samples at 1000 Hz)", (1501 - 1) / FS * 1000, 1500, 0, "Methods, Error-clamp trials (window of 1500 ms around peak velocity)")
checks.internal("traces are lateral and small: median |position| under 2 mm", float(np.nanmedian([abs(float(x.split(b',')[1])) for _, b in extra[::40] for x in b.split(b'\r\n')[2:6] if x.split(b',')[1]])) < 0.002)

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, lab="Joiner (UC Davis / George Mason)",
    data_url="https://doi.org/10.5061/dryad.hv02n (Zenodo copy: https://doi.org/10.5281/zenodo.4931545)",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes="the MATLAB cell arrays of traces were written as one CSV per participant, epoch and window (time axis added); participant codes replaced.",
    unit_note="trace of one participant (an error-clamp trial or an average of a window of error-clamp trials, 1501 samples in its own CSV)",
    how=[
        "The deposit shares participant-level data only: for each of 14 participants in each of four force-field groups (position-dependent pFF, velocity-dependent vFF, unbiased combination ucFF, position-biased combination pbFF; 56 in all), the lateral position, velocity and force of error-clamp trials in three epochs: baseline (12 single trials), adaptation (11 windows, each averaged over a window of 15 trials) and decay (30 windows, each averaged over a window of 2 trials), as the readme describes. One row is one such trace; trial numbers the traces in this order (1 to 53), the epoch is in block, the window number and the authors' index of where the error-clamp window lies (per participant and for the group) are in trial_info. Single-trial data were not shared.",
        "Each trace has 1501 samples at 1000 Hz, a window of 1500 ms centred on peak velocity (paper), so time_s runs from -0.75 to +0.75 s around peak velocity. The deposit does not name the axis or the units of its Position, Velocity and Force arrays: they are taken as the lateral (perpendicular to the movement) components in m, m/s and N, because the paper analyses the lateral force on error-clamp trials, the clamp keeps lateral displacement under 0.6 mm and the values are of that size. The sign convention (the paper counterbalanced the field direction across participants) is not documented in the deposit; values are as shared.",
        "Traces with no valid samples at all (NaN, mostly windows a participant did not have) are not written (the count is listed in the checks section); NaN samples inside a trace are empty cells. The field gains of the groups (FF_K, FF_B in the deposit) agree with the paper's coefficients times 45 N/m and 15 Ns/m.",
    ],
    differences=["The paper's adaptation coefficients, gain-space fits and time constants need the single trials (or the baseline-subtracted force and the kinematic regressors) that the deposit does not share, so only design counts and field coefficients are checked."],
)
rows = [{**dict(age='', sex='', handedness='', time_s=''), **r} for r in rows]
stdlib.release(ID, MODULE, rows, meta, checks, extra_files=extra)
print("dropped all-NaN traces:", dropped)
