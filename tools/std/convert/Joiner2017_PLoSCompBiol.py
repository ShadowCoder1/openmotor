"""Joiner, Sing & Smith (2017), PLoS Comput Biol: temporal specificity of the initial adaptive response (velocity- and
position-dependent curl fields, error-clamp trials), 11 participants x 1856 movements, in the timeseries module (one row per
movement, with the movement's 241 samples of position, velocity and force in a CSV inside the ZIP).

Run from the site folder: python3 -I tools/std/convert/Joiner2017_PLoSCompBiol.py
Raw files (untrusted data, read only) in raw_extra/Joiner2017_PLoSCompBiol/, from the Zenodo copy (record 4991069, CC0) of the
Dryad deposit https://doi.org/10.5061/dryad.1rr57: joineretal2017data.mat (info, data) and its readme (PDF, read as text).
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Joiner2017_PLoSCompBiol"
RAW = stdlib.raw_dir(ID)
MODULE = "timeseries"
checks = stdlib.Checks()
FS = 200.0  # paper, Methods: position, velocity and force measured at 200 Hz
names = [c[1] for c in stdlib.columns_for(MODULE)]

m = sio.loadmat(RAW / "joineretal2017data.mat", squeeze_me=True, struct_as_record=False)
info, data = m["info"], m["data"]
tgt = np.asarray(info.tgt_all)
bs = np.asarray(info.blocksize).astype(int)
FFMAG = float(info.FFMAG)
NT = tgt.shape[0]
assert NT == 1856 == bs.sum() and len(data.xr) == NT
CH = [("x_target_m", "xr", "m"), ("y_orthogonal_m", "yr", "m"), ("vx_target_m_s", "vxr", "m/s"), ("vy_orthogonal_m_s", "vyr", "m/s"), ("fy_orthogonal_N", "fyr", "N")]
arr = {k: np.stack([np.asarray(getattr(data, f)[t], float) for t in range(NT)]) for _, f, _ in CH for k in [f]}  # trial x sample x subject
NS = arr["xr"].shape[2]
assert arr["xr"].shape == (NT, 241, NS)
block_of = np.repeat(np.arange(1, len(bs) + 1), bs)


def fmt(v):
    return "" if v != v else "%.8g" % v


rows, extra = [], []
dropped = 0
ff_peak = []
final_x = []
for s in range(NS):
    sub = f"{ID}_S{s + 1:02d}"
    for t in range(NT):
        cols = [arr[f][t, :, s] for _, f, _ in CH]
        if np.isnan(cols[0]).all() and np.isnan(cols[1]).all():
            dropped += 1
            continue
        vco, pco, ecl, dirn = tgt[t, 1], tgt[t, 8], tgt[t, 6], tgt[t, 0]
        isff = (vco != 0 or pco != 0) and ecl == 0
        if ecl:
            ptype, cond = "error_clamp", "error clamp"
        elif vco != 0:
            ptype, cond = "force_field", "velocity-dependent field"
        elif pco != 0:
            ptype, cond = "force_field", "position-dependent field"
        else:
            ptype, cond = "none", "null"
        fname = f"S{s + 1:02d}_t{t + 1:04d}.csv"
        lines = ["time_s," + ",".join(c[0] for c in CH)]
        for i in range(241):
            lines.append(f"{i / FS:.3f}," + ",".join(fmt(c[i]) for c in cols))
        extra.append((f"samples/{fname}", ("\r\n".join(lines) + "\r\n").encode()))
        if isff:
            ff_peak.append(np.nanmax(np.abs(cols[4])) if not np.isnan(cols[4]).all() else np.nan)
        if not ecl and not np.isnan(cols[0][-1]):
            final_x.append(abs(cols[0][-1]))
        blk = int(block_of[t])
        rows.append(dict(
            dataset=ID, study=ID, experiment=f"{ID}_Main", group="", subject=sub, person=sub, subject_excluded_by_authors="",
            population="healthy", setting="lab", device="robot manipulandum", session=1, block=f"block {blk}", trial=t + 1,
            phase="practice" if blk <= 3 else "learning", task="reaching", unit="recording", condition=cond, pert_type=ptype,
            pert_value=(vco * FFMAG if vco != 0 else pco * FFMAG) if ptype == "force_field" else 0,
            pert_unit=("Ns/m" if vco != 0 else "N/m") if ptype == "force_field" else "", feedback="normal", excluded_by_authors=False,
            recording=t + 1, samples_file=f"samples/{fname}", signal_kind="hand position, velocity and applied or measured force",
            channels=";".join(c[0] for c in CH), channel_units=";".join(c[2] for c in CH), sample_rate_hz=FS, n_samples=241,
            duration_s=241 / FS, events="", processing="none (unfiltered, per the readme)",
            trial_info=f"direction_deg={int(dirn)};velocity_coefficient={int(vco)};position_coefficient={int(pco)};error_clamp={int(ecl)}"))

nrec = len(rows)
checks.internal("rows + dropped all-NaN movements = 11 x 1856", nrec + dropped == NS * NT, f"{nrec} rows, {dropped} all-NaN movements dropped")
checks.internal("blocksize sums to 1856 movements in 16 blocks", len(bs) == 16 and bs.sum() == 1856)
checks.published("participants recruited (11; one later excluded by the authors)", NS, 11, 0, "Methods, Participants ('11 recruited, 1 excluded')")
ffn = [sum(1 for r in rows if r["subject"].endswith(f"S{s + 1:02d}") and r["condition"] in ("velocity-dependent field", "position-dependent field")) for s in range(NS)]
checks.published("force-field exposures per participant (224)", max(ffn), 224, 0, "Methods, Main experiment ('224 force-field exposures')")
fam = [r for r in rows if r["subject"].endswith("S01") and r["phase"] == "practice"]
checks.published("familiarization trials (no force field), participant 1", len(fam) + sum(1 for t in range(NT) if block_of[t] <= 3 and np.isnan(arr["xr"][t, :, 0]).all()),
                 270, 0, "Methods, Familiarization (270 trials in 3 blocks)")
checks.internal("no force-field trial in the three familiarization blocks", all(r["condition"] in ("null", "error clamp") for r in fam))
checks.published("largest velocity-dependent field (Ns/m)", max(abs(r["pert_value"]) for r in rows if r["pert_unit"] == "Ns/m"), 15, 0, "Methods, Force-field sizes (alpha = +-15 N/(m/s))")
checks.published("largest position-dependent field (N/m)", max(abs(r["pert_value"]) for r in rows if r["pert_unit"] == "N/m"), 45, 0, "Methods, Force-field sizes (delta = +-45 N/m)")
checks.published("movement distance: median final x position (m)", float(np.median(final_x)), 0.10, 0.01, "Methods (targets 10 cm apart)")
checks.published("peak force on force-field trials, median over trials (N)", float(np.nanmedian(ff_peak)), 4.5, 1.0, "Deposit readme (peak force 'about 4.5N' in both field types)")
checks.internal("movement directions are 90 or 270 deg, half each", sorted(np.unique(tgt[:, 0])) == [90, 270] and (tgt[:, 0] == 90).sum() == 928)
checks.internal("every sample file has 241 samples", all(b.count(b"\r\n") == 242 for _, b in extra[:200]))

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, lab="Smith (Harvard)",
    data_url="https://doi.org/10.5061/dryad.1rr57 (Zenodo copy: https://doi.org/10.5281/zenodo.4991069)",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes="the MATLAB arrays were written as one CSV per movement; trial conditions were decoded from the authors' target file with the readme's column meanings; participant codes replaced.",
    unit_note="movement (one reach of one participant, 241 samples in its own CSV)",
    how=[
        "One row is one movement: 1856 per participant (16 blocks, the first 3 are the 270 familiarization trials with no force field), alternating between the 90 and 270 deg directions, for 11 participants. Each movement's 241 samples (200 Hz) of position, velocity and force are in samples/. Positions and velocities are relative to the target direction: x is toward the target, y is orthogonal to it, rotated 90 deg counter-clockwise from x (readme). Force is the force (N) measured on error-clamp trials, the force applied on force-field trials, and zero on null trials (readme). The deposit does not state units; metres, m/s and N follow from the 10 cm movements and the 4.5 N peak forces in the paper.",
        "trial_info holds the authors' target-file columns used (readme): direction in deg (90 or 270), the velocity-dependent coefficient (-1, 0, 1; field = coefficient x 15 Ns/m), the position-dependent coefficient (-3, 0, 3; field = coefficient x 15 N/m = +-45 N/m) and the error-clamp flag. pert_value is the field size with the sign the authors gave (counterbalanced between exposures). condition says null, error clamp, velocity- or position-dependent field.",
        "The deposit does not say which of the 11 participants the authors excluded (paper: 1 slower than the rest), so all 11 are kept and subject_excluded_by_authors is empty. Movements with no samples at all (all NaN) are not written; the recording number is the trial number, so gaps show them. Samples that are NaN inside a movement are empty cells.",
    ],
    differences=["The paper analyses 10 of the 11 participants and separates 'solitary' from 'shared' triplets; the deposit does not mark either, so neither is applied."],
)
rows = [{**dict(age='', sex='', handedness='', time_s=''), **r} for r in rows]
stdlib.release(ID, MODULE, rows, meta, checks, extra_files=extra)
print("dropped all-NaN movements:", dropped)
