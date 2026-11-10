"""Forano & Franklin (2020), PLoS Comput Biol: timescales of motor memory formation in dual-adaptation. Two experiments of 10
participants each (vBOT, forward reaches, two opposing curl fields signalled by a workspace-shift cue), one row per trial, in the
force_field_reaching module. Phases: pre-exposure, adaptation, de-adaptation, error-clamp.

Run from the site folder: python3 -I tools/std/convert/Forano2020_PLoSCompBiol.py
Raw files (untrusted data, read only) in raw_extra/Forano2020_PLoSCompBiol/ from https://zenodo.org/records/4552826 (Dryad copy):
DualAdaptation_exp1.mat (MATLAB v5, struct DA) and DualAdaptation_exp2.mat (v7.3/HDF5, struct DAv1); both hold a 11 x 4
struct array (10 participants + an empty mean row) x 4 phases. The readme (FORANO_DualAdaptation_Readme.txt) was read as text.
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Forano2020_PLoSCompBiol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
FIELDS = ("Index", "Cue", "Block", "Ttype", "Ftype", "duration")
CCF = ("MPE", "IndexMPE", "FC", "IndexFC", "BFC")


def read_exp1(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["DA"]
    out = []
    for p in range(10):
        ph = []
        for k in range(4):
            x = m[p, k]
            d = {f: np.atleast_1d(getattr(x, f)).astype(float) for f in FIELDS}
            d["cc"] = [{f: np.atleast_1d(getattr(x.CC[c], f, np.array([]))).astype(float) for f in CCF} for c in range(2)]
            ph.append(d)
        out.append(ph)
    return out


def read_exp2(path):
    import h5py
    f = h5py.File(path, "r")
    g = f["DAv1"]

    def arr(ref):
        return np.atleast_1d(np.array(f[ref][()], dtype=float)).ravel()

    out = []
    for p in range(10):
        ph = []
        for k in range(4):
            d = {fld: arr(g[fld][k, p]) for fld in FIELDS}
            cg = f[g["CC"][k, p]]
            d["cc"] = [{fld: (arr(cg[fld][c, 0]) if fld in cg else np.array([])) for fld in CCF} for c in range(2)]
            ph.append(d)
        out.append(ph)
    return out


EXPS = {1: ("DualAdaptation_exp1.mat", read_exp1), 2: ("DualAdaptation_exp2.mat", read_exp2)}
PHASES = [("pre_exposure", "baseline"), ("adaptation", "learning"), ("de_adaptation", "washout"), ("error_clamp", "test")]
data = {}
for e, (fn, rd) in EXPS.items():
    checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])
    data[e] = rd(RAW / fn)

rows = []
for e in (1, 2):
    for p, ph in enumerate(data[e]):
        # counterbalancing group: which of the deposit's two field codes (Ftype 3 or 4) went with cue 1 in the adaptation phase
        a = ph[1]
        f_cue1 = sorted(set(a["Ftype"][(a["Cue"] == 1) & (a["Ftype"] > 2)].astype(int)))
        group = f"cue1_with_field_code_{f_cue1[0]}" if len(f_cue1) == 1 else ""
        checks.internal(f"exp{e} participant {p + 1}: one field code per cue in adaptation, different for the two cues",
                        len(f_cue1) == 1 and set(a["Ftype"][(a["Cue"] == 2) & (a["Ftype"] > 2)].astype(int)) == {7 - f_cue1[0]})
        for k, (pname, phase) in enumerate(PHASES):
            d = ph[k]
            n = len(d["Index"])
            for i in range(n):
                idx = int(d["Index"][i])
                cue = int(d["Cue"][i])
                ft = int(d["Ftype"][i])
                cc = d["cc"][cue - 1]
                mpe = fc = bfc = None
                if ft == 2:
                    j = np.flatnonzero(cc["IndexFC"] == idx)
                    assert len(j) == 1 and not np.any(cc["IndexMPE"] == idx)
                    fc, bfc = cc["FC"][j[0]], cc["BFC"][j[0]]
                else:
                    j = np.flatnonzero(cc["IndexMPE"] == idx)
                    assert len(j) == 1 and not np.any(cc["IndexFC"] == idx)
                    mpe = cc["MPE"][j[0]]
                ttype = "channel" if ft == 2 else ("field" if ft in (3, 4) else "null")
                rows.append(dict(
                    dataset=ID, study=ID, experiment=f"exp{e}", group=group, subject=f"e{e}p{p + 1}", person=f"e{e}p{p + 1}",
                    subject_excluded_by_authors=False, age="", sex="", time_s="", handedness="right", population="healthy", setting="lab",
                    device="vBOT planar robotic manipulandum", session=1, block=f"{pname}_block{int(d['Block'][i])}",
                    trial=idx, phase=phase, task="reaching", unit="trial", condition=f"cue{cue}_workspace_{'left' if cue == 1 else 'right'}",
                    pert_type={"field": "force_field", "channel": "channel", "null": "none"}[ttype],
                    pert_value=10 if ttype == "field" else 0, pert_unit="N s/m (magnitude)" if ttype == "field" else "",
                    feedback="normal", excluded_by_authors=False,
                    trial_type=ttype, trial_in_condition=idx, block_in_condition=int(d["Block"][i]),
                    mpe_cm="" if mpe is None else mpe,
                    force_compensation_pct="" if fc is None else 100 * fc,
                    force_compensation_sym_pct="" if bfc is None else 100 * bfc,
                    movement_duration_s=d["duration"][i]))
checks.internal("trial index runs 1..N without gaps for each participant", all(
    [r["trial"] for r in rows if r["subject"] == s] == list(range(1, 1 + sum(r["subject"] == s for r in rows)))
    for s in sorted({r["subject"] for r in rows})))

# ---------------------------------------------------------------- published numbers
for e in (1, 2):
    checks.published(f"participants analysed, experiment {e}", len(data[e]), 10, 0, "Methods, Participants (n = 10 per experiment)")
    L = [[len(d["Index"]) for d in ph] for ph in data[e]]
    pre, ada, ec = {1: (160, 752, 320), 2: (160, 960, 160)}[e]
    checks.published(f"exp{e}: pre-exposure trials per participant (10 blocks of 16)", st.mean(l[0] for l in L), pre, 0,
                     "Methods, Experimental paradigm")
    checks.published(f"exp{e}: adaptation trials per participant", st.mean(l[1] for l in L), ada, 0, "Methods, Experimental paradigm")
    checks.published(f"exp{e}: error-clamp trials per participant", st.mean(l[3] for l in L), ec, 0, "Methods, Experimental paradigm")
    de = [l[2] for l in L]
    rep_mean, rep_se = {1: (211, 30), 2: (240, 26)}[e]
    checks.published(f"exp{e}: de-adaptation trials, mean", st.mean(de), rep_mean, 1.0, "Methods, Experimental paradigm (mean +- SE)")
    checks.published(f"exp{e}: de-adaptation trials, SE", st.stdev(de) / math.sqrt(len(de)), rep_se, 3.0,
                     "Methods, Experimental paradigm (mean +- SE)")
    checks.internal(f"exp{e}: de-adaptation lasts 64 to {400 if e == 1 else 320} trials (4 to {25 if e == 1 else 20} blocks)",
                    all(64 <= x <= (400 if e == 1 else 320) for x in de))
    checks.internal(f"exp{e}: 16-trial blocks with 8 trials per cue (except the last, partial block of de-adaptation in Exp 2, "
                    "where the criterion ended the phase mid-block)", all(
        all((d["Block"] == b).sum() == 16 and ((d["Block"] == b) & (d["Cue"] == 1)).sum() == 8
            for b in np.unique(d["Block"]) if not (k == 2 and e == 2 and b == d["Block"].max()))
        for ph in data[e] for k, d in enumerate(ph)))
    nbl = [len(np.unique(ph[1]["Block"])) for ph in data[e]]
    checks.published(f"exp{e}: adaptation blocks", st.mean(nbl), {1: 47, 2: 60}[e], 0, "Methods, Experimental paradigm")

rows_n = len(rows)
checks.internal("every trial of every participant kept", rows_n == sum(len(d["Index"]) for e in data for ph in data[e] for d in ph))
checks.internal("error-clamp phase is channel trials only; pre-exposure has no field trials", all(
    r["trial_type"] == "channel" for r in rows if r["phase"] == "test") and not any(
    r["trial_type"] == "field" for r in rows if r["phase"] == "baseline"))
checks.internal("each trial has exactly one of MPE and force compensation", all(
    (r["mpe_cm"] == "") != (r["force_compensation_pct"] == "") for r in rows))

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Franklin (TU Munich)",
    data_url="https://doi.org/10.5061/dryad.d2547d804",
    license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor format with participant codes replaced; the authors' per-trial summaries are kept "
            "(force traces, filtered kinematics and group means are not included).",
    unit_note="trial of one participant",
    how=[
        "One row is one forward reach of one participant (10 participants in each of two experiments; 1312 to 1632 trials "
        "per participant in exp1 and 1362 to 1600 in exp2, because de-adaptation ended on the authors' criterion). Phases: pre-exposure (null), adaptation to two opposing curl fields signalled by a 10 cm "
        "workspace shift (cue 1 left, cue 2 right), de-adaptation, and error-clamp (channel trials only); channel trials are "
        "scattered through the earlier phases.",
        "Null and field trials carry mpe_cm, the authors' signed maximum perpendicular error (unit not stated in the deposit; "
        "cm assumed). Channel trials carry force_compensation_pct (the authors' FC x 100, the regression slope of the lateral "
        "force on the force the field would apply, so 100 is perfect compensation) and force_compensation_sym_pct (their BFC x "
        "100, the same with the mean across the two cues subtracted, the measure plotted and modelled in the paper).",
        "The stored FC has a shared offset: in pre-exposure, where there is no field, it averages about -70 percent in experiment 1 "
        "and -50 percent in experiment 2 for both cues, which is why the paper subtracts the cue mean. The deposit does "
        "not explain this offset; treat force_compensation_sym_pct as the interpretable measure.",
        "group says which of the deposit's two field codes (3 or 4) went with cue 1 (counterbalanced across participants); the "
        "deposit does not say which code is clockwise. The 3 recruited participants the authors excluded for not adapting are "
        "not in the deposit.",
    ],
    differences=["The Zenodo and Dryad records list CC0, while the deposit's own readme states CC BY 4.0; the stricter "
                 "readme license is used here."],
)
stdlib.release(ID, "force_field_reaching", rows, meta, checks)
