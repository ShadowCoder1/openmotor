"""Heald, Lengyel & Wolpert (2021), Nature: contextual inference underlies the learning of sensorimotor repertoires (the COIN
model). 40 adults make 12 cm forward reaches with a vBOT planar manipulandum in velocity-dependent curl fields: Experiment 1
(16 participants: 8 spontaneous recovery, 8 evoked recovery; 340 trials each) and Experiment 2 (memory updating, 24
participants; 2128 trials each, a cued control point signals the field). One row per trial in the force-field reaching format.

Run from the site folder: python3 -I tools/std/convert/Heald2021_Nature.py
Raw files (untrusted data, read only) in raw_extra/Heald2021_Nature/ from the Zenodo copy of the Dryad deposit
(https://doi.org/10.5281/zenodo.5516490 = Dryad 10.5061/dryad.m63xsj42r, CC0): spontaneous_recovery_participant1-8.mat,
evoked_recovery_participant1-8.mat, memory_updating_participant1-24.mat, README.txt. Only the TrialData table of each file is
read (a MATLAB 'table' object, parsed from the file's function workspace; nothing is executed). The robot position, velocity and
force series (RobotPosition, RobotVelocity, RobotForces, 1 kHz, up to 2569 frames per trial) are NOT converted: they come to
about 4.5 GB and no trial-wise measure from them is re-derived here.
"""
import io
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Heald2021_Nature"
RAW = stdlib.raw_dir(ID)
MODULE = "force_field_reaching"
names = [c[1] for c in stdlib.columns_for(MODULE)]
checks = stdlib.Checks()


def num(x):
    if x is None or x != x:
        return ""
    return "%.8g" % (0.0 if x == 0 else float(x))


def read_table(f):
    """Final TrialData table of a file: dict column -> list or array. The file holds earlier copies of the table with extra
    columns (Trial, TargetAngle); the last one has exactly the variables of the authors' README."""
    m = sio.loadmat(str(f), variable_names=["__function_workspace__"])
    ws = m["__function_workspace__"].tobytes()
    r = MatFile5Reader(io.BytesIO(ws[8:]), byte_order="<", struct_as_record=False, squeeze_me=False)
    r.mat_stream.seek(0)
    r.initialize_read()
    hdr, _ = r.read_var_header()
    c = r.read_var_array(hdr, process=False)[0, 0].MCOS[0][2].ravel()
    tabs, i = [], 0
    while i < len(c):
        x = c[i]
        if (isinstance(x, np.ndarray) and x.dtype == object and x.ndim == 2 and x.shape[0] == 1 and x.size >= 8
                and i + 5 < len(c) and isinstance(c[i + 5], np.ndarray) and c[i + 5].dtype == object and c[i + 5].shape == x.shape):
            nm = [str(y.ravel()[0]) for y in c[i + 5].ravel()]
            cols = []
            for y in x.ravel():
                if y.dtype == object:
                    cols.append([str(z.ravel()[0]) if isinstance(z, np.ndarray) and z.size else str(z) for z in y.ravel()])
                else:
                    cols.append(np.asarray(y, dtype=float))
            tabs.append(dict(zip(nm, cols)))
            i += 6
        else:
            i += 1
    t = tabs[-1]
    for t2 in tabs[:-1]:
        for k in t:
            if k in t2:
                a, b = t[k], t2[k]
                assert (a == b) if isinstance(a, list) else np.array_equal(a, b, equal_nan=True), (f, k)
    return t


EXPS = [("spontaneous_recovery", "Experiment1", "spontaneous_recovery", 8), ("evoked_recovery", "Experiment1", "evoked_recovery", 8),
        ("memory_updating", "Experiment2", "", 24)]
rows, per = [], {}
gaps = {}


def sub_key(fname, p):
    return f"{fname}_{p}"


for fname, exp, group, n in EXPS:
    for p in range(1, n + 1):
        t = read_table(RAW / f"{fname}_participant{p}.mat")
        N = len(t["TrialNumber"])
        bn = t["BlockName"]
        ft = t["FieldType"].ravel().astype(int)
        gain = t["FieldConstants"][:, 0]
        cp = t["ControlPoint"].ravel() if "ControlPoint" in t else None
        tn_gap = int(t["TrialNumber"].ravel()[-1]) - N  # trial numbers the deposit skips
        gaps[sub_key(fname, p)] = tn_gap
        assert (np.diff(t["TrialNumber"].ravel()) > 0).all()
        assert (np.abs(t["TargetPosition"] - [0, 2]) < 1e-9).all()
        runs, off = [], 0
        for k, g in itertools.groupby(bn):
            m = len(list(g))
            runs.append((k, off, m))
            off += m
        phase = [""] * N
        if exp == "Experiment1":
            ph = {"Null": "baseline", "FieldA": "learning", "PostRest": "learning", "FieldB": "other", "Clamp150": "test", "Clamp146": "test",
                  "Clamp2": "test", "FieldA2": "test"}
            for k, a, m in runs:
                for j in range(a, a + m):
                    phase[j] = ph[k]
        else:
            first_d = next(i for i, (k, a, m) in enumerate(runs) if k == "Double")
            last_e = max(i for i, (k, a, m) in enumerate(runs) if k == "Exposure")
            for i, (k, a, m) in enumerate(runs):
                if i < first_d - (1 if runs[first_d - 1][0] == "PostRest" else 0):
                    pname = "baseline"
                elif i <= last_e + 1:
                    pname = "learning"
                else:
                    pname = "test"
                for j in range(a, a + m):
                    phase[j] = pname
        # block number within the experiment's phases: the authors' BlockName is kept in 'block'
        sub = f"{fname}_{p}"
        per[sub] = dict(exp=exp, fname=fname, ft=ft, phase=phase, runs=runs, bn=bn, gain=gain, N=N, cp=cp, t=t)
        for j in range(N):
            tt = ["null", "field", "channel"][ft[j]]
            g_ns_m = round(float(gain[j]) * 100.0, 6)
            row = dict.fromkeys(names, "")
            row.update(
                dataset=ID, study=ID, experiment=f"{ID}_{exp}", group=group, subject=sub, person=sub,
                subject_excluded_by_authors="False", handedness="right", population="healthy", setting="lab",
                device="vBOT planar robotic manipulandum", session="1", block=bn[j], trial=str(j + 1), phase=phase[j],
                task="reaching", unit="trial", condition="",
                pert_type={"null": "none", "field": "force_field", "channel": "error_clamp"}[tt],
                pert_value=num(g_ns_m) if tt == "field" else "0", pert_unit="Ns/m", feedback="normal",
                excluded_by_authors="True" if t["MissTrial"].ravel()[j] else "False",
                trial_type=tt, trial_in_condition=str(int(t["BlockTrial"].ravel()[j])),
                force_compensation_pct=num(100.0 * t["Adaptation"].ravel()[j]) if tt == "channel" else "",
                movement_duration_s=num(t["MovementDurationTime"].ravel()[j]),
                reaction_time_s=num(t["MovementReactionTime"].ravel()[j]),
                target_direction_deg="90", hand="right",
                control_point=str(int(cp[j])) if cp is not None else "")
            rows.append(row)

# ---------------------------------------------------------------- checks
S = {k: v for k, v in per.items()}
sp = [v for k, v in S.items() if v["fname"] == "spontaneous_recovery"]
ev = [v for k, v in S.items() if v["fname"] == "evoked_recovery"]
mu = [v for k, v in S.items() if v["fname"] == "memory_updating"]
checks.published("participants in total (two experiments)", len(S), 40, 0, "Methods, Participants (40 across two experiments)")
checks.published("Experiment 1 participants", len(sp) + len(ev), 16, 0, "Methods, Participants (16 in Experiment 1)")
checks.published("spontaneous recovery group", len(sp), 8, 0, "Methods, Participants (n = 8)")
checks.published("evoked recovery group", len(ev), 8, 0, "Methods, Participants (n = 8)")
checks.published("Experiment 2 (memory updating) participants", len(mu), 24, 0, "Methods, Participants (24)")


def count(v, block=None, ftype=None, phase=None):
    return sum(1 for j in range(v["N"]) if (block is None or v["bn"][j] == block) and (ftype is None or v["ft"][j] == ftype)
               and (phase is None or v["phase"][j] == phase))


for grp, vs in (("spontaneous", sp), ("evoked", ev)):
    checks.published(f"{grp}: trials per participant (50 + 125 + 15 + 150)", float(np.median([v["N"] for v in vs])), 340, 0,
                     "Methods, Experiment 1 trial structure")
    checks.published(f"{grp}: pre-exposure trials per participant", float(np.median([count(v, "Null") for v in vs])), 50, 0,
                     "Methods, Experiment 1 (pre-exposure: 50 null trials)")
    checks.published(f"{grp}: P+ exposure trials per participant", float(np.median([count(v, "FieldA") + count(v, "PostRest") for v in vs])),
                     125, 0, "Methods, Experiment 1 (exposure: 125 P+ trials)")
    checks.published(f"{grp}: counter-exposure P- trials per participant", float(np.median([count(v, "FieldB") for v in vs])), 15, 0,
                     "Methods, Experiment 1 (15 P- trials)")
    checks.published(f"{grp}: trials in the channel phase per participant", float(np.median([count(v, phase="test") for v in vs])), 150, 0,
                     "Methods, Experiment 1 (150 channel trials)")
    checks.published(f"{grp}: exposure-phase trials with the 45 s rest after trial 60 (5 P+ trials prepended)",
                     float(np.median([count(v, "PostRest") for v in vs])), 5, 0, "Methods, Experiment 1 (45 s rest, 5 extra P+ trials)")
checks.published("evoked group: P+ evoker trials replacing channel trials in the channel phase (3rd and 4th)",
                 float(np.median([count(v, phase="test", ftype=1) for v in ev])), 2, 0, "Methods, Experiment 1 (evoker trials)")
checks.published("spontaneous group: P+ trials in the channel phase", float(np.median([count(v, phase="test", ftype=1) for v in sp])), 0, 0,
                 "Methods, Experiment 1 (the spontaneous group has no evoker trials)")
checks.published("evoked group: evoker trials are the 3rd and 4th trials of the channel phase",
                 float(all([j - min(i for i in range(v["N"]) if v["phase"][i] == "test") + 1 for j in range(v["N"])
                            if v["phase"][j] == "test" and v["ft"][j] == 1] == [3, 4] for v in ev)), 1, 0,
                 "Methods, Experiment 1 (3rd and 4th channel-phase trials replaced)")
checks.published("pre-exposure: one channel trial per block of 10 (5 in 50 trials)",
                 float(np.median([count(v, "Null", 2) for v in sp + ev])), 5, 0, "Methods, Experiment 1 (one channel trial per 10-trial block)")


def triplets(v, phase):
    return sum(1 for k, a, m in v["runs"] if k.startswith("Triplet") and v["phase"][a] == phase)


checks.published("memory updating: pre-training triplets (16 blocks, one triplet each)", float(np.median([triplets(v, "baseline") for v in mu])), 16,
                 0, "Methods, Experiment 2 (pre-training: 16 blocks)")
checks.published("memory updating: training triplets (24 blocks x 2 triplets)", float(np.median([triplets(v, "learning") for v in mu])), 48, 0,
                 "Methods, Experiment 2 (training: 24 blocks, 2 triplets each)")
checks.published("memory updating: post-training triplets (32 blocks, one triplet each)", float(np.median([triplets(v, "test") for v in mu])), 32,
                 0, "Methods, Experiment 2 (post-training: 32 blocks)")
checks.published("memory updating: every triplet has two channel trials around one exposure trial",
                 float(all(all(tuple(v["ft"][a:a + m]) == (2, 1, 2) for k, a, m in v["runs"] if k.startswith("Triplet")) for v in mu)), 1, 0,
                 "Methods, Experiment 2 (triplet: two channel trials bracketing one exposure trial)")
checks.published("memory updating: force-field trials per training block", float(np.median([m for v in mu for k, a, m in v["runs"] if k == "Exposure"])),
                 32, 0, "Methods, Experiment 2 (32 force-field trials per training block)")
checks.published("memory updating: equal numbers of P1+ and P2- exposure trials (share of + gain among field trials in Exposure runs)",
                 float(np.mean([np.mean([v["gain"][j] > 0 for k, a, m in v["runs"] if k == "Exposure" for j in range(a, a + m)]) for v in mu])), 0.5, 1e-9,
                 "Methods, Experiment 2 (equal numbers of P1+ and P2-)")
checks.published("field gain magnitude (Ns/m)", float(np.median([abs(float(r["pert_value"])) for r in rows if r["trial_type"] == "field"])),
                 15, 1e-6, "Methods, Apparatus and perturbations (gain +-15 Ns/m)")
checks.internal("every memory-updating participant has 96 triplets, 25 exposure runs of 32 trials and 292 channel trials; the number of null trials varies (2128-2136 rows)",
                all(v["N"] >= 2128 and sum(1 for k, a_, m in v["runs"] if k.startswith("Triplet")) == 96 and count(v, ftype=1) == 896
                    and count(v, ftype=2) == 292 for v in mu), str(sorted({v["N"] for v in mu})))
checks.internal("rows kept = all rows of the 40 files", len(rows) == sum(v["N"] for v in S.values()))
checks.internal("channel trials carry an adaptation value; others none", all((r["force_compensation_pct"] != "") == (r["trial_type"] == "channel")
                                                                         for r in rows))
checks.internal("no miss trials flagged", all(r["excluded_by_authors"] == "False" for r in rows))
checks.internal("each participant's field sign is constant... gains are +-0.15 on all field trials",
                all(abs(abs(float(r["pert_value"])) - 15) < 1e-6 for r in rows if r["trial_type"] == "field"))

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Wolpert (Cambridge / Columbia)", data_url="https://doi.org/10.5061/dryad.m63xsj42r",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes="read the TrialData table of each participant's MATLAB file into the OpenMotor force-field reaching format; adaptation (a fraction) "
            "written as percent; participant codes replaced; the 1 kHz robot position, velocity and force series are not included.",
    unit_note="trial of one participant sequence",
    how=[
        "One row is one reach to a target 12 cm straight ahead with a vBOT manipulandum. trial_type follows the authors' FieldType (0 null, "
        "1 curl field, 2 channel). Experiment 1 (16 participants): 50 null trials, 125 trials in a curl field (P+, block FieldA, plus PostRest, "
        "the 5 trials after the 45 s rest), 15 trials in the opposite field (P-, block FieldB), then 150 channel trials (block Clamp150; "
        "the evoked recovery group's block labels are Clamp2, FieldA2 (two P+ trials in place of the 3rd and 4th) and Clamp146 or Clamp150). Experiment 2 (memory updating, 24 participants): 16 "
        "pre-training blocks and 32 post-training blocks, each a run of null trials and one triplet of channel, exposure and channel trial, "
        "and a training phase of force-field exposure runs (Double and Exposure blocks) with triplets; control_point is the cued control point "
        "(-1 left, 1 right).",
        "pert_value is the signed field gain in Ns/m (the deposit's FieldConstants times 100: 15 or -15, counterbalanced across participants "
        "and contexts); channel trials are error_clamp with 0. force_compensation_pct is the authors' Adaptation on channel trials times 100: "
        "the slope of the force into the channel wall on the force that would fully compensate the field, signed so that +100 means complete "
        "adaptation to a field with gain +15 Ns/m and -100 to one with -15 (README). trial_in_condition is the trial number within the authors' "
        "block label (BlockTrial); block keeps the authors' label; the phases (baseline, learning, other for the P- trials, test) were assigned "
        "from the block labels and the paper's design.",
        "The deposit's many-megabyte position, velocity and force series are not converted, and no error measure (MPE) is derived from them; "
        "mpe_cm is empty. Target direction is 90 deg (straight ahead) for every trial: the target is always at (0, 2) cm with the home position "
        "at (0, -10) cm.",
    ],
    differences=["The memory-updating participants have 2128 to 2136 rows, not one fixed number: the paper gives the washout (null) trials before each triplet as a variable number, and the extra rows are all null trials. For participant 13 the authors' TrialNumber jumps from 1666 to 1669 once; the rows are kept as they are.",
                 "The deposit's memory-updating files hold 25 exposure runs of 32 trials; the paper describes 24 training blocks with two triplets each "
                 "(48 triplets, which are present); the 25th run is the last exposure before the post-training blocks, with no triplets after it."],
)
stdlib.release(ID, MODULE, rows, meta, checks)
