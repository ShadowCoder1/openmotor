"""Sullivan, Buggeln, Calalo, Ngo, Semrau, Carter & Cashaback (2026), eLife: involuntary visuomotor feedback responses of two people
reaching together (cursor jumps in Experiment 1, target jumps in Experiment 2), 24 pairs per experiment, KINARM End-Point robots.
One row per participant and trial (the trial files hold both partners), format 'joint_reach_probe' (new module).

Run on the PC from the site folder: python3 -I tools/std/convert/Sullivan2026_eLife.py
Source: Figshare 10.6084/m9.figshare.30132088 (CC BY 4.0), Exp1.zip (5.8 GB) and Exp2.zip (5.9 GB), 17,496 trial CSVs plus task tables each.
The archives are NOT stored: the zip central directory and then one member at a time are read with HTTP range requests (nothing larger than
an 8 MB block is held). The authors' analysis code (github CashabackLab/Involuntary-Visuomotor-HHI-Analysis-Code) was read as text only.
Measures follow that code: 5th-order 14 Hz zero-phase Butterworth on hand kinematics and forces; movement onset at 5 cm/s; reach+hold time =
first moment the centre cursor has stayed 500 ms in the participant's own target (timing 'good' between 1400 and 1600 ms); probe force epochs
180-230, 230-300, 300-400 ms after jump command + 42 ms screen delay.
"""
import csv
import io
import json
import math
import shutil
import statistics as st
import sys
import time
import zipfile
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss

HERE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_DIR))
sys.path.insert(0, str(HERE_DIR.parent))
import stdlib  # noqa: E402
from Sullivan2026_eLife_rf import RangeFile  # noqa: E402

ID = "Sullivan2026_eLife"
CACHE = stdlib.HERE / "tmp_samples" / ID
SAMPLES = CACHE / "samples"
ZIPS = {1: ("https://ndownloader.figshare.com/files/57981328", 5772326004),
        2: ("https://ndownloader.figshare.com/files/57981664", 5933120935)}
checks = stdlib.Checks()
SCREEN_DELAY = 42
HOLD = 500
MIN_T, MAX_T = 900 + HOLD, 1100 + HOLD
REL_W, IRR_W = 0.0105, 0.2
TARGET_NAMES = {0: "joint_irrelevant", 1: "p1_relevant", 2: "p2_relevant", 3: "joint_relevant", 4: "washout"}
SELF_PARTNER = {0: ("irrelevant", "irrelevant"), 1: ("relevant", "irrelevant"), 2: ("irrelevant", "relevant"),
                3: ("relevant", "relevant"), 4: ("washout", "washout")}
WIDTH = {"relevant": REL_W, "irrelevant": IRR_W, "washout": (REL_W + IRR_W) / 2}
B, A = ss.butter(5, 14.0, fs=1000, btype="low", output="ba")
KIN = ["Right_HandX", "Right_HandY", "Right_HandXVel", "Right_HandYVel", "Right_FS_ForceX",
       "Left_HandX", "Left_HandY", "Left_HandXVel", "Left_HandYVel", "Left_FS_ForceX"]
EV = {"E_PERTURBATION_LEFT": "perturbation", "E_PERTURBATION_RIGHT": "perturbation", "E_PROBE_LEFT": "probe", "E_PROBE_RIGHT": "probe",
      "E_PROBE_NO_JUMP": "null_probe", "E_REGULAR_TRIAL": "regular"}


def num(x):
    if x is None or x != x:
        return ""
    return "%.8g" % (0.0 if float(x) == 0 else float(x))


def tf(x):
    return "True" if x else "False"


def wait_disk(need=3.2e9):
    while shutil.disk_usage(CACHE).free < need:
        print("[wait] PC disk low", flush=True)
        time.sleep(300)


def sample_csv(d, s, e, pre):
    idx = np.arange(s, e + 1, 10)
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["t_ms", "hand_x_m", "hand_y_m", "force_x_n", "center_cursor_x_m", "center_cursor_y_m"])
    hx, hy, fx = d[f"{pre}_HandX"], d[f"{pre}_HandY"], d[f"{pre}_FS_ForceX"]
    cx, cy = d[f"{'p1' if pre == 'Right' else 'p2'}_center_cursor_pos_x"], d[f"{'p1' if pre == 'Right' else 'p2'}_center_cursor_pos_y"]
    for i in idx:
        w.writerow([int(i - s), num(hx[i]), num(hy[i]), num(fx[i]), num(cx[i]), num(cy[i])])
    return out.getvalue()


def trial_rows(data, target_type, pert_type=0):
    df = pd.read_csv(io.BytesIO(data), na_values=["NaN"], dtype={"Event_Codes": str})
    df.columns = [c.replace(" ", "_") for c in df.columns]
    valid = df["Right_HandX"].notna().to_numpy()
    n = int(valid.sum())
    assert valid[:n].all() and not valid[n:].any(), "non-contiguous valid rows"
    df = df.iloc[:n].reset_index(drop=True)
    codes = df["Event_Codes"].fillna("").str.replace("\x00", "", regex=False).str.strip()
    ev = {c: int(i) for i, c in zip(codes.index, codes) if c}
    s, e = ev["E_TRIAL_START_SIGNAL"], ev["E_TRIAL_COMPLETED"]
    kind = [(k, v) for k, v in EV.items() if k in ev]
    if not kind and "E_TRIAL_TIME_OUT" in ev and pert_type == 0:
        kind = [("E_REGULAR_TRIAL", "regular")]  # a timed-out familiarization trial carries no trial-type event
        ev["E_REGULAR_TRIAL"] = ev["E_TRIAL_TIME_OUT"]
    assert len(kind) == 1, kind
    code, ttype = kind[0][0], kind[0][1]
    j_idx = ev[code]
    d = {c: df[c].to_numpy(dtype=float) for c in df.columns if c not in ("Event_Codes",)}
    f = {c: ss.filtfilt(B, A, d[c]) for c in KIN}
    out = []
    selfp, partp = SELF_PARTNER[target_type]
    for role, pre, cc in ((1, "Right", "p1"), (2, "Left", "p2")):
        mine = selfp if role == 1 else partp
        other = partp if role == 1 else selfp
        # SELF_PARTNER is written from the p1 point of view
        if role == 2:
            mine, other = partp, selfp
        speed = np.hypot(f[f"{pre}_HandXVel"][s:e + 1], f[f"{pre}_HandYVel"][s:e + 1])
        k = np.nonzero(speed > 0.05)[0]
        onset = float(k[0] + 1) if len(k) else None
        tcol = f"{cc}_target_xpos"
        if tcol in d:  # Experiment 2 logs the (jumping) target position; 0 before the first update means the nominal position
            tx = np.where(d[tcol] == 0, 0.54 if role == 1 else -0.54, d[tcol])[s:e + 1]
        else:
            tx = 0.54 if role == 1 else -0.54
        w = WIDTH[mine]
        ccx, ccy = d[f"{cc}_center_cursor_pos_x"][s:e + 1], d[f"{cc}_center_cursor_pos_y"][s:e + 1]
        inside = ((ccx <= tx + w / 2) & (ccx >= tx - w / 2) & (ccy <= 0.3 + 0.00625) & (ccy >= 0.3 - 0.00625)).astype(float)
        cs = np.concatenate([[0.0], np.cumsum(inside)])
        roll = cs[HOLD:] - cs[:-HOLD] if len(inside) >= HOLD else np.array([])
        hit = np.nonzero(roll == HOLD)[0]
        rph = float(hit[0] + HOLD) if len(hit) else None  # timepoint (1-based) of the last sample of the first full 500 ms hold
        if rph is None:
            tres = "none"
        elif MIN_T < rph < MAX_T:
            tres = "good"
        elif rph <= MIN_T:
            tres = "early"
        else:
            tres = "late"
        fi = s + int(rph) - 1 if rph is not None else e
        row = dict(role=f"p{role}", self_target=mine, partner_target=other, movement_onset_ms=onset,
                   reach_plus_hold_ms=rph, reach_time_ms=(rph - HOLD) if rph is not None else None, timing_result=tres,
                   start_x_m=d[f"{pre}_HandX"][s], final_x_m=f[f"{pre}_HandX"][fi], final_y_m=f[f"{pre}_HandY"][fi],
                   trial_type=ttype if target_type != 4 else "washout_regular", jump_direction="none", jump_onset_ms=None,
                   force_180_230_n=None, force_230_300_n=None, force_300_400_n=None)
        if ttype in ("perturbation", "probe"):
            jc = d[tcol] if tcol in d else d[f"{cc}_center_cursor_pos_x"]  # target x in Experiment 2, centre cursor x in Experiment 1
            dx = jc[j_idx + 100] - jc[j_idx - 5]
            assert abs(dx) > 0.01, dx
            row["jump_direction"] = "right" if dx > 0 else "left"
        if ttype != "regular":
            row["jump_onset_ms"] = float(j_idx - s)
        if ttype in ("probe", "null_probe"):
            fo = f[f"{pre}_FS_ForceX"]
            base = j_idx + SCREEN_DELAY
            for key, (a, b) in (("force_180_230_n", (180, 230)), ("force_230_300_n", (230, 300)), ("force_300_400_n", (300, 400))):
                row[key] = float(fo[base + a: base + b + 1].mean())
        row["_sample"] = sample_csv(d, s, e, pre)
        row["_tp"] = int(df["Trial_Protocol"].iloc[0])
        row["_block"] = int(df["Block_Number"].iloc[0])
        row["_btrial"] = int(df["Block_Trial_Number"].iloc[0])
        row["_etrial"] = int(df["Experiment_Trial_Number"].iloc[0])
        out.append(row)
    return out


# ---------------------------------------------------------------- stage 1: one member at a time over HTTP range requests
CACHE.mkdir(parents=True, exist_ok=True)
SAMPLES.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for exp in (1, 2):
    url, size = ZIPS[exp]
    if all((CACHE / f"Exp{exp}_Sub{k}.json").exists() for k in range(1, 25)):
        continue
    z = zipfile.ZipFile(RangeFile(url, size))
    tables = {}
    for i in z.infolist():
        if i.filename.startswith("task_tables/Sub") and i.filename.endswith("Trial_Table.csv"):
            sub = i.filename.split("/")[1].replace("Trial_Table.csv", "")
            rd = list(csv.DictReader(io.StringIO(z.read(i).decode("utf-8", "replace"))))
            tables[sub] = {int(r["Trial_Num"]): (int(float(r["Target Type"])), int(float(r["Perturbation Type"]))) for r in rd}
    members = defaultdict(list)
    for i in z.infolist():
        if i.filename.endswith(".csv") and not i.filename.startswith("task_tables"):
            members[i.filename.split("/")[0]].append(i)
    for sub, infos in members.items():
        cache = CACHE / f"Exp{exp}_{sub}.json"
        if cache.exists():
            continue
        rows = []
        for i in infos:
            wait_disk()
            data = z.read(i)
            # Experiment Trial Number sits in the file; the target type comes from the subject's Trial_Table
            head = data[:4000].decode("utf-8", "replace").splitlines()
            idxcol = head[0].split(",").index("Experiment Trial Number")
            en = int(float(head[1].split(",")[idxcol]))
            for r in trial_rows(data, *tables[sub][en]):
                r["target_type"] = tables[sub][en][0]
                r["_sample_name"] = None
                rows.append(r)
        for r in rows:
            r["pair"] = int(sub.replace("Sub", ""))
        for r in rows:
            samp = r.pop("_sample")
            sub_idx = 48 * (exp - 1) + 2 * (r["pair"] - 1) + (1 if r["role"] == "p1" else 2)
            name = f"{ID}_S{sub_idx:04d}_trial{r['_etrial']:03d}.csv"
            wait_disk()
            (SAMPLES / name).write_text(samp)
            r["_sample_name"] = name
        json.dump(rows, open(cache, "w"))
        print(f"Exp{exp} {sub}: {len(rows)} rows, {time.time() - t0:.0f} s", flush=True)

# ---------------------------------------------------------------- stage 2: table
all_rows = []
for exp in (1, 2):
    for k in range(1, 25):
        rs = json.load(open(CACHE / f"Exp{exp}_Sub{k}.json"))
        for r in rs:
            r["exp"] = exp
        all_rows += rs
all_rows.sort(key=lambda r: (r["exp"], r["pair"], r["_etrial"], r["role"]))
names = [c[1] for c in stdlib.columns_for("joint_reach_probe")]
rows = []
extra = []
for r in all_rows:
    pid = f"{ID}_Exp{r['exp']}_Sub{r['pair']}_{r['role']}"
    tname = TARGET_NAMES[r["target_type"]]
    ttype = r["trial_type"]
    jump = ttype in ("perturbation", "probe")
    row = dict.fromkeys(names, "")
    row.update(
        dataset=ID, study=ID, experiment=f"{ID}_Exp{r['exp']}", group="", subject=pid, person=pid,
        subject_excluded_by_authors="False", handedness="", population="healthy", setting="lab", device="kinarm",
        session="1", block=f"block{r['_block']}", trial=str(r["_etrial"]),
        phase=("practice" if r["_block"] == 1 else ("washout" if r["target_type"] == 4 else "test")),
        task="reaching", unit="trial", condition=tname,
        pert_type=(("visual_cursor_jump" if r["exp"] == 1 else "visual_target_jump") if jump or ttype == "null_probe" else "none"),
        pert_value="3" if jump else "0", pert_unit="cm", feedback="altered" if jump else "normal",
        excluded_by_authors="False", pair=f"{ID}_Exp{r['exp']}_Pair{r['pair']}", role=r["role"], trial_type=ttype,
        jump_direction=r["jump_direction"], jump_onset_ms=num(r["jump_onset_ms"]), self_target=r["self_target"],
        partner_target=r["partner_target"], block_number=str(r["_block"]), block_trial=str(r["_btrial"]),
        trial_protocol=str(r["_tp"]), movement_onset_ms=num(r["movement_onset_ms"]), reach_plus_hold_ms=num(r["reach_plus_hold_ms"]),
        reach_time_ms=num(r["reach_time_ms"]), timing_result=r["timing_result"], start_x_m=num(r["start_x_m"]),
        final_x_m=num(r["final_x_m"]), final_y_m=num(r["final_y_m"]), force_180_230_n=num(r["force_180_230_n"]),
        force_230_300_n=num(r["force_230_300_n"]), force_300_400_n=num(r["force_300_400_n"]),
        samples_file=f"samples/{r['_sample_name']}",
    )
    rows.append(row)
    extra.append((f"samples/{r['_sample_name']}", str(SAMPLES / r["_sample_name"])))

# ---------------------------------------------------------------- checks
by_p = defaultdict(list)
for r in all_rows:
    by_p[(r["exp"], r["pair"], r["role"])].append(r)
for e in (1, 2):
    checks.published(f"Experiment {e}: participants (24 pairs)", sum(1 for k in by_p if k[0] == e), 48, 0,
                     "Methods, Participants (24 pairs; 48 individuals per experiment)")
checks.published("participants across both experiments", len(by_p), 96, 0, "Methods, Participants (96 participants)")
checks.published("reaches per pair", st.mean(len(v) for v in by_p.values()), 729, 0, "Methods, Probe trials ('729 reaches')")
ok_counts = True
for v in by_p.values():
    c = Counter((r["target_type"], r["trial_type"], r["jump_direction"]) for r in v)
    for t in range(4):
        ok_counts &= (c[(t, "regular", "none")] == 81 and c[(t, "perturbation", "left")] == 20 and c[(t, "perturbation", "right")] == 20
                      and c[(t, "probe", "left")] == 10 and c[(t, "probe", "right")] == 10 and c[(t, "null_probe", "none")] == 10)
    ok_counts &= sum(c[(4, k, j)] for k in ("washout_regular",) for j in ("none",)) == 125
checks.published("experimental trials per participant (4 conditions x (81 + 40 + 30))", st.mean(
    sum(1 for r in v if r["target_type"] != 4) for v in by_p.values()), 604, 0,
    "Methods, Experimental design (81 non-perturbation, 40 perturbation, 30 probe trials per block)")
checks.published("familiarization (50) plus washout (3 x 25) trials per participant", st.mean(
    sum(1 for r in v if r["target_type"] == 4) for v in by_p.values()), 125, 0, "Methods, Experimental design")
checks.published("probe trials per participant (4 blocks x 30)", st.mean(
    sum(1 for r in v if r["trial_type"] in ("probe", "null_probe")) for v in by_p.values()), 120, 0,
    "Methods, Probe trials (120 probe trials across the four experimental blocks)")
checks.internal("each condition has 81 regular, 20+20 perturbation, 10+10 probe and 10 null-probe trials for every participant", ok_counts)
checks.internal("both partners have a row for every trial file (rows = 2 x trial files)", len(rows) == 2 * 729 * 48)
jd = Counter((r["exp"], r["role"], r["jump_direction"]) for r in all_rows if r["trial_type"] == "probe")
checks.internal("every probe has a left or right jump in the participant's display", all(k[2] in ("left", "right") for k in jd))
gp = [r["timing_result"] for r in all_rows if r["target_type"] != 4]
checks.internal("the reach+hold rule finds a held target on at least 95% of experimental trials", sum(g != "none" for g in gp) / len(gp) > 0.95,
                f"{sum(g == 'none' for g in gp)} of {len(gp)} without")


def interaction_F(exp):
    d = defaultdict(dict)
    for (e, pr, role), v in by_p.items():
        if e != exp:
            continue
        cond = {}
        for r in v:
            if r["trial_type"] == "probe" and r["force_180_230_n"] is not None:
                cond.setdefault((r["self_target"], r["partner_target"]), {}).setdefault(r["jump_direction"], []).append(r["force_180_230_n"])
        resp = {k: st.mean(x["left"]) - st.mean(x["right"]) for k, x in cond.items()}
        d[(pr, role)] = resp
    vals = []
    for resp in d.values():
        vals.append((resp[("irrelevant", "relevant")] - resp[("irrelevant", "irrelevant")])
                    - (resp[("relevant", "relevant")] - resp[("relevant", "irrelevant")]))
    n = len(vals)
    return (st.mean(vals) / (st.stdev(vals) / math.sqrt(n))) ** 2, d


F = {}
for e in (1, 2):
    F[e], dd = interaction_F(e)
    print("interaction F exp", e, F[e], flush=True)
json.dump(F, open(CACHE / "F.json", "w"))
for e, rep in ((2, 20.54),):
    checks.published(f"Experiment {e}: self x partner target interaction on the involuntary (180-230 ms) response, F[1,47]", F[e], rep, 0.005,
                     "Results, Visuomotor feedback responses")

# ---------------------------------------------------------------- release
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json", encoding="utf-8")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, lab="Cashaback (University of Delaware)",
    data_url="https://doi.org/10.6084/m9.figshare.30132088", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor format from the per-trial KINARM CSV files (one trial file holds both partners, so each gives two "
            "rows), participant codes replaced, per-trial measures recomputed with the authors' published rules, samples decimated.",
    unit_note="participant reach on one trial of a pair",
    how=[
        "One row is one participant's reach on one trial of a pair: 729 trials x 48 participants in each of the two experiments "
        "(24 pairs; Experiment 1 cursor jumps, Experiment 2 target jumps), every trial file the authors shared, in order.",
        "Measures follow the authors' analysis code and Methods: hand data low-pass filtered (5th order, 14 Hz, zero phase, applied to each "
        "trial's recording; the authors filtered the whole session); movement onset at 5 cm/s; reach+hold = first time the centre cursor "
        "stayed 500 ms in the participant's own target; on probe and null-probe trials the lateral force against the force channel is "
        "averaged 180-230, 230-300 and 300-400 ms after the jump command plus the 42 ms screen delay. Forces are in each robot's own frame; "
        "the paper's visuomotor feedback response is the left-jump minus right-jump difference of these means.",
        "p1 is the authors' Right-hand channels, p2 the Left-hand channels (mirrored display); jump direction is read from each participant's own "
        "centre cursor and is assumed to be left when it moves toward negative x in that frame. self_target and partner_target are given from "
        "each participant's own point of view.",
        "samples/*.csv hold each participant's recording from the start signal to trial completion at 100 Hz (every 10th raw sample, no filter), "
        "so the 1 kHz force epochs are only in the table, not recoverable from the samples.",
    ],
    differences=[
        "The interaction of self and partner target on the involuntary (180-230 ms) response in Experiment 1 is F[1,47] = 61.61 in the paper; "
        "recomputed from these rows (left-jump minus right-jump mean force per participant and condition, 2 x 2 interaction contrast) it is "
        "%.2f. The likely reason is that the authors filtered each participant's whole session in one pass and these rows are filtered "
        "trial by trial. Experiment 2 reproduces (20.54)." % F[1],
    ],
)
stdlib.release(ID, "joint_reach_probe", rows, meta, checks, extra_files=extra)
