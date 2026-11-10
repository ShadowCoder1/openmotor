"""Hillman, Botthof, Forrence & McDougle (2024), Psychological Science: three experiments on motor working memory (MWM) with a
KINARM End-Point robot. People felt robot-guided reaching movements without vision (the hand and the targets were never
visible) and later recalled one movement's end point with a reach, with the encoding hand or after switching hands. One row
per recall reach, in the 90-column reaching format, three experiments: _Exp1 (set size 1 or 4), _Exp2 (irrelevant movements or
a matched delay) and _Exp3 (a concurrent visuospatial dot memory task).

Run from the site folder: python3 -I tools/std/convert/Hillman2024_PsycholSci.py
Raw file (untrusted data, read only) in raw_extra/Hillman2024_PsycholSci/, from https://doi.org/10.5281/zenodo.10206635
(Zenodo record 10206635, an archive of the authors' GitHub repository hhillman231/Dissociable_Codes_in_MWM, tag v.05):
  exp1.csv, exp2.csv, exp3.csv   trial-level data of the three experiments (preprocessed by the authors; participants who were
                                 excluded and trials that timed out or were lapses are not in the files)
  README.md                      column dictionary (read as text); exp*analysis.Rmd were read as text, never run.

Frame. All positions are in the robot workspace with the origin at the home position (README: 'origin is at home position
centered in front of pt'); the authors' angular errors are counter-clockwise positive from atan2(y, x), which is the OpenMotor
convention, so hand_angle = wrap(direction of the recalled end point - direction of the target) needs no sign change (checked
against the authors' ang_err of Exp. 3 below). Recalled end points are stored as hand_angle and hand_extent (the distance from
home over the target distance); the checks below recover the authors' x, y end points and Euclidean errors from the rows.
"""
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hillman2024_PsycholSci"
RAW = stdlib.raw_dir(ID)
ZIP = "Dissociable_Codes_in_MWM-v.05.zip"
PRE = "hhillman231-Dissociable_Codes_in_MWM-c2ff1dc/"
ZENODO_MD5 = "df8f1ed210f418fb356fba7b815b3a7e"  # file checksum on the Zenodo record
RMAX = {1: 16.0, 2: 17.0, 3: 12.0}  # largest target distance (cm) in each experiment
checks = stdlib.Checks()


def wrap(a):
    """Degrees into (-180, 180]."""
    w = (a + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


def tf(x):
    return "True" if x else "False"


# ---------------------------------------------------------------- read
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
checks.internal("zip matches the downloaded hash", stdlib.sha256(RAW / ZIP) == src[ZIP]["sha256"])
checks.internal("zip matches the MD5 and size on the Zenodo record 10206635",
                hashlib.md5(open(RAW / ZIP, "rb").read()).hexdigest() == ZENODO_MD5 and (RAW / ZIP).stat().st_size == 306033)
z = zipfile.ZipFile(RAW / ZIP)
D = {k: pd.read_csv(io.BytesIO(z.read(PRE + f"exp{k}.csv"))) for k in (1, 2, 3)}
d1, d2, d3 = D[1], D[2], D[3]


def ang_of(x, y):
    return np.degrees(np.arctan2(y, x))


def wrap_arr(a):
    return (a + 180.0) % 360.0 - 180.0


for k, d in D.items():
    d["err"] = wrap_arr(ang_of(d.pt_x, d.pt_y) - ang_of(d.targ_x, d.targ_y))  # recalled direction minus target direction
    d["radius"] = np.hypot(d.targ_x, d.targ_y)

# ---------------------------------------------------------------- structure of the three files
checks.internal("rows: 1866 (Exp. 1), 2201 (Exp. 2), 1592 (Exp. 3); participants 20, 24, 14",
                (len(d1), len(d2), len(d3)) == (1866, 2201, 1592) and (d1.id.nunique(), d2.id.nunique(), d3.id.nunique()) == (20, 24, 14))
checks.internal("no missing values and no repeated (participant, block, trial) in any file",
                all(not d.isna().any().any() and not d.duplicated(["id", "block", "trial"]).any() for d in D.values()))
IN_ORDER = {}
for k, d in D.items():
    IN_ORDER[k] = (d.id.ne(d.id.shift()).sum() == d.id.nunique()  # each participant's rows are contiguous
                   and all((g.block.diff().fillna(0) >= 0).all() and (g.groupby("block").trial.diff().dropna() > 0).all()
                           for _, g in d.groupby("id", sort=False)))
checks.internal("each participant's rows are contiguous and in block, trial order in all three files", all(IN_ORDER.values()))
checks.internal("block structure: Exp. 1 blocks 1-4 of up to 24 trials, Exp. 2 blocks 1-3 of up to 32, Exp. 3 blocks 1-3 of "
                "up to 48, 48 and 24",
                set(d1.block) == {1, 2, 3, 4} and d1.trial.between(1, 24).all()
                and set(d2.block) == {1, 2, 3} and d2.trial.between(1, 32).all()
                and set(d3.block) == {1, 2, 3} and d3[d3.block < 3].trial.between(1, 48).all() and d3[d3.block == 3].trial.between(1, 24).all())
checks.internal("Exp. 3: ang_err is the recalled direction minus the target direction (counter-clockwise positive) on every row",
                np.abs(wrap_arr(d3.err - d3.ang_err)).max() < 1e-9, f"max difference {np.abs(wrap_arr(d3.err - d3.ang_err)).max():.1e} deg")
checks.internal("dist_err is the Euclidean distance between the recalled and the encoded end point on every row of all files",
                all(np.abs(np.hypot(d.pt_x - d.targ_x, d.pt_y - d.targ_y) - d.dist_err).max() < 1e-6 for d in D.values()))
checks.internal("lapse rule of the Methods holds in the shared rows: |angular error| < 30 deg for set size 1 (Exp. 1) and for "
                "all of Exp. 2, < 90 deg for set size 4 (Exp. 1) and all of Exp. 3",
                d1[d1.setsize == 1].err.abs().max() < 30 and d1[d1.setsize == 4].err.abs().max() < 90
                and d2.err.abs().max() < 30 and d3.err.abs().max() < 90,
                f"max |error| {d1[d1.setsize == 1].err.abs().max():.1f}, {d1[d1.setsize == 4].err.abs().max():.1f}, "
                f"{d2.err.abs().max():.1f}, {d3.err.abs().max():.1f} deg")
checks.internal("Exp. 1 and 2: targets 15-165 deg from the +x axis at whole-number distances (8, 12, 16 cm; 8, 11, 14, 17 cm); "
                "Exp. 3: all targets 12 cm from home",
                set(np.round(d1.radius, 4)) == {8.0, 12.0, 16.0} and set(np.round(d2.radius, 4)) == {8.0, 11.0, 14.0, 17.0}
                and ang_of(d1.targ_x, d1.targ_y).between(15 - 1e-4, 165 + 1e-4).all()
                and ang_of(d2.targ_x, d2.targ_y).between(15 - 1e-4, 165 + 1e-4).all() and np.abs(d3.radius - 12).max() < 1e-3)

# Exp. 1
ss1, ss4 = d1[d1.setsize == 1], d1[d1.setsize == 4]
enc4 = ss4[["targ1", "targ2", "targ3", "targ4"]].to_numpy()
checks.internal("Exp. 1: set size 1 recalls movement 1 and repeats one target id in targ1-targ4; set size 4 has four different ids "
                "and recall_targ is the id of the movement named by recall_move",
                (ss1.recall_move == 1).all() and (ss1[["targ2", "targ3", "targ4"]].to_numpy() == ss1[["targ1"]].to_numpy()).all()
                and (ss1.recall_targ == ss1.targ1).all() and (ss4[["targ1", "targ2", "targ3", "targ4"]].nunique(axis=1) == 4).all()
                and (enc4[np.arange(len(ss4)), ss4.recall_move.to_numpy() - 1] == ss4.recall_targ.to_numpy()).all())
pos1 = d1.groupby("recall_targ")[["targ_x", "targ_y"]].nunique()
checks.internal("Exp. 1: each of the 24 target ids (2-25) is one fixed position, and every encoded id is also a recalled id",
                (pos1 == 1).all().all() and sorted(pos1.index) == list(range(2, 26))
                and set(d1[["targ1", "targ2", "targ3", "targ4"]].to_numpy().ravel()) <= set(d1.recall_targ))
# Exp. 2
pos2 = d2.groupby("targ")[["targ_x", "targ_y"]].nunique()
xt = pd.concat([d2[[f"xtra_targ{k}", f"xtra_targ{k}_x", f"xtra_targ{k}_y"]].set_axis(["id", "x", "y"], axis=1) for k in (1, 2, 3)])
xt = xt[xt.id != 42].drop_duplicates()
n_real = (d2[["xtra_targ1", "xtra_targ2", "xtra_targ3"]] != 42).sum(axis=1)
checks.internal("Exp. 2: each of the 24 target ids (2-25) is one fixed position; irrelevant-movement ids (42 = no movement) sit at "
                "the same positions; delay trials have none, 1 or 3 irrelevant movements have 1 or 3",
                (pos2 == 1).all().all() and sorted(pos2.index) == list(range(2, 26)) and len(xt) == xt.id.nunique()
                and np.allclose(xt.set_index("id").sort_index()[["x", "y"]].to_numpy(),
                                d2.drop_duplicates("targ").set_index("targ").loc[xt.id.sort_values().unique(), ["targ_x", "targ_y"]].to_numpy())
                and (n_real[d2.type == 0] == 0).all() and (n_real[(d2.type == 1) & (d2["size"] == 0)] == 1).all()
                and (n_real[(d2.type == 1) & (d2["size"] == 1)] == 3).all())
# Exp. 3
vs = d3
karr = vs.recall_vsp.to_numpy() - 1
checks.internal("Exp. 3: the cued dot is dot recall_vsp (position and colour), dot positions lie on a 12 cm ring, vsp_ang_err is the "
                "reported direction minus the cued dot's direction, one-dot trials cue dot 1 and repeat the same values for dots 2 and 3",
                np.abs(np.stack([vs[f"vsp{j}_x"] for j in (1, 2, 3)], 1)[np.arange(len(vs)), karr] - vs.recall_vsp_x).max() == 0
                and np.abs(np.stack([vs[f"vsp{j}_y"] for j in (1, 2, 3)], 1)[np.arange(len(vs)), karr] - vs.recall_vsp_y).max() == 0
                and (np.stack([vs[f"vsp{j}_color"] for j in (1, 2, 3)], 1)[np.arange(len(vs)), karr] == vs.recall_vsp_color).all()
                and np.abs(np.hypot(vs.recall_vsp_x, vs.recall_vsp_y) - 12).max() < 1e-3
                and np.abs(np.hypot(vs.pt_vsp_x, vs.pt_vsp_y) - 12).max() < 1e-6
                and np.abs(wrap_arr(ang_of(vs.pt_vsp_x, vs.pt_vsp_y) - ang_of(vs.recall_vsp_x, vs.recall_vsp_y) - vs.vsp_ang_err)).max() < 1e-9
                and set(vs.vsp_setsize) == {1, 3} and (vs[vs.vsp_setsize == 1].recall_vsp == 1).all()
                and (vs[vs.vsp_setsize == 1][["vsp2_x", "vsp2_y", "vsp2_color"]].to_numpy()
                     == vs[vs.vsp_setsize == 1][["vsp3_x", "vsp3_y", "vsp3_color"]].to_numpy()).all())
checks.internal("Exp. 3: each participant's start hand and hand switch are balanced overall (all four combinations 386-410 trials)",
                vs.groupby(["start_hand", "switch_hands"]).size().between(380, 415).all())

# ---------------------------------------------------------------- rows
names = [c[1] for c in stdlib.reach90_columns()]
ALL_ANGLES3 = sorted(set(np.round(ang_of(d3.targ_x, d3.targ_y) % 360.0, 2)))
TID3 = {a: i for i, a in enumerate(ALL_ANGLES3)}


def base(exp, pid):
    row = dict.fromkeys(names, "")
    row.update(dataset=ID, study=ID, experiment=f"{ID}_Exp{exp}", subject=f"{ID}_Exp{exp}_{pid}", person=f"{ID}_Exp{exp}_{pid}",
               id_type="lab", person_link="assumed_unique", group="", prior_exposure="unknown",
               subject_excluded_by_authors="False", device="kinarm", setting="lab", population="healthy", session=0)
    return row


def cue(exp, r):
    hand = "right" if r["start_hand"] == 1 else "left"
    sw = "yes" if r["switch_hands"] == 1 else "no"
    if exp == 1:
        ids = [r["targ1"]] if r["setsize"] == 1 else [r[f"targ{k}"] for k in (1, 2, 3, 4)]
        return f"setsize:{r['setsize']}|recall_move:{r['recall_move']}|encode_hand:{hand}|switch_hands:{sw}|encoded_targets:" \
               + "/".join(str(int(i)) for i in ids)
    if exp == 2:
        if r["type"] == 1:
            ids = [int(r[f"xtra_targ{k}"]) for k in (1, 2, 3) if int(r[f"xtra_targ{k}"]) != 42]
            n = 1 if r["size"] == 0 else 3
            return f"maintenance:{n}_irrelevant_movement{'s' if n > 1 else ''}|irrelevant_targets:" \
                   + "/".join(map(str, ids)) + f"|encode_hand:{hand}|switch_hands:{sw}"
        return f"maintenance:{'short' if r['size'] == 0 else 'long'}_delay|encode_hand:{hand}|switch_hands:{sw}"
    return f"vsp_load:{r['vsp_setsize']}|encode_hand:{hand}|switch_hands:{sw}"


rows = []
vsp_rows = []  # Exp. 3 visuospatial task, joined to the main table by participant sequence and trial
n_missing = Counter()
for exp, d in ((1, d1), (2, d2), (3, d3)):
    recs = d.to_dict("records")
    for pid in dict.fromkeys(d.id):
        g = [r for r in recs if r["id"] == pid]
        j = 0
        for bl in dict.fromkeys(r["block"] for r in g):
            gb = {r["trial"]: r for r in g if r["block"] == bl}
            for bt in range(1, max(gb) + 1):
                row = base(exp, pid)
                row.update(trial=j, trial_orig=bt, block=f"block{bl}", phase="other", break_before=tf(bt == 1),
                           target_visible_during_move="False", target_event="none", target_jump=0, probe_kind="none",
                           instr_goal="hand_to_target", instr_strategy="unspecified", instr_disclosure="none",
                           instr_report="none", fb_cued="True", hand_measure="endpoint", fb_visible="none", fb_mode="none",
                           fb_form="none", online_cursor="none", fb_endpoint_transform_only="False")
                if bt in gb:
                    r = gb[bt]
                    th_t = math.degrees(math.atan2(r["targ_y"], r["targ_x"]))
                    rad = float(round(math.hypot(r["targ_x"], r["targ_y"])))  # design distance: the authors' x, y are within 0.0003 cm of it
                    ta = th_t % 360.0
                    right = (r["start_hand"] == 1) != (r["switch_hands"] == 1)  # recall hand: the other hand on switch trials
                    tid = int(r["recall_targ"]) if exp == 1 else int(r["targ"]) if exp == 2 else TID3[round(ta, 2)]
                    row.update(trial_kind="reach", target_angle=ta, target_id=tid, cue_elements=cue(exp, r),
                               hand_used="right" if right else "left",
                               hand_angle=wrap(math.degrees(math.atan2(r["pt_y"], r["pt_x"])) - th_t),
                               hand_extent=math.hypot(r["pt_x"], r["pt_y"]) / rad, mt_ms=int(r["move_time"]),
                               excluded_by_authors="False")
                    row["target_distance_norm"] = rad / RMAX[exp]
                    if exp == 3:
                        vsp_rows.append((row["subject"], j, bl, bt, r))
                else:
                    row.update(trial_kind="missing", excluded_by_authors="True")
                    n_missing[exp] += 1
                rows.append(row)
                j += 1

rx = {e: [r for r in rows if r["experiment"].endswith(f"Exp{e}")] for e in (1, 2, 3)}
reach = {e: [r for r in rx[e] if r["trial_kind"] == "reach"] for e in (1, 2, 3)}
checks.internal("every row the authors shared is kept, in order (1866 + 2201 + 1592 reaches)",
                [len(reach[e]) for e in (1, 2, 3)] == [1866, 2201, 1592] and len(vsp_rows) == 1592
                and [(r["trial_orig"]) for r in reach[1]] == d1.trial.tolist()
                and [(r["trial_orig"]) for r in reach[2]] == d2.trial.tolist()
                and [(r["trial_orig"]) for r in reach[3]] == d3.trial.tolist())
trailing = {1: sum(24 - g.trial.max() for _, g in d1.groupby(["id", "block"])), 2: sum(32 - g.trial.max() for _, g in d2.groupby(["id", "block"])),
            3: sum((48 if b < 3 else 24) - g.trial.max() for (i, b), g in d3.groupby(["id", "block"]))}
checks.internal("placeholder rows only for gaps in the authors' trial numbers before each block's last shared trial: "
                f"{n_missing[1]} (Exp. 1), {n_missing[2]} (Exp. 2), {n_missing[3]} (Exp. 3)",
                n_missing[1] == 53 and n_missing[2] == 97 and n_missing[3] == sum(
                    g.trial.max() - len(g) for _, g in d3.groupby(["id", "block"])))
checks.internal("rows per participant: Exp. 1 up to 96 slots, Exp. 2 up to 96, Exp. 3 up to 120",
                max(Counter(r["person"] for r in rx[1]).values()) <= 96 and max(Counter(r["person"] for r in rx[2]).values()) <= 96
                and max(Counter(r["person"] for r in rx[3]).values()) <= 120)
checks.internal("hand_angle within (-180, 180], target_angle within [0, 360), mt_ms positive",
                all(-180 < r["hand_angle"] <= 180 and 0 <= r["target_angle"] < 360 and r["mt_ms"] > 0 for e in reach for r in reach[e]))

# the authors' end points and errors are recovered from the converted rows
cols = {}
for e in (1, 2, 3):
    Rr = reach[e]
    tdn = np.array([r["target_distance_norm"] for r in Rr], float)
    R = tdn * RMAX[e]
    tt = np.radians([r["target_angle"] for r in Rr])
    ee = np.radians([r["hand_angle"] for r in Rr])
    ex = np.array([r["hand_extent"] for r in Rr], float)
    px, py = ex * R * np.cos(tt + ee), ex * R * np.sin(tt + ee)
    tx, ty = R * np.cos(tt), R * np.sin(tt)
    cols[e] = dict(px=px, py=py, tx=tx, ty=ty, euc=np.hypot(px - tx, py - ty))
    dd = D[e]
    dp = max(np.abs(px - dd.pt_x).max(), np.abs(py - dd.pt_y).max())
    dt = max(np.abs(tx - dd.targ_x).max(), np.abs(ty - dd.targ_y).max())
    de = np.abs(cols[e]["euc"] - dd.dist_err.to_numpy()).max()
    checks.internal(f"Exp. {e}: the authors' recalled x, y end points are recovered from target_angle, target_distance_norm, "
                    "hand_angle and hand_extent, and their target x, y and Euclidean error to 1e-3 cm",
                    dp < 1e-6 and dt < 1e-3 and de < 1e-3,
                    f"max differences {dp:.1e} cm (end points), {dt:.1e} cm (targets), {de:.1e} cm (Euclidean error)")

# ---------------------------------------------------------------- published numbers, from the converted rows
persons = {e: OrderedDict((r["person"], 1) for r in rx[e]) for e in (1, 2, 3)}
checks.published("Exp. 1: participants analysed", len(persons[1]), 20, 0, "Results, Experiment 1 (N = 20 after exclusions)")
checks.published("Exp. 2: participants analysed", len(persons[2]), 24, 0, "Results, Experiment 2 (N = 24 after exclusions)")
checks.published("Exp. 3: participants analysed", len(persons[3]), 14, 0, "Results, Experiment 3 (N = 14 after exclusions)")
checks.published("participants recruited (65) minus those excluded (7) = participants in the three files",
                 len(persons[1]) + len(persons[2]) + len(persons[3]), 65 - 7, 0, "Abstract and Methods, Participants")


def frame(e):
    out = []
    for r, euc in zip(reach[e], cols[e]["euc"]):
        c = dict(p.split(":", 1) for p in r["cue_elements"].split("|"))
        out.append(dict(person=r["person"], euc=euc, err=r["hand_angle"], switch=c["switch_hands"] == "yes", **c))
    return pd.DataFrame(out)


def pair_t(a, b):
    diff = np.asarray(a, float) - np.asarray(b, float)
    return diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff))), diff.mean() / diff.std(ddof=1)


def circ_sd(deg):
    x = np.radians(np.asarray(deg, float))
    return math.sqrt(-2.0 * math.log(math.hypot(np.cos(x).mean(), np.sin(x).mean())))


def bf10_paired(t, n, rscale=math.sqrt(2) / 2):
    """JZS Bayes factor for a paired t test (Rouder et al., 2009), as BayesFactor::ttestBF with its default prior."""
    nu = n - 1

    def f(g):
        return ((1 + n * g * rscale ** 2) ** -0.5 * (1 + t ** 2 / ((1 + n * g * rscale ** 2) * nu)) ** (-(nu + 1) / 2)
                * (2 * math.pi) ** -0.5 * g ** -1.5 * math.exp(-1 / (2 * g)))
    return integrate.quad(f, 0, np.inf)[0] / (1 + t ** 2 / nu) ** (-(nu + 1) / 2)


def pub_t(what, t, d, rt, rd, src, check_d=True):
    checks.published(f"{what}: paired t", t, rt, 0.005, src)
    if check_d:
        checks.published(f"{what}: Cohen's d", d, rd, 0.005, src)


F1 = frame(1)
F1["setsize"] = F1.setsize.astype(int)
F1["recall_move"] = F1.recall_move.astype(int)
S1 = "Results, Experiment 1"
a = F1.groupby(["person", "setsize"]).euc.mean().unstack()
pub_t("Exp. 1: recall error, set size 4 vs 1", *pair_t(a[4], a[1]), 7.43, 1.66, S1)
b = F1.groupby(["person", "setsize", "switch"]).euc.mean().unstack("switch")
for ss, rt, rd in ((1, 5.44, 1.22), (4, 3.53, 0.79)):
    x = b.xs(ss, level="setsize")
    pub_t(f"Exp. 1: set size {ss}, switch vs no-switch error", *pair_t(x[True], x[False]), rt, rd, S1)
pi_ = F1[((F1.setsize == 1) & (F1.recall_move == 1)) | ((F1.setsize == 4) & (F1.recall_move == 4))].groupby(["person", "setsize"]).euc.mean().unstack()
pub_t("Exp. 1: proactive interference (set size 4 recall of movement 4 vs set size 1)", *pair_t(pi_[4], pi_[1]), 1.99, 0.44, S1)
c = F1[F1.setsize == 4].groupby(["person", "recall_move", "switch"]).euc.mean().unstack("switch")
for pos, rt, rd in ((1, 0.26, 0.06), (2, -1.23, -0.28), (3, 3.14, 0.70), (4, 3.86, 0.86)):
    x = c.xs(pos, level="recall_move")
    pub_t(f"Exp. 1: switch cost of the movement recalled from position {pos}", *pair_t(x[True], x[False]), rt, rd, S1)
cost = (c[True] - c[False]).unstack()
slopes = np.array([np.polyfit([1, 2, 3, 4], cost.loc[i].to_numpy(), 1)[0] for i in cost.index])
Fv = (slopes.mean() / (slopes.std(ddof=1) / math.sqrt(len(slopes)))) ** 2
checks.published("Exp. 1: ANOVA of the switch cost on encoding position, F(1, 19)", Fv, 11.41, 0.005, S1)
checks.published("Exp. 1: ANOVA of the switch cost on encoding position, partial eta squared", Fv / (Fv + 19), 0.38, 0.005, S1)
nx = cols[1]["px"] - cols[1]["tx"]
ny = cols[1]["py"] - cols[1]["ty"]
F1["roid"] = np.hypot(nx - nx.mean(), ny - ny.mean())
ve = F1.groupby("person").agg(roid=("roid", "mean"), euc=("euc", "mean"))
r_ve = float(np.corrcoef(ve.roid, ve.euc)[0, 1])
checks.published("Exp. 1: correlation across participants of mean Euclidean error with mean dispersion", r_ve, 0.99, 0.01,
                 "Results, Experiment 1 (rho = 0.99)")

F2 = frame(2)
S2 = "Results, Experiment 2"
F2["kind"] = np.where(F2.maintenance.str.endswith("delay"), "delay", "irr")
F2["size"] = F2.maintenance.map(lambda s: 0 if s.startswith(("1_", "short")) else 1)
for kind, rt, rd, lab in (("irr", 6.59, 1.34, "interference conditions"), ("delay", 4.60, 0.94, "delay conditions")):
    x = F2[F2.kind == kind].groupby(["person", "switch"]).euc.mean().unstack()
    pub_t(f"Exp. 2: {lab}, switch vs no-switch error", *pair_t(x[True], x[False]), rt, rd, S2)
cs = F2[F2.kind == "irr"].groupby(["person", "size", "switch"]).euc.mean().unstack("switch")
cs = (cs[True] - cs[False]).unstack()
t_c, d_c = pair_t(cs[0], cs[1])
pub_t("Exp. 2: switch cost with 1 vs 3 irrelevant movements", t_c, d_c, 2.75, 0.59, S2, check_d=False)
sw = F2[(F2.kind == "irr") & F2.switch].groupby(["person", "size"]).euc.mean().unstack()
pub_t("Exp. 2: switch trials, error with 1 vs 3 irrelevant movements", *pair_t(sw[0], sw[1]), 0.99, 0.20, S2)
cd = F2[F2.kind == "delay"].groupby(["person", "size", "switch"]).euc.mean().unstack("switch")
cd = (cd[True] - cd[False]).unstack()
t_d, d_d = pair_t(cd[0], cd[1])
pub_t("Exp. 2: switch cost with the short vs the long delay", t_d, d_d, -0.11, -0.03, S2, check_d=False)

F3 = frame(3)
F3["vsp_load"] = F3.vsp_load.astype(int)
S3 = "Results, Experiment 3"
vs_df = pd.DataFrame([dict(person=sr, load=r["vsp_setsize"], err=r["vsp_ang_err"]) for sr, j, bl, bt, r in vsp_rows])
prec = vs_df.groupby(["person", "load"]).err.apply(lambda s: 1.0 / circ_sd(s)).unstack()
pub_t("Exp. 3: visuospatial recall precision (1 / circular SD), 1 vs 3 dots", *pair_t(prec[1], prec[3]), 3.46, 0.93, S3)
m3 = F3.groupby(["person", "switch", "vsp_load"]).euc.mean().unstack(["switch", "vsp_load"])
eff = {"hand switching": ((m3[(True, 1)] + m3[(True, 3)]) - (m3[(False, 1)] + m3[(False, 3)])) / 2,
       "visuospatial load": ((m3[(False, 3)] + m3[(True, 3)]) - (m3[(False, 1)] + m3[(True, 1)])) / 2,
       "load x switch interaction": (m3[(True, 3)] - m3[(False, 3)]) - (m3[(True, 1)] - m3[(False, 1)])}
Fs = {}
for nme, v in eff.items():
    Fs[nme] = (v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))) ** 2
checks.published("Exp. 3: ANOVA effect of hand switching on movement error, F(1, 13)", Fs["hand switching"], 36.05, 0.005, S3)
checks.published("Exp. 3: ANOVA effect of visuospatial load on movement error, F(1, 13)", Fs["visuospatial load"], 0.40, 0.005, S3)
checks.published("Exp. 3: ANOVA effect of visuospatial load, p", stats.f.sf(Fs["visuospatial load"], 1, 13), 0.54, 0.005, S3)
checks.published("Exp. 3: ANOVA effect of visuospatial load, partial eta squared", Fs["visuospatial load"] / (Fs["visuospatial load"] + 13),
                 0.03, 0.005, S3)
checks.published("Exp. 3: ANOVA load x switch interaction, F(1, 13)", Fs["load x switch interaction"], 0.00, 0.005, S3)
checks.published("Exp. 3: ANOVA load x switch interaction, p", stats.f.sf(Fs["load x switch interaction"], 1, 13), 0.996, 0.0005, S3)
sc = (m3[(True, 1)] - m3[(False, 1)], m3[(True, 3)] - m3[(False, 3)])
t_b, _ = pair_t(sc[0], sc[1])
checks.published("Exp. 3: Bayes factor BF10 for the load effect on the switch cost (JZS prior, scale 0.707)", bf10_paired(t_b, 14), 0.27,
                 0.005, S3)
ang3 = F3.groupby(["person", "switch", "vsp_load"]).err.apply(circ_sd).unstack(["switch", "vsp_load"])
pub_t("Exp. 3: switch cost in circular SD of the recall direction, 1 vs 3 dots",
      *pair_t(ang3[(True, 1)] - ang3[(False, 1)], ang3[(True, 3)] - ang3[(False, 3)]), 0.72, 0.19, S3)

# ---------------------------------------------------------------- the Exp. 3 visuospatial task as a second table
subj_code = OrderedDict()
for r in rows:  # same order of first appearance that stdlib.release uses for the public codes
    subj_code.setdefault(r["subject"], f"{ID}_S{len(subj_code) + 1:04d}")
VSP_COLS = [
    ("subject", "text", "", "Participant sequence code, as in the main table."),
    ("trial", "integer", "", "Row of the main table this recall belongs to: its trial column (counted from 0)."),
    ("block", "text", "", "Block label, as in the main table."),
    ("trial_orig", "integer", "", "The authors' trial number within the block, as in the main table."),
    ("vsp_setsize", "integer", "1 or 3", "Number of dots in the visuospatial stimulus. The authors' README says 0 or 1, the file holds 1 and 3."),
    ("vsp1_color", "integer", "", "The authors' integer code of the colour of dot 1 (one of six colours; the deposit does not say how a code maps to a colour)."),
    ("vsp1_x", "number", "cm", "x position of dot 1 in the robot workspace, origin at the home position, on a ring of 12 cm radius."),
    ("vsp1_y", "number", "cm", "y position of dot 1."),
    ("vsp2_color", "integer", "", "Colour code of dot 2. On one-dot trials the authors' file holds the same values for dots 2 and 3, which were not shown."),
    ("vsp2_x", "number", "cm", "x position of dot 2."),
    ("vsp2_y", "number", "cm", "y position of dot 2."),
    ("vsp3_color", "integer", "", "Colour code of dot 3."),
    ("vsp3_x", "number", "cm", "x position of dot 3."),
    ("vsp3_y", "number", "cm", "y position of dot 3."),
    ("recall_vsp", "integer", "1, 2 or 3", "Which dot the participant was asked to recall (always 1 on one-dot trials)."),
    ("recall_vsp_color", "integer", "", "Colour code of the cued dot."),
    ("recall_vsp_x", "number", "cm", "x position of the cued dot."),
    ("recall_vsp_y", "number", "cm", "y position of the cued dot."),
    ("pt_vsp_x", "number", "cm", "x position the participant reported by rotating a joystick, on the same ring."),
    ("pt_vsp_y", "number", "cm", "y position the participant reported."),
    ("vsp_ang_err", "number", "degrees", "Angular error of the report: direction of the reported position minus direction of the cued dot, counter-clockwise positive."),
    ("vsp_move_time", "integer", "ms", "Time the participant took using the joystick to report the position."),
]


def num(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return "%.8g" % float(v)


vb = io.StringIO()
vw = csv.writer(vb, lineterminator="\r\n")
vw.writerow([c[0] for c in VSP_COLS])
for sr, j, bl, bt, r in vsp_rows:
    vw.writerow([subj_code[sr], j, f"block{bl}", bt] + [num(r[c[0]]) for c in VSP_COLS[4:]])
cb = io.StringIO()
cw = csv.writer(cb, lineterminator="\r\n")
cw.writerow(["column", "type", "values_or_units", "description"])
for c in VSP_COLS:
    cw.writerow(c)
extra = [("exp3_visuospatial_task.csv", vb.getvalue().encode("utf-8")), ("exp3_visuospatial_task_columns.csv", cb.getvalue().encode("utf-8"))]

# ---------------------------------------------------------------- release
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
n_sched = {1: len(persons[1]) * 96, 2: len(persons[2]) * 96}
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="McDougle (Yale)",
    data_url="https://doi.org/10.5281/zenodo.10206635",
    license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor format, participant codes replaced, recalled end points re-expressed as direction and "
            "extent relative to the target (the authors' recalled x, y positions are recovered from them), and placeholder rows added for "
            "trial numbers missing from the authors' files.",
    unit_note="recall reach of one participant sequence (one trial)",
    how=[
        "Three experiments in which people felt a robot-guided movement without vision and recalled a movement's end point with a "
        "reach, using the encoding hand or, on switch trials, the other hand: _Exp1 (20 people; one or four movements encoded, "
        "4 blocks of 24 trials), _Exp2 (24 people; one movement encoded, then one or three irrelevant passive movements or a matched "
        "short or long delay of 3.7 or 11.1 s, 3 blocks of 32 trials) and _Exp3 (14 people; as Exp. 2 with a concurrent "
        "visuospatial memory task for one or three dots, blocks of 48, 48 and 24 trials).",
        "One row is one recall reach, in the authors' order: target_angle is the direction of the recalled movement's end point in the "
        "robot workspace (0 deg is the +x axis, counter-clockwise positive), hand_angle is the direction of the recalled end point "
        "minus the target direction (the authors' angular error), hand_extent is the distance of the recalled end point from home "
        "over the target distance (target_distance_norm is the target distance over the largest in the experiment: 16, 17 and 12 cm; the authors' target coordinates lie within 0.0003 cm of these whole-number distances), "
        "and mt_ms is the authors' move_time; the authors' recalled x, y end points are recovered from these exactly and their Euclidean errors to 0.001 cm.",
        "hand_used is the recall hand; cue_elements lists the authors' set size, recalled movement, encoding hand and switch "
        "condition with their encoded target ids (Exp. 1), the irrelevant movements or the delay (Exp. 2), or the number of dots "
        "(Exp. 3); target_id is the authors' target number in Exp. 1 and 2 and numbers the whole-degree directions in Exp. 3.",
        "Rows with trial_kind missing mark trial numbers absent from the authors' files before each block's last shared trial "
        "(timeouts when switching hands and the lapses the authors removed); exp3_visuospatial_task.csv holds the dot stimuli "
        "and the joystick recall of the cued dot for each Exp. 3 reach (one row per reach, joined by subject and trial; "
        "exp3_visuospatial_task_columns.csv says what each column means). Not available in the deposit: age, sex, handedness "
        "and reaction times.",
    ],
    differences=[
        f"Cohen's d for the change of the hand-switch cost from 1 to 3 irrelevant movements is reported as 0.59 and for the short "
        f"vs long delay as -0.03; the shared data give {d_c:.2f} and {d_d:.2f} (t = {t_c:.2f} and {t_d:.2f}, df and p reproduce).",
        "The abstract's N = 65 counts everyone recruited (22, 26 and 17); the files hold the 58 analysed participants (20, 24 and 14).",
        "The Methods give 40 trials per block for Exp. 2; the shared file has three blocks of up to 32 trials (96 per participant, "
        "as the deposit README says). The README also says 26 trials per block for Exp. 1 and 32 for Exp. 3 and codes the "
        "visuospatial set size 0 or 1; the files have blocks of 24 (Exp. 1) and 48, 48 and 24 (Exp. 3, as in the Methods) and "
        "set sizes 1 and 3.",
        f"The files lack {n_missing[1] + trailing[1]} of {n_sched[1]} scheduled trials in Exp. 1 ({(n_missing[1] + trailing[1]) / n_sched[1]:.1%}) and "
        f"{n_missing[2] + trailing[2]} of {n_sched[2]} in Exp. 2 ({(n_missing[2] + trailing[2]) / n_sched[2]:.1%}); the Methods report 2.0% and 4.4% "
        "timeouts plus lapses (1.58% and 1.48% in Exp. 1, 2.35% in Exp. 2, 2.27% in Exp. 3), which add up to more, so the "
        "reported rates cannot be matched to the missing trials. All shared errors lie inside the lapse cutoffs, so the lapses were "
        "removed before sharing; trials after a block's last shared trial cannot be told from trials not run, so no placeholders "
        f"are added there ({trailing[1]}, {trailing[2]} and {trailing[3]} trial numbers in Exp. 1, 2 and 3).",
        f"The text gives the Exp. 1 correlation between mean Euclidean error and dispersion as 0.99; the shared data give {r_ve:.4f}.",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks, extra_files=extra)

# the second table joins to the main table: same codes, same reaches
zz = zipfile.ZipFile(stdlib.OUT / f"{ID}.zip")
main = list(csv.DictReader(io.TextIOWrapper(zz.open(f"{ID}/{ID}.csv"), encoding="utf-8", newline="")))
vsp = list(csv.DictReader(io.TextIOWrapper(zz.open(f"{ID}/exp3_visuospatial_task.csv"), encoding="utf-8", newline="")))
key_main = {(r["subject"], r["trial"]) for r in main if r["experiment"].endswith("Exp3") and r["trial_kind"] == "reach"}
key_vsp = {(r["subject"], r["trial"]) for r in vsp}
assert key_main == key_vsp and len(key_vsp) == len(vsp) == 1592, "exp3_visuospatial_task.csv does not join to the main table"
assert all(m["block"] == v["block"] and m["trial_orig"] == v["trial_orig"]
           for m, v in zip([r for r in main if r["experiment"].endswith("Exp3") and r["trial_kind"] == "reach"], vsp)), "block/trial mismatch"
print("exp3_visuospatial_task.csv joins to the main table on subject and trial (1592 reaches)")
