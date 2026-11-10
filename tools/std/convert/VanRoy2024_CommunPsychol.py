"""Van Roy, Albouy, Burns & King (2024), Commun Psychol: 8-choice serial reaction time task (SRTT) learned online by
children, adolescents, young and older adults; Experiment 1 (one session, 130 participants) and Experiment 2 (three
sessions: training, 5 h and 24 h retests, 108 participants), one row per key press.

Raw files (untrusted data, read only) in raw_extra/VanRoy2024_CommunPsychol/, from Zenodo record 10365065 (CC BY 4.0):
  Experiment1.zip   Experiment1/<age group>/Exp1_<participant>.txt, one line per key press
  Experiment2.zip   Experiment2/<age group>/Exp2_<participant>.txt (session 1), Exp2_Session2_..., Exp2_Session3_...
  FiguresTables_SourceData.xlsx   the authors' per-participant source data of Tables 1-2 and every figure (ages and
                    genders are taken from its Table1 and Table2 sheets; the figure sheets are used only for checks)
  README.txt        column descriptions (its column numbers are off by one for correct and RT: the lines are
                    tab-separated as run label, [blank], block, cue, key, correct, RT, and the blank field is absent in
                    the Post-SRTT_Random lines, so the six whitespace-separated fields are always run, block, cue, key,
                    correct, RT; the authors' MATLAB indexing of importdata's numeric matrix agrees)
The authors' DataAnalysis_Experiment1.m and DataAnalysis_Experiment2.m (AnalysisScripts.zip) were read as text only, to
reproduce their outlier rule, block means, normalization, learning magnitude, micro- and macro-offline gains.

Run from the site folder: python3 -I tools/std/convert/VanRoy2024_CommunPsychol.py
"""
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "VanRoy2024_CommunPsychol"
RAW = stdlib.raw_dir(ID)
SEQ = [4, 7, 3, 8, 6, 2, 5, 1]
KEYS_PER_BLOCK, SEQ_LEN = 48, 8
GROUPS = ["Children", "Adolescents", "Young adults", "Older adults"]          # the authors' folder names and order
SHEET_GROUP = {"Child": "Children", "Adolescent": "Adolescents", "Young adult": "Young adults", "Older adult": "Older adults"}
POPULATION = {"Children": "children (7-12 years)", "Adolescents": "adolescents (13-17 years)",
              "Young adults": "young adults (18-35 years)", "Older adults": "older adults (55 years and older)"}
RUN = {  # run label -> (phase, block_type, condition, element_type)
    "Pre-SRTT_Random": ("baseline", "random", "pseudorandom", "random"),
    "SRTT_Training": ("learning", "training", "sequential", "sequence"),
    "SRTT_Test": ("test", "test", "sequential", "sequence"),
    "Post-SRTT_Random": ("test", "random", "pseudorandom", "random"),
    "Retest": ("test", "test", "sequential", "sequence"),
}
SESSION1 = (["Pre-SRTT_Random"] * 4 + ["SRTT_Training"] * 16 + ["SRTT_Test"] * 4 + ["Post-SRTT_Random"] * 4)
RETEST = ["Retest"] * 4 + ["Post-SRTT_Random"] * 4
DEVICE = ("participant's own computer keyboard, eight keys pressed with the eight fingers (no thumbs); task run online "
          "in PsyToolkit")

checks = stdlib.Checks()

# ---------------------------------------------------------------- read
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for fn in ("Experiment1.zip", "Experiment2.zip", "FiguresTables_SourceData.xlsx"):
    checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])

files = {}
for zn in ("Experiment1.zip", "Experiment2.zip"):
    with zipfile.ZipFile(RAW / zn) as zf:
        for n in zf.namelist():
            if n.endswith(".txt"):
                files[n] = zf.read(n).decode("utf-8")


def pid_of(name):
    return re.search(r"BMR[_-](\d+)\.txt$", name).group(1)


seqs = []  # (experiment, group, participant number, [(session, member name), ...]) in the authors' order
for e in (1, 2):
    for g in GROUPS:
        names = sorted(n for n in files if n.startswith(f"Experiment{e}/{g}/"))
        s1 = sorted((n for n in names if "Session" not in n), key=lambda n: n.split("/")[-1])   # MATLAB dir() order
        for n in s1:
            p = pid_of(n)
            sess = [(1, n)]
            if e == 2:
                for k in (2, 3):
                    m = [x for x in names if f"Session{k}_" in x and pid_of(x) == p]
                    assert len(m) == 1, (k, p)
                    sess.append((k, m[0]))
            seqs.append((e, g, p, sess))
checks.internal("454 text files: 130 in Experiment 1, 3 x 108 in Experiment 2, each Experiment 2 participant with one "
                "file per session", len(files) == 454 and sum(1 for s in seqs if s[0] == 1) == 130
                and sum(1 for s in seqs if s[0] == 2) == 108)

# demographics (age, gender) per experiment and participant, from the authors' Table 1 and Table 2 source data
import openpyxl  # noqa: E402

wb = openpyxl.load_workbook(RAW / "FiguresTables_SourceData.xlsx", read_only=True, data_only=True)


def sheet(name):
    """rows of a source-data sheet keyed by participant number: (group, [values after the ID column])"""
    out = {}
    for r in wb[name].iter_rows(values_only=True):
        if r and isinstance(r[1], str) and r[1].startswith("BMR"):
            out[re.search(r"(\d+)$", r[1]).group(1)] = (SHEET_GROUP[r[0]], list(r[2:]))
    return out


demo = {1: sheet("Table1 - demographics"), 2: sheet("Table2 - demographics")}
for e in (1, 2):
    ids = {s[2] for s in seqs if s[0] == e}
    checks.internal(f"Experiment {e}: the demographics sheet lists exactly the participants with data files, in the same "
                    "age groups", set(demo[e]) == ids and all(demo[e][s[2]][0] == s[1] for s in seqs if s[0] == e))
both = {s[2] for s in seqs if s[0] == 1} & {s[2] for s in seqs if s[0] == 2}
checks.internal("participants in both experiments have the same age and gender in Table 1 and Table 2 source data",
                all(demo[1][p][1][:2] == demo[2][p][1][:2] for p in both), f"{len(both)} participants")

# ---------------------------------------------------------------- rows (built column-wise: about 400,000 key presses)
COLS = ["experiment", "group", "subject", "person", "age", "sex", "population", "session", "block", "trial", "phase",
        "condition", "excluded_by_authors", "block_number", "block_type", "stimulus", "correct_key", "key_pressed",
        "correct", "rt_ms", "element_type", "sequence", "sequence_position", "repetition"]
col = {c: [] for c in COLS}
first_file = {}
for e, g, p, sess in seqs:
    age, gender = float(demo[e][p][1][0]), {"Female": "F", "Male": "M"}[demo[e][p][1][1]]
    trial = 0
    for k, name in sess:
        f = [ln.split() for ln in files[name].splitlines()]
        assert all(len(x) == 6 for x in f), name
        first_file.setdefault((e, p, k), name)
        bl_order = []
        for x in f:
            if not bl_order or bl_order[-1] != (x[0], x[1]):
                bl_order.append((x[0], x[1]))
        design = SESSION1 if k == 1 else RETEST
        assert [b[0] for b in bl_order] == design[:len(bl_order)], (name, bl_order)
        assert len(f) == KEYS_PER_BLOCK * len(bl_order) and len(set(bl_order)) == len(bl_order), name
        for bi, (run, bl) in enumerate(bl_order):
            blk_rows = f[bi * KEYS_PER_BLOCK:(bi + 1) * KEYS_PER_BLOCK]
            assert all((x[0], x[1]) == (run, bl) for x in blk_rows), (name, run, bl)
            r = np.array([int(x[5]) for x in blk_rows], dtype=float)
            m, sd = r.mean(), r.std(ddof=1)          # MATLAB mean and std over all 48 presses of the block
            out = (r > m + 3 * sd) | (r < m - 3 * sd) | (r == 0)
            phase, btype, cond, etype = RUN[run]
            is_seq = etype == "sequence"
            for j, x in enumerate(blk_rows):
                trial += 1
                for c, v in (("experiment", f"Experiment {e}"), ("group", g), ("subject", f"E{e}|{p}"), ("person", p),
                             ("age", age), ("sex", gender), ("population", POPULATION[g]), ("session", k),
                             ("block", f"{run} {bl}"), ("trial", trial), ("phase", phase), ("condition", cond),
                             ("excluded_by_authors", bool(out[j])), ("block_number", bi + 1), ("block_type", btype),
                             ("stimulus", int(x[2])), ("correct_key", x[2]), ("key_pressed", x[3]),
                             ("correct", x[4] == "1"), ("rt_ms", int(x[5])), ("element_type", etype),
                             ("sequence", "-".join(map(str, SEQ)) if is_seq else ""),
                             ("sequence_position", j % SEQ_LEN + 1 if is_seq else None),
                             ("repetition", j // SEQ_LEN + 1 if is_seq else None)):
                    col[c].append(v)
t = pd.DataFrame(col)
del col
n = len(t)
for c, v in (("dataset", ID), ("study", ID), ("subject_excluded_by_authors", False), ("handedness", ""),
             ("setting", "online"), ("device", DEVICE), ("time_s", ""), ("task", "sequence learning"), ("unit", "keypress"),
             ("pert_type", "none"), ("pert_value", 0), ("pert_unit", ""), ("feedback", "normal"), ("epoch", ""),
             ("rt_correct_ms", ""), ("interval_ms", ""), ("triplet_type", "")):
    t[c] = v

# ---------------------------------------------------------------- internal checks on the table
checks.internal("402,624 key presses: 130 x 28 blocks + 108 x (28 + 8 + 8) blocks of 48, less the missing random run "
                "of one adolescent's third session", n == 130 * 1344 + 108 * (1344 + 384 + 384) - 192, f"{n} rows")
s3 = t[(t.experiment == "Experiment 2") & (t.session == 3)].groupby("subject").size()
checks.internal("one Experiment 2 participant (an adolescent) has no random run in session 3 (paper: post-test random data "
                "of one adolescent missing)", (s3 == 192).sum() == 1 and (s3 == 384).sum() == 107
                and t[t.subject.isin(s3[s3 == 192].index)].group.iloc[0] == "Adolescents")
seq_rows = t.element_type == "sequence"
checks.internal("every sequential block repeats 4-7-3-8-6-2-5-1 six times from its first press",
                bool((t.stimulus[seq_rows].to_numpy() == np.array(SEQ)[t.sequence_position[seq_rows].astype(int) - 1]).all()))
rnd = t[~seq_rows]
cyc = rnd.groupby([rnd.subject, rnd.session, rnd.block, np.arange(len(rnd)) // 8], sort=False).stimulus.apply(
    lambda s: sorted(s) == list(range(1, 9)))
checks.internal("every pseudorandom block is six 8-press cycles that each cue all eight positions once (Methods, SRTT)",
                bool(cyc.all()), f"{len(cyc)} cycles")
rep = rnd.stimulus.to_numpy()
same_blk = (rnd.subject.to_numpy()[1:] == rnd.subject.to_numpy()[:-1]) & (rnd.block.to_numpy()[1:] == rnd.block.to_numpy()[:-1]) \
    & (rnd.session.to_numpy()[1:] == rnd.session.to_numpy()[:-1])
checks.internal("no cue repeats on consecutive presses within a pseudorandom block", bool((rep[1:][same_blk] != rep[:-1][same_blk]).all()))
checks.internal("correct is true exactly when the key pressed equals the cued position",
                bool((t.correct == (t.correct_key == t.key_pressed)).all()))
odd_rt = int((t.rt_ms <= 0).sum())
checks.internal("RT range and odd values (kept as the authors logged them)", bool(t.rt_ms.between(-1000, 20000).all()),
                f"rt_ms {t.rt_ms.min()} to {t.rt_ms.max()} ms; {int((t.rt_ms == 0).sum())} at 0 ms and "
                f"{int((t.rt_ms < 0).sum())} below 0; keys pressed {sorted(t.key_pressed.unique())}")
checks.internal("39 participants appear in both experiments; their Experiment 1 file is byte-identical to their "
                "Experiment 2 session 1 file", len(both) == 39
                and all(files[first_file[(1, p, 1)]] == files[first_file[(2, p, 1)]] for p in both))

# ---------------------------------------------------------------- the authors' measures, from the converted rows
t["run"] = t.block.str.rsplit(" ", n=1).str[0]
t["blk"] = t.block.str.rsplit(" ", n=1).str[1].astype(int)
ok_rt = t.correct & ~t.excluded_by_authors
t["rt_use"] = t.rt_ms.where(ok_rt)
blk = t.groupby(["subject", "session", "block_number"], sort=False).agg(
    run=("run", "first"), blk=("blk", "first"), rt=("rt_use", "mean"), acc=("correct", "mean"))
blk["acc"] *= 100
pre = blk[(blk.index.get_level_values("session") == 1) & (blk.run == "Pre-SRTT_Random")].groupby("subject")
pre_rt = pre.rt.apply(lambda s: s.mean(skipna=False))   # MATLAB mean(mean(rt_correct, 'omitnan')): mean of block means
pre_acc = pre.acc.mean()
blk["rt_norm"] = blk.rt / blk.index.get_level_values("subject").map(pre_rt)
blk["acc_norm"] = blk.acc / blk.index.get_level_values("subject").map(pre_acc)
info = t.drop_duplicates("subject").set_index("subject")[["experiment", "group", "person", "age", "sex"]]


class _Wide:
    """wide(col, session, runs).loc[subjects]: participants x blocks, NaN rows for participants without those blocks"""

    def __init__(self, w):
        self.w = w

    class _Loc:
        def __init__(self, w):
            self.w = w

        def __getitem__(self, subs):
            return self.w.reindex(subs)

    @property
    def loc(self):
        return self._Loc(self.w)


def wide(col, session, runs):
    b = blk[(blk.index.get_level_values("session") == session) & blk.run.isin(runs)]
    w = b[col].droplevel("session").unstack("block_number")
    return _Wide(w.reindex(columns=sorted(w.columns)))


def by_exp(e):
    return info.index[info.experiment == f"Experiment {e}"]


# micro-online and micro-offline gains: normalized mean RT of the first and the last (6th) repetition of each training block
tr = t[t.run == "SRTT_Training"]
rep_rt = tr[tr.repetition.isin([1, 6])].groupby(["subject", "blk", "repetition"]).rt_use.mean().unstack(["repetition"])
first = rep_rt[1].unstack("blk").div(pre_rt, axis=0)
last = rep_rt[6].unstack("blk").div(pre_rt, axis=0)
mon = first - last
mof = pd.DataFrame({b: last[b] - first[b + 1] for b in range(1, 16)})

measures = {}
for e in (1, 2):
    sub = by_exp(e)
    test = wide("rt_norm", 1, ["SRTT_Test"]).loc[sub]
    post = wide("rt_norm", 1, ["Post-SRTT_Random"]).loc[sub]
    lm = (post.mean(axis=1, skipna=False) - test.mean(axis=1, skipna=False)) / post.mean(axis=1, skipna=False)
    measures[e] = dict(lm=lm, mon=mon.loc[sub], mof=mof.loc[sub])
sub2 = by_exp(2)
t1 = wide("rt_norm", 1, ["SRTT_Test"]).loc[sub2].mean(axis=1)
gain = {k: t1 - wide("rt_norm", k, ["Retest"]).loc[sub2].mean(axis=1) for k in (2, 3)}

# ---------------------------------------------------------------- the same measures in the authors' source data
def src_sheet(name, e):
    s = sheet(name)
    codes = {info.loc[x, "person"]: x for x in by_exp(e)}
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):   # None, 'NaN' and '.' (missing) cells
            return np.nan

    vals = {codes[p]: [num(v) for v in vv] for p, (g, vv) in s.items()}
    width = max(len(v) for v in vals.values())
    df = pd.DataFrame({k: v + [np.nan] * (width - len(v)) for k, v in vals.items()}).T
    return df.loc[by_exp(e)]


def max_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (a.shape, b.shape)
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.abs(a - b)[~both_nan]
    return float(np.nanmax(d)) if np.isfinite(d).all() and d.size else (math.inf if d.size else 0.0)


abs1 = pd.concat([wide("rt", 1, [r]).loc[by_exp(1)] for r in ("Pre-SRTT_Random", "SRTT_Training", "SRTT_Test",
                                                               "Post-SRTT_Random")], axis=1)
acc1 = pd.concat([wide("acc", 1, [r]).loc[by_exp(1)] for r in ("Pre-SRTT_Random", "SRTT_Training", "SRTT_Test",
                                                               "Post-SRTT_Random")], axis=1)
norm1 = pd.concat([wide("rt_norm", 1, [r]).loc[by_exp(1)] for r in ("SRTT_Training", "SRTT_Test", "Post-SRTT_Random")], axis=1)
abs2 = pd.concat([wide("rt", k, [r]).loc[sub2] for k, r in ((1, "Pre-SRTT_Random"), (1, "SRTT_Training"), (1, "SRTT_Test"),
                                                           (1, "Post-SRTT_Random"), (2, "Retest"), (2, "Post-SRTT_Random"),
                                                           (3, "Retest"), (3, "Post-SRTT_Random"))], axis=1)
SD ="FiguresTables_SourceData.xlsx"
comparisons = [
    ("Experiment 1 block mean RT of correct, non-excluded presses, all 28 blocks (ms)", abs1, src_sheet("FigureS3A - Abs RT", 1).iloc[:, :28]),
    ("Experiment 1 block accuracy, % correct of all presses, all 28 blocks", acc1, src_sheet("FigureS3B - Abs acc", 1).iloc[:, :28]),
    ("Experiment 1 normalized RT, training, test and post-learning random blocks", norm1, src_sheet("Figure2A - Norm RT", 1).iloc[:, :24]),
    ("Experiment 1 learning magnitude (normalized RT)", measures[1]["lm"].to_frame(), src_sheet("Figure2C-D - Learning magnitude", 1).iloc[:, 1:2]),
    ("Experiment 1 micro-online gains per training block", measures[1]["mon"], src_sheet("Figure3A - Micro-online", 1)),
    ("Experiment 1 micro-offline gains per rest", measures[1]["mof"], src_sheet("Figure3B - Micro-offline", 1)),
]
for what, mine, theirs in comparisons:
    d = max_diff(mine.to_numpy(), theirs.to_numpy())
    checks.internal(f"{what}: equal to the authors' source data ({SD}) for every participant", d < 1e-6,
                    f"max abs difference {d:.2e}")

# Experiment 2: the source data agree with the files except (a) five young adults whose source-data rows are rotated by
# one participant (each row holds the next one's values; group statistics are unaffected) and (b) one child and one
# older adult whose deposited session 3 file gives other session 3 values than the source data (sessions 1-2 agree)
s4a = src_sheet("FigureS4A - Abs RT", 2).iloc[:, :44]
a, b = abs2.to_numpy(float), s4a.to_numpy(float)
same = lambda x, y: bool(np.all((np.abs(x - y) < 1e-6) | (np.isnan(x) & np.isnan(y))))  # noqa: E731
row_ok = np.array([same(a[i], b[i]) for i in range(len(a))])
rot = {}
for i in np.where(~row_ok)[0]:
    j = [k for k in range(len(a)) if same(a[k], b[i])]
    if j:
        rot[sub2[i]] = sub2[j[0]]
cyc_ok = bool(rot) and set(rot) == set(rot.values()) and len(rot) == 5 \
    and all(info.loc[s, "group"] == "Young adults" for s in rot)
s = next(iter(rot)) if rot else None
cycle = [s]
while rot and rot[cycle[-1]] != s and len(cycle) <= 5:
    cycle.append(rot[cycle[-1]])
cyc_ok &= len(cycle) == 5
s3_only = [sub2[i] for i in np.where(~row_ok)[0] if sub2[i] not in rot]
s3_ok = len(s3_only) == 2 and all(same(a[list(sub2).index(x), :36], b[list(sub2).index(x), :36])
                                  and not same(a[list(sub2).index(x), 36:], b[list(sub2).index(x), 36:]) for x in s3_only)
checks.internal(f"Experiment 2 block mean RT, all 44 blocks of sessions 1-3: equal to the authors' source data ({SD}) for "
                "101 of 108 participants", int(row_ok.sum()) == 101, f"{int(row_ok.sum())} equal")
checks.internal("Experiment 2 source data: the rows of five young adults are one cycle of five, each holding exactly the "
                "values computed here for the next young adult in the authors' order", cyc_ok)
checks.internal("Experiment 2 source data: for one child and one older adult only the eight session 3 blocks differ "
                "(their deposited session 3 files are not the data behind the source data)", s3_ok
                and sorted(info.loc[x, "group"] for x in s3_only) == ["Children", "Older adults"])
fig5 = src_sheet("Figure5 - Seq offline gains", 2).iloc[:, 1:3]
mine5 = pd.concat([gain[2], gain[3]], axis=1)
mine5_rot = mine5.copy()
for k, v in rot.items():
    mine5_rot.loc[k] = mine5.loc[v].to_numpy()
diff5 = np.abs(mine5_rot.to_numpy() - fig5.to_numpy())
others = ~mine5.index.isin(s3_only)
checks.internal("Experiment 2 5 h and 24 h sequential offline gains (Fig. 5 source data): equal for every participant once "
                "the five young-adult rows are rotated back, except the 24 h gains of the two participants whose session 3 "
                "files differ", float(diff5[others].max()) < 1e-9 and float(diff5[~others, 0].max()) < 1e-9
                and float(diff5[~others, 1].min()) > 1e-3, f"max abs difference elsewhere {diff5[others].max():.1e}")
gain3_src = gain[3].copy()
gain3_src.loc[s3_only] = fig5.loc[s3_only].iloc[:, 1].to_numpy()

# ---------------------------------------------------------------- published numbers, from the converted rows
T1, T2 = "Methods, Participants, Table 1", "Methods, Participants, Table 2"
REP_N = {1: (33, 33, 32, 32), 2: (27, 27, 27, 27)}
REP_F = {1: (17, 12, 23, 24), 2: (16, 12, 21, 18)}
REP_AGE = {1: ((10.2, 1.6), (15.3, 1.3), (26.3, 5.0), (64.4, 5.2)), 2: ((10.6, 1.6), (15.4, 1.4), (26.6, 4.4), (63.0, 4.4))}
REP_EXCL = {1: (1.78, (2.03, 1.76, 1.68, 1.65)), 2: (1.69, (2.0, 1.83, 1.58, 1.34))}
excl_pct = {}
for e in (1, 2):
    ie = info[info.experiment == f"Experiment {e}"]
    src = T1 if e == 1 else T2
    for g, nn, nf, (ma, sa) in zip(GROUPS, REP_N[e], REP_F[e], REP_AGE[e]):
        x = ie[ie.group == g]
        checks.published(f"Experiment {e}, {g.lower()}: participants", len(x), nn, 0, src)
        checks.published(f"Experiment {e}, {g.lower()}: female participants", (x.sex == "F").sum(), nf, 0, src)
        checks.published(f"Experiment {e}, {g.lower()}: mean age (years)", x.age.mean(), ma, 0.05, src)
        checks.published(f"Experiment {e}, {g.lower()}: SD of age (years; n in the denominator, which all eight of the "
                         "tables' SDs match)", x.age.std(ddof=0), sa, 0.05, src)
    te = t[t.experiment == f"Experiment {e}"]
    pct = te.groupby("subject").excluded_by_authors.mean() * 100
    grp = info.loc[pct.index, "group"]
    rep_all, rep_g = REP_EXCL[e]
    excl_pct[e] = [pct.mean()] + [pct[grp == g].mean() for g in GROUPS]
    for g, comp, r in zip(["all"] + GROUPS, excl_pct[e], (rep_all,) + rep_g):
        if abs(comp - r) <= (0.05 if r == 2.0 else 0.005):   # the others are listed under differences
            checks.published(f"Experiment {e}, {'all participants' if g == 'all' else g.lower()}: mean % of key presses "
                             "excluded as RT outliers", comp, r, 0.05 if r == 2.0 else 0.005,
                             "Methods, Serial Reaction Time Task")


def anova(x, e):
    g = info.loc[x.index, "group"]
    return stats.f_oneway(*[x[g == k].dropna() for k in GROUPS]).statistic


def hedges(x, a, b):
    g = info.loc[x.index, "group"]
    xa, xb = x[g == a].dropna(), x[g == b].dropna()
    na, nb = len(xa), len(xb)
    sp = math.sqrt(((na - 1) * xa.var(ddof=1) + (nb - 1) * xb.var(ddof=1)) / (na + nb - 2))
    return (xa.mean() - xb.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9))


R1 = "Results, 'Experiment 1—sequence-specific learning'"
lm = measures[1]["lm"]
checks.published("Experiment 1: one-way ANOVA F of learning magnitude across age groups", anova(lm, 1), 5.440, 0.0005, R1)
checks.published("Experiment 1: Hedges' g of learning magnitude, children vs adolescents", hedges(lm, "Children", "Adolescents"),
                 -0.714, 0.0005, R1)
checks.published("Experiment 1: Hedges' g of learning magnitude, children vs young adults", hedges(lm, "Children", "Young adults"),
                 -0.970, 0.0005, R1)
post1 = wide("rt_norm", 1, ["Post-SRTT_Random"]).loc[by_exp(1)].mean(axis=1)
checks.published("Experiment 1: group effect F on normalized RT of the post-learning random run (mixed ANOVA group "
                 "effect = one-way ANOVA of participant means)", anova(post1, 1), 5.795, 0.0005, R1)
R1d = "Results, 'Experiment 1—dynamics of performance improvements'"
for what, col, run, rep in (("normalized RT, training blocks", "rt_norm", "SRTT_Training", 1.578),
                            ("normalized RT, post-learning test blocks", "rt_norm", "SRTT_Test", 1.099),
                            ("normalized accuracy, training blocks", "acc_norm", "SRTT_Training", 0.322),
                            ("normalized accuracy, post-learning test blocks", "acc_norm", "SRTT_Test", 1.796)):
    x = wide(col, 1, [run]).loc[by_exp(1)].mean(axis=1)
    checks.published(f"Experiment 1: group effect F, {what}", anova(x, 1), rep, 0.0005, R1d)
R1m = "Results, 'Experiment 1—micro-learning across sequential training'"
for what, w, rep in (("micro-online gains (16 blocks)", measures[1]["mon"], 2.663), ("micro-offline gains (15 rests)", measures[1]["mof"], 2.490)):
    complete = w.dropna()
    checks.published(f"Experiment 1: participants with complete {what}", len(complete), 128, 0,
                     R1m + " (F(3,124); Fig. 3: 31 children)")
    checks.published(f"Experiment 1: group effect F, {what}", anova(complete.mean(axis=1), 1), rep, 0.0005, R1m)
R2 = "Results, 'Experiment 2—macro-offline performance gains'"
checks.published("Experiment 2: one-way ANOVA F of 5 h sequential offline gains", anova(gain[2], 2), 5.756, 0.0005, R2)
for a_, b_, rep in (("Children", "Young adults", 0.590), ("Adolescents", "Young adults", 0.548),
                     ("Children", "Older adults", 1.382), ("Adolescents", "Older adults", 1.290)):
    checks.published(f"Experiment 2: Hedges' g of 5 h sequential offline gains, {a_.lower()} vs {b_.lower()}",
                     hedges(gain[2], a_, b_), rep, 0.0005, R2)
# 24 h statistics: from the deposit they differ (two session 3 files); with the authors' 24 h gains for those two
# participants (Fig. 5 source data) and the converted rows for the other 106, every reported value is reproduced
REP24 = [("one-way ANOVA F of 24 h sequential offline gains", lambda g3: anova(g3, 2), 20.582),
         ("group effect F of the group x offline period ANOVA", lambda g3: anova((gain[2] + g3) / 2, 2), 14.541),
         ("group x offline period interaction F", lambda g3: anova(g3 - gain[2], 2), 3.382)] + [
    (f"Hedges' g of 24 h gains, {a_.lower()} vs {b_.lower()}", (lambda g3, a_=a_, b_=b_: hedges(g3, a_, b_)), rep)
    for a_, b_, rep in (("Children", "Young adults", 1.285), ("Children", "Older adults", 2.449),
                        ("Adolescents", "Older adults", 1.256), ("Young adults", "Older adults", 1.237))]
dep24 = [(w, f(gain[3]), r) for w, f, r in REP24]
checks.internal("Experiment 2, 24 h statistics with the authors' 24 h gains for the two participants whose session 3 "
                "files differ: all seven reported values reproduced", all(abs(f(gain3_src) - r) <= 0.0005 for w, f, r in REP24),
                "; ".join(f"{w} {f(gain3_src):.3f}" for w, f, r in REP24))

for c in checks.items:
    if c["kind"] == "published":
        print(f"  {'ok ' if c['ok'] else 'BAD'} {c['what']}: computed {c['computed']:.4f}, reported {c['reported']}")
    elif not c["ok"] or c.get("detail"):
        print(f"  {'ok ' if c['ok'] else 'FAILED'} [internal] {c['what']} {c.get('detail', '')}")

# ---------------------------------------------------------------- release
t = t.drop(columns=["run", "blk", "rt_use"])
names = [c[1] for c in stdlib.columns_for("keypress")]
for c in names:
    if c not in t.columns:      # a module column added after this converter was written stays empty
        t[c] = ""
t = t[[c for c in names if c in t.columns]]
entry = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
meta = dict(
    study=ID,
    paper={k: entry["papers"][0][k] for k in ("authors", "year", "title", "venue", "doi")},
    lab=entry["lab"],
    data_url="https://zenodo.org/records/10365065",
    license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted the per-participant key-press text files of both experiments into one table, added each "
             "participant's age and gender from the authors' Table 1 and Table 2 source data, marked the authors' RT "
             "outliers, and replaced participant identifiers with codes."),
    how=[
        "One row is one key press in the online 8-choice serial reaction time task: a cue appeared in one of eight "
        "squares (stimulus 1-8, left to right, the left little finger to the right little finger) and the participant "
        "pressed the matching key; correct_key and key_pressed use the same 1-8 codes (one logged press is coded 11), "
        "rt_ms is the authors' response time from cue onset, and every block has 48 presses.",
        "Session 1 (Experiment 1, and session 1 of Experiment 2) has 4 pseudorandom blocks (Pre-SRTT_Random, phase "
        "baseline), 16 training blocks of the sequence 4-7-3-8-6-2-5-1 (six repetitions per block, sequence_position and "
        "repetition give the place in it), 4 test blocks of the sequence after a 1 min rest (SRTT_Test) and 4 more "
        "pseudorandom blocks (Post-SRTT_Random); Experiment 2 sessions 2 and 3, about 5 h and 24 h later, have 4 blocks of "
        "the sequence (Retest) and 4 pseudorandom blocks. The unanalysed familiarisation block is not in the deposit.",
        "excluded_by_authors marks the authors' RT outliers: response times more than 3 SD (sample SD) above or below the "
        "mean of all 48 presses of the block, or 0 ms; the authors' block RT is the mean over correct presses that are not "
        "excluded, and their accuracy is the share of correct presses over all presses of the block; a few logged RTs "
        "are 0 ms or negative and are kept as logged.",
        "subject is one participant in one experiment and person is the participant: 39 people appear in both "
        "experiments, and their Experiment 1 rows are the same key presses as their Experiment 2 session 1 rows (the "
        "authors analysed them in both experiments). group is the authors' age group; age (years, to three decimals) "
        "and sex are the authors' values.",
    ],
    unit_note="key press (one cued response in the serial reaction time task)",
    differences=[
        "Mean % of key presses excluded as RT outliers, Experiment 1: {:.2f} here (children {:.2f}, adolescents {:.2f}, "
        "young adults {:.2f}, older adults {:.2f}) against 1.78 (2.03, 1.76, 1.68, 1.65) in the paper; Experiment 2 older "
        "adults: {:.2f} against 1.34. The cause is not known: the flags here reproduce the authors' per-block RT means "
        "(source data) exactly for all 130 Experiment 1 participants.".format(*excl_pct[1], excl_pct[2][4]),
        "Experiment 2, 24 h: for one child and one older adult the deposited session 3 files give other session 3 RTs "
        "than the authors' source data, so the 24 h statistics computed from this table differ from the paper: "
        + "; ".join(f"{w} {v:.3f} (paper {r})" for w, v, r in dep24)
        + ". With the authors' source-data 24 h gains for those two participants, all seven values are reproduced; the "
        "5 h statistics, which use sessions 1 and 2 only, are reproduced from the table.",
        "In the authors' Experiment 2 source data the rows of five young adults are rotated by one participant (each row "
        "holds the next one's values); rows here follow the participant numbers of the data files, and the age and sex "
        "of these five come from the Table 2 sheet by participant number, so they may be affected in the same way.",
        "The paper says 42 participants (16 children, 19 adolescents, 4 young adults, 3 older adults) took part in both "
        "experiments; the deposit has 39 participant numbers in both (14, 18, 4 and 3), whose Experiment 1 and "
        "Experiment 2 session 1 files are identical.",
        "The paper says the third session of one young adult was missing; the deposit and the authors' source data hold "
        "all three sessions for every young adult (only one adolescent's session 3 random run is missing).",
    ],
)
stdlib.release(ID, "keypress", t, meta, checks)
