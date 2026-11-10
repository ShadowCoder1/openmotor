"""Tang, Parrell & Niziolek (2022), J Neurophysiol: perturbations that push each vowel production toward (inward) or away
from (outward) the speaker's own vowel target change produced variability. Experiments 1-4 plus the control data set
(no perturbation) from Parrell & Niziolek (2021) that the paper compares against.

Raw files, raw_extra/Tang2022_JNeurophysiol/ (OSF stjc9, folder varMod/acousticdata, fetched with tools/std/fetch.py):
  datatables.mat       MATLAB table objects, one row per utterance the authors kept:
                         datatable_in (Experiment 1, inward 50%), datatable_out (Experiment 2, outward 75%),
                         datatable_out2 (Experiment 3, outward 200%), datatable_inout (Experiment 4, inward 50% then
                         outward 50%), datatable_null (control), datatable_inout_3phases (Experiment 4 again, the same
                         rows with the two halves of each exposure phase merged), and per-condition summaries
                         (datatableByCond_*, the same as the table_*.csv files)
                       columns duration, RT (s), rawF1_init, rawF2_init, rawF1_mid, rawF2_mid (mels), initdists,
                         middists, norminitdists, normmiddists, centering, cond, vowel, subj, trialinds
  table_shift_in.mat, table_shift_out.mat, table_shift_out2.mat
                       Audapter's per-trial log for Experiments 1-3, every trial including removed ones, grouped by
                         participant, phase (baseline, hold, washout) and vowel, without trial numbers: F1in, F2in
                         (produced), F1out, F2out (heard; 0 where not logged), F1diff = F1in - F1out, F2diff, distance
  table_*.csv          the authors' per participant x vowel x phase summaries (used here only for a check)
The MATLAB table objects are decoded from the MAT-file's own object store with scipy (class 'table' with properties
data, nrows, varnames...); nothing from the deposit is executed. The participant folders (dataVals.mat, formant
tracks of 24 people, 1.4 GB) are not needed: datatables.mat holds the authors' per-trial values for every experiment.
What the values are (paper, Methods; checked below): formants in mels averaged over a 50-ms window at vowel onset
(_init) and at the vowel midpoint (_mid); initdists and middists = distance in F1-F2 space from the phase median of
that vowel (recomputed here exactly from the formants, so they are not carried); norminitdists = initdists / the
baseline median distance; centering = initdists - middists; RT = trial start to vowel onset.
Run from the site folder: python3 -I tools/std/convert/Tang2022_JNeurophysiol.py
"""
import io
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from scipy.io.matlab._mio5 import MatFile5Reader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Tang2022_JNeurophysiol"
SRC = "Tang, Parrell & Niziolek 2022, J Neurophysiol 128:1469"


class MatTables:
    """Read MATLAB 'table' objects from a v7 MAT-file without MATLAB: the object store is the hidden
    __function_workspace__ variable (a MAT stream holding the FileWrapper__ cell array: cell 0 = metadata with the
    property names, class table, object table and property sets; the other cells = property values)."""

    MARK = 0xDD000000

    def __init__(self, path):
        self.top = sio.loadmat(path)
        ws = self.top["__function_workspace__"].tobytes()
        head = b"MATLAB 5.0 MAT-file".ljust(116, b" ") + b"\x00" * 8 + ws[0:4]
        rd = MatFile5Reader(io.BytesIO(head + ws[8:]), struct_as_record=True, squeeze_me=False)
        rd.initialize_read()
        rd.mat_stream.seek(128)
        hdr, _ = rd.read_var_header()
        mc = rd.read_var_array(hdr, process=True)[0, 0]["MCOS"]
        self.cells = mc[0]["arr" if "arr" in mc.dtype.names else "_ObjectMetadata"].ravel()
        meta = self.cells[0].ravel().tobytes()
        _, n_names = struct.unpack_from("<II", meta, 0)
        offs = struct.unpack_from("<8I", meta, 8)
        pos, self.names = 40, []
        while len(self.names) < n_names:
            end = meta.index(b"\x00", pos)
            self.names.append(meta[pos:end].decode("latin1"))
            pos = end + 1
        u32 = lambda a, b: np.frombuffer(meta[a:b], dtype="<u4")  # noqa: E731
        self.classes = u32(offs[0], offs[1]).reshape(-1, 4)
        assert offs[1] == offs[2], "objects saved through saveobj are not handled"
        self.objects = u32(offs[2], offs[3]).reshape(-1, 6)
        r4, i, self.props = u32(offs[3], offs[4]), 2, [None]
        while i < len(r4):
            n = int(r4[i])
            self.props.append(r4[i + 1:i + 1 + 3 * n].reshape(-1, 3))
            i += 1 + 3 * n
            i += i % 2

    def table(self, var):
        v = self.top[var]
        a = np.asarray(v[0]["arr" if "arr" in v.dtype.names else "_ObjectMetadata"]).ravel().astype(np.int64)
        assert int(a[0]) == self.MARK and int(a[1]) == 2 and int(a[2] * a[3]) == 1, a[:6]
        cid, _, _, sid, pid, _ = (int(x) for x in self.objects[int(a[4])])
        assert sid == 0 and self.names[self.classes[cid][1] - 1] == "table"
        o = {self.names[ni - 1]: self.cells[val + 2] for ni, kind, val in self.props[pid] if kind == 1}
        names = [str(np.asarray(x).ravel()[0]) for x in o["varnames"].ravel()]
        nrows = int(np.asarray(o["nrows"]).ravel()[0])
        cols = {}
        for nm, col in zip(names, o["data"].ravel()):
            col = np.asarray(col)
            if col.dtype == object:
                cols[nm] = [str(np.asarray(x).ravel()[0]) for x in col.ravel(order="F")]
            else:
                assert col.shape == (nrows, 1), (nm, col.shape)
                cols[nm] = col[:, 0]
            assert len(cols[nm]) == nrows
        return pd.DataFrame(cols)


EXPS = {  # table name -> experiment label, gain description, signed gain (negative = toward the vowel target)
    "in": ("Experiment 1", "inward-pushing 50%"),
    "out": ("Experiment 2", "outward-pushing 75%"),
    "out2": ("Experiment 3", "outward-pushing 200%"),
    "inout": ("Experiment 4", None),
    "null": ("Control (no perturbation; data set of Parrell & Niziolek 2021)", None),
}
WORD = {"iy": "bead", "ae": "bad", "aa": "bod", "eh": "bed", "uw": "booed"}
IPA = {"iy": "/i/", "ae": "/æ/", "aa": "/ɑ/", "eh": "/ɛ/", "uw": "/u/"}
DEVICE = {
    "in": "AKG C520 head-mounted microphone, Audapter, Beyerdynamic DT 770 headphones",
    "out": "AKG C520 head-mounted microphone, Audapter, Beyerdynamic DT 770 headphones",
    "out2": "Sennheiser MKE 600 desktop microphone, Audapter, Beyerdynamic DT 770 headphones",
    "inout": "Sennheiser MKE 600 desktop microphone, Audapter, Beyerdynamic DT 770 headphones",
    "null": "Audapter and headphones, processed as in the baseline phases of Experiments 1-4",
}
MASKING = "speech-shaped noise, about 60 dB SPL, mixed with the feedback (about 80 dB SPL)"
WINDOW = "f1_mel and f2_mel: 50 ms at the vowel midpoint; f1_mel_onset and f2_mel_onset: the first 50 ms of the vowel"
PHASE = {"baseline": "baseline", "hold1": "learning", "hold2": "learning", "washout": "washout",
         "holdIn1": "learning", "holdIn2": "learning", "holdOut1": "learning", "holdOut2": "learning"}


def cond_text(t, cond):
    if t == "null" or cond in ("baseline", "washout"):
        return ""
    if t == "inout":
        return "inward-pushing 50%" if cond.startswith("holdIn") else "outward-pushing 50%"
    return EXPS[t][1]


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def align_shifts(dt, sh, checks, label):
    """Attach Audapter's per-trial shift to the kept trials. The log has no trial numbers but lists every trial,
    grouped by phase and vowel in trial order, and every run of n_vowels trials holds each vowel once (checked), so
    the k-th trial of a vowel within a phase is trial ceil(trialinds / n_vowels) of that phase."""
    s = sh.copy()
    s["k"] = s.groupby(["subj", "cond", "vowel"], sort=False).cumcount()
    size = s.groupby(["subj", "cond"]).size().unstack()
    nvow = s.groupby("subj").vowel.nunique()
    d = dt.copy()
    d["P"] = d.cond.where(d.cond.isin(["baseline", "washout"]), "hold")
    parts = []
    for subj, g in d.groupby("subj", sort=False):
        nv = int(nvow[subj])
        assert not g.assign(b=np.ceil(g.trialinds / nv)).duplicated(["b", "vowel"]).any()
        nb, nh = int(size.loc[subj, "baseline"]) // nv, int(size.loc[subj, "hold"]) // nv
        start = g.P.map({"baseline": 0, "hold": nb, "washout": nb + nh})
        parts.append(g.assign(k=(np.ceil(g.trialinds / nv).astype(int) - 1 - start).to_numpy()))
    d = pd.concat(parts)

    def merged(off):
        return d.assign(k=d.k + off).merge(s.rename(columns={"cond": "P"}), on=["subj", "P", "vowel", "k"], how="left")

    m = merged(0)
    assert len(m) == len(d)
    ok, worst = True, 1.0
    for subj in d.subj.unique():  # the matched log must track the offline formants better than a one-trial slip
        r = {}
        for off in (-1, 0, 1):
            x = merged(off)
            x = x[(x.subj == subj) & x.F1in.notna()]
            r[off] = np.nanmedian([np.corrcoef(g.F1in, g.rawF1_mid)[0, 1] for _, g in x.groupby("vowel") if len(g) > 5])
        ok &= r[0] > max(r[-1], r[1]) + 0.1
        worst = min(worst, r[0] - max(r[-1], r[1]))
    checks.internal(f"{label}: Audapter's log matches the kept trials (for every participant the produced F1 it logged "
                    "correlates with the offline midpoint F1 better at the chosen alignment than one trial off)",
                    bool(ok), f"{int(m.F1in.notna().sum())} of {len(m)} rows matched; smallest margin in median r {worst:.2f}")
    return m


def build(checks):
    raw = stdlib.raw_dir(ID)
    T = MatTables(raw / "datatables.mat")
    tabs = {t: T.table(f"datatable_{t}") for t in EXPS}
    tabs3 = T.table("datatable_inout_3phases")
    key = ["subj", "trialinds"]
    a = tabs["inout"].sort_values(key).reset_index(drop=True)
    b = tabs3.sort_values(key).reset_index(drop=True)
    same = a.drop(columns=["cond", "initdists", "middists", "norminitdists", "normmiddists", "centering"]).equals(
        b.drop(columns=["cond", "initdists", "middists", "norminitdists", "normmiddists", "centering"]))
    merge_ok = (a.cond.str.replace(r"[12]$", "", regex=True) == b.cond).all()
    checks.internal("datatable_inout_3phases holds the same Experiment 4 rows as datatable_inout, with holdIn1+holdIn2 "
                    "and holdOut1+holdOut2 merged (so it is not a second set of rows)", bool(same and merge_ok))
    shifts = {"in": MatTables(raw / "table_shift_in.mat").table("table_shift_in"),
              "out": MatTables(raw / "table_shift_out.mat").table("table_shift_out"),
              "out2": MatTables(raw / "table_shift_out2.mat").table("table_shift")}
    rows = []
    for t, (exp, _) in EXPS.items():
        d = tabs[t]
        n0 = len(d)
        # the authors' derived distances, recomputed from the formants: distance to the phase median of the vowel
        ok = True
        for w in ("init", "mid"):
            c1 = d.groupby(["subj", "vowel", "cond"])[f"rawF1_{w}"].transform("median")
            c2 = d.groupby(["subj", "vowel", "cond"])[f"rawF2_{w}"].transform("median")
            ok &= bool(np.allclose(np.hypot(d[f"rawF1_{w}"] - c1, d[f"rawF2_{w}"] - c2), d[f"{w}dists"], atol=1e-9))
        bmed = d[d.cond == "baseline"].groupby(["subj", "vowel"]).initdists.median().rename("bm")
        nrm = d.join(bmed, on=["subj", "vowel"])
        ok &= bool(np.allclose(nrm.norminitdists, nrm.initdists / nrm.bm, atol=1e-9, equal_nan=True))
        ok &= bool(np.allclose(d.centering, d.initdists - d.middists, atol=1e-9))
        checks.internal(f"{exp}: the authors' initdists, middists, norminitdists and centering follow exactly from the "
                        "shared formants (distance to the median of the same vowel in the same phase), so they are "
                        "not carried as columns", ok)
        if t in shifts:
            d = align_shifts(d, shifts[t], checks, exp)
        d = d.sort_values(["subj", "trialinds"], kind="stable")
        assert len(d) == n0
        for r in d.itertuples(index=False):
            hold = PHASE[r.cond] == "learning" and t != "null"
            sf1 = sf2 = mag = None
            if not hold:
                sf1 = sf2 = mag = 0.0
            elif t in shifts and not pd.isna(r.F1out) and r.F1out != 0:
                sf1, sf2, mag = float(r.F1out - r.F1in), float(r.F2out - r.F2in), float(r.distance)
            excluded = (t == "null" and r.vowel == "uw") or (t == "out" and r.subj == 19 and r.cond == "washout")
            rows.append({
                "dataset": ID, "study": ID, "experiment": exp, "group": "", "subject": f"{t}|{int(r.subj)}",
                "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
                "setting": "lab", "device": DEVICE[t], "session": 1, "block": r.cond, "trial": int(r.trialinds),
                "phase": "other" if (t == "null" and r.cond != "baseline") else PHASE[r.cond], "time_s": "",
                "task": "speech", "unit": "utterance", "condition": cond_text(t, r.cond),
                "pert_type": "formant_shift" if hold else "none", "pert_value": mag, "pert_unit": "mel",
                "feedback": "altered" if hold else "normal", "excluded_by_authors": bool(excluded),
                "stimulus": WORD[r.vowel], "segment": f"vowel {IPA[r.vowel]} of {WORD[r.vowel]}",
                "measure_window": WINDOW, "duration_ms": float(r.duration) * 1000, "f0_hz": "", "f1_hz": "", "f2_hz": "",
                "f1_mel": float(r.rawF1_mid), "f2_mel": float(r.rawF2_mid), "f1_mel_onset": float(r.rawF1_init),
                "f2_mel_onset": float(r.rawF2_init), "intensity_au": "", "shift_f1": sf1, "shift_f2": sf2,
                "masking_noise": MASKING, "feedback_latency_ms": 18, "reaction_time_ms": float(r.RT) * 1000,
                "_t": t, "_subj": int(r.subj), "_vowel": r.vowel,
            })
    return rows, shifts


def rm_anova(wide):
    """One-way repeated-measures ANOVA F (participants x phases)."""
    X = wide.to_numpy(float)
    n, k = X.shape
    gm = X.mean()
    ssp = n * ((X.mean(0) - gm) ** 2).sum()
    sss = k * ((X.mean(1) - gm) ** 2).sum()
    sse = ((X - gm) ** 2).sum() - ssp - sss
    return (ssp / (k - 1)) / (sse / ((k - 1) * (n - 1))), k - 1, (k - 1) * (n - 1)


def phase_means(a, col, three_phase=False):
    """Variability (or centering) per participant and phase the authors' way: distance of each production from the
    median of its vowel in its phase, averaged per vowel, then over vowels."""
    a = a.copy()
    if three_phase:  # Experiment 4: one median per whole exposure phase (datatable_inout_3phases)
        a["blk"] = a.block.str.replace(r"[12]$", "", regex=True)
    else:
        a["blk"] = a.block
    for w, f1, f2 in (("onset", "f1_mel_onset", "f2_mel_onset"), ("mid", "f1_mel", "f2_mel")):
        g = a.groupby(["subject", "_vowel", "blk"])
        a[f"dist_{w}"] = np.hypot(a[f1] - g[f1].transform("median"), a[f2] - g[f2].transform("median"))
    a["centering"] = a.dist_onset - a.dist_mid
    return a.groupby(["subject", "blk", "_vowel"])[col].mean().groupby(["subject", "blk"]).mean().unstack("blk")


def checks_all(a, shifts, checks):
    sizes = a.groupby("experiment").size()
    per = a.groupby("experiment").subject.nunique()
    exp = {t: EXPS[t][0] for t in EXPS}
    checks.internal("rows per experiment = the authors' tables (12600, 11581, 11160, 14269, 11344)",
                    [int(sizes[exp[t]]) for t in EXPS] == [12600, 11581, 11160, 14269, 11344], str(sizes.to_dict()))
    checks.internal("participants per experiment = 24, 22, 21, 20 (paper, Methods) and 25 in the control data set",
                    [int(per[exp[t]]) for t in EXPS] == [24, 22, 21, 20, 25], str(per.to_dict()))
    checks.internal("subject x trial is unique and trials increase within every participant",
                    not a.duplicated(["subject", "trial"]).any()
                    and all((np.diff(g.trial.to_numpy()) > 0).all() for _, g in a.groupby("subject", sort=False)))
    designs = {t: g.groupby("block").trial.agg(["min", "max"]) for t, g in a.groupby("_t")}
    lim = {"in": 640, "out": 540, "out2": 540, "inout": 720, "null": 460}
    checks.internal("trial numbers within the design (540 trials in Experiments 1-3, 720 in Experiment 4, 460 in the "
                    "control; one Experiment 1 participant ran 640 trials with a fourth word, booed, that the authors "
                    "did not keep)", all(int(designs[t]["max"].max()) <= lim[t] for t in lim))
    vow = a.groupby("_t")._vowel.unique().map(sorted).to_dict()
    checks.internal("words: bead, bad, bod in Experiments 1, 2, 4; bead, bad, bed in Experiment 3; plus booed in the control",
                    vow == {"in": ["aa", "ae", "iy"], "out": ["aa", "ae", "iy"], "out2": ["ae", "eh", "iy"],
                            "inout": ["aa", "ae", "iy"], "null": ["aa", "ae", "iy", "uw"]}, str(vow))
    rng = dict(f1_mel=(250, 1200), f2_mel=(850, 2050), f1_mel_onset=(250, 1200), f2_mel_onset=(850, 2050),
               duration_ms=(100, 750), reaction_time_ms=(50, 4500))
    for c, (lo, hi) in rng.items():
        v = a[c].astype(float)
        checks.internal(f"{c} present on every row and within {lo}-{hi}", bool(v.notna().all() and v.between(lo, hi).all()),
                        f"{v.min():.1f}-{v.max():.1f}")
    hold = a.feedback == "altered"
    known = hold & a.pert_value.notna()
    checks.internal("perturbation: 0 with normal feedback on baseline, washout and all control rows; on exposure rows of "
                    "Experiments 1-3 the logged shift (heard minus produced, mels) is present except where Audapter "
                    "logged no heard formants; Experiment 4 has no shift log",
                    bool((a.loc[~hold, ["pert_value", "shift_f1", "shift_f2"]] == 0).all().all()
                         and a.loc[hold & (a._t == "inout"), "pert_value"].isna().all()
                         and (a.loc[known, "pert_value"] - np.hypot(a.loc[known, "shift_f1"], a.loc[known, "shift_f2"])).abs().max() < 1e-6),
                    f"{int(known.sum())} exposure rows with a logged shift, {int((hold & ~known & (a._t != 'inout')).sum())} without")
    ex = a.excluded_by_authors
    checks.internal("excluded_by_authors: the control's booed rows (not analysed) and the Experiment 2 washout of the one "
                    "participant whose washout the authors' summary leaves empty",
                    int(ex.sum()) == int(((a._t == "null") & (a._vowel == "uw")).sum() + ((a._t == "out") & (a._subj == 19) & (a.block == "washout")).sum()),
                    f"{int(ex.sum())} rows")
    raw = stdlib.raw_dir(ID)
    # the authors' summary table equals the per-vowel means of the kept rows
    csv = pd.read_csv(raw / "table_in.csv")
    e1 = a[a._t == "in"]
    pm = phase_means(e1, "dist_onset").stack().rename("v").reset_index()
    pm["subj"] = pm.subject.str.split("|").str[1].astype(float)
    rt = e1.groupby(["_subj", "block", "_vowel"]).reaction_time_ms.mean().rename("rt").reset_index()
    mg = csv.merge(rt, left_on=["subj", "cond", "vowel"], right_on=["_subj", "block", "_vowel"])
    checks.internal("Experiment 1: the authors' summary table (table_in.csv) RT = the mean of the converted rows' "
                    "reaction_time_ms per participant, vowel and phase", len(mg) == 216 and bool(np.allclose(mg.RT * 1000, mg.rt)))

    # ---- published numbers
    # trials removed (Methods, Data analysis): kept rows here; planned trials from Audapter's logs and the design
    planned = {t: shifts[t][shifts[t].vowel != "uw"].groupby("subj").size() for t in shifts}
    kept = {t: a[(a._t == t)].groupby("_subj").size() for t in EXPS}
    pct = {t: 100 * (1 - kept[t] / planned[t]) for t in shifts}
    pct["inout"] = 100 * (1 - kept["inout"] / 720)
    checks.published("Experiment 1: % of trials removed (mean over participants)", pct["in"].mean(), 1.9, 0.05,
                     f"{SRC}, Methods (Data analysis)")
    checks.published("Experiment 2: % of trials removed (mean over participants)", pct["out"].mean(), 2.2, 0.05,
                     f"{SRC}, Methods (Data analysis)")
    checks.published("Experiment 4: % of trials removed (mean over participants)", pct["inout"].mean(), 1.0, 0.1,
                     f"{SRC}, Methods (Data analysis)")
    tot_k = sum(kept[t].sum() for t in ("in", "out", "out2", "inout"))
    tot_p = sum(planned[t].sum() for t in shifts) + 720 * len(kept["inout"])
    checks.published("Experiments 1-4: % of all trials removed", 100 * (1 - tot_k / tot_p), 1.6, 0.05,
                     f"{SRC}, Methods (Data analysis)")
    out = {"pct_exp3": pct["out2"].mean()}

    def change(t, ph, col="dist_onset", drop=(), three=False):
        g = phase_means(a[(a._t == t) & ~a.excluded_by_authors], col, three)
        g = g.drop(index=[s for s in g.index if int(s.split("|")[1]) in drop])
        g = g[["baseline", ph]].dropna()  # only the two phases compared: a participant missing another phase stays in
        dlt = g[ph] - g["baseline"]
        tt = stats.ttest_rel(g[ph], g["baseline"])
        return dlt, float(tt.statistic), g

    V, C = "variability change from baseline (mels)", "centering change from baseline (mels)"
    R = f"{SRC}, Results"
    for t, ph, mean_rep, t_rep, d_rep, src in [
        ("in", "hold2", 4.9, 2.98, 0.609, "Experiment 1 and Figure 5"), ("in", "washout", 4.4, 3.32, 0.677, "Experiment 1"),
        ("out", "hold2", 4.4, 3.53, 0.75, "Experiment 2 and Figure 5"),
    ]:
        dlt, tv, g = change(t, ph)
        lab = f"{EXPS[t][0]}, {'late exposure' if ph != 'washout' else 'washout'}"
        checks.published(f"{lab}: {V}, mean", dlt.mean(), mean_rep, 0.05, f"{R}, {src}")
        checks.published(f"{lab}: paired t", tv, t_rep, 0.005, f"{R}, {src}")
        checks.published(f"{lab}: Cohen's d (t / sqrt n)", tv / math.sqrt(len(dlt)), d_rep, 0.005 if d_rep != 0.75 else 0.005,
                         f"{R}, {src}")
    dlt, tv, g = change("out", "washout")
    checks.published("Experiment 2, washout: variability change, mean, without the participant whose washout the authors' "
                     "summary leaves empty (mels)", dlt.mean(), 1.6, 0.05, f"{R}, Experiment 2")
    checks.published("Experiment 2, washout: paired t (21 participants)", tv, 1.57, 0.03, f"{R}, Experiment 2")
    for t, ph, mrep, trep, drep in [("inout", "holdIn", 4.1, 4.0, 0.89), ("inout", "holdOut", 4.1, 4.04, 0.90)]:
        dlt, tv, g = change(t, ph, three=True)
        lab = f"Experiment 4, {'inward' if ph == 'holdIn' else 'outward'} phase"
        checks.published(f"{lab}: {V}, mean", dlt.mean(), mrep, 0.05, f"{R}, Experiment 4")
        checks.published(f"{lab}: paired t", tv, trep, 0.005 if trep != 4.0 else 0.05, f"{R}, Experiment 4")
        checks.published(f"{lab}: Cohen's d (t / sqrt n)", tv / math.sqrt(len(dlt)), drep, 0.005, f"{R}, Experiment 4")
    # repeated-measures ANOVAs over baseline, late exposure, washout (Experiment 4: baseline, inward, outward)
    for t, cols, col, rep, src in [
        ("in", ["baseline", "hold2", "washout"], "dist_onset", 7.55, "Experiment 1"),
        ("out", ["baseline", "hold2", "washout"], "dist_onset", 7.93, "Experiment 2"),
        ("out2", ["baseline", "hold2", "washout"], "dist_onset", 1.39, "Experiment 3"),
        ("inout", ["baseline", "holdIn", "holdOut"], "dist_onset", 9.71, "Experiment 4"),
        ("null", ["baseline", "hold2", "washout"], "dist_onset", 0.434, "control data set"),
        ("in", ["baseline", "hold2", "washout"], "centering", 0.06, "Experiment 1, centering"),
        ("out", ["baseline", "hold2", "washout"], "centering", 3.94, "Experiment 2, centering"),
        ("out2", ["baseline", "hold2", "washout"], "centering", 0.87, "Experiment 3, centering"),
        ("inout", ["baseline", "holdIn", "holdOut"], "centering", 3.15, "Experiment 4, centering"),
        ("null", ["baseline", "hold2", "washout"], "centering", 1.45, "control data set, centering"),
    ]:
        g = phase_means(a[(a._t == t) & ~a.excluded_by_authors], col, t == "inout")[cols].dropna()
        F, df1, df2 = rm_anova(g)
        tol = 0.0005 if rep == 0.434 else 0.005
        checks.published(f"{EXPS[t][0]}: repeated-measures ANOVA F for phase, {'variability' if col == 'dist_onset' else 'centering'}"
                         f" (df {df1}, {df2})", F, rep, tol, f"{R}, {src}")
    for t, ph, mrep, trep in [("out", "hold2", 2.2, 2.68), ("inout", "holdOut", 2.7, 1.92)]:
        dlt, tv, g = change(t, ph, col="centering", three=t == "inout")
        lab = f"{EXPS[t][0]}, {'late exposure' if t == 'out' else 'outward phase'}"
        checks.published(f"{lab}: {C}, mean", dlt.mean(), mrep, 0.05, f"{R}, {EXPS[t][0]} (centering)")
        checks.published(f"{lab}: centering, paired t", tv, trep, 0.005, f"{R}, {EXPS[t][0]} (centering)")
    dlt, tv, g = change("inout", "holdIn", col="centering", three=True)
    checks.published("Experiment 4, inward phase: centering, paired t", tv, 0.32, 0.005, f"{R}, Experiment 4 (centering)")
    # exposure change against the control data set (Welch t, Cohen's d with the pooled SD)
    cn, _, _ = change("null", "hold2")
    ccn, _, _ = change("null", "hold2", col="centering")
    for t, ph, col, trep, drep, dfrep in [("in", "hold2", "dist_onset", 2.91, 0.84, None),
                                          ("out", "hold2", "dist_onset", 3.19, 0.94, None),
                                          ("out", "hold2", "centering", 3.4, 0.99, 43.6)]:
        ce, _, _ = change(t, ph, col=col)
        ref = cn if col == "dist_onset" else ccn
        w = stats.ttest_ind(ce, ref, equal_var=False)
        sp = math.sqrt(((len(ce) - 1) * ce.var() + (len(ref) - 1) * ref.var()) / (len(ce) + len(ref) - 2))
        what = "variability" if col == "dist_onset" else "centering"
        checks.published(f"{EXPS[t][0]} vs control: {what} change in late exposure, Welch |t|", abs(w.statistic), trep,
                         0.005 if trep != 3.4 else 0.05, f"{R}, {EXPS[t][0]}")
        checks.published(f"{EXPS[t][0]} vs control: {what} change, Cohen's d", (ce.mean() - ref.mean()) / sp, drep,
                         0.01 if (t, col) == ("out", "centering") else 0.005,  # 0.998 here, 0.99 in the paper (see differences)
                         f"{R}, {EXPS[t][0]}")
        if dfrep:
            checks.published(f"{EXPS[t][0]} vs control: {what}, Welch degrees of freedom", w.df, dfrep, 0.05, f"{R}, {EXPS[t][0]}")
    # mean F1 perturbation in Experiment 3 (produced minus heard F1 over exposure, per participant, then averaged)
    h = a[(a._t == "out2") & (a.feedback == "altered") & a.shift_f1.notna()]
    m = (-h.shift_f1).groupby(h.subject).mean()
    checks.published("Experiment 3: mean F1 perturbation over exposure, produced minus heard (mels)", m.mean(), 13.4, 0.05,
                     f"{R}, Perturbation magnitude")
    out["pert_t_exp3"] = float(stats.ttest_1samp(m, 0).statistic)
    for t, ph in (("out2", "hold2"), ("inout", "holdIn"), ("inout", "holdOut")):
        ce, _, _ = change(t, ph, three=t == "inout")
        out[f"vs_ctrl_{t}_{ph}"] = stats.ttest_ind(ce, cn, equal_var=False)
    checks.published("Experiment 3 vs control: variability change in late exposure, Welch |t|",
                     abs(out["vs_ctrl_out2_hold2"].statistic), 1.49, 0.005, f"{R}, Experiment 3")
    return out


def main():
    checks = stdlib.Checks()
    rows, shifts = build(checks)
    a = pd.DataFrame(rows)
    for c in ("pert_value", "shift_f1", "shift_f2"):
        a[c] = pd.to_numeric(a[c])
    extra = checks_all(a, shifts, checks)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
        elif c["kind"] == "published":
            print(f"  ok {c['what']}: {c['computed']:.4f} vs {c['reported']}")
    print("  [not a check] Experiment 3 trials removed (mean %):", round(extra["pct_exp3"], 3),
          "; Experiment 3 perturbation t:", round(extra["pert_t_exp3"], 3))
    for k, v in extra.items():
        if k.startswith("vs_ctrl"):
            print(f"  [not a check] {k}: Welch t {v.statistic:.3f}, df {v.df:.1f}")
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Decoded the authors' MATLAB table objects, merged the five experiments' per-trial tables into one table "
                 "in trial order (the authors' tables group trials by phase and vowel), attached each exposure trial's "
                 "logged feedback shift (Experiments 1-3), converted seconds to milliseconds, left out the distance and "
                 "centering columns that follow from the formants, and replaced participant numbers with codes."),
        how=[
            "One row is one word (bead, bad, bod, or bed in Experiment 3; booed also in the control) spoken by one "
            "participant, in trial order; trial is the authors' trial number and its gaps are trials they removed for "
            "production errors or formant-tracking errors (1.6% overall).",
            "f1_mel and f2_mel are F1 and F2 in mels averaged over 50 ms at the vowel midpoint, f1_mel_onset and "
            "f2_mel_onset over the first 50 ms of the vowel; duration_ms is the vowel duration and reaction_time_ms the "
            "time from the start of the trial's recording (just before the word appeared) to vowel onset. The paper's "
            "variability is the distance of the onset formants from the median of the same vowel in the same phase "
            "(block), and centering is that distance at onset minus the one at midpoint.",
            "During exposure (blocks hold1 and hold2, or holdIn1/holdIn2 then holdOut1/holdOut2 in Experiment 4) "
            "Audapter moved each production's F1 and F2 toward (inward) or away from (outward) the participant's "
            "baseline median for that vowel by the percentage in condition; shift_f1 and shift_f2 are that trial's "
            "heard minus produced formants from Audapter's log (mels, positive raises the formant heard) and pert_value "
            "their combined size. The deposit has no shift log for Experiment 4, so those cells are empty there.",
            "Baseline and washout had unaltered feedback; the control data set (Parrell & Niziolek 2021) had no "
            "perturbation at all, so its blocks after baseline are phase 'other', kept under the labels the authors "
            "used to time-match them. The control's booed rows (not analysed in the paper) and the Experiment 2 "
            "washout of the participant whose washout the authors' summary table leaves empty are marked "
            "excluded_by_authors. Each experiment had its own participants; whether any control participant also took "
            "part in Experiments 1-4 is not stated, so they are coded as separate people.",
        ],
        unit_note="spoken word (utterance) of one participant",
        differences=[
            "Trials removed in Experiment 3: the shared tables give 1.59% (the paper reports 1.5%).",
            "Experiment 2 against the control, centering change in late exposure: Cohen's d is 0.998 here (paper 0.99); the "
            "paper's Welch t (3.40) and degrees of freedom (43.6) for the same comparison reproduce, so the paper's d may be "
            "truncated rather than rounded.",
            f"Mean F1 perturbation in Experiment 3: the logs give 13.35 mels with t(20) = {extra['pert_t_exp3']:.2f} "
            "(paper 13.4 mels, t(20) = 2.47); the paper's means for the other experiments (2.5-13.4 mels) are not "
            "matched one to one because the paper does not say which experiment each belongs to.",
            "Experiment 4 against the control (inward and outward phase): the shared rows give Welch t = "
            f"{abs(extra['vs_ctrl_inout_holdIn'].statistic):.2f} and {abs(extra['vs_ctrl_inout_holdOut'].statistic):.2f} "
            "(paper 2.56 and 2.23); the within-experiment statistics for the same data, and the Experiment 3 comparison "
            "with the control (Welch t 1.49), reproduce.",
        ],
    )
    out = a.drop(columns=["_t", "_subj", "_vowel"])
    return stdlib.release(ID, "speech", out.to_dict("records"), meta, checks)  # records: module columns this dataset lacks stay empty


if __name__ == "__main__":
    main()
