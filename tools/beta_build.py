"""Data for the Beta page: preview curves for every standardized experiment, and a cross-study summary.

Reads the published standardized files (source/release/*.zip) and writes
  source/beta/<dataset>.json  preview curves for one dataset; the page loads it when someone opens that study
  source/beta/index.json      a short summary of every dataset, and the points for "What the data show"

Every number is an average over participants; no participant codes are written.
Hand angles (and aim reports) are flipped for each participant so that moving against their first
perturbation (a rotation or a clamp) is positive. The perturbation is flipped the same way, so a hand
that fully cancels a rotation sits on the rotation line. Trials and participants that the authors
excluded are left out, as in their analyses.

Run: python3 tools/beta_build.py   (about two minutes; then tools/build_page.py copies the files into dist/beta/)
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
REL = HERE / "source/release"
OUT = HERE / "source/beta"
COLS = ["experiment", "subject", "group", "trial", "phase", "trial_kind", "fb_visible", "fb_mode", "rotation", "clamp",
        "hand_angle", "aim_report", "rt_ms", "excluded_by_authors", "subject_excluded_by_authors", "instr_strategy",
        "instr_goal", "setting", "population", "prior_exposure"]
MAX_SERIES = 8      # lines per experiment
LATE_N = 20         # feedback trials averaged at the end of the first perturbation block
MIN_BLOCK = 40      # trials in that block, for the cross-study summary
MIN_N = 5           # participants per line, for the cross-study summary
GENERIC_GROUPS = {"Arrighi2016_PLoSOne"}  # group labels never shown for these datasets
PHASE_ORDER = ["practice", "baseline", "learning", "aftereffect", "washout", "relearning", "probe"]
# Datasets whose file puts several experiments under one experiment id: the experiment is read from the group name instead
# (dataset -> (pattern on the group name, the experiment id in the file)). Tsay et al. 2023 (Proc R Soc B) codes Exp 1
# (90-trial blocks, clamps of 30, 45 and 60 deg) and Exp 2 (270-trial blocks, four training schedules) as one experiment,
# "All"; its group names start with "Exp1" or "Exp2".
EXPERIMENT_FROM_GROUP = {"Tsay2023_ProcRSocB": (r"^(Exp\d+)\b", "Tsay2023_ProcRSocB_All")}


def split_experiments(name, df):
    """df with its experiment column split by group name where EXPERIMENT_FROM_GROUP says so (the same table otherwise)"""
    rule = EXPERIMENT_FROM_GROUP.get(name)
    if not rule:
        return df
    pre = df["group"].astype("string").fillna("").str.extract(rule[0], expand=False)
    hit = pre.notna() & df["experiment"].astype(str).eq(rule[1])
    df = df.copy()
    df.loc[hit, "experiment"] = name + "_" + pre[hit].astype(str)
    return df


def source_experiment(name, exp):
    """the experiment id in the dataset's file for an experiment id made by split_experiments()"""
    rule = EXPERIMENT_FROM_GROUP.get(name)
    return rule[1] if rule and re.fullmatch(re.escape(name) + r"_Exp\d+", str(exp)) else exp


def r1(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 1)


def rle(values):
    """[[start, length, value], ...]; NaN becomes null."""
    out = []
    for i, v in enumerate(values):
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            v = None
        if out and out[-1][2] == v:
            out[-1][1] += 1
        else:
            out.append([i, 1, v])
    return out


def mode_or_none(s):
    s = s.dropna()
    return None if s.empty else s.value_counts().index[0]


def exp_label(ds, exp):
    tail = exp[len(ds):].lstrip("_") if exp.startswith(ds) else exp
    m = re.fullmatch(r"Exp(\d+)([A-Za-z]?)", tail)
    if m:
        return "Experiment " + m.group(1) + m.group(2).lower()
    return tail.replace("_", " ") or "Experiment"


def kind_word(mode):
    return {"rotated": "rotation", "clamped": "clamp"}.get(mode, "")


def schedule_text(sig, mode_at):
    """'rotation 30°' or 'clamp 3°, then 30°' from a participant's schedule [(start, value), ...]."""
    vals, starts = [], []
    for start, v in sig:
        if v not in (None, 0.0) and (not vals or vals[-1] != v):
            vals.append(v)
            starts.append(start)
    if not vals:
        return "no rotation or clamp"
    word = kind_word(mode_at(starts[0])) or "perturbation"
    txt = lambda v: f"{v:g}°"
    return word + " " + ", then ".join(txt(v) for v in vals[:3]) + (", ..." if len(vals) > 3 else "")


def read_dataset(zip_path, name, cols=COLS):
    """The trial table of one standardized reaching ZIP (any reach90 dataset: the first release in source/release/,
    or a later one downloaded from its release URL), with only the columns asked for."""
    with zipfile.ZipFile(zip_path) as z:
        member = f"{name}/{name}.csv"
        if member not in z.namelist():  # one data CSV next to columns.csv and the README
            member, = [m for m in z.namelist() if m.endswith(".csv") and not m.endswith("columns.csv")]
        return pd.read_csv(z.open(member), usecols=list(cols), low_memory=False)


def build_dataset(name, paradigms, zip_path=None, extra_cols=(), on_line=None):
    """Preview lines of one dataset. zip_path defaults to source/release/<name>.zip. extra_cols are read as well
    (for callers, unused here); on_line(**context) is called for every line, with the line's trials (tools/explore_build.py)."""
    df = read_dataset(zip_path or REL / f"{name}.zip", name, COLS + [c for c in extra_cols if c not in COLS])
    df["group"] = df["group"].astype("string").fillna("")
    df = split_experiments(name, df)
    out = {"name": name, "experiments": []}
    summary_points = []
    for exp, de in df.groupby("experiment", sort=True):
        n_all = de["subject"].nunique()
        excl_subj = de.groupby("subject")["subject_excluded_by_authors"].any()
        keep = excl_subj.index[~excl_subj.values]
        de = de[de["subject"].isin(keep)].sort_values(["subject", "trial"]).copy()
        if de.empty:
            continue
        pert = np.where(de["fb_mode"].eq("clamped"), de["clamp"], de["rotation"]).astype(float)
        de["pert"] = pert
        nz = de[np.isfinite(pert) & (np.abs(pert) > 0)]
        first = nz.groupby("subject")["pert"].first()
        flip = (-np.sign(first)).reindex(de["subject"].unique()).fillna(1.0)
        f = de["subject"].map(flip).astype(float)
        ok = de["trial_kind"].eq("reach") & ~de["excluded_by_authors"].astype(bool)
        de["hand_n"] = np.where(ok, f * de["hand_angle"], np.nan)
        de["aim_n"] = np.where(~de["excluded_by_authors"].astype(bool), f * de["aim_report"], np.nan)
        de["rt"] = np.where(ok, de["rt_ms"], np.nan)
        de["pert_n"] = np.round(-f * de["pert"], 1) + 0.0  # + 0.0 turns -0.0 into 0.0

        # participants whose perturbation schedule is the same form one line within their group
        # (a schedule is the sequence of perturbation values; trials without a value, such as no-feedback
        # trials, do not count, so one missing value does not make a participant's schedule different)
        sig = {}
        for s, d in de.groupby("subject", sort=False):
            segs = []
            for a, _, v in rle(d["pert_n"].tolist()):
                if v is not None and (not segs or segs[-1][1] != v):
                    segs.append((a, v))
            sig[s] = tuple(segs)
        grp_of = de.groupby("subject", sort=False)["group"].first()
        groups = grp_of.value_counts()
        plan = []  # (group, schedule or None, subjects, varies)
        odd = 0    # participants left out because their schedule matched no line of their group
        for g in groups.index:
            subs = grp_of.index[grp_of.values == g]
            cohorts = pd.Series([sig[s] for s in subs], index=subs)
            vc = cohorts.value_counts()
            big = [c for c in vc.index if vc[c] >= max(2, 0.1 * len(subs))]
            rest = len(subs) - sum(int(vc[c]) for c in big)
            if vc.iloc[0] >= 0.9 * len(subs):
                plan.append((g, None, list(subs), False))
            elif 2 <= len(big) <= 4 and rest <= 0.25 * len(subs):
                for c in big:
                    plan.append((g, c, [s for s in subs if sig[s] == c], False))
                odd += rest
            else:
                plan.append((g, None, list(subs), True))
        if len(plan) > MAX_SERIES:  # too many schedules: one line per group instead
            plan = [(g, None, list(grp_of.index[grp_of.values == g]), True) for g in groups.index]
            odd = 0
        omitted = max(0, len(plan) - MAX_SERIES)
        plan = sorted(plan, key=lambda p: -len(p[2]))[:MAX_SERIES]
        many_groups = len(groups) > 1 or (len(groups) == 1 and groups.index[0] != "")
        gnames = {g: (f"Group {i + 1}" if name in GENERIC_GROUPS else (g or "All participants")) for i, g in enumerate(groups.index)}

        series = []
        for g, c, subs, varies in plan:
            d = de[de["subject"].isin(subs)]
            T = int(d["trial"].max()) + 1
            by = d.groupby("trial")
            idx = pd.RangeIndex(T)
            h = by["hand_n"].agg(["mean", "std", "count"]).reindex(idx)
            sem = h["std"] / np.sqrt(h["count"].where(h["count"] > 1))
            sched = by["pert_n"].median().round(1).reindex(idx)
            modes = by["fb_mode"].agg(mode_or_none).reindex(idx)
            nofb = by["fb_visible"].agg(lambda s: bool((s == "none").mean() >= 0.5)).reindex(idx)
            phases = by["phase"].agg(mode_or_none).reindex(idx)
            mode_at = lambda t: modes.iloc[t] if 0 <= t < T else None
            label = gnames[g] if many_groups else ""
            if c is not None:
                label = (label + ", " if label else "") + schedule_text(c, mode_at)
            ser = {"label": label or "All participants", "n": len(subs), "trials": T,
                   "hand": [r1(v) for v in h["mean"]], "sem": [r1(v) for v in sem],
                   "nt": rle([int(v) for v in h["count"].fillna(0)]),
                   "pert": rle([None if pd.isna(v) else float(v) + 0.0 for v in sched]),
                   "mode": rle([v if v is not None and not pd.isna(v) else None for v in modes]),
                   "nofb": rle([bool(v) if not pd.isna(v) else False for v in nofb]),
                   "phase": rle([v if v is not None and not pd.isna(v) else None for v in phases])}
            if varies:
                ser["varies"] = True
            rt = by["rt"].median().reindex(idx)
            if rt.notna().sum() >= 10:
                ser["rt"] = [None if pd.isna(v) else int(round(v)) for v in rt]
            aim = by["aim_n"].mean().reindex(idx)
            if aim.notna().sum() >= 5:
                ser["aim"] = [r1(v) for v in aim]
            series.append(ser)
            if on_line is not None:
                on_line(exp=exp, exp_label=exp_label(name, exp), group=g, schedule=c, subs=subs, varies=varies, d=d,
                        sched=sched, modes=modes, nofb=nofb, phases=phases, series=ser, paradigms=paradigms.get(exp, []))

            # cross-study summary: the first block of a constant rotation or clamp
            if varies or len(subs) < MIN_N or "cue_context" in paradigms.get(exp, []):
                continue
            pt = summary_point(name, exp, d, sched, modes, nofb, ser["label"], len(subs))
            if pt:
                summary_points.append(pt)
        out["experiments"].append({"id": exp, "label": exp_label(name, exp), "n": int(n_all),
                                   "n_excluded": int(n_all - len(keep)), "series": series, "omitted": omitted, "odd": int(odd)})
    return out, summary_points


def summary_point(name, exp, d, sched, modes, nofb, label, n):
    vals = sched.to_numpy(dtype=float)
    start = next((t for t, v in enumerate(vals) if np.isfinite(v) and v != 0), None)
    if start is None:
        return None
    kind = kind_word(modes.iloc[start])
    if not kind:
        return None
    P = vals[start]
    end = start
    for t in range(start, len(vals)):
        v = vals[t]
        if np.isfinite(v) and v == P:
            end = t
        elif not np.isfinite(v) and bool(nofb.iloc[t]):
            continue
        else:
            break
    if P <= 0 or end - start + 1 < MIN_BLOCK:
        return None
    later = [v for v in vals[end + 1:] if np.isfinite(v)]
    if later and later[0] > P:  # the rotation kept growing: a gradual ramp, not one constant perturbation
        return None
    blk = d[(d["trial"] >= start) & (d["trial"] <= end)]
    fbk = blk[blk["fb_mode"].isin(["rotated", "clamped"]) & blk["hand_n"].notna()]
    if fbk.empty:
        return None
    goals = {"rotation": {"cursor_to_target"}, "clamp": {"cursor_to_target", "hand_to_target"}}[kind]  # clamp studies ask people to ignore the cursor
    if mode_or_none(fbk["instr_strategy"]) == "given_with_angle" or mode_or_none(fbk["instr_goal"]) not in goals:
        return None
    if mode_or_none(blk["population"]) != "healthy":
        return None
    if (blk.groupby("subject")["prior_exposure"].first().astype(str).str.lower() == "true").mean() > 0.5:
        return None  # most had adapted before (for example, a second run): not a first exposure
    per = fbk.groupby("subject")["hand_n"].apply(lambda s: s.tail(LATE_N).mean() if len(s) >= 10 else np.nan).dropna()
    if len(per) < MIN_N:
        return None
    late, sd = float(per.mean()), float(per.std(ddof=1))
    early = fbk.groupby("subject")["hand_n"].apply(lambda s: s.iloc[1:11].mean() if len(s) >= 11 else np.nan).dropna()
    return {"ds": name, "exp": exp, "label": label, "kind": kind, "size": round(float(P), 2), "n": int(len(per)),
            "late": round(late, 2), "sem": round(sd / np.sqrt(len(per)), 2),
            "early": round(float(early.mean()), 2) if len(early) >= MIN_N else None,
            "block": int(end - start + 1), "fb": mode_or_none(fbk["fb_visible"]), "setting": mode_or_none(blk["setting"])}


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    manifest = {m["name"]: m for m in json.load(open(REL / "manifest.json")).values()}
    only = sys.argv[1:]
    index, points = {}, []
    old = OUT / "index.json"
    if only and old.exists():
        prev = json.load(open(old))
        index, points = prev["datasets"], [p for p in prev["points"] if p["ds"] not in only]
    for name, m in sorted(manifest.items()):
        if only and name not in only:
            continue
        data, pts = build_dataset(name, m["paradigms"])
        (OUT / f"{name}.json").write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        index[name] = {"study": m["catalog_study"], "file": m["file"], "bytes": m["bytes"], "rows": m["rows"], "persons": m["persons"],
                       "experiments": [{"id": e["id"], "label": e["label"], "n": e["n"], "lines": len(e["series"])} for e in data["experiments"]]}
        points += pts
        size = (OUT / f"{name}.json").stat().st_size
        print(f"{name:42s} {len(data['experiments'])} exp  {sum(len(e['series']) for e in data['experiments']):2d} lines  {size / 1e3:6.1f} kB  {len(pts)} points", flush=True)
    json.dump({"datasets": index, "points": points}, open(old, "w"), separators=(",", ":"), ensure_ascii=False)
    print(len(index), "datasets,", len(points), "summary points")
