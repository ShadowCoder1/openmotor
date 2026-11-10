"""PhysioNet 'Unconstrained and Metronomic Walking' 1.0.0 (Hausdorff et al. 1996, J Appl Physiol 80:1448-1457): stride intervals of
10 young healthy men (si01-si10) who walked about 1 h at normal, slow and fast pace, free (.norm .slow .fast) and in time with a
metronome (.metnrm .metslw .metfst). Each file is one column of stride intervals (s), one per line, no time column. One row per line.

Raw file (untrusted data, read with zipfile only): raw_extra/Hausdorff1996_JApplPhysiol/umwdb.zip
Run from the site folder: python3 -I tools/std/convert/Hausdorff1996_JApplPhysiol.py
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hausdorff1996_JApplPhysiol"
MODULE = "walking"
CONDS = [("norm", "normal pace, free walking", "none"), ("slow", "slow pace, free walking", "none"), ("fast", "fast pace, free walking", "none"),
         ("metnrm", "normal pace, in time with a metronome", "metronome"), ("metslw", "slow pace, in time with a metronome", "metronome"),
         ("metfst", "fast pace, in time with a metronome", "metronome")]


def dfa(x, nmin=8):
    y = np.cumsum(x - x.mean())
    ns = np.unique(np.round(np.logspace(np.log10(nmin), np.log10(len(y) // 4), 14)).astype(int))
    F = []
    for n in ns:
        k = len(y) // n
        seg = y[:k * n].reshape(k, n)
        t = np.arange(n)
        F.append(np.sqrt(np.mean([np.mean((s - np.polyval(np.polyfit(t, s, 1), t)) ** 2) for s in seg])))
    return np.polyfit(np.log(ns), np.log(F), 1)[0]


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(ID) / "umwdb.zip")
    ck = stdlib.Checks()
    names = {}
    for n in z.namelist():
        m = re.fullmatch(r".*/si(\d\d)\.(norm|slow|fast|metnrm|metslw|metfst)", n)
        if m:
            names[(int(m.group(1)), m.group(2))] = n
    subs = sorted({k[0] for k in names})
    frames, people, al = [], [], {c[0]: [] for c in CONDS}
    for s in subs:
        for si, (ext, cond, pt) in enumerate(CONDS, start=1):
            x = np.loadtxt(io.StringIO(z.read(names[(s, ext)]).decode()))
            assert x.ndim == 1
            n = len(x)
            al[ext].append(dfa(x))
            frames.append(pd.DataFrame(dict(
                dataset=ID, study="Hausdorff et al. 1996, J Appl Physiol", experiment="", group="", subject=f"s{s:02d}",
                subject_excluded_by_authors=False, age=np.nan, sex="M", handedness="", population="healthy young men", setting="lab",
                device="ultra-thin force-sensitive switches taped inside one shoe, ankle-worn recorder (300 Hz)", session=1,
                block=ext, trial=np.arange(1, n + 1), phase="other", time_s=np.nan, task="walking", unit="stride",
                condition=cond, pert_type=pt, pert_value=np.nan if pt == "metronome" else 0, pert_unit="", feedback="normal",
                excluded_by_authors=False, fast_leg="unknown", walking_context="overground", surface_incline_deg=0.0,
                walking_trial=si, stride_time_s=x)))
    df = pd.concat(frames, ignore_index=True)
    df["session"] = 1
    df["trial"] = df.groupby(["subject", "block"]).cumcount() + 1
    print("DFA alpha by condition (informational):", {k: round(float(np.mean(v)), 2) for k, v in al.items()})
    src = "PhysioNet database page / paper abstract"
    ck.published("subjects", len(subs), 10, 0, src + " (ten healthy subjects)")
    ck.published("files per subject (3 free + 3 metronome paces)", len(names) / len(subs), 6, 0, src + ", Data Description")
    per = df.groupby(["subject", "block"]).size()
    ck.internal("each walk has at least 1000 strides", bool((per >= 1000).all()), f"{per.min()}-{per.max()} strides")
    ck.internal("stride times positive", bool((df.stride_time_s > 0).all()))
    ck.internal("every line of every file kept", len(df) == sum(len(f) for f in frames), f"{len(df)} rows")
    f = {k: np.mean(v) for k, v in al.items()}
    ck.internal("free walking is long-range correlated (alpha > 0.7) and metronome walking is less so (abstract)",
                all(f[k] > 0.7 for k in ("norm", "slow", "fast")) and all(f[k] < f["norm"] for k in ("metnrm", "metslw", "metfst")), str({k: round(float(v), 2) for k, v in f.items()}))
    meta = dict(
        study=ID,
        paper=dict(authors="Hausdorff JM, Purdon PL, Peng CK, Ladin Z, Wei JY, Goldberger AL", year=1996,
                   title="Fractal dynamics of human gait: stability of long-range correlations in stride interval fluctuations",
                   venue="J Appl Physiol 80(5):1448-1457", doi="10.1152/jappl.1996.80.5.1448"),
        lab="Hausdorff / Goldberger (Beth Israel Deaconess Medical Center, Harvard)",
        data_url="https://physionet.org/content/umwdb/1.0.0/", license="ODC-By 1.0", license_url="https://opendatacommons.org/licenses/by/1-0/",
        changes="Kept every line of every file in order; added the subject, pace and metronome condition from the file name. Nothing filtered.",
        how=["One row is one stride interval (stride_time_s, seconds, from one heel strike to the next of the same foot), in the order walked. The deposit has "
             "no time column, so time_s and stride_start_s are empty; trial counts strides within a walk.",
             "Each of 10 young men walked about 1 h per condition on an oval, obstacle-free path (225 or 400 m) with force-sensitive switches in one shoe: normal, "
             "slow and fast self-chosen paces (block norm, slow, fast) and in time with a metronome at the normal, slow and fast paces (metnrm, metslw, metfst). "
             "walking_trial numbers the six conditions in that order. Metronome rate is not in the deposit (pert_value empty).",
             "Age and sex per subject are not shared (the page gives mean age 21.7 years, range 18-29, for the group of men); sex is therefore M for all."],
        unit_note="stride",
        differences=["The deposit shares no per-subject ages, no time column and no metronome rates, so the paper's DFA exponents were not used as checks; "
                     "the checks are the subject and file counts and the direction of the effect."])
    cols_ = [c[1] for c in stdlib.columns_for(MODULE)]
    df = df.assign(**{c: np.nan for c in cols_ if c not in df.columns})[cols_]
    stdlib.release(ID, MODULE, df, meta, ck)


if __name__ == "__main__":
    main()
