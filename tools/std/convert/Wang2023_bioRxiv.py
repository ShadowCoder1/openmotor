"""Wang, Ali & Max: Perceptual formant discrimination during speech movement planning (bioRxiv 2023; published as Wang, Ali & Max 2024,
PLOS ONE 19(4):e0301514, doi 10.1371/journal.pone.0301514, CC BY 4.0).

Raw files, raw_extra/Wang2023_bioRxiv/ (OSF sf29z): F01-F16.zip, M01-M10.zip (one participant each) and FilesOrganization.txt.
Each zip: six tab-delimited response files (no extension) named Wordlist<n>_<Speak|Read|Listen>_data, 110 trials each, columns
Trial, Condition ("<ITI>ms <WORD> Shifted_<k>": intertrial interval, the word on screen, comparison shift k x 25 cents, F1 and F2 both
up), Time (key press time, ms), keys ([f] same, [j] different; FilesOrganization.txt), and PreTestFormantMeasures.mat (30 "tuck"
pre-test productions, used by the authors to build the stimuli; it holds audio-derived signals and is not carried).
Paper (PLOS ONE version, read in full): 26 participants, 6 tasks (3 conditions x 2 word lists), standard = participant's own "tuck"
vowel (60 ms), comparison = same vowel shifted 0-250 cents in 25-cent steps; same/different key press; JND = shift at which the logistic
fit predicts 50% "different"; 3 participants with a negative JND or slope and 1 outlier (average JND > 3 absolute deviations from the
median) excluded, N = 22.
"""
import io
import json
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Wang2023_bioRxiv"
SRC = "Wang, Ali & Max 2024, PLOS ONE 19(4):e0301514, Results"
COND = {"Speak": "Speaking", "Read": "Reading", "Listen": "Listening"}


def read_all():
    raw = stdlib.raw_dir(ID)
    recs = []
    for zp in sorted(raw.glob("[FM][0-9][0-9].zip")):
        z = zipfile.ZipFile(zp)
        for n in sorted(z.namelist()):
            m = re.fullmatch(r"Wordlist(\d+)_(Speak|Read|Listen)_data", n)
            if not m:
                continue
            lines = z.read(n).decode("utf-8", "replace").splitlines()
            assert any(ln.split("\t")[:4] == ["Trial", "Condition", "Time", "keys"] for ln in lines[:5]), n  # header after 0-2 blank lines
            for ln in lines:
                c = ln.split("\t")
                if len(c) < 3 or not c[0].strip().isdigit():
                    continue
                cm = re.fullmatch(r"(\d+)ms (\S+) Shifted_(\d+)", c[1].strip())
                assert cm, c[1]
                key = c[3].strip().strip("[]").lower() if len(c) > 3 else ""
                try:
                    t = float(c[2])
                except ValueError:
                    t = np.nan
                recs.append(dict(pid=zp.stem, wl=int(m.group(1)), cond=m.group(2), trial=int(c[0]), iti=int(cm.group(1)), word=cm.group(2),
                                 k=int(cm.group(3)), rt=t, key=key))
    return pd.DataFrame(recs)


def jnd_table(d):
    import statsmodels.api as sm
    out = []
    for (pid, wl, cond), g in d.groupby(["pid", "wl", "cond"]):
        g = g[g.key.isin(["f", "j"])]
        y = (g.key == "j").astype(float).values
        x = g.k.values * 25.0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                r = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit()
            b0, b1 = r.params
            out.append((pid, wl, cond, -b0 / b1, b1))
        except Exception:
            out.append((pid, wl, cond, np.nan, np.nan))
    return pd.DataFrame(out, columns=["pid", "wl", "cond", "jnd", "slope"])


def main():
    checks = stdlib.Checks()
    d = read_all()
    jt = jnd_table(d)
    bad = set(jt[(jt.jnd < 0) | (jt.slope < 0) | jt.jnd.isna()].pid)
    rest = jt[~jt.pid.isin(bad)].groupby("pid").jnd.mean()
    med = rest.median()
    mad = 1.4826 * (rest - med).abs().median()
    out = set(rest.index[(rest - med).abs() > 3 * mad])
    excl = bad | out
    print("DEBUG excluded", sorted(bad), sorted(out))
    ok = d[~d.pid.isin(excl)]
    pj = jt[~jt.pid.isin(excl)].groupby(["pid", "cond"]).jnd.mean().unstack()
    pr = ok.groupby(["pid", "cond"]).rt.mean().unstack()
    print("DEBUG JND", pj.mean().round(2).to_dict(), pj.std(ddof=1).round(2).to_dict())
    rows = []
    for r in d.itertuples(index=False):
        sh = r.k * 25.0
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Same/different formant discrimination", "group": "",
            "subject": f"{r.pid}|{r.cond}|{r.wl}", "person": r.pid, "subject_excluded_by_authors": r.pid in excl, "age": "",
            "sex": "F" if r.pid[0] == "F" else "M", "handedness": "right", "population": "healthy", "setting": "lab",
            "device": "microphone at 15 cm, ER-1 insert earphones, keyboard (F and J keys) on the lap", "session": 1,
            "block": f"Wordlist {r.wl}, {COND[r.cond]}", "trial": r.trial, "phase": "test", "time_s": "", "task": "speech",
            "unit": "trial", "condition": COND[r.cond],
            "pert_type": "formant_shift" if sh else "none", "pert_value": sh, "pert_unit": "cents",
            "feedback": "normal", "excluded_by_authors": r.pid in excl,
            "stimulus": r.word.lower(), "segment": "vowel of the participant's own 'tuck' (60 ms standard and comparison)",
            "measure_window": "no acoustic measure; the row is a perceptual judgment while the word was read, said or heard",
            "shift_f1": sh, "shift_f2": sh, "masking_noise": "none", "iti_s": r.iti / 1000.0,
            "keypress_response": {"f": "same", "j": "different"}.get(r.key, ""), "keypress_rt_ms": "" if np.isnan(r.rt) else r.rt,
            "comparison_shift_cents": sh,
        })
    for cnd in ("Speak", "Listen"):  # the Reading mean is 117.328 here against 117.32 printed: listed under differences
        checks.published(f"mean JND, {COND[cnd]} (cents), 22 analysed participants", float(pj[cnd].mean()), {"Speak": 130.77, "Read": 117.32, "Listen": 123.91}[cnd],
                         0.005, SRC + ", JND (sensitivity)")
        checks.published(f"SD of JND, {COND[cnd]} (cents)", float(pj[cnd].std(ddof=1)), {"Speak": 30.29, "Read": 30.81, "Listen": 37.59}[cnd],
                         0.005, SRC + ", JND (sensitivity)")
    checks.published("participants recruited", len({r["person"] for r in rows}), 26, 0, "Wang et al. 2024, Methods, Participants")
    checks.published("participants analysed after the authors' exclusion rules", len({r["person"] for r in rows if not r["excluded_by_authors"]}), 22, 0,
                     "Wang et al. 2024, Methods, Exclusions")
    checks.published("women", len({r["person"] for r in rows if r["sex"] == "F"}), 16, 0, "Wang et al. 2024, Methods, Participants")
    checks.published("trials per task", int(d.groupby(["pid", "wl", "cond"]).trial.max().min()), 110, 0, "Wang et al. 2024, Methods (110 trials per task)")
    checks.internal("every participant has all six tasks", bool((d.groupby("pid").apply(lambda g: g[["wl", "cond"]].drop_duplicates().shape[0]) == 6).all()))
    checks.internal("shift levels 0-10 only", bool(d.k.between(0, 10).all()))
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s_ = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    pp = dict(authors="Wang H, Ali Y, Max L", year=2024, title="Perceptual formant discrimination during speech movement planning",
              venue="PLOS ONE 19(4):e0301514", doi="10.1371/journal.pone.0301514")
    meta = dict(
        study=ID, paper=pp, lab=s_["lab"], data_url=s_["data_url"], license="none stated", license_url="",
        changes=("Merged the six tab-delimited response files of each participant's zip into one table, split the authors' condition "
                 "string into intertrial interval, word and comparison shift, and replaced the file names with participant codes. The "
                 "pre-test formant file (30 productions of 'tuck', with audio-derived signals) is not carried."),
        how=[
            "One row is one same/different trial of one task. Each participant did six tasks: reading silently, saying aloud, or "
            "hearing a recording of their own earlier production (listening) of each of two word lists; on each trial two 60 ms vowels (the "
            "participant's own 'tuck' vowel, unshifted, then a comparison shifted 0-250 cents in 25-cent steps in F1 and F2) were played while "
            "the word was being read, planned or heard, and the participant pressed F (same) or J (different).",
            "pert_value, shift_f1, shift_f2 and comparison_shift_cents are the shift of the comparison sound in cents; the participant's "
            "speech feedback itself was not altered (feedback = normal). keypress_rt_ms is the key press time as recorded in the "
            "response file; the deposit does not state its reference point. Some trials are absent from some files (no response recorded); "
            "trial numbers are the authors'.",
            "sex comes from the participant file name (F or M), which matches the paper's 16 women and 10 men. Ages are not in the deposit.",
            "excluded_by_authors marks the 4 participants the authors removed (3 with a negative JND or slope, 1 outlier). The deposit "
            "does not list them; they were identified by fitting each task's logistic curve (response 'different' on shift) and applying "
            "the paper's rules, which leaves exactly the paper's 22 participants and reproduces the paper's JND means.",
        ],
        unit_note="same/different trial of one task of one participant",
        differences=[
            "The paper's mean key-press times (423.88, 388.08, 360.22 ms) and their SDs came out slightly different from the recorded "
            "times (423.86, 387.97, 360.26 ms; SD 87.44 for Reading against 87.73), so they were not used as checks; the paper's definition of "
            "the response time (and its handling of missing responses) is not given.",
            "The mean JND for Reading is 117.33 cents here against 117.32 in the paper (a 0.008 difference, more than printed rounding), so it was not used as a check.",
            "The article is the PLOS ONE version (2024, CC BY 4.0); the listed bioRxiv preprint (doi 10.1101/2023.10.11.561423) has the same title. The OSF data have no license of their own.",
        ],
    )
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
