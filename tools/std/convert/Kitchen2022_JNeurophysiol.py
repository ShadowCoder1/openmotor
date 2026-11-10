"""Kitchen, Kim, Wang, Hermosillo & Max (2022), J Neurophysiol 128(3):696-710: speech auditory-motor adaptation (all formants raised
250 cents; words talk / tuck / tech) with a gradual or a sudden onset, in 24 women who also did a visuomotor reaching task.

Raw files, raw_extra/Kitchen2022_JNeurophysiol/ (OSF 7uvzd), speech task only (the reaching files of the same repository are not part
of this release):
  Speech_<participant>_<G|S>_.txt  195 rows x 6 columns, tab-delimited, header WordHertz F1Hertz F2Hertz WordCents F1Cents F2Cents.
     Columns 1-3: word, F1 and F2 in Hz in the order the trials were performed (NaN for rejected trials). Columns 4-6: the same
     productions in cents grouped by word (all talk, all tech, all tuck in the order given by WordCents), order within a word
     preserved, rejected trials interpolated, cents relative to the median of trials 4-10 of the word (SpeechAdaptationData_ReadMe.txt).
  Speech_BehavCorrels_All.txt  per participant and condition: Extent, BLVar, BLDur, AdptDur, AdptWTCrct, BLCentering (readme).
Paper schedule (Methods, read at PMC9484989): 3-trial epochs, word order random within an epoch, 6 familiarization trials not in the files.
  Gradual: 10 baseline epochs, ramp over 20 epochs, 250 cents held to epoch 50, 15 washout epochs. Sudden: 20 baseline epochs, 250 cents
  from epoch 21 to 50, 15 washout epochs. 65 epochs = 195 trials. Extent of adaptation = mean of the last 5 perturbation epochs, as a
  percentage of the perturbation (a shift of -250 cents = 100%).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Kitchen2022_JNeurophysiol"
SRC = "Kitchen et al. 2022, J Neurophysiol 128(3):696-710, Results"
SPEC = {"G": dict(name="Gradual", base=10, ramp_end=30, hold_end=50), "S": dict(name="Sudden", base=20, ramp_end=20, hold_end=50)}


def read_file(p):
    t = p.read_bytes().decode("utf-8", "replace").splitlines()
    assert t[0].split("\t") == ["WordHertz", "F1Hertz", "F2Hertz", "WordCents", "F1Cents", "F2Cents"]
    r = [ln.split("\t") for ln in t[1:] if ln.strip()]
    assert len(r) == 195, (p.name, len(r))
    return r


def num(x):
    try:
        v = float(x)
    except ValueError:
        return np.nan
    return v


def build(checks):
    raw = stdlib.raw_dir(ID)
    rows, ext, trials = [], {}, []
    pids = sorted({p.name.split("_")[1] for p in raw.glob("Speech_CF*_*.txt")}, key=lambda s: int(s[2:]))
    for pid in pids:
        for c in "GS":
            fs = [p for p in raw.glob(f"Speech_{pid}_{c}*.txt")]
            assert len(fs) == 1, (pid, c, fs)
            r = read_file(fs[0])
            words = [x[0].strip().upper() for x in r]
            cw = [x[3].strip().upper() for x in r]
            # cents blocks: start row of each word's block, then the k-th occurrence of the word in performance order
            start = {w: cw.index(w) for w in set(cw)}
            assert all(cw[start[w]:start[w] + 65] == [w] * 65 for w in start) and sorted(start) == ["TALK", "TECH", "TUCK"]
            seen = {}
            sp = SPEC[c]
            cents = np.full((195, 2), np.nan)
            for i, w in enumerate(words):
                k = seen.get(w, 0)
                seen[w] = k + 1
                cents[i] = [num(r[start[w] + k][4]), num(r[start[w] + k][5])]
            assert all(v == 65 for v in seen.values())
            pct = []
            for i in range(195):
                ep = i // 3 + 1
                if ep <= sp["base"]:
                    block, ph, pert = "Baseline", "baseline", 0.0
                elif ep <= sp["ramp_end"]:
                    block, ph, pert = "Ramp", "learning", np.nan
                elif ep <= sp["hold_end"]:
                    block, ph, pert = "Hold", "learning", 250.0
                else:
                    block, ph, pert = "Washout", "aftereffect", 0.0
                f1, f2 = num(r[i][1]), num(r[i][2])
                rows.append({
                    "dataset": ID, "study": ID, "experiment": "Speech adaptation", "group": "", "subject": f"{pid}|{sp['name']}", "person": pid,
                    "subject_excluded_by_authors": False, "age": "", "sex": "F", "handedness": "right", "population": "healthy",
                    "setting": "lab", "device": "Shure SM58 microphone, TC Helicon VoiceOne, Etymotic ER-3A insert earphones", "session": 1,
                    "block": f"{block}, epoch {ep}", "trial": i + 1, "phase": ph, "time_s": "", "task": "speech", "unit": "utterance",
                    "condition": sp["name"], "pert_type": "formant_shift" if ph == "learning" else "none", "pert_value": pert,
                    "pert_unit": "cents", "feedback": "altered" if ph == "learning" else "normal", "excluded_by_authors": False,
                    "stimulus": words[i].lower(), "segment": "vowel of the word", "measure_window": "5-25% into the vowel",
                    "f1_hz": "" if np.isnan(f1) else f1, "f2_hz": "" if np.isnan(f2) else f2,
                    "shift_f1": pert, "shift_f2": pert, "masking_noise": "none reported", "feedback_latency_ms": 10,
                    "f1_cents": cents[i, 0], "f2_cents": cents[i, 1],
                    "change_reference": "median of trials 4-10 of the same word; cents of rejected trials are interpolated by the authors",
                })
                trials.append((pid, c, ep, (cents[i, 0] + cents[i, 1]) / 2))
    t = pd.DataFrame(trials, columns=["pid", "c", "ep", "cents"])
    # extent: mean of the last 5 perturbation epochs (46-50) of the epoch means, as % of the 250-cent perturbation (shift down = +)
    em = t.groupby(["pid", "c", "ep"]).cents.mean().reset_index()
    ex = em[em.ep.between(46, 50)].groupby(["pid", "c"]).cents.mean() * -100 / 250
    return rows, pids, ex


def main():
    checks = stdlib.Checks()
    rows, pids, ex = build(checks)
    a = pd.DataFrame(rows)
    raw = stdlib.raw_dir(ID)
    bc = pd.read_csv(raw / "Speech_BehavCorrels_All.txt", sep="\t")
    bc["pid"] = bc.PPant.str.split("_").str[0]
    bc["c"] = bc.PPant.str.split("_").str[1]
    print("DEBUG", ex.groupby("c").agg(["mean", "std"]).round(2).to_dict(), "pooled", ex.groupby("pid").mean().agg(["mean", "std"]).round(2).to_dict())
    w = ex.unstack()
    m = bc.set_index(["pid", "c"]).Extent
    checks.internal("extent computed from the trial rows (mean of epochs 46-50, % of 250 cents) equals the deposit's Extent column",
                    float((ex - m).abs().max()) < 1e-9, f"max difference {float((ex - m).abs().max()):.1e}, {len(m)} values")
    checks.published("participants analysed", a.person.nunique(), 24, 0, SRC + " (26 recruited, 24 analysed; Methods)")
    checks.published("trials per condition", int(a.groupby("subject").size().iloc[0]), 195, 0, "Kitchen et al. 2022, Methods (195 trials, 65 per word)")
    checks.published("adaptation extent, pooled over both conditions, mean (% of perturbation)", float(ex.mean()), 29.2, 0.05, SRC)
    checks.published("adaptation extent, pooled over both conditions, SD (%)", float(ex.std(ddof=1)), 19.8, 0.05, SRC)
    checks.published("correlation of speech adaptation extent between the gradual and sudden conditions, r", float(np.corrcoef(w.G, w.S)[0, 1]),
                     0.17, 0.005, SRC)
    checks.internal("65 trials of each word in every sequence", bool((a.groupby(["subject", "stimulus"]).size() == 65).all()))
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    pc = []
    bc["name"] = bc.c.map({"G": "gradual", "S": "sudden"})
    cols = [("Extent", "extent_pct", "percent of perturbation", "Adaptation extent: mean over the last 5 perturbation epochs, as % of the 250-cent shift (100% = fully opposing)"),
            ("BLVar", "baseline_variability_cents", "cents", "Baseline variability: standard deviation of the normalized values over baseline trials 4-10"),
            ("BLDur", "baseline_trial_duration", "as shared", "Average trial duration during baseline trials 4-10 (the authors' BLDur; unit not stated in the deposit)"),
            ("AdptDur", "adaptation_trial_duration", "as shared", "Average trial duration during the adaptation phase (the authors' AdptDur)"),
            ("AdptWTCrct", "within_trial_correction", "as shared", "Average within-trial correction during adaptation (negative values: correction opposite to the perturbation)"),
            ("BLCentering", "baseline_centering", "as shared", "Centering measure for baseline trials (the authors' BLCentering)")]
    pcols = []
    for pid in pids:
        rec = {"person": pid}
        for nm in ("gradual", "sudden"):
            g = bc[(bc.pid == pid) & (bc.name == nm)].iloc[0]
            for src, col, u, d in cols:
                rec[f"{col}_{nm}"] = float(g[src])
        pc.append(rec)
    for nm in ("gradual", "sudden"):
        for src, col, u, d in cols:
            pcols.append((f"{col}_{nm}", "number", u, f"{d} ({nm} condition). As shared; the authors excluded some values as outliers (see paper)."))
    s_ = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    pp = s_["papers"][0]
    meta = dict(
        study=ID, paper={k: pp[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s_["lab"], data_url=s_["data_url"],
        license="none stated", license_url="",
        changes=("Joined each participant's Hz columns (performance order) to their cents columns (grouped by word) trial by trial, added "
                 "the epoch, phase and feedback shift from the paper, kept the authors' per-participant behavioural variables as "
                 "participants.csv, and replaced file names with participant codes. The reaching files of the same repository are not part of this release."),
        how=[
            "One row is one spoken word (talk, tuck or tech) of one participant in the gradual or the sudden condition, in the order performed "
            "(195 trials = 65 three-trial epochs; the six familiarization trials are not in the deposit). f1_hz and f2_hz are the authors' "
            "values over 5-25% of the vowel (empty for rejected trials); f1_cents and f2_cents are the same production in cents relative to the "
            "median of the word's trials 4-10, with rejected trials interpolated by the authors.",
            "The deposit lists the cents columns grouped by word, not in performance order; the k-th production of a word was matched to the "
            "k-th cents row of that word (the word labels agree and the counts are 65 each). The Hz and cents values are of the same trial but "
            "the match rests on this ordering rule stated in the deposit's readme.",
            "pert_value is 250 cents (all formants up) in the hold epochs (gradual 31-50, sudden 21-50), 0 in baseline and washout, and empty "
            "during the gradual ramp (epochs 11-30), where the paper says 'equal increments across 20 epochs' without giving the values.",
            "participants.csv has the authors' per-participant behavioural variables for each condition (extent of adaptation, baseline "
            "variability, trial durations, within-trial correction, centering). Ages are not in the deposit (paper: 24 women, mean 23.0 years).",
        ],
        unit_note="spoken word of one participant in one condition",
        differences=[
            "The two excluded participants (one with curved reaches, one who followed the shift) and the authors' outlier values are not "
            "identified in the deposit, so no rows are marked as excluded.",
            "The paper's table gives baseline centering for the gradual condition as 15.7 +- 1.1 cents; the shared values have SD 17.1, so that "
            "SD was not used as a check.",
        ],
    )
    for r in rows:
        pass
    return stdlib.release(ID, "speech", rows, meta, checks, participants=pc, participant_columns=pcols)


if __name__ == "__main__":
    main()
