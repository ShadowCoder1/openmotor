"""Miller, Brickhouse, Hochberg, Nieto-Castanon, Tourville, Dickerson & Guenther (2026), Neurobiology of Language 7:NOL.a.281:
speech motor sequence learning of pseudowords with non-native consonant clusters in primary progressive aphasia (PPA).

Raw files in raw_extra/Miller2026_NeurobiolLang/ (OSF p2xuc, no license stated; the paper is CC BY 4.0):
  learning.zip   per-participant text logs, one tab-separated line per utterance: SEQFTD##_Training-analysis-log.txt (270 lines: 180 on
                 day 1 'Training1', 90 on day 2 'Training2') and SEQFTD##_Test-analysis-log.txt (90 lines), plus README.txt that gives the
                 nine columns (file name, overall accuracy, sequencing accuracy, error type, onset sequencing accuracy, onset error type,
                 cluster duration, cluster + vowel duration, comment).
  subjects.csv   sex, diagnosis, age, laterality score, years since symptom onset, education (34 people; 30 have learning logs)
  clinical.xlsx  clinical and motor speech measures per participant, with a 'key' sheet that describes them
Not carried: neuro.xlsx and neuro_controlData.xlsx (cortical thickness from MRI) and SEQFTD_Acoustic_Features_SMR.xlsx (a different
task, sequential motion rate syllable repetition).
Run from the site folder: python3 -I tools/std/convert/Miller2026_NeurobiolLang.py
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Miller2026_NeurobiolLang"
SRC = "Miller et al. 2026, Neurobiology of Language"
RAW = stdlib.raw_dir(ID)
DEV = "microphone recordings over Zoom of cued repetition of pseudowords with non-native consonant clusters (auditory model plus text)"
NAME = re.compile(r"^(SEQFTD\d+)_(\d)_(Training1A?|Training2|Test)_(\d+)_+([A-Za-z]+)(?:_\w+)?$")
NUM = re.compile(r"^\s*(NaN|-?\d+(?:\.\d+)?)\s*(.*)$")


def num(s):
    s = s.strip()
    return np.nan if s in ("", "NaN") else float(s)


def parse_line(line):
    """Nine tab-separated fields; a few lines have extra tabs in the comment or a space instead of a tab before it."""
    f = line.rstrip("\r\n").split("\t")
    if len(f) < 9:
        m = NUM.match(f[-1])
        assert len(f) == 8 and m, line[:120]
        f = f[:-1] + [m.group(1), m.group(2)]
    return f[:8] + [" ".join(x.strip() for x in f[8:] if x.strip())]


def read_logs():
    z = zipfile.ZipFile(RAW / "learning.zip")
    out = {}
    for info in z.infolist():
        m = re.fullmatch(r"learning/(SEQFTD\d+)_(Training|Test)-analysis-log\.txt", info.filename)
        if not m:
            continue
        lines = [l for l in z.read(info).decode("utf-8", "replace").split("\n") if l.strip()]
        out[(m.group(1), m.group(2))] = [parse_line(l) for l in lines]
    return out


def code(sid):
    return f"P{int(sid[6:]):03d}"


def main():
    checks = stdlib.Checks()
    logs = read_logs()
    sids = sorted({k[0] for k in logs})
    assert all((s, "Training") in logs and (s, "Test") in logs for s in sids)

    sub = pd.read_csv(RAW / "subjects.csv").rename(columns={"Subject ID": "sid"}).set_index("sid")
    clin = pd.read_excel(RAW / "clinical.xlsx", sheet_name=0).set_index("SubjID")
    key = pd.read_excel(RAW / "clinical.xlsx", sheet_name="key", header=None)
    keymap = {str(a): str(b) for a, b in zip(key[0], key[1]) if isinstance(a, str) and isinstance(b, str)}
    keymap["BNT"] = "Boston Naming Test, 30 item version from the Uniform Data Set Version 3"

    recs = []
    for sid in sids:
        order = 0
        trained = set()
        for part in ("Training", "Test"):
            for f in logs[(sid, part)]:
                m = NAME.match(f[0])
                assert m and m.group(1) == sid, f[0]
                if part == "Training" and m.group(5).upper() not in ("XXX", "NOBEEP"):
                    trained.add(m.group(5).upper())
        assert len(trained) == 3, (sid, trained)
        for part in ("Training", "Test"):
            for f in logs[(sid, part)]:
                m = NAME.match(f[0])
                lst, label, tn, stim = int(m.group(2)), m.group(3), int(m.group(4)), m.group(5).upper()
                stim = "XXX" if stim == "NOBEEP" else stim
                order += 1
                acc, seq, onset = num(f[1]), num(f[2]), num(f[4])
                ok_seq = seq == 1
                recs.append(dict(
                    sid=sid, list=lst, label=label, auth_trial=tn, stim=stim, trial=order, acc=acc, seq=seq, onset=onset,
                    err=f[3] if f[3] != "NaN" else "", onset_err=f[5] if f[5] != "NaN" else "",
                    clus=num(f[6]) if ok_seq else np.nan, dur=num(f[7]) if ok_seq else np.nan, comment=f[8],
                    cond="" if stim == "XXX" else ("trained" if stim in trained else "novel")))
    a = pd.DataFrame(recs)

    # blocks of 45 trials: day 1 has four (Training1), day 2 two (Training2) and two test blocks
    def block(r):
        if r.label == "Training1A":
            return "Training1A (repeated trials)"
        n = 4 if r.label == "Training1" else 2
        return f"{r.label} block {min((r.auth_trial - 1) // 45 + 1, n)}"
    a["block"] = a.apply(block, axis=1)
    a["session"] = np.where(a.label.str.startswith("Training1"), 1, 2)
    a["phase"] = np.where(a.label == "Test", "test", "learning")

    rows = []
    for r in a.itertuples(index=False):
        s = sub.loc[r.sid]
        age = pd.to_numeric(s.Age, errors="coerce")
        scored = not np.isnan(r.acc)
        rows.append(dict(
            dataset=ID, study=SRC, experiment=f"stimulus list {r.list}", group=s.Dx, subject=code(r.sid), person=code(r.sid),
            subject_excluded_by_authors=bool(s.Dx == "AD"), age=age, sex=s.Sex, handedness="", population="primary progressive aphasia"
            if s.Dx != "AD" else "Alzheimer's disease (PPA protocol cohort)", setting="home", device=DEV,
            session=r.session, block=r.block, trial=r.trial, phase=r.phase, time_s="", task="sequence learning", unit="utterance",
            condition=r.cond, pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=not scored,
            stimulus=r.stim.lower() if r.stim != "XXX" else "", segment="whole pseudoword",
            duration_ms=r.dur, word_correct=bool(r.acc == 1) if scored else "",
            sequencing_correct=bool(r.seq == 1) if scored else "", onset_cluster_correct=bool(r.onset == 1) if scored else "",
            error_type_code=r.err, onset_error_type_code=r.onset_err, cluster_dur_ms=r.clus,
            response_transcription=r.comment, authors_trial_number=r.auth_trial,
            measure_window="" if np.isnan(r.dur) else "first consonant onset to end of vowel (hand-marked)"))

    # ---------------------------------------------------------------- the participant file
    pcols = [("diagnosis", "text", "", "Diagnosis as the authors list it: svPPA semantic, lvPPA logopenic, agPPA agrammatic, PPAOS primary progressive apraxia of speech, "
              "agPPA+AOS both, PPA-O other PPA, AD Alzheimer's disease."),
             ("laterality_score", "number", "-100 to 100", "The authors' handedness laterality score; its sign convention is not stated in the deposit (most values are -100)."),
             ("years_post_onset", "number", "years", "Years since symptom onset."),
             ("education_years", "number", "years", "Years of education."),
             ("scan_available", "true/false", "", "A structural MRI scan exists for the participant (not shared here).")]
    for c in clin.columns:
        base = c.split("_")[0] if c.startswith("EWIL") or c.startswith("PVI") else c
        desc = keymap.get(c) or keymap.get(base) or "Clinical measure (see the authors' key sheet)"
        pcols.append((("clin_" + re.sub(r"[^A-Za-z0-9]+", "_", c)).lower(), "number", "as shared", f"{desc}. Authors' column {c}."))
    plist = []
    for sid in sids:
        s = sub.loc[sid]
        d = dict(person=code(sid), diagnosis=s.Dx, laterality_score=s.Handedness, years_post_onset=s.YearsPostSymptomOnset,
                 education_years=s.Education, scan_available=bool(s.ScanAvail))
        for c in clin.columns:
            d[("clin_" + re.sub(r"[^A-Za-z0-9]+", "_", c)).lower()] = clin.loc[sid, c] if sid in clin.index else np.nan
        plist.append(d)

    # ---------------------------------------------------------------- checks
    per = a.groupby("sid")
    checks.published("participants with learning data (Methods, Participants: 30 adults with PPA recruited)", len(sids), 30, 0, f"{SRC}, Methods, Participants")
    # protocol counts (Experimental Protocol): training blocks of 45 trials, 4 on day 1 and 2 on day 2, 90 test trials
    tr1 = a[a.label.str.startswith("Training1")]
    checks.published("day-1 training trials per participant, median (4 blocks of 45)", tr1.groupby("sid").size().median(), 180, 0, f"{SRC}, Methods, Experimental Protocol")
    checks.published("day-2 training trials per participant, median (2 blocks of 45)", a[a.label == "Training2"].groupby("sid").size().median(), 90, 0,
                     f"{SRC}, Methods, Experimental Protocol")
    checks.published("test trials per participant (2 blocks of 45)", a[a.label == "Test"].groupby("sid").size().median(), 90, 0, f"{SRC}, Methods, Experimental Protocol")
    checks.published("trained pseudowords per participant", np.mean([len({x for x in g.stim[g.phase == "learning"] if x != "XXX"}) for _, g in per]), 3, 0,
                     f"{SRC}, Methods, Experimental Protocol")
    nov = a[(a.phase == "test") & (a.cond == "novel")]
    checks.published("novel pseudowords in the test phase per participant", nov.groupby("sid").stim.nunique().median(), 9, 0, f"{SRC}, Methods, Experimental Protocol")
    non_ad = [x for x in sids if sub.loc[x, "Dx"] != "AD"]
    checks.published("males among the 26 participants of Table 1 (15 in the text, Participants)", sum(sub.loc[x, "Sex"] == "M" for x in non_ad), 15, 0, f"{SRC}, Methods, Participants")
    checks.published("participants in Table 1 (8 + 6 + 9 + 3)", len(non_ad), 26, 0, f"{SRC}, Table 1")
    ag = {"svPPA": ("svPPA",), "nfvPPA": ("agPPA", "PPAOS", "agPPA+AOS"), "PPA-O": ("PPA-O",)}
    for grp, want in (("svPPA", 63.1), ("nfvPPA", 70.1), ("PPA-O", 75.3)):
        ages = [float(sub.loc[x, "Age"]) for x in non_ad if sub.loc[x, "Dx"] in ag[grp]]
        checks.published(f"mean age of the {grp} group (Table 1)", np.mean(ages), want, 0.05, f"{SRC}, Table 1")
    dx = pd.Series([sub.loc[s, "Dx"] for s in sids]).value_counts()
    checks.published("svPPA participants (Table 1: 6)", dx.get("svPPA", 0), 6, 0, f"{SRC}, Table 1")
    checks.published("lvPPA participants (Table 1: 9)", dx.get("lvPPA", 0), 9, 0, f"{SRC}, Table 1")
    checks.published("nfvPPA participants (agPPA, PPAOS, agPPA+AOS; Table 1: 8)", int(dx.reindex(["agPPA", "PPAOS", "agPPA+AOS"]).fillna(0).sum()), 8, 0, f"{SRC}, Table 1")
    checks.published("PPA-Other participants (Table 1: 3)", dx.get("PPA-O", 0), 3, 0, f"{SRC}, Table 1")

    # Table 3 learning measures (seq. errors only; first 15 / last 15 productions of each trained word; first 5 of each test word)
    def rate_dur(g, n, which, drop_nan=False, nan_err=False, dur_mode="pool"):
        er, du, per_stim = [], [], []
        for _, h in g.groupby("stim"):
            if drop_nan:
                h = h[~h.acc.isna()]
            h = h.iloc[:n] if which == "first" else h.iloc[-n:]
            e = (h.seq == 0) | (h.seq.isna() & nan_err)
            sc = h.seq.notna() | nan_err
            er.append(100 * e.sum() / max(sc.sum(), 1))
            d = h.dur[h.seq == 1]
            du.extend(d)
            if len(d):
                per_stim.append(d.mean())
        dd = (np.mean(du) if du else np.nan) if dur_mode == "pool" else (np.mean(per_stim) if per_stim else np.nan)
        return np.mean(er), dd

    def table3(subset, **kw):
        res = {k: [] for k in ("s", "e", "n", "t")}
        for sid in subset:
            g = a[(a.sid == sid)]
            trn = g[(g.phase == "learning") & (g.stim != "XXX")]
            tst = g[g.phase == "test"]
            res["s"].append(rate_dur(trn, 15, "first", **kw))
            res["e"].append(rate_dur(trn, 15, "last", **kw))
            res["n"].append(rate_dur(tst[tst.cond == "novel"], 5, "first", **kw))
            res["t"].append(rate_dur(tst[tst.cond == "trained"], 5, "first", **kw))
        return {k: (np.nanmean([x[0] for x in v]), np.nanmean([x[1] for x in v])) for k, v in res.items()}
    t26 = table3(non_ad)
    T3 = dict(s=(61.5, 447.3), e=(50.2, 396.8), n=(52.1, 418.8), t=(48.1, 379.6))
    names = dict(s="start of training", e="end of training", n="novel words in the test", t="trained words in the test")
    diffs = []
    for k in T3:
        for i, (lab, unit) in enumerate((("error rate", "%"), ("duration of accurate productions", "ms"))):
            what = f"{lab}, {names[k]}, {unit} (Table 3, full cohort, the 26 participants of Table 1)"
            if abs(t26[k][i] - T3[k][i]) <= 0.05:
                checks.published(what, t26[k][i], T3[k][i], 0.05, f"{SRC}, Table 3")
            else:
                diffs.append(f"Table 3, {lab}, {names[k]}: {t26[k][i]:.2f} {unit} here against {T3[k][i]} {unit} in the paper "
                             "(the learning measures are rebuilt from the rows by the paper's description and come close but not within its rounding).")
    print("\n".join(diffs))
    checks.internal("every participant has 270 or 274 training lines and 90 test lines",
                    bool(a[a.phase == "learning"].groupby("sid").size().isin([270, 274]).all() and (a[a.phase == "test"].groupby("sid").size() == 90).all()))
    checks.internal("overall accuracy implies sequencing accuracy", bool(((a.acc != 1) | (a.seq == 1)).all()))
    checks.internal("an accurately sequenced utterance has a measured duration (all but at most 20 of about 5,000 have one; the rest have a missing field)",
                    int(a.dur[a.seq == 1].isna().sum()) <= 20, f"{int(a.dur[a.seq == 1].isna().sum())} without")
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), c.get("computed", ""), c.get("reported", ""))

    s0 = json.load(open(stdlib.HERE / "source/extra_studies.json"))
    s = next(x for x in s0 if x["id"] == ID)
    p = dict(s["papers"][0])
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Joined the 60 per-participant log files into one table (training log, then test log), split the nine log columns, turned the "
                 "0/1 accuracy columns into true/false, left the duration empty where the authors did not measure it, recoded participant "
                 "numbers as codes, and added age, sex, diagnosis and clinical scores from the deposit's participant tables."),
        how=["One row is one utterance in the non-native cluster learning protocol: the participant heard a pseudoword (CCVC, with an "
             "initial consonant cluster that English does not allow) while reading it, and repeated it. 30 people with PPA or a related diagnosis "
             "took part in two sessions on consecutive days: 180 training trials on day 1 (session 1), 90 training trials on day 2 and a 90-trial "
             "test (session 2) of the 3 trained words mixed with 9 novel words. block follows the authors' file labels in blocks of 45 trials; "
             "trial counts the rows of the participant in file order (training log first). Some files hold a few extra lines (Training1A) "
             "where trials were repeated.",
             "word_correct is the authors' overall accuracy (epenthesis counts as an error); sequencing_correct is their sequencing accuracy "
             "(epenthesis ignored), which the paper uses; onset_cluster_correct is their accuracy over the initial cluster only. "
             "error_type_code and onset_error_type_code are their numeric error codes (see columns.csv), response_transcription is their "
             "comment (what the participant said when it was wrong, or a note such as 'no response'). condition is trained or novel: a word is "
             "trained if it was practised in that participant's training logs.",
             "duration_ms is the hand-marked duration from the first consonant to the end of the vowel, cluster_dur_ms is the duration of the "
             "cluster alone. The authors measured them only for accurately sequenced utterances; on all other lines their columns hold a number "
             "that is about the length of the recording (4 to 11 s), which is not a measured duration, so those cells are left empty here. "
             "excluded_by_authors is true for lines with no score (no response, skipped or unusable trial).",
             "group is the diagnosis (svPPA, lvPPA, agPPA, PPAOS, agPPA+AOS, PPA-O, AD). participants.csv gives diagnosis, laterality score, "
             "years since onset, education and the clinical measures of the authors' clinical table."],
        unit_note="utterance", differences=diffs + [
            "The paper analysed 26 participants (Table 1) of 30 who were enrolled, but the deposit does not say which four were left out. The four with "
            "a diagnosis of Alzheimer's disease (AD) are the only ones that Table 1 (diagnosis groups 8 + 6 + 9 + 3) and the text (15 males) do not "
            "count, and without them Table 3 is reproduced closely, so subject_excluded_by_authors is true for them; this is inferred, not stated.",
            "Age is empty for the participant that the authors give only as 90+ (the paper's mean age uses the true age)."])
    stdlib.release(ID, "speech", rows, meta, checks, participants=plist, participant_columns=pcols)


main()
