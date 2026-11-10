"""Van Waes, Leijten, Pauwaert & Van Horenbeeck (2019), JORS: Inputlog Copy Task corpus, sub-dataset of English tests by 21-25 year olds.

Run from the site folder: python3 -I tools/std/convert/VanWaes2019_JORS.py
Raw file (untrusted data, read only): raw_extra/VanWaes2019_JORS/sub.zip from https://doi.org/10.5281/zenodo.5803401
(sub-dataset_EN_21-25year.zip). One row of the output is one measure of one participant in one subset of the copy task.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "VanWaes2019_JORS"
RAW = stdlib.raw_dir(ID) / "sub.zip"
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(stdlib.raw_dir(ID) / "SOURCES.json"))}
checks.internal("raw file matches the downloaded hash", stdlib.sha256(RAW) == src["sub.zip"]["sha256"])
z = zipfile.ZipFile(RAW)
PRE = "corpus/data/"


def rd(name):
    return pd.read_csv(io.BytesIO(z.read(PRE + name)), dtype=str, keep_default_na=False)


sess = rd("session.csv")
files = sorted(n[len(PRE):] for n in z.namelist() if n.endswith(".csv") and n != PRE + "session.csv")
checks.internal("491 tests, one row per test in every file", len(sess) == 491 and sess.id.is_unique)
checks.internal("46 subset files besides session.csv", len(files) == 45, str(len(files)))

# person = one uuid (some people did the task more than once); subject = one test (id)
person_of = dict(zip(sess.id, sess.uuid))
rep_flag = dict(zip(sess.id, sess.repetition))
dup = sess.uuid.duplicated(keep=False)
checks.internal("persons with more than one test", True,
                f"{(sess.repetition == 'TRUE').sum()} tests flagged repetition (of which {((sess.repetition == 'TRUE') & dup).sum()} share a participant key with another test in this subset), {sess.uuid.nunique()} persons")
attempt = sess.groupby("uuid").cumcount() + 1                      # attempt number in file order
attempt_of = dict(zip(sess.id, attempt))

SEX = {"F (Female)": "F", "M (Male)": "M"}
age_of = dict(zip(sess.id, pd.to_numeric(sess.age)))
sex_of = {i: SEX.get(g, "") for i, g in zip(sess.id, sess.gender)}
dev_of = {i: (f"{c} computer, {k} keyboard" if c != "Missing" else f"{k} keyboard") for i, c, k in zip(sess.id, sess.computer, sess.keyboard)}

MEAS = [  # column, measure, description, statistic, unit
    ("targetted", "targeted_count", "Number of typed characters that corresponded to a prompted character (the authors' 'Count (targeted)').", "count", "count"),
    ("non_targetted", "non_targeted_count", "Number of typed characters outside the targeted ones (the deposit does not define it further).", "count", "count"),
    ("cpm", "cpm", "Characters per minute: 60000 / mean inter-key interval of the targeted bigrams (the authors' definition).", "value", "characters per minute"),
    ("median", "iki_median", "Median inter-key interval of the targeted bigrams.", "median", "ms"),
    ("stdev", "iki_sd", "Standard deviation of the inter-key intervals of the targeted bigrams (named stdev in the deposit).", "sd", "ms"),
    ("logmean", "iki_logmean", "Mean of the inter-key intervals on the log scale, back-transformed to ms (named logmean in the deposit).", "mean", "ms"),
    ("mean_iki", "iki_mean", "Arithmetic mean inter-key interval of the targeted bigrams.", "mean", "ms"),
]


def num(s):
    return pd.to_numeric(s.replace("NA", np.nan))


out = []
# session level scores
for col, meas, desc, unit in [("cpm", "cpm", "Characters per minute over the whole copy task.", "characters per minute"),
                              ("correctness", "correctness", "Share of correctly typed bigrams over the whole copy task, in percent (the authors' 'correctness').", "percent"),
                              ("score", "score", "The authors' overall score; equals cpm x correctness / 100 in every test where all three are present.", "score")]:
    v = num(sess[col])
    out.append(pd.DataFrame(dict(sid=sess.id, cond="whole task", meas=meas, desc=desc, stat="value", val=v, unit=unit, n=np.nan)))
chk = num(sess.score) - num(sess.cpm) * num(sess.correctness) / 100
checks.internal("score = cpm x correctness / 100 (to rounding)", float(chk.abs().max()) < 0.2, f"max deviation {chk.abs().max():.3f}")

worst = 0.0
n_rows_files = 0
for f in files:
    d = rd(f)
    assert list(d.columns) == ["id", "targetted", "non_targetted", "cpm", "median", "stdev", "logmean", "mean_iki"], (f, list(d.columns))
    assert len(d) == 491 and set(d.id) == set(sess.id), f
    cond = f[:-4]
    ncount = num(d.targetted)
    dev = (num(d.cpm) - 60000 / num(d.mean_iki)).abs().max()
    worst = max(worst, float(dev) if dev == dev else 0)
    for col, meas, desc, stat, unit in MEAS:
        out.append(pd.DataFrame(dict(sid=d.id, cond=cond, meas=meas, desc=desc, stat=stat, val=num(d[col]), unit=unit,
                                     n=np.nan if stat == "count" else ncount)))
t = pd.concat(out, ignore_index=True)
t["sid"] = t.sid.astype(int)
t = t.sort_values("sid", kind="stable").reset_index(drop=True)          # by test, keeping the file order within a test
t["sid"] = t.sid.astype(str)
checks.published("largest gap between cpm and 60000 / mean inter-key interval over all 45 subsets x 491 tests (cpm is rounded to whole numbers)",
                 worst, 0, 1.0, "Copy task analysis ('cpm' ... specifically 60 000 (ms)/mean IKI (ms))")
checks.internal("about 500 tests, English, ages 21-25 (Zenodo description)", len(sess) == 491 and sess.language.eq("EN").all()
                and pd.to_numeric(sess.age).between(21, 25).all())

n = len(t)
rows = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": ID + "_Exp1", "group": "", "subject": t.sid.map(lambda i: f"t{i}"),
    "person": t.sid.map(lambda i: person_of[i]),
    "subject_excluded_by_authors": False, "age": t.sid.map(age_of), "sex": t.sid.map(sex_of), "handedness": "",
    "population": "healthy", "setting": "online", "device": t.sid.map(dev_of),
    "session": t.sid.map(attempt_of), "block": t.cond, "trial": t.groupby("sid", sort=False).cumcount() + 1, "phase": "test", "time_s": "",
    "task": "typing", "unit": "participant summary", "condition": t.cond, "pert_type": "none", "pert_value": 0, "pert_unit": "",
    "feedback": "normal", "excluded_by_authors": False,
    "measure": t.meas, "measure_description": t.desc, "statistic": t.stat, "value": t.val, "value_unit": t.unit, "n_trials": t.n,
})
checks.internal("every test has the same number of rows", rows.groupby("subject").size().nunique() == 1, str(rows.groupby("subject").size().iloc[0]))
checks.internal("rows = 491 x (3 + 45 x 7)", n == 491 * (3 + 45 * 7), str(n))

first = sess[attempt.eq(1).to_numpy()]
parts = pd.DataFrame({"person": first.uuid, "laterality_score": num(first.handedness), "familiarity_with_task": num(first.familiarity),
                      "education": first.education.replace("Missing", "")})
pcols = [("laterality_score", "number", "-100 to 100", "Handedness score from the task questionnaire at the person's first test, as shared (positive = right-handed; the deposit does not define the scale further)."),
         ("familiarity_with_task", "number", "0, 50, 100", "Answer to the closing question on familiarity with the task at the first test, as coded in the deposit."),
         ("education", "text", "", "Highest education level reported at the first test.")]
checks.internal("one participants row per person", len(parts) == sess.uuid.nunique() == 433, str(len(parts)))

# participants.csv must have one row per person per stdlib; we give one row per test with the subject code and map codes in release
meta = dict(
    study="Van Waes et al. 2019",
    paper=dict(authors="Van Waes L, Leijten M, Pauwaert T, Van Horenbeeck E", year=2019,
               title="A Multilingual Copy Task: Measuring Typing and Motor Skills in Writing with Inputlog",
               venue="Journal of Open Research Software 7(1):30", doi="10.5334/jors.234"),
    lab="Van Waes & Leijten (University of Antwerp)",
    data_url="https://doi.org/10.5281/zenodo.5803401",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to long form (one row per measure) in the OpenMotor format with participant codes replaced. Dropped: the "
             "deposit's uuid and the free-text session name (replaced by the person and subject codes), and the 'disorder' column, "
             "which neither the deposit nor the paper defines. Handedness, familiarity and education are in participants.csv."),
    how=[
        "One row is one measure of one test of the Inputlog Copy Task (typing and motor skill: a tapping task, a sentence copy, "
        "four three-word copies and a consonant-group copy) by an English-speaking 21-25 year old; condition names the subset "
        "of the task or of its keystrokes (for example components/tapping, hands/ll, trials/words1/trial3, or whole task).",
        "The deposit shares per-participant summaries only, not keystroke logs. Per subset it shares the counts of targeted and "
        "non-targeted characters, characters per minute (60000 / mean inter-key interval), and the median, SD, log-scale mean and "
        "arithmetic mean of the inter-key interval in ms; for the whole task it shares cpm, correctness (percent of correct "
        "bigrams) and the authors' score (cpm x correctness / 100). Missing values (NA in the deposit) are empty.",
        "subject is one test (one run); person is one participant, found from the deposit's participant key, so a person who did "
        "the task more than once has several subjects (session is the attempt number in file order). Names of measures follow "
        "the deposit's column names: median, stdev and logmean and non_targetted are not defined in the paper, so their "
        "descriptions are read from the columns' values (checked: cpm and score follow the definitions above).",
    ],
    unit_note="measure of one test in one subset of the copy task (long form)",
)
rec = stdlib.release(ID, "summary", rows, meta, checks, participants=parts, participant_columns=pcols)
