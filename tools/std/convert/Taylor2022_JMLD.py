"""Taylor, Fawver, Thomas, Williams & Lohse (2022), J Mot Learn Dev: time-estimation (hold a mouse button for a target
duration), blocked vs random practice, 84 participants: 5 pre-test trials, 210 practice trials, 40 retention/transfer trials.

Run from the site folder: python3 -I tools/std/convert/Taylor2022_JMLD.py
Raw file (untrusted data, read only): raw_extra/Taylor2022_JMLD/repo.zip, the Zenodo copy (10.5281/zenodo.7674907) of the
authors' GitHub repository; only Data/data_PRE_MASTER.csv, data_ACQ_MASTER.csv and data_POST_MASTER.csv are used (the other
files are the authors' derived tables and R script, read as text only).
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Taylor2022_JMLD"
P = "keithlohse-taylor_2022_CI_sequential_effects-6ef9fec/Data/"
z = zipfile.ZipFile(stdlib.raw_dir(ID) / "repo.zip")
rd = lambda n: pd.read_csv(io.BytesIO(z.read(P + n)))
pre, acq, post = rd("data_PRE_MASTER.csv"), rd("data_ACQ_MASTER.csv"), rd("data_POST_MASTER.csv")
ck = stdlib.Checks()


def grp(g, e):
    est = "error estimation" if str(e).strip().lower().startswith("estimate") else "no error estimation"
    return f"{g.strip()} schedule, {est}"


def common(d, session, phase, block, ph_device):
    return pd.DataFrame({
        "dataset": ID, "study": ID, "experiment": ID, "group": [grp(g, e) for g, e in zip(d.group, d.estimation)],
        "subject": d.participant.astype(str), "subject_excluded_by_authors": False, "age": "", "sex": "",
        "handedness": "", "population": "healthy", "setting": "lab", "device": "mouse button held with the index finger of the dominant hand",
        "session": session, "block": block, "phase": phase, "time_s": "", "task": "finger tapping and timing", "unit": "trial",
        "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": ph_device,
    })


def finish(df, d):
    df["stimulus"] = [f"{t} ms" for t in d.target]
    df["target_interval_ms"] = d.target.to_numpy()
    return df


# --- pre-test (day 1, before practice): the deposit gives constant error only
a = common(pre, 1, "baseline", pre.session.to_numpy(), "none")
a["condition"] = "pre-test"; a["excluded_by_authors"] = False
a["tapping_trial"] = pre.trial.to_numpy()
a["constant_error_ms"] = pre.constant_error.to_numpy()
a = finish(a, pre); a["_p"] = pre.participant.to_numpy(); a["_o"] = 0

# --- practice
ab = acq.absolute_error.to_numpy()
b = common(acq, 1, "learning", ["Set " + str(x) for x in acq.block], "altered")  # signed error feedback after every trial
b["condition"] = "practice"; b["excluded_by_authors"] = ab >= 1000
b["tapping_trial"] = acq.trial_total.to_numpy()
b["hold_ms"] = acq.hold_time.to_numpy(); b["constant_error_ms"] = acq.constant_error.to_numpy()
b["absolute_error_ms"] = ab
b["estimated_error_ms"] = acq.estimated_error.to_numpy()
b["estimate_minus_actual_ms"] = acq.error_estimate_CE.to_numpy(); b["estimate_abs_error_ms"] = acq.error_estimate_AE.to_numpy()
b = finish(b, acq); b["_p"] = acq.participant.to_numpy(); b["_o"] = 1

# --- retention / transfer (day 2, no feedback)
pb = post.absolute_error.to_numpy()
c = common(post, 2, "test", ["retention/transfer test"] * len(post), "none")
c["phase"] = np.where(post.test_type == "Transfer", "transfer", "test")
c["condition"] = post.test_type.str.lower().to_numpy(); c["excluded_by_authors"] = pb >= 1000
c["tapping_trial"] = post.trial.to_numpy()
c["hold_ms"] = post.hold_time.to_numpy(); c["constant_error_ms"] = post.constant_error.to_numpy(); c["absolute_error_ms"] = pb
c["pre_trial_time_ms"] = post.pre_trial_time.to_numpy(); c["response_time_ms"] = post.response_time.to_numpy()
c = finish(c, post); c["_p"] = post.participant.to_numpy(); c["_o"] = 2

t = pd.concat([a, b, c], ignore_index=True)
t["_i"] = np.arange(len(t))
t = t.sort_values(["_p", "_o", "_i"], kind="stable").reset_index(drop=True)   # participants in id order; pre, practice, tests
t["trial"] = t.groupby("_p").cumcount() + 1
t["subject"] = t["subject"].astype(str)
# subject sequences: first appearance order is by participant id, as in the authors' files
# --- checks
ck.internal("84 participants in each of the three files", pre.participant.nunique() == acq.participant.nunique() == post.participant.nunique() == 84)
ck.internal("constant error = hold time - target on every practice and test row",
            bool((acq.constant_error == acq.hold_time - acq.target).all() and (post.constant_error == post.hold_time - post.target).all()))
ck.internal("absolute error = |constant error|", bool((acq.absolute_error == acq.constant_error.abs()).all() and (post.absolute_error == post.constant_error.abs()).all()))
ck.internal("rows kept: 420 + 17640 + 3360", len(t) == 420 + 17640 + 3360)
ck.internal("pre-test, practice and test files hold the same 84 people", set(pre.participant) == set(acq.participant) == set(post.participant))
per = acq.groupby("participant").size()
ck.published("practice trials per participant (3 sets x 70)", per.iloc[0], 210, 0, "Task and Stimuli: 210 practice trials")
ck.internal("every participant has 210 practice trials", bool((per == 210).all()))
posts = post.groupby("participant").size()
ck.published("retention and transfer test trials per participant", posts.iloc[0], 40, 0, "Task and Stimuli: 40 test trials (5 targets x 4 trials x 2 sets)")
ck.internal("every participant has 40 test trials", bool((posts == 40).all()))
g = acq.drop_duplicates("participant").group.value_counts()
ck.published("participants with a blocked schedule", g["Blocked"], 41, 0, "Participants: n = 41 blocked")
ck.published("participants with a random schedule", g["Random"], 43, 0, "Participants: n = 43 random")
ck.published("participants in total", acq.participant.nunique(), 84, 0, "Abstract / Participants: N = 84")
ok = acq[acq.absolute_error < 1000]
ck.published("share of practice trials within +/-50 ms (no KR), after outlier exclusion (%)", 100 * (ok.absolute_error <= 50).mean(), 20.8, 0.05, "Results, Changes following KR versus No KR (20.8%)")
ck.published("same, blocked schedule (%)", 100 * (ok[ok.group == "Blocked"].absolute_error <= 50).mean(), 24, 0.5, "Results (24% blocked)")
ck.published("same, random schedule (%)", 100 * (ok[ok.group == "Random"].absolute_error <= 50).mean(), 18, 0.5, "Results (18% random)")
ck.internal("targets in practice are 1500, 1700, 1900 ms; tests also 1600, 1800 ms", set(acq.target) == {1500, 1700, 1900} and set(post.target) == {1500, 1600, 1700, 1800, 1900})

out = t.drop(columns=["_p", "_o", "_i"])
names = [c[1] for c in stdlib.columns_for("tapping")]
for c in names:
    if c not in out.columns and c != "person":
        out[c] = np.nan if c.endswith("_ms") else ""
out = out[[c for c in names if c != "person"]]
for ch in ck.items:
    if ch["kind"] == "published": print(f"  {ch['what']}: {ch['computed']:.4f} vs {ch['reported']} ok={ch['ok']}")
    elif not ch["ok"]: print("  FAILED", ch["what"])

meta = dict(
    paper=dict(authors="Taylor S, Fawver B, Thomas JL, Williams AM, Lohse KR", year=2022,
               title="Practice Schedules Affect How Learners Correct Their Errors: Secondary Analysis From a Contextual Interference Study",
               venue="Journal of Motor Learning and Development 10(3):388-411", doi="10.1123/jmld.2022-0021"),
    lab="Lohse (Washington University in St. Louis)", data_url="https://doi.org/10.5281/zenodo.7674907",
    license="Other (Open), as stated on the Zenodo record (id other-open); the GitHub repository carries no license file",
    license_url="https://zenodo.org/records/7674907",
    changes=("Three authors' tables (pre-test, practice, retention/transfer) were stacked into one table in the order pre-test, "
             "practice, test for each participant; participant codes replaced the ids; the authors' derived tables "
             "(determinants, summaries) and R script were not copied; pre_feedback_time (0 on every row) was dropped."),
    how=["One row is one trial of a time-estimation task: the participant held a mouse button down with the index finger of the dominant hand for the "
         "target duration shown on the screen (1500, 1700 or 1900 ms in practice; also 1600 and 1800 ms in the transfer test). "
         "hold_ms is the produced duration, constant_error_ms is hold minus target (positive: too long), all in ms.",
         "Order per participant: 5 pre-test trials (constant error only; the authors label 5 of the 420 rows 'Training' instead of 'Pre-Testing', kept as given in block), "
         "210 practice trials in three sets of 70 (blocked groups practised one target per set, random groups interleaved the targets; signed error feedback "
         "after every trial, shown as 0 within +/-50 ms) and, about 24 hours later, 40 retention (1500, 1700, 1900 ms) and transfer (1600, 1800 ms) trials "
         "without feedback. group gives the practice schedule and whether the participant also estimated their own error after some practice trials.",
         "excluded_by_authors marks trials with absolute error of 1000 ms or more, which the authors' script removes before analysis."],
    unit_note="trial of the time-estimation task",
)
stdlib.release(ID, "tapping", out.to_dict("records"), meta, ck)
