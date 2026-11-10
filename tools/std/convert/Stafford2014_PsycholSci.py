"""Stafford & Dewar (2014), Psychol Sci: scores of the 854,064 players of the online game Axon, one row per play.

Run from the site folder: python3 -I tools/std/convert/Stafford2014_PsycholSci.py
Raw files (untrusted data, read only) in raw_extra/Stafford2014_PsycholSci/, from https://github.com/tomstafford/axongame:
  data_by_cookie.json   one object per browser cookie (machine identity): attempt number '00001'... -> [score, YYYYMMDD, hour]
  LICENSE               the repository's MIT licence, copied into the ZIP
The authors' Python 2 scripts (ps_fig*.py, sup_*.py) were read as text, never run; the published checks redo their
deterministic analyses (player counts, resters vs goers, explore/exploit) on the converted rows.
"""
import bisect
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import mstats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Stafford2014_PsycholSci"
RAW = stdlib.raw_dir(ID)
FINAL = "final submitted version in the deposit (stafford_and_dewar_revision.pdf)"
checks = stdlib.Checks()

src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for fn in ("data_by_cookie.json", "LICENSE"):
    checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])
data = json.load(open(RAW / "data_by_cookie.json"))

# ---------------------------------------------------------------- compact columns of the converted rows
n_players = len(data)
n_rows = sum(len(v) for v in data.values())
subj = np.empty(n_rows, np.int32)
att = np.empty(n_rows, np.int32)
score = np.empty(n_rows, np.int64)
date = np.empty(n_rows, np.int32)
hour = np.empty(n_rows, np.int16)
i = 0
ok_vals = ok_keys = True
for p, v in enumerate(data.values()):
    items = sorted(((int(k), val) for k, val in v.items()), key=lambda x: x[0])  # attempts in the order of their number
    ok_keys &= len({k for k, _ in items}) == len(items) and all(len(k) == 5 and k.isdigit() for k in v)
    for a, val in items:
        ok_vals &= isinstance(val, list) and len(val) == 3 and all(isinstance(x, int) for x in val)
        subj[i], att[i], score[i], date[i], hour[i] = p, a, val[0], val[1], val[2]
        i += 1
del data
checks.internal("every play kept: one row per (cookie, attempt), 3,099,705 rows", i == n_rows == 3099705, f"{n_rows} rows")
checks.internal("attempt keys are 5-digit numbers, unique within a cookie; every value is [score, YYYYMMDD, hour] integers",
                ok_keys and ok_vals)
counts = np.bincount(subj, minlength=n_players)
checks.internal("every cookie has at least one play", bool((counts >= 1).all()))
dates = sorted(set(date.tolist()))
checks.internal("ranges: scores positive, hours 0-23, 60 distinct dates", bool(score.min() > 0 and hour.min() >= 0
                and hour.max() <= 23) and len(dates) == 60, f"scores {score.min()}-{score.max()}")
first_attempt = att[np.r_[0, np.flatnonzero(np.diff(subj)) + 1]]
gaps = int(sum(1 for s, c in zip(np.split(att, np.flatnonzero(np.diff(subj)) + 1), counts)
               if not (s[0] == 1 and s[-1] == c)))
checks.internal("attempt numbers: most cookies have 1..n, some lack attempts (as the paper says: 'first play (or second "
                "play where this data was missing)')", gaps < 0.03 * n_players,
                f"{gaps} cookies whose attempts are not exactly 1..n; {int((first_attempt != 1).sum())} lack attempt 1")

# ---------------------------------------------------------------- published numbers
checks.published("players (browser cookies)", n_players, 854064, 0, f"Abstract and Method ({FINAL})")
checks.published("players with more than 9 plays", int((counts > 9).sum()), 45672, 0, f"Results, Practice amount ({FINAL})")
checks.published("players with more than 14 plays", int((counts > 14).sum()), 21575, 0,
                 f"Results, Practice spacing ({FINAL})")
checks.published("first date of play (YYYYMMDD)", dates[0], 20120314, 0, f"Method, '14th of March' 2012 ({FINAL})")
checks.published("last date of play (YYYYMMDD)", dates[-1], 20120513, 0, f"Method, '13th of May 2012' ({FINAL})")

starts = np.r_[0, np.cumsum(counts)[:-1]]


def plays(p):
    s = slice(starts[p], starts[p] + counts[p])
    return dict(zip(att[s].tolist(), zip(score[s].tolist(), date[s].tolist(), hour[s].tolist())))


def when(v):
    return dt.datetime.combine(dt.datetime.strptime(str(v[1]), "%Y%m%d"), dt.time(v[2]))


# resters vs goers (sup_resters_v_goers.py): players with more than 14 plays that have attempts 1, 10 and 11-15;
# spread = time from attempt 1 to attempt 10; outcome = best score on attempts 11-15; rester if spread > 24 h
rest, go = [], []
for p in np.flatnonzero(counts > 14):
    v = plays(p)
    if 1 not in v or 10 not in v or any(a not in v for a in range(11, 16)):
        continue
    best = max(v[a][0] for a in range(11, 16))
    (rest if when(v[10]) - when(v[1]) > dt.timedelta(hours=24) else go).append(best)
rest, go = np.array(rest, float), np.array(go, float)
t, _ = stats.ttest_ind(rest, go)
SRC_RG = f"Results, Practice spacing ({FINAL})"
checks.published("resters vs goers: degrees of freedom of the t test", len(rest) + len(go) - 2, 20354, 0, SRC_RG)
checks.published("resters vs goers: t", t, 6.219, 0.0005, SRC_RG)
checks.published("goers: mean of best score on plays 11-15", go.mean(), 44050, 0.5, SRC_RG)
checks.published("resters: mean of best score on plays 11-15", rest.mean(), 47264, 0.5, SRC_RG)
checks.published("goers: SD", go.std(), 26882, 0.5, SRC_RG)
checks.published("resters: SD", rest.std(), 29461, 0.5, SRC_RG)
checks.published("Cohen's d (pooled as in the authors' script)", (rest.mean() - go.mean())
                 / math.sqrt(0.5 * (rest.std() ** 2 + go.std() ** 2)), 0.11, 0.005, SRC_RG)

# explore/exploit (sup_ee_observed.py + sup_make_eeheatmap.py): players with more than 9 plays; variance of the scores of
# plays 1-5 and mean of plays 6-10, each turned into percentile ranks (mstats.scoreatpercentile at 0..99, bisect)
var1, av2, var2, av1 = [], [], [], []
for p in np.flatnonzero(counts > 9):
    v = plays(p)
    f = [v[a][0] for a in range(1, 6) if a in v]
    s = [v[a][0] for a in range(6, 11) if a in v]
    var1.append(np.var(f)), av1.append(np.mean(f)), var2.append(np.var(s)), av2.append(np.mean(s))


def pct_rank(x):
    cut = [float(mstats.scoreatpercentile(np.array(x), q)) for q in range(100)]
    return [bisect.bisect(cut, a) for a in x]


r_text = stats.pearsonr(pct_rank(var1), pct_rank(av2))[0]
r_script = stats.pearsonr(pct_rank(var2), pct_rank(av1))[0]
checks.internal("explore/exploit correlation with the pairing the paper's text describes (variance of plays 1-5 with "
                "mean of plays 6-10)", True, f"r = {r_text:.3f}")
checks.published("explore/exploit: Pearson r of percentile ranks, paired as in the deposited sup_ee_observed.py "
                 "(score variance on plays 6-10, mean score on plays 1-5)", r_script, 0.59, 0.005,
                 f"Results, Exploration versus exploitation ({FINAL})")
mode = int(np.bincount(counts).argmax())
checks.internal("most common number of plays per cookie", True, f"{mode} plays ({int((counts == mode).sum())} cookies; "
                f"{int((counts == 1).sum())} with 1 play); most plays by one cookie {int(counts.max())}")

for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']}: " + (f"{c['computed']:.6g} vs {c['reported']}" if
                                                                   c["kind"] == "published" else c.get("detail", "")))

# ---------------------------------------------------------------- rows
DATE = {d: f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}" for d in dates}
UNIT = "points (the game's score: total branch length of the player's neuron, in micrometres)"
BASE = dict(dataset=ID, study=ID, experiment=f"{ID}_Main", group="", subject_excluded_by_authors=False, age="", sex="",
            handedness="", population="healthy", setting="online",
            device="own computer and web browser (mouse clicks on targets)", session="", block="", phase="learning",
            time_s="", task="sports and games", unit="trial", condition="", pert_type="none", pert_value=0, pert_unit="",
            feedback="normal", excluded_by_authors=False, game="Axon", score_unit=UNIT)


class Rows:
    def __iter__(self):
        trial = 0
        prev = -1
        for s, a, sc, d, h in zip(subj.tolist(), att.tolist(), score.tolist(), date.tolist(), hour.tolist()):
            trial = trial + 1 if s == prev else 1
            prev = s
            yield {**BASE, "subject": f"c{s}", "trial": trial, "attempt": a, "score": sc, "date": DATE[d], "hour": h}


paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Stafford (University of Sheffield)",
    data_url="https://github.com/tomstafford/axongame",
    license="MIT", license_url="https://github.com/tomstafford/axongame/blob/master/LICENSE",
    license_text=(RAW / "LICENSE").read_text(encoding="utf-8"),
    changes=("Converted to the OpenMotor format: browser cookie identifiers replaced by codes, each play unpacked into a "
             "row, plays sorted by attempt number within each cookie, and dates written as YYYY-MM-DD."),
    how=[
        "One row is one play of the online game Axon (Wellcome Trust, 2012), in which players grow a neuron by rapid mouse "
        "clicks on targets; score is the game's score for that play (total branch length in micrometres), date and hour "
        "are when it was played (hour resolution, as tracked by the game through Google Analytics), and attempt is the "
        "authors' play number.",
        "subject and person are one browser cookie (the paper's 'machine identity'): the authors note that one cookie can "
        "hold plays by more than one person and one person can have several cookies.",
        "The JSON stores a cookie's plays in no particular order, so rows are sorted by attempt number within each cookie "
        "(cookies keep the file's order); some cookies lack some attempt numbers, which are kept as gaps in attempt.",
        "Every play in the deposit is kept; the authors analysed subsets (for example cookies with more than 9 plays), "
        "so no row or player is flagged as excluded.",
    ],
    unit_note="play of the game (one attempt by one browser cookie)",
    differences=[
        f"The paper says the modal number of plays is 1; in the deposit it is {mode} ({int((counts == mode).sum())} "
        f"cookies against {int((counts == 1).sum())} with one play).",
        f"The paper says some played up to 1000 times; the most plays by one cookie is {int(counts.max())}.",
        f"The paper's explore/exploit r = .59 is reproduced ({r_script:.3f}) with the pairing of the deposited script "
        f"(variance of plays 6-10 with mean of plays 1-5); the pairing the paper's text describes and the bootstrap script "
        f"uses (variance of plays 1-5 with mean of plays 6-10) gives r = {r_text:.3f}.",
    ],
)
stdlib.release(ID, "game", Rows(), meta, checks)
