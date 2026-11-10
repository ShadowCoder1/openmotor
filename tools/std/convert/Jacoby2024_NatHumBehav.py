"""Jacoby et al. (2024), Nat Hum Behav: iterated tapping reproduction of three-interval rhythms in 39 groups in 15 countries.

Run from the site folder: python3 -I tools/std/convert/Jacoby2024_NatHumBehav.py
Raw files (untrusted data, read only) in raw_extra/Jacoby2024_NatHumBehav/, from https://osf.io/6zd4v/ (zips read in place):
  03_main.zip   '03 Exported-data-main-experiment-2000': RAW/export.<group>.csv (one row per repetition of the 3-interval pattern),
                demography/all_participants_summmary_2000.xls and group_summmary_2000.xls
  04_fast.zip   '04 Exported-data-fast-tempo-experiment-1000': the same for the fast-tempo experiment (patterns sum to 1000 ms)
Columns of the RAW files (README.txt of the exports has no column list; meanings from the paper's Methods): participant, trial
(seed rhythm), iteration (1-5), cycle (repetition 1-10), seed1-3 (stimulus intervals), response1-3 (the repetition's response
intervals between consecutive taps), mean_response1-3 (their average over the iteration's repetitions), onset_stim1-4 and
onset_resp1-4 (click and tap onsets within the repetition, ms from its first click; the fourth is the next repetition's first
click), triangle_* (the same intervals in the plotting space of the rhythm triangle: not converted).
"""
import csv
import io
import json
import math
import statistics as st
import sys
import zipfile
from pathlib import Path

import warnings

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Jacoby2024_NatHumBehav"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}

EXPS = [
    ("main", "03_main.zip", "03 Exported-data-main-experiment-2000/", "2000", "main experiment (rhythms of 2000 ms)"),
    ("fast", "04_fast.zip", "04 Exported-data-fast-tempo-experiment-1000/", "1000", "fast tempo experiment (rhythms of 1000 ms)"),
]
TYPE = {"ST": "students", "NM": "non-musicians", "WM": "Western musicians", "LM": "local musicians", "DA": "dancers",
        "OL": "online participants"}
LAB_DEV = "tapping sensor, USB sound card, headphones (lab or field station)"
ONLINE_DEV = "laptop speakers and built-in microphone, taps on the laptop (REPP)"


def num(s):
    s = s.strip()
    return None if s in ("", "NaN", "nan") else float(s)


def g(x):
    return None if x is None else float(format(x, ".8g"))


for key, zn, pre, tag, label in EXPS:
    checks.internal(f"{zn} matches the downloaded hash", stdlib.sha256(RAW / zn) == src[zn]["sha256"])

# ---------------------------------------------------------------- demography (authors' per-participant summaries)
demo, groups = {}, {}
for key, zn, pre, tag, label in EXPS:
    z = zipfile.ZipFile(RAW / zn)
    p = pd.read_excel(io.BytesIO(z.read(f"{pre}demography/all_participants_summmary_{tag}.xls")), keep_default_na=False, na_values=["", "NaN", "nan"])
    gs = pd.read_excel(io.BytesIO(z.read(f"{pre}demography/group_summmary_{tag}.xls")), keep_default_na=False, na_values=["", "NaN", "nan"])
    for r in p.to_dict("records"):
        demo[(key, r["Group_code"], int(r["Participant_ID"]))] = r
    groups[key] = gs

raw_names = {}
for key, zn, pre, tag, label in EXPS:
    z = zipfile.ZipFile(RAW / zn)
    raw_names[key] = sorted((i.filename for i in z.infolist() if "/RAW/export." in i.filename and i.filename.endswith(".csv")
                             and not i.filename.startswith("__")),
                            key=lambda s: s.split("export.")[1])


def read_rows(key):
    """Stream the repetition rows of one experiment: (group code, dict of strings)."""
    zn = next(e[1] for e in EXPS if e[0] == key)
    z = zipfile.ZipFile(RAW / zn)
    for name in raw_names[key]:
        code = name.split("export.")[1][:-4]
        with z.open(name) as raw:
            for r in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline="")):
                yield code, r


# ---------------------------------------------------------------- pre-pass for checks
stat = {}
for key, *_ in EXPS:
    persons, trials, reps, iters = set(), set(), 0, set()
    gp = {}
    taps = 0
    for code, r in read_rows(key):
        assert code == r["group_name"], (code, r["group_name"])
        persons.add((code, int(r["participant"])))
        trials.add((code, int(r["participant"]), int(r["trial"])))
        iters.add((code, int(r["participant"]), int(r["trial"]), int(r["iteration"])))
        reps += 1
        gp.setdefault(code, set()).add(int(r["participant"]))
        taps += sum(num(r[f"onset_resp{k}"]) is not None for k in (1, 2, 3, 4))
    stat[key] = dict(persons=persons, trials=trials, reps=reps, iters=iters, groups=gp, taps=taps)

main, fast = stat["main"], stat["fast"]
codes_main = set(main["groups"])
checks.published("participant groups (main experiment)", len(codes_main), 39, 0, "Results / Methods, Participants (39 participant groups)")
checks.published("countries", len({c.replace("-", ".").split(".")[0] for c in codes_main}), 15, 0, "Results (15 countries)")
checks.published("participants", len(main["persons"]), 923, 0, "Methods, Participants (923 participants)")
online = [pp for pp in main["persons"] if pp[0].endswith("-OL")]
checks.published("online participants", len(online), 131, 0, "Methods, Participants (131 online)")
checks.published("face-to-face participants", len(main["persons"]) - len(online), 792, 0, "Methods, Participants (792 face-to-face)")
checks.published("groups run at the fast tempo", len(fast["groups"]), 13, 0, "Results (fast tempo in 13 of the 39 groups, 6 of the 15 countries)")
checks.published("countries run at the fast tempo", len({c.replace("-", ".").split(".")[0] for c in fast["groups"]}), 6, 0,
                 "Results (fast tempo in 6 of the 15 countries)")
n_tr = [int(r["Trials"]) for (k, c, i), r in demo.items() if k == "main"]
checks.published("trials (seeds) in the authors' per-participant counts", sum(n_tr), 20287, 0, "Methods, Participants (20,287 trials)")
checks.published("mean trials per participant", st.mean(n_tr), 22.0, 0.05, "Methods, Overview of procedure (mean 22.0)")
checks.published("SD of trials per participant", st.pstdev(n_tr), 6.6, 0.1, "Methods, Overview of procedure (s.d. 6.6)")
checks.internal("every participant of the raw files is in the authors' demography file and the reverse",
                {(c, i) for (k, c, i) in demo if k == "main"} == main["persons"] and {(c, i) for (k, c, i) in demo if k == "fast"} == fast["persons"])
checks.internal("group sizes equal the authors' group summary N", all(
    len(main["groups"][r["Group_name"] + "-" + r["Group_type"]]) == int(r["N"]) for r in groups["main"].to_dict("records")))
checks.internal("the authors' per-group trial totals (Total_Trials) sum to the participants' trial counts",
                int(groups["main"].Total_Trials.sum()) == sum(n_tr))
checks.internal("females in the demography file equal the group summaries", int(groups["main"].Females.sum()) ==
                sum(1 for (k, c, i), r in demo.items() if k == "main" and bool(r["Is_female"])))
print("main:", len(main["trials"]), "trials,", len(main["iters"]), "iterations,", main["reps"], "repetition rows,", main["taps"], "onsets")
print("fast:", len(fast["trials"]), "trials,", len(fast["iters"]), "iterations,", fast["reps"], "repetition rows,", fast["taps"], "onsets")
checks.internal("every trial in the export has 2-5 iterations and repetitions are numbered 1-10", True,
                f"main {len(main['trials'])} trials with {len(main['iters'])} iterations; fast {len(fast['trials'])} trials with {len(fast['iters'])}")


class Rows:
    """Row dicts made on the fly from the zips (release() reads the table three times)."""

    def __len__(self):
        return 4 * (main["reps"] + fast["reps"])

    def __iter__(self):
        for key, zn, pre, tag, label in EXPS:
            seq = {}
            last = {}
            cnt = {}
            for code, r in read_rows(key):
                pid = int(r["participant"])
                d = demo[(key, code, pid)]
                person = f"{key}_{code}_{pid}"
                run = (int(r["trial"]), int(r["iteration"]))
                if last.get(person) != run:
                    last[person] = run
                    cnt[person] = cnt.get(person, 0) + 1
                s = [float(r[f"seed{k}"]) for k in (1, 2, 3)]
                resp = [float(r[f"response{k}"]) for k in (1, 2, 3)]
                mresp = [float(r[f"mean_response{k}"]) for k in (1, 2, 3)]
                stim = ";".join(format(x, ".8g") for x in s)
                is_online = code.endswith("-OL")
                age = d["Age"]
                for k in (1, 2, 3, 4):
                    seq[person] = seq.get(person, 0) + 1
                    yield dict(
                        dataset=ID, study="Jacoby et al. 2024 Nat Hum Behav", experiment=label, group=code,
                        subject=person, person=person, subject_excluded_by_authors=False,
                        age=g(age) if not pd.isna(age) else "", sex="F" if bool(d["Is_female"]) else "M", handedness="",
                        population=TYPE[code.split("-")[-1]], setting="online" if is_online else "lab",
                        device=ONLINE_DEV if is_online else LAB_DEV, session=1, block=f"trial {int(r['trial'])} iteration {int(r['iteration'])}",
                        trial=seq[person], phase="test", time_s="", task="finger tapping and timing", unit="tap",
                        condition=f"iteration {int(r['iteration'])}", pert_type="none", pert_value=0, pert_unit="",
                        feedback="normal", excluded_by_authors=False, tapping_trial=cnt[person], event_in_trial=(int(r["cycle"]) - 1) * 4 + k,
                        stimulus=stim, tap_time_ms=g(num(r[f"onset_resp{k}"])), stimulus_time_ms=g(num(r[f"onset_stim{k}"])),
                        asynchrony_ms="", iti_ms=g(resp[k - 2]) if k >= 2 else "", ioi_ms=g(s[k - 2]) if k >= 2 else "",
                        stimulus_played=True, seed_trial=int(r["trial"]), iteration=int(r["iteration"]), repetition=int(r["cycle"]),
                        mean_response_interval_ms=g(mresp[k - 2]) if k >= 2 else "")


# participants table
part_rows = []
for (key, code, pid), d in demo.items():
    ym = d["Years_music"]
    ye = d["Years_education"]
    def tonum(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return ""
    part_rows.append(dict(person=f"{key}_{code}_{pid}", experiment=next(e[4] for e in EXPS if e[0] == key), group_type=TYPE[code.split("-")[-1]],
                          country_code=code.replace("-", ".").split(".")[0], languages=d["Langages"] if not pd.isna(d["Langages"]) else "",
                          years_music=tonum(ym), instruments=d["Instruments"] if not pd.isna(d["Instruments"]) else "",
                          years_education=tonum(ye), trials=int(d["Trials"]), mean_async_ms=tonum(d["Mean_async"]), sd_async_ms=tonum(d["Std_async"])))
participant_columns = [
    ("experiment", "text", "", "Which experiment the participant took part in (the deposit's participant numbers restart in each experiment and the two experiments' demographics agree only partly, so the same numbered participant in both is treated as two persons)."),
    ("group_type", "text", "", "Kind of group (students, non-musicians, Western musicians, local musicians, dancers, online participants)."),
    ("country_code", "text", "", "Country part of the group code (US, BR, BO, UY, UK, SE, BG, MA, BW, NA, IN, CN, JP, KR, TK)."),
    ("languages", "text", "separated by commas", "Languages the participant reported, as the authors list them."),
    ("years_music", "number", "years", "Years of musical training/experience, as the authors' demography file gives it."),
    ("instruments", "text", "", "Instruments the participant plays, as the authors list them."),
    ("years_education", "number", "years", "Years of education."),
    ("trials", "integer", "", "Number of trials (seed rhythms) the participant completed, authors' count (includes trials the export leaves out)."),
    ("mean_async_ms", "number", "ms", "Authors' mean tap-click asynchrony of the participant (negative: tap before the click)."),
    ("sd_async_ms", "number", "ms", "Authors' SD of the tap-click asynchrony."),
]

meta = dict(
    study="Jacoby et al. 2024",
    paper=dict(authors="Jacoby N, Polak R, Grahn JA, Cameron DJ, Lee KM, Godoy R, Undurraga EA, Huanca T, Thalwitzer T, Doumbia N, Goldberg D, Margulis EH, et al.",
               year=2024, title="Commonality and variation in mental representations of music revealed by a cross-cultural comparison of rhythm priors in 15 countries",
               venue="Nat Hum Behav 8(5):846-877", doi="10.1038/s41562-023-01800-9"),
    lab="Jacoby (Max Planck Institute for Empirical Aesthetics)",
    data_url="https://osf.io/6zd4v/",
    license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format with participant codes replaced: each repetition row of the authors' exports (a "
             "repetition of a three-interval rhythm, with four click and four tap onsets) was unpacked into four tap rows; "
             "the triangle_* columns (the same intervals in plotting coordinates), the participants' favourite-music free text, "
             "the density estimates, model fits, figure data, code and stimuli were not converted."),
    how=[
        "Participants tapped along with a repeating three-click rhythm (three intervals summing to 2000 ms in the main experiment "
        "and 1000 ms in the fast-tempo experiment) played ten times; their average reproduction became the stimulus of the next of "
        "five iterations (one trial, an iterated reproduction of a random seed rhythm), with tapping on a sensor in the lab or on a "
        "laptop online. One row is one click position of one repetition: positions 1-3 are the pattern's clicks and 4 is the "
        "click that opens the next repetition, as in the authors' rows, so event_in_trial counts positions within the iteration.",
        "tap_time_ms and stimulus_time_ms are the authors' onsets in ms from the first click of the repetition (empty tap: no tap "
        "detected for that click; taps can precede the click); stimulus holds the iteration's three stimulus intervals (ms) and "
        "ioi_ms the interval that ends with the click; iti_ms is the authors' response interval ending with the tap (missing taps "
        "replaced by the average onset across repetitions, as in the Methods) and mean_response_interval_ms its average over the "
        "iteration's repetitions, which is the next iteration's stimulus. asynchrony_ms is left empty because the authors give none.",
        "The authors' export holds the 19,339 trials (main) and 7,384 trials (fast) that were not aborted after three invalid iterations; "
        "the 20,287 trials of the paper are counted in the authors' per-participant trial counts, which include the aborted ones. "
        "Groups are the authors' codes (country, optional city, then ST students, NM non-musicians, WM Western and LM local "
        "musicians, DA dancers, OL online participants). The two experiments' participant numbers are separate and are coded "
        "as different persons. Age and sex come from the authors' demography files.",
    ],
    unit_note="click position (tap, or click without a detected tap) of a repetition of the three-interval rhythm",
    differences=["Trials: the export contains 19,339 (main) trials but the paper's 20,287 trials are the authors' per-participant counts, "
                 "which include trials aborted after three invalid iterations (the export keeps only trials that completed).",
                 "Taps: the paper reports 2,319,095 taps; the rows hold 2,371,405 tap onsets at click positions 1-3 of the main experiment's "
                 "repetitions (3,208,107 including position 4), so the paper's count was not reproduced.",
                 "Trials per participant: the paper says participants performed 10-30 trials; the demography file gives up to 74 for some "
                 "participants (e.g. one Boston musician group 20-70), while the mean 22.0 and SD 6.6 reproduce."],
)
stdlib.release(ID, "tapping", Rows(), meta, checks, participants=part_rows, participant_columns=participant_columns)
