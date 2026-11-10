"""Anglada-Tort, Harrison & Jacoby (2022), Behav Res Methods (REPP): Experiments 2 and 3, one row per tap.

Run from the site folder: python3 -I tools/std/convert/AngladaTort2022_BehavResMethods.py
Raw files (untrusted data, read only) in raw_extra/AngladaTort2022_BehavResMethods/, from https://osf.io/r2pxd/:
  Exp2_raw_data.csv, Exp3_raw_data.csv  one row per 30-s tapping trial; the aligned tap and stimulus onsets are arrays in
                                        the tcontent column (REPP's Ra, Sa, ea, ra, sa, Rraw), the stimulus in stim_info
  Exp3_Figure4A.csv                     the authors' per-participant test and retest scores (internal check only)
The array names follow the REPP package (gitlab.com/computational-audition/repp, read as text, never run): Ra / Sa = tap /
stimulus onset paired at each stimulus position (NaN when no tap was paired), ea = Ra - Sa, ra / sa = interval from the
previous position, Rraw = every detected tap. The published checks recompute the paper's SD-of-asynchrony scores from the
converted rows.
"""
import ast
import csv
import json
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "AngladaTort2022_BehavResMethods"
RAW = stdlib.raw_dir(ID)
E2, E3 = f"{ID}_Exp2", f"{ID}_Exp3"
STIMS = {"iso": ("iso_600ms", "iso_800ms"), "music": ("train1.unfiltered.wav", "train7.unfiltered.wav")}
OS_NAME = {"0": "Chrome OS", "1": "Windows", "2": "macOS", "3": "other OS"}
DEV_REPP = "laptop speakers and built-in microphone, taps on the laptop surface (REPP)"
DEV_LAB = "headphones, tapping sensor and sound card with a loop-back cable (in-lab method)"

checks = stdlib.Checks()


def read(name):
    with open(RAW / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def arrays(text):
    """A trial's arrays: JSON with NaN, or (in-lab trials of Experiment 2) a Python dict literal with nan. Parsed as data
    only (json.loads, ast.literal_eval); missing values become None."""
    d = json.loads(text) if text.lstrip().startswith('{"') else ast.literal_eval(re.sub(r"\bnan\b", "None", text))
    return {k: ([None if isnan(x) else x for x in v] if isinstance(v, list) else (None if isnan(v) else v))
            for k, v in d.items()}


def flags(si):
    out = []
    for p in si["onsets_played"]:
        assert isinstance(p, bool) or p in ("true", "false"), p
        out.append(p if isinstance(p, bool) else p == "true")
    return out


def ms(x):
    if x is None:
        return None
    return int(x) if float(x).is_integer() else float(x)


def sd0(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    return (sum((a - mx) * (b - my) for a, b in zip(x, y))
            / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)))


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


# ---------------------------------------------------------------- read
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for name in ("Exp2_raw_data.csv", "Exp3_raw_data.csv", "Exp3_Figure4A.csv"):
    checks.internal(f"{name} matches the downloaded hash", stdlib.sha256(RAW / name) == src[name]["sha256"])
e2, e3 = read("Exp2_raw_data.csv"), read("Exp3_raw_data.csv")

trials = []
for r in e2:
    si = json.loads(r["stim_info"])
    task = "music" if si["stim_name"].startswith("train") else "iso"
    assert task == ("music" if r["stim"].startswith("music") else "iso") and r["condition"] in ("psynet", "matlab")
    ts = arrays(r["tstats"])
    trials.append(dict(exp=E2, pid=r["id"], test=r["part"], task=task, stim=si["stim_name"], method=r["condition"],
                       batch="", os="", si=si, tc=arrays(r["tcontent"]), n_resp=ts["number_of_responses_all"],
                       n_raw=ts["raw_tapping_detected_onsets_number"], raw=r))
for r in e3:
    si = json.loads(r["stim_info"])
    assert si["stim_name"] == r["stim_name"] and r["condition"] == f"{r['test']}-{r['task']}"
    trials.append(dict(exp=E3, pid=r["participant_id"], test=r["test"], task=r["task"], stim=r["stim_name"], method="",
                       batch=r["batch"], os=r["OS"], si=si, tc=arrays(r["tcontent"]),
                       n_resp=float(r["number_of_responses_all"]), n_raw=float(r["raw_tapping_detected_onsets_number"]),
                       raw=r))
for t in trials:
    col = "notplayed" if t["task"] == "music" else "over_all"
    t["sd_auth"] = float(t["raw"][f"std_async_computed_{col}"])
    t["mean_auth"] = float(t["raw"][f"mean_async_computed_{col}"])

# ---------------------------------------------------------------- what the deposit holds
c2 = Counter((r["id"], r["condition"], r["part"], r["stim"]) for r in e2)
checks.internal("Experiment 2: 20 participants x 2 methods x 2 tests x 4 stimuli x 2 trials = 640 trials (Methods, "
                "Experiment 2: four isochronous and four music trials per battery, done twice with each method)",
                len(e2) == 640 and len({k[0] for k in c2}) == 20 and len(c2) == 320 and set(c2.values()) == {2})
c3 = Counter((r["participant_id"], r["test"], r["stim_name"]) for r in e3)
checks.internal("Experiment 3: at most 2 trials per participant, test and stimulus (16 per participant)",
                max(c3.values()) == 2, f"{len(e3)} trials; trials per participant "
                f"{min(Counter(r['participant_id'] for r in e3).values())}-{max(Counter(r['participant_id'] for r in e3).values())}")
ratio = [float(r["ratio_taps_to_metronomes"]) for r in e3]
checks.internal("Experiment 3 deposit holds only trials that pass the authors' criteria (Supplementary, Failing "
                "Criteria): none failed, all six markers found, marker error at most 5 ms, 50-200% taps per stimulus",
                all(r["is_failed"] == "FALSE" and r["marker_onsets"] == "6" and r["marker_missed"] == "0"
                    and float(r["marker_max_difference"]) <= 5 for r in e3) and 50 <= min(ratio) and max(ratio) <= 200,
                f"max marker error {max(float(r['marker_max_difference']) for r in e3):.2f} ms; taps per stimulus "
                f"{min(ratio):.0f}-{max(ratio):.0f}%")
per_p = defaultdict(set)
for r in e3:
    per_p[r["participant_id"]].add((r["OS"], r["batch"], r["musical_training"]))
checks.internal("Experiment 3: OS, batch and musical training are constant within each participant; batch is the "
                "participant code's suffix", all(len(v) == 1 for v in per_p.values())
                and all(r["participant_id"].endswith("-" + r["batch"]) for r in e3))

# operating-system codes are not documented: codes 1 and 2 are identified by the paper's marker-error comparison
me = {o: [float(r["marker_max_difference"]) for r in e3 if r["OS"] == o] for o in OS_NAME}
n1, n2 = len(me["1"]), len(me["2"])
sp = math.sqrt(((n1 - 1) * st.variance(me["1"]) + (n2 - 1) * st.variance(me["2"])) / (n1 + n2 - 2))
tstat = (st.mean(me["1"]) - st.mean(me["2"])) / (sp * math.sqrt(1 / n1 + 1 / n2))
checks.internal("OS code 1 = Windows and 2 = macOS: marker errors M = 1.74 and 1.48 ms, t(2812) = 9.36 (Results, "
                "Experiment 3; Fig. 4c)", abs(st.mean(me["1"]) - 1.74) < 0.006 and abs(st.mean(me["2"]) - 1.48) < 0.006
                and n1 + n2 - 2 == 2812 and abs(tstat - 9.36) < 0.01,
                f"Windows {st.mean(me['1']):.3f} ({n1} trials), macOS {st.mean(me['2']):.3f} ({n2}), t = {tstat:.3f}")
laptops0 = {r["participant_id"]: r["laptop"].lower() for r in e3 if r["OS"] == "0"}
checks.internal("OS code 0 = Chrome OS: every laptop model these participants reported is a Chromebook or Pixelbook",
                all(("chrome" in v) or ("pixelbook" in v) for v in laptops0.values()), f"{len(laptops0)} participants")

# ---------------------------------------------------------------- the OpenMotor table
NAMES = [c[1] for c in stdlib.columns_for("tapping") if c[1] != "person"]  # person = subject (stdlib fills it)
cols = {n: [] for n in NAMES}
codes = {}
rows_of, trials_of = Counter(), Counter()
trial_keys = []
bad = Counter()
n_round = n_slots = n_unpaired = n_taps = 0
for t in trials:
    tc, si = t["tc"], t["si"]
    Ra, Sa, ea, ra, sa, Rraw = (tc[k] for k in ("Ra", "Sa", "ea", "ra", "sa", "Rraw"))
    played = flags(si)
    N = len(Sa)
    if not all(len(x) == N for x in (Ra, ea, ra, sa, played, si["stim_onsets"], si["shifted_onsets"])):
        bad["array lengths"] += 1
    if any(x is None for x in Sa) or Sa != tc["Sraw"] or Rraw != tc["tapping_detected_onsets"]:
        bad["stimulus or raw tap arrays"] += 1
    shift = tc["first_marker_onset"] - si["marker_onsets"][0]
    if max(abs(a - (b + shift)) for a, b in zip(Sa, si["shifted_onsets"])) > 1:
        bad["stimulus onsets not on the recording clock"] += 1
    for j in range(N):
        if (Ra[j] is None) != (ea[j] is None) or (Ra[j] is not None and abs(ea[j] - (Ra[j] - Sa[j])) > 1):
            bad["asynchrony != tap - stimulus"] += 1
        if j and ((ra[j] is None) != (Ra[j] is None or Ra[j - 1] is None)
                  or (ra[j] is not None and abs(ra[j] - (Ra[j] - Ra[j - 1])) > 1)):
            bad["iti != difference of paired taps"] += 1
        if j and (sa[j] is None or abs(sa[j] - (Sa[j] - Sa[j - 1])) > 1):
            bad["ioi != difference of stimuli"] += 1
    if ra[0] is not None or sa[0] is not None:
        bad["first interval not empty"] += 1
    if any(Rraw[i] >= Rraw[i + 1] for i in range(len(Rraw) - 1)):
        bad["raw taps not increasing"] += 1
    # pair each paired tap with its detected tap (equal, or 1 ms apart where the deposit rounded them differently)
    used = [False] * len(Rraw)
    where = {v: k for k, v in enumerate(Rraw)}
    ks = []
    for j in range(N):
        if Ra[j] is None:
            continue
        k = where.get(Ra[j])
        if k is None or used[k]:
            cand = [i for i, v in enumerate(Rraw) if not used[i] and abs(v - Ra[j]) <= 1]
            assert cand, "paired tap without a detected tap"
            k = min(cand, key=lambda i: abs(Rraw[i] - Ra[j]))
            n_round += 1
        used[k] = True
        ks.append(k)
    if ks != sorted(ks):
        bad["paired taps out of order"] += 1
    if len(ks) != t["n_resp"] or len(Rraw) != t["n_raw"]:
        bad["tap counts differ from the authors' per-trial counts"] += 1
    ev = [((Ra[j] if Ra[j] is not None else Sa[j]), 0, j) for j in range(N)]
    ev += [(Rraw[k], 1, k) for k in range(len(Rraw)) if not used[k]]
    ev.sort(key=lambda e: (e[0], e[1]))
    if [e[2] for e in ev if e[1] == 0] != list(range(N)):
        bad["time order changes the order of stimulus positions"] += 1
    n_slots += N
    n_unpaired += len(Rraw) - len(ks)
    n_taps += len(Rraw)

    key = (t["exp"], t["pid"])
    if key not in codes:
        codes[key] = f"{ID}_S{len(codes) + 1:03d}"
    subj = codes[key]
    trials_of[subj] += 1
    trial_keys.append((subj, trials_of[subj]))
    if t["exp"] == E2:
        device, cond, setting = (DEV_REPP if t["method"] == "psynet" else DEV_LAB), f"{t['method']} {t['task']}", "lab"
    else:
        device, cond, setting = f"{DEV_REPP}; {OS_NAME[t['os']]}", t["task"], "online"
    for pos, (_, kind, i) in enumerate(ev, 1):
        rows_of[subj] += 1
        if kind == 0:
            tap, stim, asy, iti, ioi, pl = ms(Ra[i]), ms(Sa[i]), ms(ea[i]), ms(ra[i]), ms(sa[i]), played[i]
            excl = t["task"] == "music" and played[i]
        else:
            tap, stim, asy, iti, ioi, pl, excl = ms(Rraw[i]), None, None, None, None, None, False
        row = dict(dataset=ID, study=ID, experiment=t["exp"], group=t["batch"], subject=subj,
                   subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy",
                   setting=setting, device=device, session=1, block=t["test"], trial=rows_of[subj], phase="test",
                   time_s="", task="finger/hand", unit="tap", condition=cond, pert_type="none", pert_value=0,
                   pert_unit="", feedback="normal", excluded_by_authors=excl, tapping_trial=trials_of[subj],
                   event_in_trial=pos, stimulus=t["stim"], tap_time_ms=tap, stimulus_time_ms=stim, asynchrony_ms=asy,
                   iti_ms=iti, ioi_ms=ioi, stimulus_played=pl)
        for n in NAMES:
            cols[n].append(row[n])

checks.internal("every trial's arrays are consistent: equal lengths, stimulus onsets on the recording clock (shifted "
                "by the first detected marker), asynchrony = tap - stimulus, iti and ioi = differences of consecutive "
                "positions (all within the deposit's 1-ms rounding), paired taps in order, counts equal to the "
                "authors' per-trial counts", not bad, str(dict(bad)))
n = len(cols["subject"])
n_paired = sum(1 for a, b in zip(cols["tap_time_ms"], cols["stimulus_time_ms"]) if a is not None and b is not None)
checks.internal("each paired tap is one of the detected taps", True,
                f"{n_round} of {n_paired} paired taps are 1 ms from their detected tap (rounding); the row keeps the "
                "paired value")
checks.internal("rows = stimulus positions + taps paired with no stimulus; every detected tap appears once",
                n == n_slots + n_unpaired and n - cols["tap_time_ms"].count(None) == n_taps,
                f"{n} rows: {n_slots} stimulus positions, {n_unpaired} unpaired taps, "
                f"{cols['tap_time_ms'].count(None)} stimulus positions without a tap")
seen, order = set(), []
for i in range(n):
    k = (cols["subject"][i], cols["tapping_trial"][i])
    if k not in seen:
        seen.add(k)
        order.append(k)
same_trial = [i > 0 and (cols["subject"][i], cols["tapping_trial"][i]) == (cols["subject"][i - 1],
                                                                          cols["tapping_trial"][i - 1]) for i in range(n)]
checks.internal("every trial kept, each as one contiguous run of rows, in the authors' file order (Experiment 2 file, "
                "then Experiment 3); event_in_trial counts up from 1 within each trial",
                order == trial_keys and len(order) == len(e2) + len(e3) and len(seen) == sum(not x for x in same_trial)
                and all(cols["event_in_trial"][i] == (cols["event_in_trial"][i - 1] + 1 if same_trial[i] else 1)
                        for i in range(n)))
last = {}
ok_trial = True
for i in range(n):
    s = cols["subject"][i]
    ok_trial &= cols["trial"][i] == last.get(s, 0) + 1
    last[s] = cols["trial"][i]
checks.internal("trial counts each participant's rows from 1 without gaps", ok_trial)
asy = [a for a in cols["asynchrony_ms"] if a is not None]
taps = [a for a in cols["tap_time_ms"] if a is not None]
iois = [a for a in cols["ioi_ms"] if a is not None]
checks.internal("ranges: tap and stimulus times 2-38 s into the recording, |asynchrony| < 400 ms, stimulus intervals "
                "400-1200 ms (isochronous trials exactly 600 or 800 ms)",
                2000 <= min(taps) and max(taps) <= 38000 and max(abs(a) for a in asy) < 400
                and 400 <= min(iois) and max(iois) <= 1200
                and all(cols["ioi_ms"][i] in (None, int(cols["stimulus"][i][4:7]))
                        for i in range(n) if cols["stimulus"][i].startswith("iso")),
                f"taps {min(taps)}-{max(taps)} ms, asynchrony {min(asy)} to {max(asy)} ms, ioi {min(iois)}-{max(iois)} ms")
excl = sum(cols["excluded_by_authors"])
checks.internal("excluded_by_authors = metronome-marked beats of the music trials", True,
                f"{excl} of {n} rows ({100 * excl / n:.1f}%)")

# ---------------------------------------------------------------- per-trial scores from the converted rows
tr_asy, tr_meta = defaultdict(list), {}
for i in range(n):
    k = (cols["subject"][i], cols["tapping_trial"][i])
    if k not in tr_meta:
        tr_meta[k] = (cols["experiment"][i], cols["block"][i], cols["condition"][i], cols["stimulus"][i],
                      cols["device"][i])
    if cols["asynchrony_ms"][i] is not None and not cols["excluded_by_authors"][i]:
        tr_asy[k].append(cols["asynchrony_ms"][i])
checks.internal("every trial has at least two analysed asynchronies", all(len(tr_asy[k]) >= 2 for k in tr_meta))
tr_sd = {k: sd0(tr_asy[k]) for k in tr_meta}
d_sd = [abs(tr_sd[k] - t["sd_auth"]) for k, t in zip(trial_keys, trials)]
d_mean = [abs(st.mean(tr_asy[k]) - t["mean_auth"]) for k, t in zip(trial_keys, trials)]
checks.internal("per trial, the SD (population SD, as REPP computes it) and mean of the analysed asynchronies equal the "
                "authors' per-trial values (all onsets for isochronous trials, beats without the metronome for music) "
                "within 0.5 ms, the most that rounding every asynchrony to 1 ms can move a mean or an SD (plus 1e-6 for "
                "the 10 significant digits the authors' file keeps)",
                max(d_sd) <= 0.5 + 1e-6 and max(d_mean) <= 0.5 + 1e-6,
                f"max difference {max(d_sd):.3f} ms (SD), {max(d_mean):.3f} ms (mean); median "
                f"{st.median(d_sd):.3f} and {st.median(d_mean):.3f} ms")

# Experiment 3 (Results and Discussion): score = SD of asynchrony, averaged over trials of a stimulus, then stimuli, then
# the two tasks; test-retest uses participants with a trial for every stimulus in both tests
cell = defaultdict(list)
for k, (exp, block, cond, stim, dev) in tr_meta.items():
    if exp == E3:
        cell[(k[0], block, cond, stim)].append(tr_sd[k])
subj3 = sorted({k[0] for k, m in tr_meta.items() if m[0] == E3})
subj2 = sorted({k[0] for k, m in tr_meta.items() if m[0] == E2})
full = [s for s in subj3 if all((s, te, ta, sm) in cell for te in ("pre", "post") for ta in STIMS for sm in STIMS[ta])]
score = {s: {te: st.mean(st.mean(st.mean(cell[(s, te, ta, sm)]) for sm in STIMS[ta]) for ta in STIMS)
             for te in ("pre", "post")} for s in full}
pre3, post3 = [score[s]["pre"] for s in full], [score[s]["post"] for s in full]
fig4a = {r["participant_id"]: (float(r["PRE"]), float(r["POST"])) for r in read("Exp3_Figure4A.csv")}
orig = {c: pid for (exp, pid), c in codes.items() if exp == E3}
dev4a = max(max(abs(score[s]["pre"] - fig4a[orig[s]][0]), abs(score[s]["post"] - fig4a[orig[s]][1])) for s in full)
r_fig = pearson([v[0] for v in fig4a.values()], [v[1] for v in fig4a.values()])
checks.internal("Experiment 3 test and retest scores equal the authors' Figure 4A data file, participant by participant",
                set(orig[s] for s in full) == set(fig4a) and dev4a < 0.3,
                f"max difference {dev4a:.3f} ms; Pearson r here {pearson(pre3, post3):.3f}, in the authors' file "
                f"{r_fig:.3f} (the paper prints r = .80)")
checks.internal("Experiment 3 Pearson test-retest r agrees with the authors' figure data",
                abs(pearson(pre3, post3) - r_fig) < 0.002)

SRC3 = "Results, Experiment 3 (and Supplementary, Participants)"
checks.published("Experiment 3 participants with valid tapping data", len(subj3), 226, 0, SRC3)
checks.published("Experiment 3 tapping trials recorded on Windows or macOS laptops", sum(
    1 for k, m in tr_meta.items() if m[0] == E3 and m[4].endswith(("Windows", "macOS"))), 2814, 0,
    "Fig. 4 legend (2814 trials in total) and Results, Experiment 3 (t(2812))")
checks.published("Experiment 3 participants in the test-retest analysis (a valid trial for each stimulus in each task "
                 "and test)", len(full), 166, 0, "Results, Experiment 3")
checks.published("Experiment 3 test-retest Spearman rho of SD of asynchrony", spearman(pre3, post3), 0.81, 0.005,
                 "Results, Experiment 3; Fig. 4a")
iso = [st.mean([x for te in ("pre", "post") for sm in STIMS["iso"] for x in cell[(s, te, "iso", sm)]]) for s in full]
mus = [st.mean([x for te in ("pre", "post") for sm in STIMS["music"] for x in cell[(s, te, "music", sm)]]) for s in full]
SRCD = "Discussion, third paragraph (N = 166; labelled Experiment 2 there, the online sample is Experiment 3)"
checks.published("Experiment 3 isochronous task: mean SD of asynchrony (ms)", st.mean(iso), 40.68, 0.01, SRCD)
checks.published("Experiment 3 isochronous task: SD across participants (ms)", st.stdev(iso), 11.98, 0.01, SRCD)
checks.published("Experiment 3 music task: mean SD of asynchrony (ms)", st.mean(mus), 42.68, 0.01, SRCD)
checks.published("Experiment 3 music task: SD across participants (ms)", st.stdev(mus), 15.82, 0.01, SRCD)
checks.published("Experiment 3 correlation of the isochronous and music scores", pearson(iso, mus), 0.58, 0.005, SRCD)

# Experiment 2: score per test and method = mean over the two tasks of the mean trial SD
c2s = defaultdict(list)
for k, (exp, block, cond, stim, dev) in tr_meta.items():
    if exp == E2:
        method, task = cond.split()
        c2s[(k[0], method, block, task)].append(tr_sd[k])


def test_score(s, m, te):
    return st.mean(st.mean(c2s[(s, m, te, ta)]) for ta in STIMS)


SRC2 = "Results, Experiment 2; Fig. 3b-c"
checks.published("Experiment 2 participants", len(subj2), 20, 0, "Methods, Experiment 2; Supplementary, Participants")
for m, label, r_rep in (("psynet", "REPP", 0.87), ("matlab", "in-lab loop-back method", 0.89)):
    pre2 = [test_score(s, m, "pre") for s in subj2]
    post2 = [test_score(s, m, "post") for s in subj2]
    checks.published(f"Experiment 2 test-retest Pearson r, {label}", pearson(pre2, post2), r_rep, 0.005, SRC2)
    if m == "matlab":
        checks.published(f"Experiment 2 test-retest Spearman rho, {label}", spearman(pre2, post2), 0.83, 0.005, SRC2)
    else:
        rho_rows, rows_score = spearman(pre2, post2), {"pre": dict(zip(subj2, pre2)), "post": dict(zip(subj2, post2))}
# REPP rho (.81 in the paper) is not reproducible from the rows at two decimals: the deposit rounds every asynchrony to
# 1 ms, which can swap the ranks of participants whose scores are very close. The authors' unrounded per-trial SDs are used.
a2 = defaultdict(list)
for r in e2:
    if r["condition"] == "psynet":
        task = "music" if r["stim"].startswith("music") else "iso"
        a2[(r["id"], r["part"], task)].append(float(r[f"std_async_computed_{'notplayed' if task == 'music' else 'over_all'}"]))
ids2 = sorted({k[0] for k in a2})
auth_score = {te: {p: st.mean(st.mean(a2[(p, te, ta)]) for ta in STIMS) for p in ids2} for te in ("pre", "post")}
rho_auth = spearman([auth_score["pre"][p] for p in ids2], [auth_score["post"][p] for p in ids2])
gaps = [abs(auth_score[te][p] - auth_score[te][q]) for te in ("pre", "post") for i, p in enumerate(ids2) for q in ids2[i + 1:]
        if (auth_score[te][p] - auth_score[te][q]) * (rows_score[te][codes[(E2, p)]] - rows_score[te][codes[(E2, q)]]) < 0]
checks.internal("Experiment 2 REPP test-retest Spearman rho = .81 (Results, Experiment 2) from the authors' per-trial SDs",
                abs(rho_auth - 0.81) < 0.005,
                f"{rho_auth:.3f} from their per-trial values, {rho_rows:.3f} from the rows: the deposit's 1-ms rounding "
                f"of the asynchronies swaps {len(gaps)} pair(s) of participants whose scores are "
                f"{', '.join(f'{g:.3f}' for g in gaps)} ms apart")
overall = {m: [st.mean(test_score(s, m, te) for te in ("pre", "post")) for s in subj2] for m in ("matlab", "psynet")}
checks.published("Experiment 2 concurrent validity r (REPP vs in-lab method)", pearson(overall["matlab"],
                 overall["psynet"]), 0.94, 0.005, SRC2)

for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    else:
        print(f"  [{'ok' if ch['ok'] else 'FAILED'}] {ch['what']} {ch.get('detail', '')}")

# ---------------------------------------------------------------- release
meta = dict(
    study="Anglada-Tort et al. 2022",
    paper=dict(authors="Anglada-Tort M, Harrison PMC, Jacoby N", year=2022,
               title="REPP: A robust cross-platform solution for online sensorimotor synchronization experiments",
               venue="Behav Res 54(5):2271-2285", doi="10.3758/s13428-021-01722-2"),
    lab="Jacoby (Max Planck Institute for Empirical Aesthetics)",
    data_url="https://osf.io/r2pxd/",
    license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format with participant codes replaced: each trial's arrays of tap and stimulus "
             "onsets were unpacked into one row per tap (or per stimulus onset without a tap); the authors' per-trial "
             "summaries (mean and SD of asynchrony, vector length, model fits), the self-reported laptop model and the "
             "per-participant musical training score were dropped, and Experiment 1 (a hardware timing calibration) "
             "and the figure files were not converted."),
    how=[
        "One row is one tap, or one stimulus onset that received no tap, in the 30-s tapping trials of Experiment 2 "
        "(lab, 20 participants, each tested twice with REPP on a laptop, condition psynet, and twice with the authors' "
        "in-lab loop-back method, condition matlab) and Experiment 3 (online, 226 participants, REPP; block pre and "
        "post are test and retest); a trial is isochronous tapping to a metronome (iso_600ms, iso_800ms) or tapping to "
        "the beat of a music excerpt (train1, train7).",
        "tap_time_ms and stimulus_time_ms are the authors' onsets in ms from the start of the trial's recording, "
        "aligned on the first marker sound, and asynchrony_ms is their tap minus stimulus (negative when the tap came "
        "first); a row without a stimulus is a tap their alignment paired with no stimulus, a row without a tap is a "
        "stimulus that got none, iti_ms is given only between taps paired with consecutive stimuli, and all values "
        "are whole ms as in the deposit.",
        "In the music trials a metronome click marked the opening and closing beats (stimulus_played true) and the "
        "authors analysed only the beats without it, so those rows are excluded_by_authors; the deposit holds only "
        "trials that passed their quality criteria (all six marker sounds found, marker error within 5 ms, 50-200% as "
        "many taps as stimuli), so no other rows are excluded.",
        "Rows keep the authors' file order, which is sorted by stimulus rather than by time, so tapping_trial is not "
        "the order in which trials were run (the deposit does not record it); in Experiment 3 group is the "
        "recruitment batch and device names the laptop's operating system, where the deposit's codes 1 and 2 are "
        "Windows and macOS (their marker errors reproduce the paper's), code 0 is Chrome OS (every laptop reported is "
        "a Chromebook) and code 3 is shown as other OS.",
    ],
    unit_note="tap, or stimulus onset that received no tap, in a 30-s tapping trial",
    differences=["Experiment 3 test-retest r: the rows and the authors' own Figure 4A data file both give 0.82; the paper prints .80.",
                 "Experiment 2 REPP test-retest rho: 0.800 from the rows against .81 in the paper, because the deposit rounds asynchronies to 1 ms (the authors' unrounded values give .812)."],
)


class Rows:
    """The table as a re-iterable sequence of row dicts made on the fly (release() reads it three times)."""

    def __init__(self, columns):
        self.names = list(columns)
        self.cols = [columns[c] for c in self.names]
        self.n = len(self.cols[0])

    def __len__(self):
        return self.n

    def __iter__(self):
        for vals in zip(*self.cols):
            yield dict(zip(self.names, vals))


stdlib.release(ID, "tapping", Rows(cols), meta, checks)
