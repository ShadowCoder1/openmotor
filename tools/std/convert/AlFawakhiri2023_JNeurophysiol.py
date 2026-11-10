"""Al-Fawakhiri, Ma, Taylor & Kim (2023), J Neurophysiol: task success and implicit adaptation in four error-clamp
experiments (204 participants), one row per reach, in the 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/AlFawakhiri2023_JNeurophysiol.py
Raw files (untrusted data, read only) in raw_extra/AlFawakhiri2023_JNeurophysiol/, from
https://github.com/kimoli/TaskSuccessImplicitAdapt (no license file in the repository):
  data_fig1_kimrep.csv                 Experiment 1, replication of Kim et al. (2019): Hit (16 mm) or Miss (6 mm) target,
                                       3.5 deg clamp, 8 targets
  data_fig2_soundtargetsize.csv        Experiment 2: the same with a Strict or Lenient chime / knock sound (2 x 2 groups)
  data_fig3_tsayrep.csv                Experiment 3, replication of Tsay et al.: one target, random +-4 deg clamp with
                                       No Jump, Jump-To, Jump-Away and Jump-In-Place trials
  data_fig4_TargetSizeTargetJump.csv   Experiment 4: one target, clamps of 1.75-10.5 deg in both directions, Hit, Miss
                                       and Jump-To trials interleaved
The authors' analysis_fig1-4 .Rmd scripts were read as text to learn the variables and the analysis, never run. Their
output_fig*.csv files (the numbers behind the figures) were read on GitHub as text; the checks below reproduce them
from the converted rows with the authors' own trial exclusions.

Signs.
* Experiments 3 and 4: HandError is the hand angle relative to the target at 90 deg (Experiment 4 also gives HandAngle,
  which equals 90 + HandError on every row), counter-clockwise positive as atan2 with y up; ClampAngle is in the same
  frame (the next-trial change of the hand has a negative slope on the clamp in every participant). TargetJump is in the
  clamp's frame in Experiment 3 (Jump-To trials have TargetJump = ClampAngle, as the authors' Rmd reads it) but in the
  opposite sense in Experiment 4, whose Jump-To trials (TrialLabel 'Jump To', the target jumping onto the cursor) all
  have TargetJump = -ClampAngle; target_jump is therefore -TargetJump in Experiment 4. TargetJump 0.01 codes
  Jump-In-Place (the target blanked for one frame): target_event flicker, target_jump 0.
* Experiments 1 and 2: the clamp direction was counterbalanced across participants (paper, Methods), but the files have
  no clamp column and every participant's hand angle goes positive during the clamp: the authors stored the angles
  sign-normalized so that adaptation is positive. Which participants had a counter-clockwise clamp is not in the deposit
  (their baseline direction biases are mirror images of the others', see the checks), so hand angles are kept as stored
  and the clamp is written as -3.5 deg in that same frame for everyone. For about half of these participants the frame
  is the mirror image of the screen; no angle was flipped here.
"""
import csv
import json
import math
import statistics as st
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "AlFawakhiri2023_JNeurophysiol"
RAW = stdlib.raw_dir(ID)
NAN = float("nan")
RADIUS_MM = 80.0
checks = stdlib.Checks()

FILES = {"Exp1": "data_fig1_kimrep.csv", "Exp2": "data_fig2_soundtargetsize.csv", "Exp3": "data_fig3_tsayrep.csv",
         "Exp4": "data_fig4_TargetSizeTargetJump.csv"}
# Experiment 1 (Rmd: miss = condition 1, hit = condition 2); Experiment 2 (Rmd: 1 'hit knock', 2 'hit ding', 3 'miss knock',
# 4 'miss ding'; knock = the Strict sound rule, ding = Lenient, confirmed by the hits flags: conditions 1 and 3 count a hit
# within 1.79 deg, 2 and 4 within about 5.4 deg)
GROUP = {"Exp1": {"1": "Miss_6mm", "2": "Hit_16mm"},
         "Exp2": {"1": "Hit_16mm_Strict", "2": "Hit_16mm_Lenient", "3": "Miss_6mm_Strict", "4": "Miss_6mm_Lenient"}}
SIZE_MM = {"Hit": 16.0, "Miss": 6.0}  # Experiments 1 and 2, target diameter (Methods)
E4_SIZE_MM = {"62": 31.0, "6": 4.5}   # Experiment 4: the file's TgtSize codes; diameters from the paper's Methods


def wrap(a):
    w = (a + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


def r8(x):
    """The value as release() writes it (8 significant digits), so the checks use exactly the written numbers."""
    return float(format(x, ".8g")) if math.isfinite(x) else x


def fin(v):
    return [x for x in v if x is not None and math.isfinite(x)]


def rmean(v):
    v = fin(v)
    return sum(v) / len(v) if v else NAN


def rsd(v):
    v = fin(v)
    return st.stdev(v) if len(v) > 1 else NAN


def rmedian(v):
    v = fin(v)
    return st.median(v) if v else NAN


def r_smooth(y):
    """The authors' smooth() (Rmd: 'dupe MATLAB's smooth function with window = 5'), reproduced as R evaluates it:
    y[i-2:i+2] is y[i - (2:i) + 2], so the middle values are the mean of y[2..i] (1-based), not a 5-point window."""
    n = len(y)
    h = [y[0], rmean(y[0:3])]
    t = [rmean(y[n - 3:n]), y[n - 1]]
    m = [rmean(y[1:i]) for i in range(3, n - 1)]
    return h + m + t


def authors_bad(y, false_start=None):
    """The authors' trial rule: |angle - smooth| > 3 SD of that difference, |angle| > 90, or (Experiments 3 and 4) a
    false start."""
    sm = r_smooth(y)
    det = [a - b for a, b in zip(y, sm)]
    sd = rsd(det)
    # R skips missing angles (if (!is.na(...))), so a trial without a hand angle is never flagged
    return [math.isfinite(a) and (abs(d) > 3 * sd or abs(a) > 90 or bool(false_start and false_start[i]))
            for i, (a, d) in enumerate(zip(y, det))]


def load(fn):
    return list(csv.DictReader(open(RAW / fn, newline="", encoding="utf-8-sig")))


def fl(s):
    return NAN if s in ("", "NaN", "nan", "NA") else float(s)


src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
data = {}
for exp, fn in FILES.items():
    checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])
    data[exp] = load(fn)

names = [c[1] for c in stdlib.reach90_columns()]
rows = []
meta_rows = []  # per row: (experiment, subject key, label) for the checks
target_ids = {}


def base_row(exp, skey, group):
    r = dict.fromkeys(names, "")
    r.update(dataset=ID, experiment=f"{ID}_{exp}", study=ID, subject=f"{exp}_{skey}", person=f"{exp}_{skey}",
             id_type="lab", person_link="assumed_unique", group=group, prior_exposure="unknown",
             subject_excluded_by_authors=False, device="tablet", setting="lab", population="healthy", session=0,
             trial_kind="reach", target_distance_norm=1, target_visible_during_move=True, probe_kind="none",
             instr_report="none", fb_cued=True, fb_endpoint_transform_only=False)
    return r


# ---------------------------------------------------------------- Experiments 1 and 2
BLOCKS12 = [("baseline_nofb", 1, 40, "baseline", False), ("baseline_fb", 41, 120, "baseline", True),
            ("clamp", 121, 760, "learning", True), ("washout_nofb", 761, 800, "aftereffect", False),
            ("washout_fb", 801, 880, "washout", True)]
for exp in ("Exp1", "Exp2"):
    raw = data[exp]
    subs = OrderedDict()
    for r in raw:
        subs.setdefault(r["subject"], []).append(r)
    n_sub = {"Exp1": 48, "Exp2": 96}[exp]
    checks.internal(f"{exp}: {n_sub} participants x 880 trials, trials 1-880 in order, cycles of 8 reaches to the 8 targets",
                    len(subs) == n_sub and all([int(r["trialNum"]) for r in v] == list(range(1, 881)) for v in subs.values())
                    and all(int(r["bin"]) == (int(r["trialNum"]) - 1) // 8 + 1 for v in subs.values() for r in v)
                    and all(len({r["targAngle"] for r in v[i:i + 8]}) == 8 for v in subs.values() for i in range(0, 880, 8)))
    checks.internal(f"{exp}: condition constant within participant, 24 participants per condition",
                    all(len({r["condition"] for r in v}) == 1 for v in subs.values())
                    and set(Counter(v[0]["condition"] for v in subs.values()).values()) == {24})
    late_pos = [rmedian([fl(r["maxvelAngle"]) for r in v[680:760]]) > 0 for v in subs.values()]
    checks.internal(f"{exp}: every participant's late-clamp hand angle is positive although the clamp side was "
                    "counterbalanced (the authors stored the angles with adaptation positive)", all(late_pos))
    # the authors' trial exclusions: per participant and target, on the peak-velocity angle
    bad = {}
    for s, v in subs.items():
        for tg in sorted({r["targAngle"] for r in v}):
            idx = [i for i, r in enumerate(v) if r["targAngle"] == tg]
            for i, b in zip(idx, authors_bad([fl(v[i]["maxvelAngle"]) for i in idx])):
                bad[(s, i)] = b
    frac = sum(bad.values()) / len(bad)
    checks.internal(f"{exp}: the authors' outlier rule removes under 1% of trials (Methods)", frac < 0.01, f"{100 * frac:.2f}%")
    for s, v in subs.items():
        cond = v[0]["condition"]
        group = GROUP[exp][cond]
        size = SIZE_MM["Hit" if group.startswith("Hit") else "Miss"]
        lenient = group.endswith("Lenient")
        for j, r in enumerate(v):
            t = int(r["trialNum"])
            block, phase, fb = next((b, ph, f) for b, lo, hi, ph, f in BLOCKS12 if lo <= t <= hi)
            clamped = block == "clamp"
            ta = fl(r["targAngle"]) % 360.0
            tid = target_ids.setdefault((exp, round(ta, 6)), len([k for k in target_ids if k[0] == exp]))
            hand = r8(wrap(fl(r["maxvelAngle"])))
            ep = r8(wrap(fl(r["reachAngle"])))
            row = base_row(exp, s, group)
            row.update(trial=j, trial_orig=t, block=block, phase=phase, break_before=t in (1, 121, 761, 801),
                       target_angle=ta, target_id=tid,
                       target_size_deg=math.degrees(math.atan(size / 2.0 / RADIUS_MM)), target_event="none", target_jump=0,
                       instr_goal="hand_to_target",
                       instr_strategy="forbidden" if t > 120 else "unspecified",
                       instr_disclosure="exact" if t > 120 else "none",
                       hand_angle=hand, hand_angle_fb=ep, hand_measure="peak_velocity",
                       rt_ms=1000.0 * fl(r["rt"]), mt_ms=1000.0 * fl(r["mt"]),
                       excluded_by_authors=bad[(s, j)],
                       fb_visible="online" if fb else "none",
                       fb_mode=("clamped" if clamped else "veridical") if fb else "none",
                       fb_form="cursor" if fb else "none",
                       online_cursor=("perturbed" if clamped else "veridical") if fb else "none",
                       rotation=0 if fb and not clamped else "", clamp=-3.5 if clamped else "",
                       cursor_error=(-3.5 if clamped else ep) if fb else "",
                       fb_delay_ms=0 if fb else "", fb_duration_ms=50 if fb else "",
                       hit=(1 if group.startswith("Hit") else 0) if clamped else "")
            chan = ["cursor"] if fb else []
            if exp == "Exp2":
                chan.append("reward_tone")
                # chime = 1, knock = 0. Baseline and washout: the sound followed the hand (paper), the authors' hits flag;
                # clamp: it followed the clamped cursor, so Lenient groups always heard the chime and Strict groups the knock
                row.update(reward=(1 if lenient else 0) if clamped else int(r["hits"]),
                           reward_space="cursor" if clamped else "hand")
            row["outcome_channels"] = "|".join(chan)
            rows.append(row)
            meta_rows.append((exp, s, group))

# ---------------------------------------------------------------- Experiments 3 and 4
frac90 = {}
for exp in ("Exp3", "Exp4"):
    raw = data[exp]
    subs = OrderedDict()
    for r in raw:
        subs.setdefault(r["Subject"], []).append(r)
    n_sub, n_tr = {"Exp3": (18, 904), "Exp4": (42, 968)}[exp]
    checks.internal(f"{exp}: {n_sub} participants x {n_tr} trials in order: 100 baseline (TrialType 1, no clamp), "
                    "3 tutorial (TrialType 2, 45 deg clamp), then clamp trials (TrialType 3)",
                    len(subs) == n_sub and all([int(r["TrialNum"]) for r in v] == list(range(1, n_tr + 1)) for v in subs.values())
                    and all((r["TrialType"], r["ClampApplied"]) == (("1", "0") if int(r["TrialNum"]) <= 100 else
                                                                    ("2", "1") if int(r["TrialNum"]) <= 103 else ("3", "1"))
                            for v in subs.values() for r in v)
                    and all(abs(fl(r["ClampAngle"])) == 45 for v in subs.values() for r in v if r["TrialType"] == "2"))
    if exp == "Exp4":
        checks.internal("Exp4: HandAngle = 90 + HandError on every row (target at 90 deg)",
                        max(abs(wrap(fl(r["HandAngle"]) - 90.0 - fl(r["HandError"]))) for r in raw) < 1e-9)
        checks.internal("Exp4: Jump-To trials (TrialLabel) all have TargetJump = -ClampAngle, Hit and Miss trials no jump; "
                        "Hit trials have the large target (TgtSize 62), Miss and Jump-To the small one (6)",
                        all(fl(r["TargetJump"]) == -fl(r["ClampAngle"]) and r["TgtSize"] == "6" for r in raw if r["TrialLabel"] == "Jump To")
                        and all(fl(r["TargetJump"]) == 0 for r in raw if r["TrialLabel"] in ("Hit", "Miss"))
                        and all(r["TgtSize"] == ("62" if r["TrialLabel"] == "Hit" else "6") for r in raw if r["TrialLabel"] in ("Hit", "Miss")))
    else:
        checks.internal("Exp3: clamp trials have TargetJump = ClampAngle (Jump-To), -ClampAngle (Jump-Away), 0.01 "
                        "(Jump-In-Place) or 0, about a quarter each",
                        all(fl(r["TargetJump"]) in (fl(r["ClampAngle"]), -fl(r["ClampAngle"]), 0.01, 0.0) for r in raw if r["TrialType"] == "3"))
    checks.internal(f"{exp}: BadReachAngle (Exp3 only) marks |HandError| > 90 exactly",
                    exp == "Exp4" or all((r["BadReachAngle"] == "1") == (abs(fl(r["HandError"])) > 90) for r in raw))
    slopes_neg = 0
    for v in subs.values():
        x, y = [], []
        for a, b in zip(v[:-1], v[1:]):
            ha, hb = fl(a["HandError"]), fl(b["HandError"])
            if a["TrialType"] == b["TrialType"] == "3" and a["FalseStart"] == "0" and abs(ha) < 90 and abs(hb) < 90:
                x.append(fl(a["ClampAngle"]))
                y.append(fl(b["HandError"]) - fl(a["HandError"]))
        mx, my = st.mean(x), st.mean(y)
        slopes_neg += sum((p - mx) * (q - my) for p, q in zip(x, y)) / sum((p - mx) ** 2 for p in x) < 0
    checks.internal(f"{exp}: the next-trial change of the hand has a negative slope on ClampAngle in every participant "
                    "(clamp and hand in one frame)", slopes_neg == len(subs), f"{slopes_neg} of {len(subs)}")
    bad = {}
    for s, v in subs.items():
        idx = [i for i, r in enumerate(v) if int(r["TrialNum"]) > 103]
        flags = authors_bad([fl(v[i]["HandError"]) for i in idx], [v[i]["FalseStart"] == "1" for i in idx])
        for i, b in zip(idx, flags):
            bad[(s, i)] = b
    frac90[exp] = sum(1 for (s, i), b in bad.items() if b and subs[s][i]["FalseStart"] == "0") / len(bad)
    n_missing = sum(1 for r in raw if not math.isfinite(fl(r["HandError"])))
    checks.internal(f"{exp}: trials without a hand angle (HandError NaN) are kept with hand_angle empty",
                    n_missing < 0.01 * len(raw), f"{n_missing} of {len(raw)}")
    for s, v in subs.items():
        for j, r in enumerate(v):
            t = int(r["TrialNum"])
            tt = r["TrialType"]
            block, phase = {"1": ("baseline", "baseline"), "2": ("tutorial", "practice"), "3": ("clamp", "learning")}[tt]
            tj_raw = fl(r["TargetJump"])
            if tj_raw == 0.01:
                event, jump = "flicker", 0.0
            elif tj_raw != 0:
                event, jump = "jump", (tj_raw if exp == "Exp3" else -tj_raw)
            else:
                event, jump = "none", 0.0
            clamped = tt != "1"
            ca = fl(r["ClampAngle"])
            hand = r8(wrap(fl(r["HandError"])))
            label = r.get("TrialLabel", "")
            size_deg = ""
            if exp == "Exp4":
                size_deg = math.degrees(math.atan(E4_SIZE_MM[r["TgtSize"]] / 2.0 / RADIUS_MM))
            row = base_row(exp, s, "")
            row.update(trial=j, trial_orig=t, block=block, phase=phase, trial_kind="demo" if tt == "2" else "reach",
                       break_before=t in (1, 101, 104), target_angle=90, target_id=0, target_size_deg=size_deg,
                       target_event=event, target_jump=jump,
                       instr_goal="hand_to_target",
                       instr_strategy={"1": "unspecified", "2": "unspecified", "3": "forbidden"}[tt],
                       instr_disclosure="none" if tt == "1" else "exact",
                       hand_angle=hand, hand_angle_fb=hand, hand_measure="fixed_distance",
                       rt_ms=1000.0 * fl(r["RT"]), mt_ms=1000.0 * fl(r["MT"]),
                       excluded_by_authors=bad.get((s, j), False),
                       fb_visible="online", fb_mode="clamped" if clamped else "veridical", fb_form="cursor",
                       online_cursor="perturbed" if clamped else "veridical",
                       rotation="" if clamped else 0, clamp=ca if clamped else "",
                       cursor_error=ca if clamped else hand, fb_delay_ms=0, fb_duration_ms=50, outcome_channels="cursor",
                       hit={"Hit": 1, "Jump To": 1, "Miss": 0}.get(label, ""),
                       speed_warning=int(r["TooSlow"]))
            rows.append(row)
            meta_rows.append((exp, s, label))

checks.internal("every row the authors shared is kept, in order (42240 + 84480 + 16272 + 40656)",
                len(rows) == sum(len(v) for v in data.values()) == 183648)
checks.internal("hand_angle within (-180, 180] where recorded, target_angle within [0, 360), rt and mt recorded on the rows "
                "that have a hand angle",
                all(-180 < r["hand_angle"] <= 180 for r in rows if math.isfinite(r["hand_angle"]))
                and all(0 <= r["target_angle"] < 360 for r in rows)
                and all(math.isfinite(r["rt_ms"]) and math.isfinite(r["mt_ms"]) for r in rows if math.isfinite(r["hand_angle"])),
                f"{sum(not math.isfinite(r['hand_angle']) for r in rows)} rows without a hand angle")
checks.internal("clamp trials: cursor_error equals the clamp; veridical trials: cursor_error equals hand_angle_fb",
                all(r["cursor_error"] == r["clamp"] for r in rows if r["fb_mode"] == "clamped")
                and all(r["cursor_error"] == r["hand_angle_fb"] or not math.isfinite(r["hand_angle_fb"])
                        for r in rows if r["fb_mode"] == "veridical"))
persons = {(r["experiment"], r["subject"]) for r in rows}
checks.published("participants, all four experiments", len(persons), 204, 0, "Methods, Participants (204 participants)")
for exp, n in (("Exp1", 48), ("Exp2", 96), ("Exp3", 18), ("Exp4", 42)):
    checks.published(f"participants, {exp}", sum(1 for e, s in persons if e.endswith(exp)), n, 0, "Methods, Participants")
checks.published("Exp4: clamp trials per participant", sum(1 for r in rows if r["experiment"].endswith("Exp4") and r["block"] == "clamp") / 42,
                 865, 0, "Methods, Experiment 4 (865 clamp trials)")


# ---------------------------------------------------------------- published numbers, from the converted rows
def by_subject(exp):
    out = OrderedDict()
    for r, m in zip(rows, meta_rows):
        if m[0] == exp:
            out.setdefault(r["subject"], []).append((r, m))
    return out


def ttest_ind(a, b):
    na, nb = len(a), len(b)
    sp = ((na - 1) * st.variance(a) + (nb - 1) * st.variance(b)) / (na + nb - 2)
    return (st.mean(a) - st.mean(b)) / math.sqrt(sp * (1 / na + 1 / nb))


def ttest_rel(a, b):
    d = [x - y for x, y in zip(a, b)]
    return st.mean(d) / (st.stdev(d) / math.sqrt(len(d)))


def measures12(exp, base_hi):
    """Early learning rate, late asymptote and retention per participant, as in analysis_fig1/fig2 .Rmd: excluded trials
    removed, each target's baseline bias (feedback baseline, trials 41 to base_hi) subtracted, means per 8-trial bin."""
    out = {}
    for s, v in by_subject(exp).items():
        ang = [NAN if r["excluded_by_authors"] else r["hand_angle"] for r, m in v]
        bias = {}
        for tg in {r["target_angle"] for r, m in v}:
            bias[tg] = rmean([a for a, (r, m) in zip(ang, v) if r["target_angle"] == tg and 41 <= r["trial_orig"] <= base_hi])
        corr = [a - bias[r["target_angle"]] for a, (r, m) in zip(ang, v)]
        bins = {b: rmean(corr[8 * (b - 1):8 * b]) for b in range(1, 111)}
        early = st.mean(bins[b] for b in range(18, 23)) / 5.0
        asym = rmean([bins[b] for b in range(86, 96)])
        ret = bins[100] / bins[95]
        out[s] = (v[0][1][2], early, asym, ret)
    return out


F1 = "authors' file output_fig1Panel{}_meanSEM.csv (the data behind Fig 1{})"
m1 = measures12("Exp1", 119)  # Rmd fig1: trialNum > 40 & trialNum < 120
g1 = {g: [x for x in m1.values() if x[0] == g] for g in ("Hit_16mm", "Miss_6mm")}
for g, key, vals in (("Hit_16mm", "Hit", (1.39194411106015, 17.2401428035099, 0.704447840901554)),
                     ("Miss_6mm", "Miss", (1.34409876032794, 18.8675527323169, 0.811783564963534))):
    for k, (what, panel) in enumerate((("early learning rate (deg/cycle)", "B_earlyLR"), ("asymptote, last 10 clamp cycles (deg)", "C_asym1"),
                                       ("retention, last no-feedback washout cycle / last clamp cycle", "D_wash"))):
        checks.published(f"Exp1 {key}: {what}, mean", st.mean(x[1 + k] for x in g1[g]), vals[k], 1e-6,
                         F1.format(panel, panel[0]))
checks.published("Exp1 Hit: SD of the asymptote", st.stdev(x[2] for x in g1["Hit_16mm"]), 7.46, 0.005, "Results, Experiment 1 (17.24 +- 7.46 deg)")
checks.published("Exp1 Miss: SD of the asymptote", st.stdev(x[2] for x in g1["Miss_6mm"]), 9.33, 0.005, "Results, Experiment 1 (18.88 +- 9.33 deg)")
for k, what, rep in ((1, "early learning", -0.30), (2, "asymptote", 0.67), (3, "retention", 0.85)):
    checks.published(f"Exp1: t(46), Miss vs Hit, {what}", ttest_ind([x[k] for x in g1["Miss_6mm"]], [x[k] for x in g1["Hit_16mm"]]),
                     rep, 0.005, "Results, Experiment 1")

m2 = measures12("Exp2", 120)  # Rmd fig2: trialNum > 40 & trialNum < 121
F2 = "authors' file output_fig2_{}_meanSEM.csv (the data behind Fig 2)"
rep2 = {"Hit_16mm_Lenient": (1.37469160385848, 17.0844960534743, 0.492924438662217),
        "Hit_16mm_Strict": (1.2258995026358, 17.9826266062891, 0.550580637923302),
        "Miss_6mm_Lenient": (1.20002569712323, 19.6214929123232, 0.442113565190106),
        "Miss_6mm_Strict": (1.43292512743835, 19.1375364523072, 0.756610423852361)}
for g, vals in rep2.items():
    xs = [x for x in m2.values() if x[0] == g]
    for k, (what, f) in enumerate((("early learning rate", "earlyLR"), ("asymptote", "asym1"), ("retention", "wash"))):
        checks.published(f"Exp2 {g}: {what}, mean", st.mean(x[1 + k] for x in xs), vals[k], 1e-6, F2.format(f))
strict = [x[3] for x in m2.values() if x[0].endswith("Strict")]
lenient = [x[3] for x in m2.values() if x[0].endswith("Lenient")]
checks.published("Exp2: retention, Strict mean", st.mean(strict), 0.65, 0.005, "Results, Experiment 2 (0.65 +- 0.08)")
checks.published("Exp2: retention, Lenient mean", st.mean(lenient), 0.47, 0.005, "Results, Experiment 2 (0.47 +- 0.03)")
checks.published("Exp2: t(94), retention, Strict vs Lenient", ttest_ind(strict, lenient), 2.23, 0.005, "Results, Experiment 2")


def stl_table(exp, kind_of):
    """Single-trial learning as in analysis_fig3/fig4 .Rmd: on clamp trials, the next trial's hand angle minus this one's
    (excluded trials removed; none after the last trial), median per participant, clamp and trial kind."""
    med = {}
    for s, v in by_subject(exp).items():
        cl = [(r, m) for r, m in v if r["block"] == "clamp"]
        ang = [NAN if r["excluded_by_authors"] else r["hand_angle"] for r, m in cl]
        stl = [ang[i + 1] - ang[i] for i in range(len(ang) - 1)] + [NAN]
        cells = {}
        for x, (r, m) in zip(stl, cl):
            cells.setdefault((r["clamp"], kind_of(r, m)), []).append(x)
        med[s] = {k: rmedian(x) for k, x in cells.items()}
    return med


def kind3(r, m):
    if r["target_event"] == "flicker":
        return "Jump In Place"
    if r["target_event"] == "jump":
        return "Jump To" if r["target_jump"] == r["clamp"] else "Jump Away" if r["target_jump"] == -r["clamp"] else "?"
    return "No Jump"


med3 = stl_table("Exp3", kind3)
J3 = ("Jump Away", "No Jump", "Jump In Place", "Jump To")
ign_jump = {s: {c: rmean([d.get((c, j), NAN) for j in J3]) for c in (-4.0, 4.0)} for s, d in med3.items()}
ign_dir = {s: {j: rmean([-d.get((4.0, j), NAN), d.get((-4.0, j), NAN)]) for j in J3} for s, d in med3.items()}
checks.internal("Exp3: every clamp trial is one of the four jump kinds", all(k[1] != "?" for d in med3.values() for k in d))
FB3 = "authors' file output_fig3Panel{}_STLXJump_meanSEM.csv (the data behind Fig 3{})"
for c, rep in ((-4.0, 1.26291992433214), (4.0, -1.18308933771036)):
    checks.published(f"Exp3: single-trial learning after a {c:+g} deg clamp, mean", st.mean(x[c] for x in ign_jump.values()), rep, 1e-6, FB3.format("B", "B"))
for j, rep in (("Jump Away", 1.57329747516538), ("Jump In Place", 1.35813500741364), ("Jump To", 0.457261805567801), ("No Jump", 1.50332423593818)):
    checks.published(f"Exp3: single-trial learning, {j}, mean", st.mean(x[j] for x in ign_dir.values()), rep, 1e-6, FB3.format("C", "C"))
t_dir = ttest_rel([x[4.0] for x in ign_jump.values()], [x[-4.0] for x in ign_jump.values()])
checks.published("Exp3: F(1,17) for clamp direction (squared paired t)", t_dir ** 2, 94.7, 0.05, "Results, Experiment 3")
for a, b, rep in (("No Jump", "Jump To", 3.36), ("Jump Away", "Jump To", 2.89), ("Jump In Place", "Jump To", 3.12), ("No Jump", "Jump In Place", 1.96)):
    # the last one is 1.9651 here and with the authors' script (whose means match their figure files exactly): printed 1.96
    checks.published(f"Exp3: t(17), {a} vs {b}", ttest_rel([x[a] for x in ign_dir.values()], [x[b] for x in ign_dir.values()]), rep,
                     0.006 if b == "Jump In Place" else 0.005, "Results, Experiment 3")


def kind4(r, m):
    k = "Jump To" if r["target_event"] == "jump" else "Hit" if r["target_size_deg"] > 5 else "Miss"
    return k


checks.internal("Exp4: the trial kind read back from target size and jump equals the authors' TrialLabel on every clamp trial",
                all(kind4(r, m) == m[2] for r, m in zip(rows, meta_rows) if m[0] == "Exp4" and r["block"] == "clamp"))
med4 = stl_table("Exp4", kind4)
CL4 = (1.75, 3.5, 5.25, 7.0, 8.75, 10.5)
S4 = ("Miss", "Hit", "Jump To")
FB4 = "authors' file output_fig4Panel{} (the data behind Fig 4{})"
for c, rep in zip([-x for x in CL4[::-1]] + list(CL4),
                  (1.44688853634613, 1.44273923718018, 1.29086398856185, 1.17444827986219, 0.769204662823823, 0.419615849917394,
                   -0.538318969499486, -0.776525192244187, -1.00862955296447, -1.19216355786108, -1.30373475057967, -1.44487120860929)):
    checks.published(f"Exp4: single-trial learning after a {c:+g} deg clamp, mean",
                     st.mean(rmean([d.get((c, k), NAN) for k in S4]) for d in med4.values()), rep, 1e-6, FB4.format("A_STLXClamp_meanSEM.csv", "A"))
ign4 = {s: {(k, c): rmean([-d.get((c, k), NAN), d.get((-c, k), NAN)]) for k in S4 for c in CL4} for s, d in med4.items()}
for k, rep in (("Hit", 1.07350415577691), ("Jump To", 0.825325445727916), ("Miss", 1.3031713451076)):
    checks.published(f"Exp4: single-trial learning, {k} trials, mean", st.mean(x[(k, c)] for x in ign4.values() for c in CL4), rep, 1e-6,
                     FB4.format("B_STLXSuccess_meanSEM.csv", "B"))
rep4c = {"Hit": (0.433187030349509, 0.746595486272525, 1.06977784867074, 1.24944573312396, 1.46979201072519, 1.47222682551956),
         "Jump To": (0.375337090592546, 0.522109771404975, 0.771179999797034, 1.01254212264108, 1.10068292088824, 1.17010076904362),
         "Miss": (0.628378108183265, 1.04988952492451, 1.43365890077221, 1.46255346386934, 1.54923605002634, 1.69531202286995)}
for k, reps in rep4c.items():
    for c, rep in zip(CL4, reps):
        checks.published(f"Exp4: single-trial learning, {k} trials, {c:g} deg clamp, mean", st.mean(x[(k, c)] for x in ign4.values()), rep, 1e-6,
                         FB4.format("C_STLXSuccessClamp_meanSEM.csv", "C"))
per = {k: [rmean([x[(k, c)] for c in CL4]) for x in ign4.values()] for k in S4}
for a, b, rep in (("Miss", "Hit", 4.04), ("Miss", "Jump To", 5.36), ("Hit", "Jump To", 2.79)):
    checks.published(f"Exp4: t(41), {a} vs {b}", ttest_rel(per[a], per[b]), rep, 0.005, "Results, Experiment 4")

# ---------------------------------------------------------------- release
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Taylor (Princeton)",
    data_url="https://github.com/kimoli/TaskSuccessImplicitAdapt",
    license="none stated",
    license_url="",
    changes="converted to the OpenMotor format and participant codes replaced; Experiment 4's target jump sign reversed "
            "to match its clamp (see How).",
    unit_note="trial of one participant sequence",
    how=[
        "One row is one reach, all 183,648 the authors shared (204 participants): Experiments 1 and 2 have 880 reaches to 8 "
        "targets (5 cycles without a cursor, 10 with, 80 with a 3.5 deg error clamp, 5 without and 10 with a cursor), "
        "Experiments 3 and 4 have 100 baseline reaches, 3 tutorial reaches with a 45 deg clamp and 801 or 865 clamp reaches "
        "to one target at 90 deg.",
        "Experiments 3 and 4 are counter-clockwise positive (hand and clamp share the authors' frame; Experiment 4's TargetJump "
        "has the opposite sense to its clamp, so target_jump = -TargetJump there). In Experiments 1 and 2 the clamp side was "
        "counterbalanced but is not in the files, and the authors stored hand angles with adaptation positive: they are kept "
        "as stored, the clamp is written as -3.5 deg in that frame for everyone, and for roughly half of those participants "
        "(not identified) the frame is a mirror image of the screen.",
        "hand_angle is the authors' measure: the angle at peak velocity in Experiments 1-2 (hand_angle_fb is the endpoint "
        "angle), at the target distance in Experiments 3-4; excluded_by_authors marks the authors' trial exclusions as their "
        "Rmd scripts compute them (3 SD from their running mean, beyond 90 deg, false starts in Experiments 3-4).",
        "In Experiment 2, reward is the chime (1) or knock (0): on baseline and washout trials the authors' hits flag (the hand "
        "within the group's sound zone), on clamp trials the group's sound (Lenient chime, Strict knock). Target sizes in "
        "Experiment 4 follow the paper (31 and 4.5 mm; the file codes them 62 and 6); Experiment 3 gives none.",
    ],
    differences=[
        "Experiment 1, Miss group: the paper prints an asymptote of 18.88 deg; the rows and the authors' own Fig 1C file give "
        "18.8676 (the SD, 9.33, matches).",
        "Experiment 3 has 801 clamp trials per participant in the file; the paper says 800.",
        "Experiment 3, No Jump vs Jump-In-Place: t(17) = 1.9651 from the rows (with the authors' exclusions, whose condition "
        "means match their Fig 3 files exactly); the paper prints 1.96.",
        f"Experiments 3 and 4: the authors' 3 SD / 90 deg rule removes {100 * frac90['Exp3']:.1f}% and {100 * frac90['Exp4']:.1f}% "
        "of clamp trials as their scripts compute it; the paper says under 1% of trials.",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks)
