"""Lazzari et al. (2024), Brain Sci: finger tapping with delayed auditory feedback in people who stutter, one row per tap or click.

Run from the site folder: python3 -I tools/std/convert/Lazzari2024_BrainSci.py
Raw files (untrusted data, read only) in raw_extra/Lazzari2024_BrainSci/, from https://osf.io/gb5j4/:
  Tapping Data.zip           one text file per trial from the authors' tapping pad (Teensy 3.2 + force-sensitive resistor):
                             a "# config" line (AF = auditory feedback on, DELAY = feedback delay in ms, INTVL = metronome
                             interval, NCLICK_PREDELAY / NCLICK / NCONT = clicks before the delay, clicks, continuation taps) and
                             lines "message_number type onset_t offset_t max_force_t max_force n_missed_frames" with type click
                             (metronome), tap, or feedback (the sound a tap triggered); times are ms on the device clock.
                             Folders: SCT with DAF (48 trials per participant, file suffix _tap or _sound = the instruction),
                             SCT without feedback (16 trials), Spontaneus tapping (1 trial), Delay detection task - sound.
  spontaneus_results.xlsx    the authors' per-participant spontaneous tapping results (read for the checks only)
  README.docx, Stutterers_analyses.R   read as text (never run)
Not converted: the two delay detection tasks (a verbal first/second judgement per trial, which the tapping format has no column
for), the circular-statistics spreadsheets (derived values) and the speech syllable spreadsheet.
"""
import json
import re
import statistics as st
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Lazzari2024_BrainSci"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for n in ("Tapping Data.zip", "spontaneus_results.xlsx"):
    checks.internal(f"{n} matches the downloaded hash", stdlib.sha256(RAW / n) == src[n]["sha256"])
z = zipfile.ZipFile(RAW / "Tapping Data.zip")

EXP = {"SCT with DAF": "SCT with delayed auditory feedback", "SCT without feedback": "SCT without feedback",
       "Spontaneus tapping": "Spontaneous tapping"}
DEVICE = ("force-sensitive resistor tapping pad (Interlink FSR, Teensy 3.2), noise-reduction earphones, 30-ms woodblock "
          "click as metronome and as feedback sound")

# ---------------------------------------------------------------- read the trial files
trials = defaultdict(list)  # participant -> [(date, time, folder, instruction, config, events)]
n_files = Counter()
for i in z.infolist():
    if i.is_dir():
        continue
    parts = i.filename.split("/")
    folder, name = parts[1], parts[-1]
    if folder not in EXP:
        continue
    m = re.fullmatch(r"([CS]\d\d)_(\d{8})_(\d{6})(?:_(tap|sound))?\.txt", name)
    assert m, name
    pid, date, tm, instr = m.groups()
    cfg, ev = None, []
    for l in z.read(i).decode("utf-8", "replace").splitlines():
        if l.startswith("# config"):
            cfg = dict(kv.split("=") for kv in l[len("# config "):].split())
            cfg = {k: int(v) for k, v in cfg.items()}
        else:
            p = l.split()
            if len(p) == 7 and p[0].isdigit():
                assert p[1] in ("click", "tap", "feedback"), p
                ev.append(dict(kind=p[1], on=int(p[2]), off=None if p[3] == "NA" else int(p[3]),
                               force=None if p[5] == "NA" else int(p[5]), miss=int(p[6])))
    assert cfg is not None and ev, name
    trials[pid].append((date, tm, folder, instr, cfg, ev))
    n_files[folder] += 1
pids = sorted(trials, key=lambda s: (s[0] == "S", s))   # controls first, then people who stutter (README order)
checks.internal("files read: 24 participants, 1152 + 384 + 24 trial files", len(pids) == 24
                and n_files == Counter({"SCT with DAF": 1152, "SCT without feedback": 384, "Spontaneus tapping": 24}), str(dict(n_files)))

# ---------------------------------------------------------------- rows
rows = []
bad = Counter()
spont_pct = {}
spont_fb = Counter()
click_ioi = []
trial_cfg = defaultdict(Counter)
n_missed = 0
n_nofb = 0
orphan_fb = 0
for pid in pids:
    grp = "PWS" if pid[0] == "S" else "PNS"
    dates = sorted({t[0] for t in trials[pid]})
    row_no = 0
    for k, (date, tm, folder, instr, cfg, ev) in enumerate(sorted(trials[pid], key=lambda t: (t[0], t[1])), 1):
        clicks = sorted(e["on"] for e in ev if e["kind"] == "click")
        taps = sorted((e for e in ev if e["kind"] == "tap"), key=lambda e: e["on"])
        fbs = Counter(e["on"] for e in ev if e["kind"] == "feedback")
        n_missed += sum(e["miss"] for e in ev)
        # a tap's feedback sound is logged exactly 0 or DELAY ms after the tap (taps are in time order); some taps have none
        if cfg["AF"] == 0 and fbs:
            bad["feedback in a trial without feedback"] += 1
        for t in taps:
            for dl in ((cfg["DELAY"], 0) if cfg["DELAY"] else (0,)):
                if fbs[t["on"] + dl] > 0:
                    fbs[t["on"] + dl] -= 1
                    t["fb"] = t["on"] + dl
                    break
        orphan_fb += sum(fbs.values())
        n_nofb += sum(1 for t in taps if "fb" not in t) if cfg["AF"] == 1 else 0
        delays = [t["fb"] - t["on"] for t in taps if "fb" in t]
        if cfg["AF"] == 1 and cfg["DELAY"] > 0 and not (delays and delays[0] == 0 and delays[-1] == cfg["DELAY"]
                                                         and delays == sorted(delays)):
            bad["delayed trial does not start undelayed and end delayed"] += 1
        click_ioi += [b - a for a, b in zip(clicks, clicks[1:])]
        trial_cfg[folder][(cfg["AF"], cfg["DELAY"], cfg["INTVL"], cfg["NCLICK_PREDELAY"] + cfg["NCLICK"], instr)] += 1
        # spontaneous tapping: the authors drop inter-tap intervals under 100 ms or over 1.5 x the trial's median
        itis = [b["on"] - a["on"] for a, b in zip(taps, taps[1:])]
        excl_tap = set()
        if folder == "Spontaneus tapping":
            med = st.median(itis)
            excl_tap = {j + 1 for j, x in enumerate(itis) if x < 100 or x > 1.5 * med}
            spont_pct[pid] = 100 * len(excl_tap) / len(itis)
        events = [("c", c, None) for c in clicks] + [("t", t["on"], t) for t in taps]
        events.sort(key=lambda e: (e[1], e[0]))  # time order; a click first when it coincides with a tap
        tap_i = click_i = 0
        prev_tap = prev_click = None
        cond = ("spontaneous tapping" if folder == "Spontaneus tapping" else "no feedback" if folder == "SCT without feedback"
                else f"{instr} instruction, feedback delay {cfg['DELAY']} ms")
        for pos, (kind, t, tap) in enumerate(events, 1):
            row_no += 1
            fbk = "none" if cfg["AF"] == 0 else "normal" if cfg["DELAY"] == 0 else "altered"
            row = dict(dataset=ID, study=ID, experiment=EXP[folder], group=grp, subject=pid, subject_excluded_by_authors=False,
                       age="", sex="", handedness="right", population="people who stutter" if grp == "PWS" else "adults who do not stutter",
                       setting="lab", device=DEVICE, session=dates.index(date) + 1, block=instr or "", trial=row_no, phase="baseline"
                       if cfg["DELAY"] == 0 or folder != "SCT with DAF" else "test", time_s="", task="finger/hand", unit="tap",
                       condition=cond, pert_type="none", pert_value=0, pert_unit="", feedback=fbk, excluded_by_authors=False,
                       tapping_trial=k, event_in_trial=pos,
                       stimulus=(f"clicks logged by the device, {cfg['INTVL']} ms apart" if folder == "Spontaneus tapping"
                                 else f"metronome {cfg['INTVL']} ms"))
            if kind == "c":
                row.update(stimulus_time_ms=t, ioi_ms=None if prev_click is None else t - prev_click,
                           stimulus_played=None if folder == "Spontaneus tapping" else True)
                prev_click = t
            else:
                d = tap.get("fb", None)
                dl = None if d is None else d - t
                if dl:
                    row.update(pert_type="delay", pert_value=dl, pert_unit="ms")
                row.update(feedback="none" if d is None else "altered" if dl else "normal", tap_time_ms=t,
                           tap_offset_ms=tap["off"], tap_force=tap["force"], feedback_time_ms=d,
                           iti_ms=None if prev_tap is None else t - prev_tap, excluded_by_authors=tap_i in excl_tap)
                prev_tap = t
                tap_i += 1
            rows.append(row)
checks.internal("every logged feedback sound belongs to a tap 0 or the configured delay earlier (delayed trials: undelayed "
                "taps first, then the delay); %d taps of the feedback trials have no logged feedback sound, and %d logged feedback "
                "sound has no tap record" % (n_nofb, orphan_fb), not bad and orphan_fb == 1, str(dict(bad)))
checks.internal("every metronome interval is the configured 600 ms (clicks of every trial)",
                set(click_ioi) <= set(range(595, 606)) and abs(st.median(click_ioi) - 600) < 1,
                f"min {min(click_ioi)}, median {st.median(click_ioi)}, max {max(click_ioi)} ms over {len(click_ioi)} intervals")
checks.internal("no frames were reported missed by the pad", n_missed == 0, f"{n_missed} missed frames")

# ---------------------------------------------------------------- published numbers
SRC_M = "Methods (Participants; Tasks)"
tr_of = defaultdict(set)
for r in rows:
    tr_of[(r["subject"], r["experiment"])].add(r["tapping_trial"])
checks.published("participants who stutter", sum(1 for p in pids if p[0] == "S"), 11, 0, "Results / Methods, Participants")
checks.published("participants who do not stutter", sum(1 for p in pids if p[0] == "C"), 13, 0, "Results / Methods, Participants")
sct = {p: len(tr_of[(p, "SCT without feedback")]) for p in pids}
checks.published("SCT without feedback: trials per participant (all 24 have 16)", min(sct.values()) if min(sct.values()) == max(sct.values()) else -1,
                 16, 0, "Methods, Synchronization-continuation task, 16 trials")
daf = {p: len(tr_of[(p, "SCT with delayed auditory feedback")]) for p in pids}
checks.published("SCT with delayed auditory feedback: trials per participant (all 24 have 48 = 2 instructions x 24)",
                 min(daf.values()) if min(daf.values()) == max(daf.values()) else -1, 48, 0,
                 "Methods, SCT with DAF: 24 trials per instruction")
per_delay = Counter()
for p in pids:
    for (date, tm, folder, instr, cfg, ev) in trials[p]:
        if folder == "SCT with DAF":
            per_delay[(instr, cfg["DELAY"])] += 1
checks.published("trials per instruction at 0 ms delay, summed over the 24 participants (8 per participant)",
                 per_delay[("tap", 0)], 8 * 24, 0, "Methods, SCT with DAF (8 trials at 0 ms)")
checks.published("trials per instruction at each other delay (50, 100, 200, 300 ms), summed over participants (4 per participant)",
                 min(per_delay[(i, d)] for i in ("tap", "sound") for d in (50, 100, 200, 300)), 4 * 24, 0,
                 "Methods, SCT with DAF (4 trials per delay)")
checks.internal("every instruction x delay cell has the paper's count",
                all(per_delay[(i, d)] == (192 if d == 0 else 96) for i in ("tap", "sound") for d in (0, 50, 100, 200, 300)),
                str(dict(per_delay)))
checks.published("metronome clicks per SCT trial (24)", Counter(len([1 for e in ev if e["kind"] == "click"])
                                                              for p in pids for (_, _, folder, _, _, ev) in trials[p]
                                                              if folder != "Spontaneus tapping").most_common(1)[0][0], 24, 0,
                 "Methods, SCT (24 metronome clicks per trial)")
checks.published("spontaneous tapping: percentage of inter-tap intervals removed (under 100 ms or over 1.5 x the trial median; "
                 "mean over participants)", st.mean(spont_pct.values()), 5.93, 0.005, "Methods, Data analysis (5.93% of ITIs)")
# the authors' spreadsheet gives each participant's percentage
import openpyxl  # noqa: E402
ws = openpyxl.load_workbook(RAW / "spontaneus_results.xlsx", read_only=True).worksheets[0]
xr = list(ws.iter_rows(values_only=True))
auth = {r[4]: dict(zip(xr[0], r)) for r in xr[1:]}
dmax = max(abs(spont_pct[p] - auth[p]["percentage_excluded"]) for p in pids)
checks.internal("per participant, the share of spontaneous inter-tap intervals flagged excluded equals the authors' "
                "percentage_excluded (spontaneus_results.xlsx)", set(auth) == set(pids) and dmax < 1e-6, f"max difference {dmax:.2e}")
# group means of the spontaneous tapping interval, from the rows (the authors' per-participant values are not reproduced)
by_p = defaultdict(list)
for r in rows:
    if r["experiment"] == "Spontaneous tapping" and r.get("iti_ms") is not None and not r["excluded_by_authors"]:
        by_p[r["subject"]].append(r["iti_ms"])
g_mean = {g: st.mean(st.mean(by_p[p]) for p in pids if p[0] == g) for g in "SC"}
for c in checks.items:
    print(f"  [{'ok' if c['ok'] else 'FAILED'}] {c['what']} " + (f"computed {c['computed']:.4f} reported {c['reported']}" if c["kind"] == "published" else c.get("detail", "")))
print("  group mean ITI from rows (not a check): PWS %.2f, PNS %.2f (paper 616.05, 597.47)" % (g_mean["S"], g_mean["C"]))

meta = dict(
    study="Lazzari et al. 2024",
    paper=dict(authors="Lazzari G, van de Vorst R, van Vugt FT, Lega C", year=2024,
               title="Subtle Patterns of Altered Responsiveness to Delayed Auditory Feedback during Finger Tapping in People Who Stutter",
               venue="Brain Sciences 14(5):472", doi="10.3390/brainsci14050472"),
    lab="Lega (University of Pavia) / van Vugt (Universite de Montreal)",
    data_url="https://osf.io/gb5j4/",
    license="none stated", license_url="",
    changes=("Converted to the OpenMotor format with participant codes replaced: every trial file of the synchronization-"
             "continuation tasks (with and without delayed auditory feedback) and of the spontaneous tapping task became one "
             "row per tap and one row per metronome click, in time order, with each tap's feedback sound time added from the "
             "same file. The file names' dates and times were used only to order the trials. The two delay detection tasks, "
             "the circular-statistics spreadsheets and the syllable spreadsheet were not converted."),
    how=[
        "One row is one tap on the force-sensitive pad (tap_time_ms, the release time tap_offset_ms and the peak force "
        "tap_force as the pad logged them) or one metronome click (stimulus_time_ms, 600 ms apart, stimulus_played true), "
        "with times in ms on the device's running clock, which is not reset between trials. Trials are the synchronization-"
        "continuation task without feedback (16 per participant, 24 clicks), the same task with the tap's sound delayed by 0, "
        "50, 100, 200 or 300 ms under the instruction to synchronize the taps or the delayed sound with the metronome (48 per "
        "participant, block tap or sound), and one spontaneous tapping trial. The spontaneous trial's file also logs 24 clicks "
        "600 ms apart at its start, which the paper's description of unpaced tapping does not mention and the deposit does "
        "not explain; they are kept as stimulus rows with stimulus_played empty.",
        "In delayed-feedback trials the first 8 taps are undelayed and the later ones carry the delay: pert_value is each "
        "tap's own logged delay (feedback_time_ms minus tap_time_ms), so a '300 ms' trial starts with rows of delay 0; the "
        "few taps with no logged feedback sound (218 of the feedback trials' taps) have feedback none and no delay. "
        "Participants tap from the fifth click on and keep tapping after the metronome stops, so many taps have no click "
        "and the first clicks have no tap; asynchrony_ms is empty because the authors analyse each tap's circular phase "
        "within the 600 ms cycle rather than pairing taps with clicks.",
        "excluded_by_authors marks, in the spontaneous trial, taps that end an inter-tap interval under 100 ms or over 1.5 "
        "times the trial's median, the rule the authors state (it reproduces their percentage_excluded for every "
        "participant); group is PWS (people who stutter) or PNS (people who do not stutter), as the deposit's participant codes mark. Age and sex are not in the deposit.",
    ],
    unit_note="tap, or metronome click that received no tap, in a tapping trial",
    differences=["Spontaneous tapping group means do not reproduce from the rows: the mean inter-tap interval is %.2f ms for PWS and "
                 "%.2f ms for PNS here against 616.05 and 597.47 in the paper, because the authors' spreadsheet values (which give "
                 "the paper's numbers) differ from a recomputation from the tap times: dropping the last tap and using population "
                 "SDs reproduces the authors' mean for 10 of the 24 participants, not for the others." % (g_mean["S"], g_mean["C"])],
)
stdlib.release(ID, "tapping", rows, meta, checks)
