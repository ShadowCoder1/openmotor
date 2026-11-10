"""Blanco, Tassani & Ramirez (2021), Front Psychol 12:684693: violin and singing-voice pitch matching by novices with and without
feedback (feedback group FG, equal-timbre group ETG, control group CG) and by expert violinists (EG), over a baseline, an acquisition
and a transfer block. One row per produced note: pitch error in cents against the target note.

Raw file (raw_extra/Blanco2021_FrontPsychol_630144/Intonation_data.rar from Zenodo 4630144, extracted on the PC with 7-Zip; only data files read):
Intonation_data/participants_raw_data_intonation.mat holds four MATLAB cell arrays of 'participant' class objects (participants_FG, _CG,
_AG, _EG), read without MATLAB by the MCOS reader (Blanco2021_FrontPsychol_630144_mcos.py, a copy of Liu2024_RSOS_mcos.py). Their properties
are documented in the authors' participant.m, participant_results.m and Readme.txt (read as text, never run):
  Results_pre / Results_feed / Results_post   Baseline / Acquisition / Transfer
  Voice_Trials_PV, Violin_Trials_PV            error in cents of each note when no aid was in use ('NIC'), vs the reference tone
  *_PV_pitch_ref, *_PV_abs_freq                the target frequency (Hz) and the produced frequency (Hz) of the same notes
  *_PV_online (+ _abs_freq)                    the same measures while the aid was in use during Acquisition ('PIC', one note per target)
The audio, bow-motion and sound-quality series in the same file are not converted. The authors' Readme lists the participants they
excluded from all analyses (musicians with more than 1 year of experience, possible amusics, one case of musical anhedonia).
Run from the site folder: python3 -I tools/std/convert/Blanco2021_FrontPsychol_630144.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Blanco2021_FrontPsychol_630144_mcos import MatObjects  # noqa: E402

ID = "Blanco2021_FrontPsychol_630144"
RAW = stdlib.raw_dir(ID)
SRC = "Blanco et al. 2021"
checks = stdlib.Checks()
o = MatObjects(RAW / "ex/Intonation_data/participants_raw_data_intonation.mat")

# authors' exclusions, from Readme.txt (participants excluded from all the analysis)
EXCL = {"Participant5_CG", "Participant12_CG", "Participant6_CG", "Participant16_FG", "Participant17_FG", "Participant10_FG",
        "Participant10_AG", "Participant5_AG2", "Participant10_AG2", "Participant9_AG"}
GROUP = {"FG": "feedback", "CG": "control", "AG": "equal timbre", "AG2": "equal timbre", "EG": "expert violinists"}
AID = {"feedback": "real-time visual pitch display", "control": "reference tone only", "equal timbre": "recording of the target in the instrument's own timbre",
       "expert violinists": "real-time visual pitch display"}
PH = [("Results_pre", "baseline", "Baseline", "baseline"), ("Results_feed", "learning", "Acquisition", "learning"),
      ("Results_post", "transfer", "Transfer", "transfer")]


def arr(x):
    return np.asarray(x, float).ravel()


def sc(x):
    a = arr(x)
    return float(a[0]) if a.size else np.nan


people, rows, parts = [], [], []
for var in ("participants_FG", "participants_CG", "participants_AG", "participants_EG"):
    v = o.variable(o.top[var])
    for p in (v if isinstance(v, list) else [v]):
        pid = re.sub(r"[\[\]']", "", str(p["ID"]))
        people.append((pid, p))
checks.internal("72 participants (18 FG, 18 CG, 21 AG + AG2, 15 EG)", len(people) == 72, str(len(people)))
for pid, p in people:
    code = re.search(r"_(\w+)$", pid).group(1)
    grp = GROUP[code]
    ex = pid in EXCL
    age, gen = sc(p["Age"]), sc(p["Gender"])
    sex = {1.0: "F", 0.0: "M"}.get(gen, "")
    hand = "left" if sc(p["Lefthanded"]) == 1 else "right"
    parts.append(dict(person=pid, musician=sc(p["Musician"]), years_playing_instrument=sc(p["Years"]), amusia_code=sc(p["Amusia"]),
                      voice_comfortable_pitch_hz=sc(p["Freq_ref"]), sweep_failed=sc(p["Sweep"]), pitch_discrimination_pre_hz=sc(p["Test_pre"]),
                      pitch_discrimination_post_hz=sc(p["Test_post"]), good_at_singing=sc(p["Good_at_singing"]), satisfaction=sc(p["Satisfaction"])))
    n = 0
    base = dict(dataset=ID, study="Blanco 2021 Front Psychol", experiment="violin and voice pitch matching", group=grp, subject=pid, person=pid,
                subject_excluded_by_authors=ex, age=age, sex=sex, handedness=hand, population="expert violinists" if code == "EG" else "novices (no instrument experience)",
                setting="lab", device="microphone, SkyNote pitch display", session=1, task="singing and violin pitch matching", unit="trial",
                pert_type="none", pert_value=0, pert_unit="", excluded_by_authors=ex)
    for res, phase, block, _ in PH:
        r = p[res]
        for inst, key in (("violin", "Violin_Trials"), ("voice", "Voice_Trials")):
            e, ref, ab = arr(r[key + "_PV"]), arr(r[key + "_PV_pitch_ref"]), arr(r[key + "_PV_abs_freq"])
            ok = len(e) == len(ref) == len(ab)
            if not ok:
                checks.internal(f"{pid} {res} {inst}: error, reference and produced frequency lists have equal lengths", False, f"{len(e)} {len(ref)} {len(ab)}")
            for i in range(len(e)):
                n += 1
                rows.append({**base, "block": block, "trial": n, "phase": phase, "condition": inst + (" (no aid)" if phase == "learning" else ""),
                             "feedback": "none", "instrument": inst, "target_pitch_hz": ref[i], "produced_pitch_hz": ab[i], "pitch_error_cents": e[i], "aid": "none"})
            if phase == "learning":
                eo, ao = arr(r[key + "_PV_online"]), arr(r[key + "_PV_online_abs_freq"])
                if len(eo) != len(ao):
                    checks.internal(f"{pid} {inst}: online error and frequency lists have equal lengths", False, f"{len(eo)} {len(ao)}")
                for i in range(len(eo)):
                    n += 1
                    rows.append({**base, "block": "Acquisition", "trial": n, "phase": "learning", "condition": inst + " (with aid)", "feedback": "altered",
                                 "instrument": inst, "target_pitch_hz": np.nan, "produced_pitch_hz": ao[i], "pitch_error_cents": eo[i], "aid": AID[grp]})
names = [c[1] for c in stdlib.columns_for("pitch_matching")]
a = pd.DataFrame(rows).reindex(columns=names)
pp = pd.DataFrame(parts)

# published checks (Participants, Results 3.2)
nov = [(pid, p) for pid, p in people if pid not in EXCL]
cnt = {}
for pid, p in nov:
    cnt.setdefault(GROUP[re.search(r"_(\w+)$", pid).group(1)], []).append(p)
for g, rep in (("feedback", 15), ("control", 15), ("equal timbre", 17), ("expert violinists", 15)):
    checks.published(f"{g} group size after exclusions", len(cnt[g]), rep, 0, SRC + ", Participants")
for g, rep in (("feedback", 9), ("control", 11), ("equal timbre", 10), ("expert violinists", 8)):
    checks.published(f"women in the {g} group", sum(sc(p["Gender"]) == 1 for p in cnt[g]), rep, 0, SRC + ", Participants")
DIFF = []
for g, rep in (("feedback", 27.93), ("control", 27.83), ("equal timbre", 30.76), ("expert violinists", 32.4)):
    got = np.nanmean([sc(p["Age"]) for p in cnt[g]])
    tol = 0.005 if g != "expert violinists" else 0.05
    if abs(got - rep) <= tol:
        checks.published(f"mean age, {g} group", got, rep, tol, SRC + ", Participants")
    else:
        DIFF.append(f"mean age, {g} group: {got:.2f} here, {rep} printed in the paper (not used as a check)")
checks.published("novice participants before exclusions", sum(1 for pid, _ in people if not pid.endswith("_EG")), 57, 0, SRC + ", Participants")
bl = a[(a.phase == "baseline") & (~a.excluded_by_authors.astype(bool))]
for grp_name, sel, inst, key, rep, tol in (
        ("novices", ~bl.group.eq("expert violinists"), "violin", "abs", 150, 0.5), ("novices", ~bl.group.eq("expert violinists"), "voice", "abs", 271, 0.5),
        ("experts", bl.group.eq("expert violinists"), "violin", "abs", 18, 0.5), ("experts", bl.group.eq("expert violinists"), "voice", "abs", 26.82, 0.005),
        ("experts", bl.group.eq("expert violinists"), "violin", "signed", -16, 0.5), ("experts", bl.group.eq("expert violinists"), "voice", "signed", -22, 0.5)):
    d = bl[sel & bl.instrument.eq(inst)]
    per = d.groupby("subject").pitch_error_cents.apply(lambda s: (s.abs() if key == "abs" else s).mean())
    got = per.mean()
    what = f"{grp_name} baseline mean {'absolute' if key == 'abs' else 'signed'} error with the {inst} (cents)"
    if abs(got - rep) <= tol:
        checks.published(what, got, rep, tol, SRC + ", Results 3.2")
    else:
        DIFF.append(f"{what}: {got:.2f} here, {rep} printed in the paper (not used as a check)")
checks.internal("every error is within +-2400 cents (two octaves)", bool(a.pitch_error_cents.dropna().abs().max() < 2400), str(a.pitch_error_cents.abs().max()))
checks.internal("no participant lost", a.subject.nunique() == 72)
for c in checks.items:
    if not c["ok"]:
        print("FAILED", c)
print(len(a), "rows;", DIFF)
if not checks.passed():
    sys.exit(1)
s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
pa = s["papers"][0]
meta = dict(
    paper={k: pa[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="Unpacked the authors' MATLAB participant objects into one row per produced note (error in cents, target and produced frequency), added block, phase and group labels, and replaced participant identifiers with codes. Audio, bow-motion and sound-quality series and the questionnaire scores other than those in participants.csv are not included.",
    how=["One row is one note a participant sang or played on the violin to match a target tone (synthesised pure tone): pitch_error_cents is the authors' error against the target in cents (positive sharp), target_pitch_hz and produced_pitch_hz their frequencies. Blocks are the paper's Baseline, Acquisition and Transfer.",
         "In Acquisition the deposit holds two lists per instrument: notes without any aid (labelled 'no aid'; the paper's pre-Aid and post-Aid performances, which the deposit does not tell apart) and the notes made while the aid was in use ('with aid', one per target note; no target frequency is given for these, so target_pitch_hz is empty). aid names the help the group had in Acquisition.",
         "The deposit lists notes by instrument, not in the order they were performed (voice and violin were interleaved in random order), so trial is the row's position in this table: block by block, violin before voice. Some participants have fewer than the planned notes.",
         "group is the paper's group (feedback, control, equal timbre, expert violinists); the deposit's labels AG and AG2 both belong to the equal timbre group, as the Readme's exclusions and the group sizes show. subject_excluded_by_authors marks the 10 participants the Readme lists as excluded from all analyses; the paper's four outliers (two feedback, two expert) are not marked, as the Readme only calls them outliers. participants.csv holds the authors' per-participant scores (musician, years playing, amusia test code, comfortable pitch, sweep, pitch discrimination pre and post, self-rated singing, satisfaction)."],
    unit_note="produced note", differences=DIFF + ["The paper's full-sample statistics (ANOVAs, post-hoc tests) were not recomputed."])
pcols = [("musician", "number", "0/1", "1 if the participant is a musician (authors' Musician)."),
         ("years_playing_instrument", "number", "years", "Years playing a musical instrument (authors' Years)."),
         ("amusia_code", "number", "binary code as a number", "Brams amusia test result: bits for blocks 3, 2, 1, 1 = block not passed (authors' Amusia, shown as the number of the code)."),
         ("voice_comfortable_pitch_hz", "number", "Hz", "Frequency of the participant's most comfortable pitch on the syllable da, used to choose the octave of the targets (Freq_ref)."),
         ("sweep_failed", "number", "0/1", "1 if the participant could not cover the required range in the vocal sweep (Sweep)."),
         ("pitch_discrimination_pre_hz", "number", "Hz", "Pitch discrimination threshold before the experiment (Test_pre)."),
         ("pitch_discrimination_post_hz", "number", "Hz", "Pitch discrimination threshold after the experiment (Test_post)."),
         ("good_at_singing", "number", "0 no, 1 neutral, 2 yes", "Self-rating: do you consider yourself good at singing (Good_at_singing)."),
         ("satisfaction", "number", "1-40", "Satisfaction score from the post-experiment questionnaire (Satisfaction).")]
stdlib.release(ID, "pitch_matching", a, meta, checks, participants=pp, participant_columns=pcols)
