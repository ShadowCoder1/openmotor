"""Franke, Falk, Benker & Hoole (2025), Journal of Phonetics 112:101432: inter-gestural coupling of the onset consonant and the vowel
(CV-lag) and predictive timing in 10 adults who stutter (PWS) and 10 matched adults who do not (PWNS), reading German words with
bilabial onsets in a carrier phrase while unpaced, tapping a finger, speaking to a metronome, or both; electromagnetic articulography
(Carstens AG501, 1250 Hz).

Raw files, raw_extra/Franke2025_JPhonetics/ (OSF tvwx6, fetched with tools/std/fetch.py; no license stated on the OSF project):
  cv_data_all_final.csv          2,749 tokens (one row each): Subject, trialnumber, the lip-aperture (LipAp) gesture landmarks in seconds from the
        start of the token (gesture onset, maximum velocity, nucleus onset, maximum constriction onset, nucleus offset, maximum velocity
        of the opening, gesture offset), Label (carrier phrase, word, condition U/T/M/B, repetition, word-list number; '!!' marks
        trials the authors exclude), word, condition, rep
  CV_acoustics_all_final.csv     2,744 tokens: acoustic landmarks and durations (C1_dur consonant, V_dur vowel, CV_dur) in seconds
  worddur_all.csv                word durations (s) from the aligned TextGrids, keyed by Subject and trialnumber
  CV_lag_stats_final.csv         1,815 analysed tokens (vowels /o:/ and /u:/ only, no '!!' trials): word_dur and seven CV-lag variants in s
        (tongue-back landmark minus lip-aperture landmark); the one the paper uses is CV_lag_TBACK_nucleusOnset_LipAp_nucleusOnset
  syncdata_10pairs_stats.csv     2,086 tokens of the tapping, metronome and combined conditions: articulator_time (= LipAp nucleus onset),
        tap_time, met_time and the offsets tap_offset, met_offset (time of tap or beat minus articulator_time, s)
  participant_metadata.csv       one row per participant (semicolon-separated, Latin-1); only sex, age, handedness, stuttering onset age and
        the stuttering severity score are used (free-text fields and the data-use agreement fields are not)
  *.R, *.Rmd                     the authors' preparation and analysis code, read as text
Not used: cv_kin_stats_final.csv and LipAp_TBack_all_final.csv (positions at the landmarks, mm), gamm_data_all_cv.csv (time-normalized
tongue-back trajectories, 53 MB).
Paper: Franke et al. 2025 (open access, CC BY 4.0 at epub.ub.uni-muenchen.de/128843), sections 2.1-2.6 and 3.1-3.2, Tables 1 and 3.
Run from the site folder: python3 -I tools/std/convert/Franke2025_JPhonetics.py
"""
import csv
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Franke2025_JPhonetics"
SRC = "Franke et al. 2025, Journal of Phonetics"
RAW = stdlib.raw_dir(ID)
PAPER = "Franke et al. 2025"
DEVICE = "Carstens AG501 electromagnetic articulograph (1250 Hz), Sennheiser super-cardioid microphone (25.6 kHz)"
COND = {"U": "Unpaced", "T": "Tapping", "M": "Metronome", "B": "Metronome+Tapping"}
WORD = {"ma:l": "Mahl", "ma:s": "Ma\u00df", "ba:dn": "Baden", "mo:s": "Moos", "mo:n": "Mohn", "bo:dn": "Boden", "mu:s": "Mus",
        "bu:dn": "Buden", "bu:n@": "Buhne"}  # paper, Table 2 (the data code, as the authors label it, to the German word)
SSI_PAPER = {"S01": 22, "S02": 16, "S03": 31, "S04": 25, "S05": 14, "S06": 19, "S07": 5, "S08": 5, "S09": 37, "S11": 31}  # Table 1 (S11 = the paper's S10)
LAGS = ["CV_lag_TBACK_maxcon_LipAp_gestureOnset", "CV_lag_TBACK_maxcon_LipAp_maxvelOnset", "CV_lag_TBACK_maxcon_LipAp_maxconOnset",
        "CV_lag_TBACK_maxcon_LipAp_nucleusOnset", "CV_lag_TBACK_gestureOnset_LipAp_gestureOnset",
        "CV_lag_TBACK_nucleusOnset_LipAp_nucleusOnset", "CV_lag_TBACK_maxvelOnset_LipAp_maxvelOnset"]
norm = lambda s: re.sub(r"\s+", "", str(s))


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def ms(v):
    return "" if pd.isna(v) else round(float(v) * 1000, 6)


def main():
    ck = stdlib.Checks()
    cv = pd.read_csv(RAW / "cv_data_all_final.csv")
    ac = pd.read_csv(RAW / "CV_acoustics_all_final.csv")
    wd = pd.read_csv(RAW / "worddur_all.csv").dropna(subset=["trialnumber"])
    lag = pd.read_csv(RAW / "CV_lag_stats_final.csv")
    sy = pd.read_csv(RAW / "syncdata_10pairs_stats.csv")
    md = pd.read_csv(io.StringIO((RAW / "participant_metadata.csv").read_bytes().decode("latin-1")), sep=";", keep_default_na=False)
    md.columns = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in md.columns]
    meta_by = {r["Subject"]: r for r in md.to_dict("records")}
    cv["L"] = cv.Label.map(norm)
    lag["L"] = lag.Label.map(norm)
    wd["trialnumber"] = wd.trialnumber.astype(int)
    ack = {(r.Subject, r.trialnumber): r for r in ac.itertuples(index=False)}
    wdk = {(r.Subject, r.trialnumber): r.word_dur for r in wd.itertuples(index=False)}
    syk = {(r.Subject, r.trial_number): r for r in sy.itertuples(index=False)}
    # CV-lag rows by (participant, normalized label): attached only where the token is unique on both sides
    lag_by = {}
    for i, r in enumerate(lag.itertuples(index=False)):
        lag_by.setdefault((r.Subject, r.L), []).append(i)
    cv_key_n = cv.groupby(["Subject", "L"]).size().to_dict()
    cv["trial_rank"] = cv.groupby("Subject").trialnumber.rank(method="first").astype(int)

    rows, helper, attached = [], [], set()
    for r in cv.itertuples(index=False):
        sub = r.Subject
        m = meta_by[sub]
        pws = sub.startswith("S")
        a, sync = ack.get((sub, r.trialnumber)), syk.get((sub, r.trialnumber))
        wdur = wdk.get((sub, r.trialnumber))
        li = lag_by.get((sub, r.L), [])
        lrow = lag.iloc[li[0]] if len(li) == 1 and cv_key_n[(sub, r.L)] == 1 else None
        if lrow is not None:
            attached.add(li[0])
        code = r.word.strip("_")
        target = r.poi == "V" and code in WORD
        blk = re.search(r"_B(\d+)\s*$", r.Label)
        flagged = "!!" in r.Label
        rows.append(dict(
            dataset=ID, study=SRC, experiment="", group="PWS" if pws else "PWNS", subject=sub, person=sub,
            subject_excluded_by_authors=False, age=float(m["Age"]), sex="M" if m["Sex"] == "m" else "F", handedness=m["Handedness"],
            population="adults who stutter" if pws else "healthy", setting="lab", device=DEVICE, session=1,
            block=f"word list {int(blk.group(1))}" if blk else "", trial=int(r.trial_rank), phase="other", time_s="", task="speech",
            unit="utterance", condition=COND[r.condition], pert_type="none", pert_value=0, pert_unit="", feedback="normal",
            excluded_by_authors=bool(flagged or not target),
            stimulus=f"{WORD[code]} [{code}]" if code in WORD else code, target_sound=code[0], trial_number_authors=int(r.trialnumber),
            recording_name=r.Label.strip(), gesture_signal="LA", gesture_order=1, gesture_segment=code[0],
            gons_ms=ms(r.LipAp_gesture_on), pvel_ms=ms(r.LipAp_maxvel_on), nons_ms=ms(r.LipAp_nucleus_on), maxc_ms=ms(r.LipAp_maxcon_on),
            noffs_ms=ms(r.LipAp_nucleus_off), pvel2_ms=ms(r.LipAp_maxvel_off), goffs_ms=ms(r.LipAp_gesture_off),
            plateau_duration_ms=ms(r.LipAp_nucleus_dur),
            vowel_duration_ms=ms(a.V_dur) if a is not None else "", consonant_duration_ms=ms(a.C1_dur) if a is not None else "",
            word_duration_ms=ms(wdur) if wdur is not None else "",
            cv_lag_ms=ms(lrow["CV_lag_TBACK_nucleusOnset_LipAp_nucleusOnset"]) if lrow is not None else "",
            tap_offset_ms=ms(sync.tap_offset) if sync is not None else "", metronome_offset_ms=ms(sync.met_offset) if sync is not None else ""))
        helper.append(dict(sub=sub, grp="PWS" if pws else "PWNS", cond=COND[r.condition], code=code, in_lag=lrow is not None, flagged=flagged,
                           target=target, rep=r.rep, trialnumber=r.trialnumber,
                           sync_ok=None if sync is None else bool(abs(sync.articulator_time - r.LipAp_nucleus_on) < 1e-9
                                                                  if not pd.isna(r.LipAp_nucleus_on) else True),
                           tapok=None if sync is None or pd.isna(sync.tap_time) else bool(abs(sync.tap_offset - (sync.tap_time - sync.articulator_time)) < 1e-9),
                           plateau=r.LipAp_nucleus_off - r.LipAp_nucleus_on, word_s=wdur if wdur is not None else np.nan))
    h = pd.DataFrame(helper)
    df = pd.DataFrame(rows)

    # participants table (person = the converter's code; release() assigns the public codes in the same first-appearance order)
    people = list(dict.fromkeys(df.person))
    pcode = {p: f"{ID}_P{i + 1:04d}" for i, p in enumerate(people)}
    parts = []
    for p in people:
        m = meta_by[p]
        onset = pd.to_numeric(m["Stuttering_start_age"], errors="coerce")
        parts.append(dict(person=p, ssi_score=pd.to_numeric(m["SSI_score"], errors="coerce"), ssi_severity=m["SSI_StutteringSeverity"].replace("_", " "),
                          stuttering_onset_age=onset))
    pcols = [("ssi_score", "number", "", "Stuttering Severity Instrument (SSI-4) score of an adult who stutters, as in the paper's Table 1; empty for controls."),
             ("ssi_severity", "text", "", "SSI-4 severity category (very mild, mild, moderate, very severe) of an adult who stutters; empty for controls."),
             ("stuttering_onset_age", "number", "years", "Age in years at which the participant's stuttering started, as reported; empty for controls.")]

    # the authors' lag table as shared, with participant codes (rows not attached to a token of the main table included)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(["person", "group", "condition", "recording_name", "word", "vowel", "word_duration_s", "mean_duration_3words_s"] + [c.lower() + "_s" for c in LAGS])
    for r in lag.itertuples(index=False):
        row = [pcode[r.Subject], "PWS" if r.group == "S" else "PWNS", COND[r.condition], r.Label.strip(), r.word, r.vowel,
               stdlib._fmt(float(r.word_dur), "number"), stdlib._fmt(float(r.meanDur_3words), "number")]
        row += [stdlib._fmt(float(getattr(r, c)), "number") if not pd.isna(getattr(r, c)) else "" for c in LAGS]
        w.writerow(row)
    extra = [("cv_lag_stats.csv", buf.getvalue().encode())]

    # ---- internal checks ---------------------------------------------------------------------------------------------------
    ck.internal("every token of the landmark table kept, in the file's order", len(df) == len(cv) == 2749, f"{len(df)} rows")
    ck.internal("20 participants, none lost", df.person.nunique() == 20)
    ck.internal("trial runs 1..n in each participant", all(sorted(g.trial) == list(range(1, len(g) + 1)) for _, g in df.groupby("person")))
    ck.internal("sync table: the articulator time is the lip-aperture nucleus onset on every joined token", bool(h.sync_ok.dropna().all()),
                f"{int(h.sync_ok.notna().sum())} tokens")
    ck.internal("sync table: tap offset = tap time minus articulator time on every token with a tap", bool(h.tapok.dropna().all()))
    ck.internal("landmark order gons <= pvel <= nons <= maxc <= noffs <= pvel2 <= goffs on at least 99.5% of tokens",
                float((df[["gons_ms", "pvel_ms", "nons_ms", "maxc_ms", "noffs_ms", "pvel2_ms", "goffs_ms"]].replace("", np.nan).astype(float).diff(axis=1)
                       .iloc[:, 1:] >= 0).all(axis=1).mean()) > 0.995)
    n_att = len(attached)
    ck.internal("CV-lag rows attached to a token of the landmark table: all but a few that cannot be told apart", 1800 <= n_att <= 1815,
                f"{n_att} of {len(lag)}")

    # ---- published numbers -------------------------------------------------------------------------------------------------
    per = df.drop_duplicates("person")
    ck.published("participants", len(per), 20, 0, f"{PAPER}, 2.1 Participants")
    for g in ("PWS", "PWNS"):
        d = per[per.group == g]
        ck.published(f"{g}: participants", len(d), 10, 0, f"{PAPER}, 2.1 Participants")
        ck.published(f"{g}: mean age (years)", d.age.mean(), 23.1, 0.005, f"{PAPER}, 2.1 Participants")
        ck.published(f"{g}: SD of age (years)", d.age.std(), {"PWS": 3.18, "PWNS": 4.04}[g], 0.005, f"{PAPER}, 2.1 Participants")
        ck.published(f"{g}: youngest (years)", d.age.min(), {"PWS": 20, "PWNS": 19}[g], 0, f"{PAPER}, 2.1 Participants (range)")
        ck.published(f"{g}: oldest (years)", d.age.max(), {"PWS": 30, "PWNS": 32}[g], 0, f"{PAPER}, 2.1 Participants (range)")
        ck.published(f"{g}: males", int((d.sex == "M").sum()), 5, 0, f"{PAPER}, 2.1 Participants")
        ck.published(f"{g}: left-handed participants", int((d.handedness == "left").sum()), 2, 0, f"{PAPER}, 2.1 Participants")
    on = pd.Series([p["stuttering_onset_age"] for p in parts]).dropna()
    ck.published("PWS: mean age of stuttering onset (years)", on.mean(), 6, 0.005, f"{PAPER}, 2.1 Participants")
    ck.published("PWS: SD of age of stuttering onset (years)", on.std(), 2.67, 0.005, f"{PAPER}, 2.1 Participants")
    for pid, v in SSI_PAPER.items():
        pp = next(p for p in parts if p["person"] == pid)
        ck.published(f"SSI-4 score of {'S10' if pid == 'S11' else pid}", pp["ssi_score"], v, 0, f"{PAPER}, Table 1")
    ck.published("different target words", h[h.target].code.nunique(), 9, 0, f"{PAPER}, 2.2 Speech material, Table 2")
    ck.published("most target words produced by one participant (4 repetitions x 4 conditions x 9 words)", h[h.target].groupby("sub").size().max(), 144, 0,
                 f"{PAPER}, 2.3 Procedure (a maximum of 144 target words)")
    an = h[h.in_lag]
    T3 = {("Unpaced", "PWNS"): (0.52, 0.09), ("Unpaced", "PWS"): (0.56, 0.14), ("Tapping", "PWNS"): (0.52, 0.07), ("Tapping", "PWS"): (0.60, 0.10),
          ("Metronome", "PWNS"): (0.60, 0.09), ("Metronome", "PWS"): (0.60, 0.10), ("Metronome+Tapping", "PWNS"): (0.60, 0.10),
          ("Metronome+Tapping", "PWS"): (0.62, 0.09)}
    for (c, g), (mean_r, sd_r) in T3.items():
        x = an[(an.cond == c) & (an.grp == g)].word_s
        ck.published(f"target word duration, {c}, {g}: mean (s)", x.mean(), mean_r, 0.005, f"{PAPER}, Table 3")
        ck.published(f"target word duration, {c}, {g}: SD (s)", x.std(), sd_r, 0.005, f"{PAPER}, Table 3")
    for g, r_ in (("PWNS", 0.074), ("PWS", 0.073)):
        ck.published(f"bilabial closure duration (nucleus offset minus onset), {g}: mean (s)", an[an.grp == g].plateau.mean(), r_, 0.0005,
                     f"{PAPER}, 3.1 (footnote 5)")
    for x in ck.items:
        if not x["ok"]:
            print("FAILED:", x)
    print(f"rows {len(df)}, persons {len(per)}, flagged {int(h.flagged.sum())}, non-target {int((~h.target).sum())}, lag attached {n_att} of {len(lag)}")

    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Joined the authors' tables on participant and trial number (landmarks cv_data_all_final.csv, acoustic durations, word "
                 "durations, tap and metronome offsets) and, through the recording label, the CV-lag; times and durations in ms (they "
                 "were in seconds); sex, age and handedness from the participant table; participant codes instead of the authors' "
                 "participant numbers; the authors' lag table also as cv_lag_stats.csv (seconds)."),
        how=[
            "One row is one spoken target word (an utterance) of one participant, in the order of the authors' landmark table "
            "cv_data_all_final.csv: German words with a bilabial onset ([m] or [b]) and the vowel /a:/, /o:/ or /u:/, read in a carrier "
            "phrase in four pacing conditions (Unpaced, Tapping, Metronome, Metronome+Tapping). The row holds the lip-aperture "
            "gesture of the onset consonant (gesture_signal LA): its landmark times in ms from the start of the token, with "
            "plateau_duration_ms the authors' bilabial closure (nucleus offset minus nucleus onset).",
            "cv_lag_ms is the authors' CV-lag, the time from the nucleus onset of the lip-aperture gesture to the nucleus onset of the "
            "tongue-back vowel gesture, shared for the vowels /o:/ and /u:/ only; word_duration_ms is the target word duration; "
            "vowel_duration_ms and consonant_duration_ms are acoustic durations; tap_offset_ms and metronome_offset_ms are the "
            "asynchronies of the finger tap and of the closest metronome beat from the lip-aperture nucleus onset (positive when "
            "the tap or beat came after it). The tongue-back landmark times themselves are not in the deposit.",
            "excluded_by_authors marks the tokens the authors leave out: those with a '!!' flag in the recording label (stuttering, "
            "prosody, a slip of the tongue and the like) and two tokens that are not target words. The authors' analyses also use "
            "only /o:/ and /u:/ words for CV-lag; the tokens with a cv_lag_ms value are their analysed set. Not included: the "
            "tongue-back and lip positions at the landmarks and the time-normalized tongue-back trajectories.",
            "participants.csv has the stuttering severity (SSI-4) and onset age; sex, age and handedness are in the main table. "
            "The authors' own lag table is in cv_lag_stats.csv with the participant codes (seven CV-lag variants and word durations, "
            "in seconds).",
        ],
        unit_note="spoken target word (utterance) of one participant, with its lip-aperture gesture",
        differences=[
            "The paper excludes 94 trials (80 in PWS, 14 in PWNS) before analysis; the landmark table flags only "
            f"{int(h.flagged.sum())} tokens with '!!' and holds {len(df)} of the at most 2,880 tokens (20 participants x 144), so "
            "the other exclusions were made before the tables were built and cannot be reproduced here.",
            f"{len(lag) - n_att} of the 1,815 rows of the authors' CV-lag table could not be attached to a single token of the landmark table (the "
            "recording label is cut or shared by two tokens); they are in cv_lag_stats.csv.",
            "The paper says each target word was repeated four times per condition; one token of the landmark table has repetition 5.",
        ],
    )
    if not ck.passed():
        sys.exit("checks failed")
    return stdlib.release(ID, "articulography", rows, meta, ck, participants=parts, participant_columns=pcols, extra_files=extra)


if __name__ == "__main__":
    main()
