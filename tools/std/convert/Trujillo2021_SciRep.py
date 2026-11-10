"""Trujillo, Ozyurek, Holler & Drijvers (2021), Sci Rep: speakers exhibit a multimodal Lombard effect in noise. Pairs of
Dutch speakers played a communication game; the producer (the one recorded with a Kinect) got 20 action verbs across to
a partner while hearing no noise, 4-talker or 8-talker babble, which changed from round to round.

Raw files, raw_extra/Trujillo2021_SciRep/ (OSF b2zq8, fetched with tools/std/fetch.py):
  Lowlands_Production_full.txt  sbj (pair), Trial (round 1-20), then A1_*/A2_* = attempt 1 / attempt 2 hand kinematics
                                (Rmax, Lmax, VelAvg, Handed, Rsubs, Lsubs, HoldCount, Holdtime, Dist, VertAmp, PeakR,
                                PeakL, duration, Gesture, Speech, multimod) and Verb; 1180 rows = 59 pairs x 20 rounds
  face_data_full.txt            sbj, Trial, then <measure>_A1 / <measure>_A2: mouth, brow, lip and speech measures
  noise_data_full.txt           PAIR, PPN (participant 1 or 2 of the pair), then the noise level (0 clear, 4 or 8 talkers)
                                of rounds 0-20; the authors' analysis uses PPN 1 (R script, lines 22-31)
  modality.csv                  pp (pair), t (round), G, S, M and Modality: 1 speech only, 2 gesture only, 3 both
                                (hand-checked by the authors)
What the values are: authors' MATLAB (KFE_analyze_Lombard.m, read as text: attempts are the first two tries of a round;
Rmax = greatest hand distance from its starting position, VelAvg = mean hand speed, Rsubs = peaks in hand speed after
Meyer 1988, Holdtime = time the hands were still, Dist = path of both hands, VertAmp = 0-5 height scale) and the paper
(Methods, Data processing; the Kinect and Praat measures). The face and voice measures are summarised elsewhere than in the
shared scripts, so their names and the paper's description are the documentation.
Run from the site folder: python3 -I tools/std/convert/Trujillo2021_SciRep.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Trujillo2021_SciRep"
SRC = "Trujillo et al. 2021, Sci Rep 11:16721"
KIN = [("Rmax", "reach_right_max"), ("Lmax", "reach_left_max"), ("VelAvg", "hand_speed_mean"), ("Handed", "hand_used"),
       ("Rsubs", "submovements_right"), ("Lsubs", "submovements_left"), ("HoldCount", "hold_count"),
       ("Holdtime", "hold_time"), ("Dist", "hand_path_total"), ("VertAmp", "vertical_amplitude"),
       ("PeakR", "peak_speed_right"), ("PeakL", "peak_speed_left"), ("duration", "attempt_duration_s"),
       ("Gesture", "gesture_flag"), ("Speech", "speech_flag"), ("multimod", "multimodal_flag")]
FACE = [("MeanMouthHeight", "mouth_height_mean"), ("MaxMouthHeight", "mouth_height_max"),
        ("MeanMouthArea", "mouth_area_mean"), ("MaxMouthArea", "mouth_area_max"), ("BrowNum", "brow_movements"),
        ("BrowDur", "brow_duration"), ("MeanLipMove", "lip_movement_mean"), ("TotalLipMovePS", "lip_movement_total_per_s"),
        ("MeanLipVel_Low", "lip_speed_mean_lower"), ("MeanLipVel_Up", "lip_speed_mean_upper"),
        ("PeakLipVel_Low", "lip_speed_peak_lower"), ("PeakLipVel_Up", "lip_speed_peak_upper"),
        ("MeanSpeechAmp", "speech_intensity_mean_db"), ("MaxSpeechAmp", "speech_intensity_max_db"),
        ("SpeechF0", "speech_f0_max_hz")]
INT_COLS = {"hand_used", "submovements_right", "submovements_left", "hold_count", "vertical_amplitude", "brow_movements"}
FLAG_COLS = {"gesture_flag", "speech_flag", "multimodal_flag"}
MODALITY = {1: "speech", 2: "gesture", 3: "speech+gesture"}
CONDITION = {0: "clear (no noise)", 4: "4-talker babble", 8: "8-talker babble"}
DEVICE = "Kinect markerless depth camera, microphone, headphones playing babble noise"
EXCLUDED_PAIR = 85  # the authors drop it: a missing button press makes its noise levels unknown (R script, line 50)


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def nz(v):
    return None if pd.isna(v) else v


def build(checks):
    raw = stdlib.raw_dir(ID)
    k = pd.read_csv(raw / "Lowlands_Production_full.txt", sep="\t")
    f = pd.read_csv(raw / "face_data_full.txt", sep="\t")
    n = pd.read_csv(raw / "noise_data_full.txt", sep="\t")
    m = pd.read_csv(raw / "modality.csv", sep=";")
    cols = [str(i) for i in range(21)]
    noise = {}
    for r in n[n.PPN == 1].to_dict("records"):
        for t in range(1, 21):
            noise[(int(r["PAIR"]), t)] = r[str(t)]
    mod = {(int(r.pp), int(r.t)): r for r in m.itertuples(index=False)}
    kk = {(int(r["sbj"]), int(r["Trial"])): r for r in k.to_dict("records")}
    ff = {(int(r["sbj"]), int(r["Trial"])): r for r in f.to_dict("records")}
    keys = sorted(set(kk) | set(ff) | set(mod))
    # consistency of the hand-checked modality codes
    bad = [(r.G, r.S, r.M, r.Modality) for r in m.itertuples(index=False)
           if not pd.isna(r.Modality) and ((r.G, r.S) != {1: (0, 1), 2: (1, 0), 3: (1, 1)}[int(r.Modality)])]
    checks.internal("modality file: Modality 1/2/3 = speech only / gesture only / both, as its G and S columns say",
                    not bad, f"{len(bad)} inconsistent rows")
    rows = []
    for key in keys:
        pair, rnd = key
        kr, fr, mr = kk.get(key), ff.get(key), mod.get(key)
        for att in (1, 2):
            vals = {}
            if kr is not None:
                for a, c in KIN:
                    vals[c] = nz(kr.get(f"A{att}_{a}"))
            if fr is not None:
                for a, c in FACE:
                    vals[c] = nz(fr.get(f"{a}_A{att}"))
            has = any(v is not None for v in vals.values())
            if att == 2 and not has:
                continue
            if att == 1 and not has and kr is None and mr is None and fr is None:
                continue
            for c in FLAG_COLS:
                if vals.get(c) is not None:
                    vals[c] = bool(vals[c])
            for c in INT_COLS:
                if vals.get(c) is not None:
                    vals[c] = int(vals[c])
            modality = ""
            if att == 1 and mr is not None and not pd.isna(mr.Modality):
                modality = MODALITY[int(mr.Modality)]
            nv = noise.get(key)
            nv = None if nv is None or pd.isna(nv) else int(nv)
            excl = pair == EXCLUDED_PAIR or nv is None or mr is None or pd.isna(mr.Modality)
            row = {
                "dataset": ID, "study": ID, "experiment": "Lowlands communication game", "group": "",
                "subject": f"P{pair}", "subject_excluded_by_authors": pair == EXCLUDED_PAIR, "age": "", "sex": "",
                "handedness": "", "population": "healthy", "setting": "lab", "device": DEVICE, "session": 1,
                "block": f"round {rnd}", "trial": 0, "phase": "test", "time_s": "", "task": "speech",
                "unit": "trial", "condition": CONDITION.get(nv, ""),
                "pert_type": "none" if nv == 0 else ("babble_noise" if nv else ""),
                "pert_value": nv if nv is not None else "", "pert_unit": "talkers" if nv is not None else "",
                "feedback": "" if nv is None else ("normal" if nv == 0 else "masked"), "excluded_by_authors": bool(excl),
                "stimulus": nz(kr.get("Verb")) if kr is not None else "", "round": rnd, "attempt": att,
                "modality": modality,
            }
            for _, c in KIN + FACE:
                row[c] = vals.get(c, "")
                if row[c] is None:
                    row[c] = ""
            rows.append(row)
    seq = {}
    for r in rows:
        seq[r["subject"]] = seq.get(r["subject"], 0) + 1
        r["trial"] = seq[r["subject"]]
    return rows, noise


def checks_all(a, checks, noise):
    k = pd.read_csv(stdlib.raw_dir(ID) / "Lowlands_Production_full.txt", sep="\t")
    n2 = int(k.A2_duration.notna().sum())
    n1 = int(k.A1_duration.notna().sum())
    kin = a[a.attempt.eq(1) & a.attempt_duration_s.notna()]
    checks.internal("attempt 1 rows with kinematics = the authors' A1 rows with a duration (1159), attempt 2 rows = A2 (519)",
                    len(kin) == n1 == 1159 and int(a[a.attempt.eq(2) & a.attempt_duration_s.notna()].shape[0]) == n2 == 519,
                    f"{len(kin)} and {n2}")
    checks.internal("every one of the 1180 production rows (59 pairs x 20 rounds) has an attempt 1 row",
                    int(a[a.subject.isin(set(a.subject))].drop_duplicates(["subject", "round"]).shape[0]) >= 1180
                    and set(zip(k.sbj, k.Trial)) <= set(zip(a.subject.str[1:].astype(int), a["round"])))
    checks.internal("subject x round x attempt is unique", not a.duplicated(["subject", "round", "attempt"]).any())
    checks.internal("noise levels are 0, 4 or 8 talkers", set(a.pert_value.dropna().unique()) <= {0, 4, 8})
    checks.internal("vertical amplitude within 0-5 and rounds within 1-20",
                    bool(a.vertical_amplitude.dropna().between(0, 5).all() and a["round"].between(1, 20).all()))
    for c, lo, hi in [("speech_intensity_max_db", 50, 110), ("speech_f0_max_hz", 40, 2000), ("attempt_duration_s", 0.3, 15)]:
        v = a[c].dropna().astype(float)
        checks.internal(f"{c} within {lo}-{hi}", bool(v.between(lo, hi).all()), f"{v.min():.3g}-{v.max():.3g}")
    # --- published
    kin_pairs = a[a.attempt_duration_s.notna() & (a.subject != f"P{EXCLUDED_PAIR}")].subject.nunique()
    checks.published("participants with kinematic data after the excluded pair", kin_pairs, 58, 0,
                     f"{SRC}, Methods (Participants: final sample of 58 native Dutch speakers)")
    import statsmodels.formula.api as smf
    s = a[(~a.excluded_by_authors) & a.modality.isin(["speech", "speech+gesture"]) & a.speech_intensity_max_db.notna()].copy()
    s["nf"] = s.pert_value.astype(int).astype(str)
    s["one"] = 1
    s["Verb"] = s.stimulus
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mm = smf.mixedlm("speech_intensity_max_db ~ C(nf)", s, groups=s["one"],
                         vc_formula={"pair": "0+C(subject)", "verb": "0+C(Verb)"}).fit(reml=True)
    est, se = float(mm.params["C(nf)[T.8]"]), float(mm.bse["C(nf)[T.8]"])
    checks.published("speech intensity, 8-talker minus clear, mixed model with random pair and verb intercepts (dB)",
                     est, 0.29, 0.015, f"{SRC}, Results (Speech: +0.29 dB)")
    checks.published("speech intensity, 8-talker minus clear: standard error (dB)", se, 0.09, 0.005,
                     f"{SRC}, Results (Speech: 0.29 +- 0.09 dB)")
    return dict(n_speech=len(s), n_pairs=s.subject.nunique(), est4=float(mm.params["C(nf)[T.4]"]), t8=est / se)


def main():
    checks = stdlib.Checks()
    rows, noise = build(checks)
    a = pd.DataFrame(rows)
    for _, c in KIN + FACE:
        if c not in FLAG_COLS:
            a[c] = pd.to_numeric(a[c].replace("", np.nan))
    a["pert_value"] = pd.to_numeric(a.pert_value.replace("", np.nan))
    extra = checks_all(a, checks, noise)
    for c in checks.items:
        print(("ok     " if c["ok"] else "FAILED "), c["what"], c.get("computed", ""), c.get("reported", ""))
    print("  [info]", extra, "rows", len(a), "pairs", a.subject.nunique())
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Merged the authors' kinematics, face and voice tables, the hand-checked modality file and the noise level "
                 "of each round (participant 1 of the pair) into one table with one row per attempt, turned the A1/A2 "
                 "column pairs into attempt 1 and attempt 2 rows, and replaced pair numbers with codes."),
        how=[
            "One row is one attempt by the producer of a pair (the participant the Kinect recorded) to get one action verb "
            "across to the partner; round is the verb number (1-20) and attempt is 1, or 2 when the first try did not succeed "
            "(the authors measured at most two). Rows are in round order; every round of the authors' tables has an attempt 1 row, "
            "also when the recording gave no values.",
            "Hand, face and voice columns are the authors' per-attempt summaries under their own names; hand distances and speeds "
            "are kept in the deposit's unit, which the deposit does not state (probably mm and mm/s), and the face measures have "
            "no stated unit. speech_intensity_*_db and speech_f0_max_hz come from Praat as the paper describes (maximum intensity "
            "in dB, maximum f0).",
            "pert_value is the babble the producer heard in that round: 0 for none, 4 or 8 talkers (the partner always heard "
            "4-talker babble); the noise level comes from the authors' noise file for participant 1 of the pair, which their "
            "analysis script uses. Rows without a noise level or a hand-checked modality, and every row of pair 85 (whose noise "
            "levels the authors could not determine), are marked excluded_by_authors, as the script drops them.",
            "modality (speech, gesture, speech+gesture) is the authors' hand-checked code for the round; they merged it with the "
            "attempt 1 values, so it is on attempt 1 rows only. The paper's models use attempt 1 only; attempt 2 rows are the "
            "extra values in the deposit. Two pairs have a modality and noise level but no recording and give one row each round.",
        ],
        unit_note="attempt to communicate one verb",
        differences=[
            "The OSF project states no license; the paper itself is CC BY 4.0. The paper's Table 1 counts (243 speech-only, 732 "
            "gesture-only and 327 multimodal attempts, 1302 in all) do not reproduce from the deposit: the hand-checked modality "
            "file has 71 speech-only, 254 gesture-only and 734 speech-and-gesture rounds (1059 in all), and the paper does not "
            "say how it counted attempts.",
            "Speech-intensity analysis: the deposit gives "
            f"{extra['n_speech']} rounds with speech and an intensity from {extra['n_pairs']} pairs (paper: 243 + 327 = 570 attempts); "
            "the 8-talker effect reproduces (0.28 dB against 0.29 dB) but the 4-talker contrast is "
            f"{extra['est4']:.2f} dB here. The paper's submovement ratios (1.142 and 1.093 times clear) come from Poisson "
            "mixed models and were not reproduced.",
        ],
    )
    for r in rows:
        r["person"] = r["subject"]
    return stdlib.release(ID, "communication", rows, meta, checks)


if __name__ == "__main__":
    main()
