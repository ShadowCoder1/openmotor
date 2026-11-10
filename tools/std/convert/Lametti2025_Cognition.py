"""Lametti, Wheeler, Palatinus, Hocine & Shiller (2025), Cognition 254:106010: English-French bilinguals adapt to opposing
F1/F2 feedback shifts tied to the language they speak (Experiments 1, 2, 3, 5) or to text colour (Experiment 4).

Raw files, raw_extra/Lametti2025_Cognition/ (OSF dt4be): EX<e>_P<nn>.mat, one per participant (5 experiments x 15), each
holding (read with scipy, nothing executed):
  fmtKeep_EN / fmtKeep_FR   30 sentences x [F1, F2] x 7 sentence blocks (S1..S7): median F1, F2 per sentence, mel
  condition_EN / condition_FR   30 x [language code (EN 2, FR 1), text colour (1 red, 2 blue)]
  fmtKeepENv / fmtKeepFRv   48 words x [F1, F2] x 2 transfer blocks (T1, T2); conditionENv / conditionFRv: 48 x
                            [language, colour, vowel code 1-4] x 2
  age, sex (1/2), shiftDir (1/2), AoA_EN/FR, proficiency_EN/FR (self rating 1-10), Sentences_TRAIN_* , R_ord_TRAIN_*
  fmtKeep_RED/BLU and the ...v RED/BLU arrays regroup the same sentences by colour (verified), so they are not repeated.
Design (preprint PsyArXiv 56zwg, Methods; authors' data_analysis.m read as text): practice, S1, S2, T1, S3-S6 (altered
feedback), T2, S7. Shifts of 56.57 mel in F1 and in F2 (80 mel in F1-F2 space), opposite between the two cues.
Run from the site folder: python3 -I tools/std/convert/Lametti2025_Cognition.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Lametti2025_Cognition"
SRC = "Lametti et al. (preprint PsyArXiv 56zwg), Results"
S = 56.57
ORDER = ["S1", "S2", "T1", "S3", "S4", "S5", "S6", "T2", "S7"]
PHASE = dict(S1="baseline", S2="baseline", T1="baseline", S3="learning", S4="learning", S5="learning", S6="learning",
             T2="transfer", S7="aftereffect")
LANG = {2: "English", 1: "French"}
COLOUR = {1: "red", 2: "blue"}
DEVICE = "head-mounted Shure WH20 microphone, Sennheiser HD 280 Pro headphones, Audapter"


TRUNCATED = {(1, 'English_adapt', 0, ', mean'), (1, 'English_transfer', 0, ', SD'), (2, 'English_adapt', 1, ', mean')}


def shift(exp, sd, lang, colour):
    """(F1, F2) feedback shift in mel for a language/colour on altered-feedback sentences."""
    if exp == 4:
        sgn = 1 if colour == "red" else -1          # Methods, Experiment 4: red +F1 -F2, blue -F1 +F2
    else:
        sgn = -1 if lang == "English" else 1         # Experiment 1/3/5: English -F1 +F2, French +F1 -F2
        if exp == 2:
            sgn = -sgn                               # Experiment 2: reversed
    return sgn * S, -sgn * S


def load():
    raw = stdlib.raw_dir(ID)
    out = []
    for p in sorted(raw.glob("EX*_P*.mat")):
        e, pn = int(p.name[2]), p.name[4:7]
        out.append((e, pn, sio.loadmat(p)))
    return out


def build(files, checks):
    rows, parts, comp = [], [], {}
    zero_blocks = {}
    for e, pn, m in files:
        sd = int(m["shiftDir"].item())
        sub = f"EX{e}_{pn}"
        sex = {1: "F", 2: "M"}[int(m["sex"].item())]
        parts.append(dict(person=sub, aoa_english=float(m["AoA_EN"].item()), aoa_french=float(m["AoA_FR"].item()),
                          proficiency_english=float(m["proficiency_EN"].item()),
                          proficiency_french=float(m["proficiency_FR"].item()), shift_dir_code=sd))
        trial = 0
        arr = {"English": (m["fmtKeep_EN"], m["condition_EN"]), "French": (m["fmtKeep_FR"], m["condition_FR"])}
        warr = {"English": (m["fmtKeepENv"], m["conditionENv"]), "French": (m["fmtKeepFRv"], m["conditionFRv"])}
        for b in ORDER:
            sent = b.startswith("S")
            k = int(b[1]) - 1
            for lang in ("English", "French"):
                F, C = arr[lang] if sent else warr[lang]
                n = F.shape[0]
                for i in range(n):
                    trial += 1
                    f1, f2 = (F[i, 0, k], F[i, 1, k])
                    if f1 == 0 and f2 == 0:  # S1 was not stored for these participants (whole block holds zeros)
                        f1 = f2 = np.nan
                        zero_blocks.setdefault((sub, b), 0)
                        zero_blocks[(sub, b)] += 1
                    colour = COLOUR[int(C[i, 1] if sent else C[i, 1, k])]
                    assert int(C[i, 0] if sent else C[i, 0, k]) == (2 if lang == "English" else 1)
                    alt = b in ("S3", "S4", "S5", "S6")
                    s1, s2 = shift(e, sd, lang, colour) if alt else (0.0, 0.0)
                    nan = bool(np.isnan(f1) or np.isnan(f2))
                    rows.append({
                        "dataset": ID, "study": ID, "experiment": f"Experiment {e}", "group": "", "subject": sub,
                        "subject_excluded_by_authors": False, "age": float(m["age"].item()), "sex": sex, "handedness": "",
                        "population": "healthy English-French bilinguals", "setting": "lab", "device": DEVICE, "session": 1,
                        "block": b, "trial": trial, "phase": PHASE[b], "time_s": "", "task": "speech", "unit": "utterance",
                        "condition": f"{lang} {'sentence' if sent else 'word'}, {colour} text",
                        "pert_type": "formant_shift" if alt else "none", "pert_value": 80.0 if alt else 0.0,
                        "pert_unit": "mel (length of the F1-F2 shift)",
                        "feedback": "altered" if alt else ("masked" if (not sent and e != 3) else "normal"),
                        "excluded_by_authors": nan,
                        "stimulus": f"{lang} sentence {i + 1}" if sent else f"{lang} word {i + 1} of 48",
                        "segment": "vowels of the sentence" if sent else "vowel of the word",
                        "measure_window": ("median over all voiced segments of the sentence" if sent
                                           else "25 ms at the hand-marked midpoint of the vowel"),
                        "f1_mel": f1, "f2_mel": f2, "shift_f1": s1, "shift_f2": s2,
                        "masking_noise": ("speech-shaped noise replacing feedback, scaled to speech volume"
                                          if (not sent and e != 3) else "60 dB speech-shaped noise mixed into feedback"),
                        "produced_language": lang, "text_colour": colour,
                        "vowel_id": "" if sent else int(C[i, 2, k]),
                    })
        # per-participant changes as in the authors' data_analysis.m
        def chg(F, a, b_):  # mean over sentences of block b_ minus block a
            return np.nanmean(F[:30, :2, b_] - F[:30, :2, a], axis=0)
        c = {}
        for lang, key in (("English", "EN"), ("French", "FR")):
            Fs = m[f"fmtKeep_{key}"]
            c[lang + "_adapt"] = chg(Fs, 1, 5)
            c[lang + "_after"] = chg(Fs, 1, 6)
            Fv, Cv = m[f"fmtKeep{key}v"], m[f"condition{key}v"]
            vm = np.array([[np.nanmean(Fv[Cv[:, 2, b_] == v, :, b_], axis=0) for v in range(1, 5)] for b_ in range(2)])
            c[lang + "_transfer"] = (vm[1] - vm[0]).mean(axis=0)
        comp[sub] = dict(exp=e, sd=sd, sex=sex, age=float(m["age"].item()), **{k_: v for k_, v in c.items()})
    checks.internal("zeros in the shared arrays are exactly whole S1 sentence blocks (60 sentences) of 7 participants: not recorded, emptied",
                    len(zero_blocks) == 7 and all(b == 'S1' and n == 60 for (_, b), n in zero_blocks.items()), str(len(zero_blocks)))
    return rows, pd.DataFrame(parts), comp


def main():
    files = load()
    checks = stdlib.Checks()
    rows, parts, comp = build(files, checks)
    a = pd.DataFrame(rows)
    C = pd.DataFrame.from_dict(comp, orient="index")
    checks.internal("75 participants, 15 in each of 5 experiments", len(C) == 75 and (C.groupby("exp").size() == 15).all(),
                    str(C.groupby("exp").size().to_dict()))
    checks.internal("sentence rows: 60 sentences x 7 blocks per participant; word rows: 96 x 2",
                    bool((a[a.block.str.startswith("S")].groupby("subject").size() == 420).all()
                         and (a[a.block.str.startswith("T")].groupby("subject").size() == 192).all()))
    checks.internal("formants in plausible mel range (F1 100-1500, F2 600-2200) where present",
                    bool(a.f1_mel.dropna().between(100, 1500).all() and a.f2_mel.dropna().between(600, 2200).all()),
                    f"F1 {a.f1_mel.min():.0f}-{a.f1_mel.max():.0f}, F2 {a.f2_mel.min():.0f}-{a.f2_mel.max():.0f}")
    checks.internal("in every experiment the group mean change in English F1 (S6-S2) opposes the English F1 shift in force",
                    all(np.sign(C[C.exp == e].English_adapt.apply(lambda v: v[0]).mean()) !=
                        np.sign(shift(e, 1, "English", "red")[0]) for e in (1, 2, 3, 5)))
    # sex codes: 1 = female / 2 = male checked by the paper's counts below
    for e, f, mm in ((1, 11, 4), (2, 11, 4), (3, 12, 3)):
        x = C[C.exp == e]
        checks.published(f"Experiment {e}: women", (x.sex == "F").sum(), f, 0, "preprint Methods, Experiments 1-3")
        checks.published(f"Experiment {e}: men", (x.sex == "M").sum(), mm, 0, "preprint Methods, Experiments 1-3")
    checks.published("women in all five experiments", (C.sex == "F").sum(), 60, 0, "preprint Methods, Participants")
    checks.published("men in all five experiments", (C.sex == "M").sum(), 15, 0, "preprint Methods, Participants")
    checks.published("participants in all experiments", len(C), 75, 0, "preprint Methods, Participants")

    def stat(e, key, j):
        v = np.array([x[j] for x in C[C.exp == e][key]])
        return v.mean(), v.std(ddof=1)

    paper = {  # preprint Results: (experiment, language, F1/F2, phase) -> mean, sd
        (1, "English_adapt", 0): (17.95, 18.10), (1, "French_adapt", 0): (-16.99, 17.05),
        (1, "English_adapt", 1): (-19.24, 16.63), (1, "French_adapt", 1): (14.22, 21.09),
        (1, "English_transfer", 0): (8.97, 20.97), (1, "French_transfer", 0): (-2.62, 22.44),
        (1, "English_transfer", 1): (-3.47, 15.31), (1, "French_transfer", 1): (2.87, 13.65),
        (2, "English_adapt", 0): (-16.40, 19.61), (2, "French_adapt", 0): (15.33, 24.10),
        (2, "English_adapt", 1): (20.57, 17.95), (2, "French_adapt", 1): (-15.31, 17.24),
        (2, "English_transfer", 0): (-6.55, 21.91), (2, "French_transfer", 0): (-2.22, 14.98),
        (2, "English_transfer", 1): (6.53, 11.62), (2, "French_transfer", 1): (5.92, 17.14),
    }
    for (e, key, j), (mu, sdv) in paper.items():
        m_, s_ = stat(e, key, j)
        what = f"Experiment {e}: {key.replace('_', ' ')} F{j + 1} change (mel)"
        for lab_, got, rep in ((", mean", m_, mu), (", SD", s_, sdv)):
            if (e, key, j, lab_) in TRUNCATED:   # printed value is the computed one cut, not rounded: see differences
                continue
            checks.published(what + lab_, got, rep, 0.005, SRC)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = s["papers"][0]
    meta = dict(
        study=ID, paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
        license="none stated", license_url="",
        changes=("Unpacked each participant's MATLAB file into one row per sentence or word production, added block, phase, "
                 "feedback shift (from the paper's Methods), language and text colour, and replaced participant file names with "
                 "codes. Age, sex, age of acquisition and self-rated proficiency are given per participant."),
        how=[
            "One row is one spoken sentence (blocks S1-S7) or one word of the transfer task (T1, T2) by one participant; f1_mel and "
            "f2_mel are the authors' F1 and F2 in mel (the median over the sentence, or 25 ms at the vowel midpoint for words).",
            "The shared files do not give the order of trials, only the sentence or word number, so rows run in block order "
            "(S1, S2, T1, S3-S6, T2, S7) with the English sentences of a block before the French ones; trial is the row's position "
            "in that order, not the presentation order. Each sentence was spoken once per block.",
            "In S3-S6 the feedback was shifted by 56.57 mel in F1 and 56.57 mel in F2 (shift_f1, shift_f2, signed), 80 mel in "
            "F1-F2 space (pert_value); signs depend on the experiment (language-contingent in Experiments 1, 2, 3 and 5, "
            "colour-contingent in 4). The paper does not describe a ramp, so the shift is coded as full size from S3.",
            "Transfer words were produced with noise-masked feedback in all experiments except 3 (normal feedback). vowel_id is the "
            "authors' vowel code 1-4 (which vowel each code is is not documented). In Experiment 5 the sentences of S1 and S7 differ "
            "from those of S2-S6 (per the paper), so sentence numbers there refer to different sentences.",
        ],
        unit_note="spoken sentence or word of one participant",
        differences=[
            "The published numbers used for checking are those of the authors' preprint (PsyArXiv 56zwg, 2023); the journal "
            "version (Cognition 2025) could not be read here and may differ.",
            "Three preprint values are the computed ones cut rather than rounded, so they are not used as checks: Experiment 1 "
            "English sentence F1 change mean 17.956 (printed 17.95), Experiment 1 English transfer F1 SD 20.976 (printed 20.97), "
            "Experiment 2 English sentence F2 change mean 20.577 (printed 20.57).",
            "Sex: the files give 14 women and 1 man in Experiment 4 and 12 women and 3 men in Experiment 5; the preprint text says "
            "13 and 2 in each (60 women and 15 men in all five experiments, which the files reproduce). The mean age in the files is "
            "22.48 years (preprint: 22.89), so one or two files may be assigned to a different experiment, or ages differ.",
            "Seven participants have no S1 (first baseline) data: the arrays hold zeros for the whole block, which are emptied here "
            "(S1 is used only in Experiment 5's new-sentence after-effect measure).",
        ],
    )
    for r in rows:
        r["person"] = r["subject"]
    pcols = [("aoa_english", "number", "years", "Self-reported age of acquisition of English (0 = from birth)."),
             ("aoa_french", "number", "years", "Self-reported age of acquisition of French."),
             ("proficiency_english", "number", "1-10", "Self-rated English proficiency (mean of the authors' 1-10 ratings)."),
             ("proficiency_french", "number", "1-10", "Self-rated French proficiency."),
             ("shift_dir_code", "integer", "1, 2", "The authors' code for which shift direction the participant had (1 or 2).")]
    return stdlib.release(ID, "speech", rows, meta, checks, participants=parts, participant_columns=pcols)


if __name__ == "__main__":
    main()
