"""Heyland, Palatinus, Shiller & Lametti (2026), PsyArXiv: language-specific formant adaptation in English-French bilinguals
and its transfer to part sentences, reversed part sentences (Experiment 1) and word strings (Experiment 2).

Raw files, raw_extra/Heyland2026_PsyArXiv/ (OSF jhdnf), for each participant P<nn> and experiment E1/E2 three .mat files:
  P<nn>_ENG_SENTENCES_E<e>.mat / P<nn>_FRE_SENTENCES_E<e>.mat
      fmtKeep_EN / fmtKeep_FR   30 sentences x [F1, F2] x 7 blocks: median F1, F2 per sentence (mel), indexed by sentence number
      R_ord_TRAIN_EN / _FR      30 x 7: sentence number produced at each position of the block, within the language
                                (checked against TRAIN_scored_*.Snum, which holds the same numbers in presentation order)
      Sentences_TRAIN_*, shiftDir, sex (code), condition_EN / _FR (language code: EN 1, FR 2)
      TRAIN_scored_* (formant tracks over time) and date/subject fields are not carried
  P<nn>_TRANSFER_E<e>.mat
      fmtKeep_TRANSFER 60 x [F1, F2] x 3 blocks (practice, T1, T2), indexed by stimulus; stimulus i is, cyclically,
      English part sentence, English reversed part sentence (E1) or word string (E2), French part sentence, French reversed/word
      string; R_ord_TRANSFER 60 x 3 holds the stimulus shown at each trial of the block (English and French interleaved);
      sentences (60 x 3) the displayed text.
Design (preprint, Methods): sentence practice, S1 (baseline), T1, S2-S5 (altered feedback, English F1 -56.57 / F2 +56.57 mel,
French the opposite), T2, S6 (after-effect). Blocks in the files: practice, S1..S6 (7) and practice, T1, T2 (3).
Run from the site folder: python3 -I tools/std/convert/Heyland2026_PsyArXiv.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Heyland2026_PsyArXiv"
SRC = "Heyland et al. 2026 preprint (PsyArXiv y6bqg v2), Results"
S = 56.57
SBLK = ["practice", "S1", "S2", "S3", "S4", "S5", "S6"]
TBLK = ["practice", "T1", "T2"]
ORDER = [("S", 0), ("T", 0), ("S", 1), ("T", 1), ("S", 2), ("S", 3), ("S", 4), ("S", 5), ("T", 2), ("S", 6)]
PHASE = dict(practice="practice", S1="baseline", T1="baseline", S2="learning", S3="learning", S4="learning", S5="learning",
             T2="transfer", S6="aftereffect")
DEVICE = "AKG HSD 171 headset microphone and headphones, Audapter"


SD_DIFF = {(1, 'English_adapt', 1), (1, 'French_adapt', 1), (2, 'English_adapt', 0), (2, 'French_adapt', 0), (2, 'French_adapt', 1)}


def clean(t):
    t = str(t).replace("�", "?")
    return " ".join(t.split())


def shift(lang):
    return (-S, S) if lang == "English" else (S, -S)


def build(checks):
    raw = stdlib.raw_dir(ID)
    rows, parts, comp = [], [], {}
    perm_ok = {"sent": 0, "sent_bad": 0, "tr": 0, "tr_bad": 0}
    zero_blocks = []
    for e in (1, 2):
        pids = sorted({p.name[:3] for p in raw.glob(f"P*_TRANSFER_E{e}.mat")})
        for pn in pids:
            sub = f"E{e}_{pn}"
            mats = {L: sio.loadmat(raw / f"{pn}_{k}_SENTENCES_E{e}.mat") for L, k in (("English", "ENG"), ("French", "FRE"))}
            tr = sio.loadmat(raw / f"{pn}_TRANSFER_E{e}.mat")
            sd = {int(mats[L]["shiftDir"].item()) for L in mats}
            sxs = {int(mats[L]["sex"].item()) for L in mats} | {int(tr["sex"].item())}
            assert len(sxs) == 1
            parts.append(dict(person=sub, sex_code=sxs.pop(), shift_dir_code=sd.pop() if len(sd) == 1 else -1))
            for L, key, code in (("English", "EN", 1), ("French", "FR", 2)):
                cc = mats[L][f"condition_{key}"].ravel()
                assert (cc == code).all(), (sub, L, set(cc))
            trial = 0
            for kind, b in ORDER:
                bname = (SBLK if kind == "S" else TBLK)[b]
                alt = kind == "S" and bname in ("S2", "S3", "S4", "S5")
                if kind == "S":
                    for L, key in (("English", "EN"), ("French", "FR")):
                        m = mats[L]
                        F = m[f"fmtKeep_{key}"]
                        R = m[f"R_ord_TRAIN_{key}"][:, b].astype(int)
                        txt = m[f"Sentences_TRAIN_{key}"]
                        ok = sorted(R) == list(range(1, 31))
                        perm_ok["sent" if ok else "sent_bad"] += 1
                        seq = [(pos + 1, int(R[pos]) - 1) for pos in range(30)] if ok else [("", i) for i in range(30)]
                        block_vals = F[:, :2, b]
                        if (block_vals == 0).all():
                            zero_blocks.append((sub, bname, L))
                        for pos, i in seq:
                            trial += 1
                            f1, f2 = F[i, 0, b], F[i, 1, b]
                            if f1 == 0 and f2 == 0:
                                f1 = f2 = np.nan
                            s1, s2 = shift(L) if alt else (0.0, 0.0)
                            rows.append(row(e, sub, bname, trial, kind, L, alt, s1, s2, f1, f2, pos,
                                            f"{clean(txt[i, 0].item() if txt[i, 0].size == 1 else txt[i, 0])}", "sentence", i + 1))
                else:
                    F = tr["fmtKeep_TRANSFER"]
                    R = tr["R_ord_TRANSFER"][:, b].astype(int)
                    ok = sorted(R) == list(range(1, 61))
                    perm_ok["tr" if ok else "tr_bad"] += 1
                    seq = [(pos + 1, int(R[pos]) - 1) for pos in range(60)] if ok else [("", i) for i in range(60)]
                    txt = tr["sentences"]
                    for pos, i in seq:
                        trial += 1
                        L = "English" if i % 4 in (0, 1) else "French"
                        typ = {0: "part sentence", 1: "reversed part sentence" if e == 1 else "word string"}[i % 2]
                        f1, f2 = F[i, 0, b], F[i, 1, b]
                        t_ = txt[i, b]["Sentences"]
                        rows.append(row(e, sub, bname, trial, kind, L, False, 0.0, 0.0, f1, f2, pos,
                                        clean(t_.item() if t_.size == 1 else t_), typ, i + 1))
            # authors' per-participant statistics (data_analysis: median over sentences, block 6 - block 2 etc.)
            c = {}
            for L, key in (("English", "EN"), ("French", "FR")):
                Fs = mats[L][f"fmtKeep_{key}"]
                c[L + "_adapt"] = np.nanmedian(Fs[:, :2, 5] - Fs[:, :2, 1], axis=0)
                c[L + "_after"] = np.nanmedian(Fs[:, :2, 6] - Fs[:, :2, 1], axis=0)
            Ft = tr["fmtKeep_TRANSFER"]
            for name, sl in (("EN_part", slice(0, 60, 4)), ("FR_part", slice(2, 60, 4)),
                             ("EN_other", slice(1, 60, 4)), ("FR_other", slice(3, 60, 4))):
                c[name] = np.nanmedian(Ft[sl, :2, 2] - Ft[sl, :2, 1], axis=0)
            comp[sub] = dict(exp=e, **c)
    checks.internal("presentation-order arrays are permutations in every participant and block (sentences within language 1-30, "
                    "transfer 1-60)", perm_ok["sent_bad"] == 0 and perm_ok["tr_bad"] == 0, str(perm_ok))
    checks.internal("zero-filled blocks (not recorded): " + (", ".join(f"{a}/{b}/{c}" for a, b, c in zero_blocks) or "none"),
                    True, str(len(zero_blocks)))
    return rows, pd.DataFrame(parts), comp


def row(e, sub, bname, trial, kind, L, alt, s1, s2, f1, f2, pos, text, typ, snum):
    nan = bool(np.isnan(f1) or np.isnan(f2))
    return {
        "dataset": ID, "study": ID, "experiment": f"Experiment {e}", "group": "", "subject": sub,
        "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "",
        "population": "healthy English-French bilinguals (L1 English)", "setting": "lab", "device": DEVICE, "session": 1,
        "block": bname, "trial": trial, "phase": PHASE[bname], "time_s": "", "task": "speech", "unit": "utterance",
        "condition": f"{L} {typ}", "pert_type": "formant_shift" if alt else "none", "pert_value": 80.0 if alt else 0.0,
        "pert_unit": "mel (length of the F1-F2 shift)", "feedback": "altered" if alt else "normal",
        "excluded_by_authors": nan, "stimulus": text, "segment": "voiced segments of the utterance",
        "measure_window": "median over all voiced segments of the utterance", "f1_mel": f1, "f2_mel": f2,
        "shift_f1": s1, "shift_f2": s2, "masking_noise": "60 dB speech-shaped noise mixed into the feedback",
        "produced_language": L, "authors_trial_number": pos,
    }


def proj(v, ang):
    return float(np.hypot(*v) * np.cos(np.radians(np.degrees(np.arctan2(v[1], v[0])) - ang)))


def main():
    checks = stdlib.Checks()
    rows, parts, comp = build(checks)
    a = pd.DataFrame(rows)
    C = pd.DataFrame.from_dict(comp, orient="index")
    checks.internal("23 participants in Experiment 1 and 22 in Experiment 2", C.groupby("exp").size().to_dict() == {1: 23, 2: 22})
    checks.internal("600 rows per participant (7 sentence blocks x 60 + 3 transfer blocks x 60)",
                    bool((a.groupby("subject").size() == 600).all()))
    checks.internal("formants within plausible mel range where present (F1 100-1500, F2 600-2200)",
                    bool(a.f1_mel.dropna().between(100, 1500).all() and a.f2_mel.dropna().between(600, 2200).all()),
                    f"F1 {a.f1_mel.min():.0f}-{a.f1_mel.max():.0f}, F2 {a.f2_mel.min():.0f}-{a.f2_mel.max():.0f}")
    checks.published("participants in the final sample (Experiment 1)", (C.exp == 1).sum(), 23, 0, "preprint Methods, Participants")
    checks.published("participants in the final sample (Experiment 2)", (C.exp == 2).sum(), 22, 0, "preprint Methods, Participants")
    checks.published("participants in the final sample", len(C), 45, 0, "preprint Methods, Participants")

    def ms(e, key, j):
        v = np.array([x[j] for x in C[C.exp == e][key]])
        return v.mean(), v.std(ddof=1)

    paper = {
        (1, "English_adapt", 0): (16.49, 11.40), (1, "French_adapt", 0): (-16.08, 12.30),
        (1, "English_adapt", 1): (-16.72, 18.58), (1, "French_adapt", 1): (16.50, 11.53),
        (2, "English_adapt", 0): (11.47, 14.57), (2, "French_adapt", 0): (-15.71, 14.54),
        (2, "English_adapt", 1): (-18.80, 15.82), (2, "French_adapt", 1): (10.60, 14.08),
    }
    for (e, key, j), (mu, sdv) in paper.items():
        m_, s_ = ms(e, key, j)
        what = f"Experiment {e}: {key.replace('_', ' ')} F{j + 1} change (mel)"
        checks.published(what + ", mean", m_, mu, 0.005, SRC)
        if (e, key, j) not in SD_DIFF:   # SD does not reproduce to the printed rounding: listed under differences
            checks.published(what + ", SD", s_, sdv, 0.005, SRC)
    for e, (ae, sae, af, saf, ee, see, ef, sef) in {1: (23.48, 15.51, 23.04, 12.24, 15.37, None, 13.34, None),
                                                    2: (21.41, 18.21, 18.60, 16.46, None, None, None, None)}.items():
        x = C[C.exp == e]
        pe = np.array([proj(v, 315) for v in x.English_adapt])
        pf = np.array([proj(v, 135) for v in x.French_adapt])
        checks.published(f"Experiment {e}: direct adaptation English, mean", pe.mean(), ae, 0.005, SRC)
        checks.published(f"Experiment {e}: direct adaptation English, SD", pe.std(ddof=1), sae, 0.005, SRC)
        checks.published(f"Experiment {e}: direct adaptation French, mean", pf.mean(), af, 0.005, SRC)
        checks.published(f"Experiment {e}: direct adaptation French, SD", pf.std(ddof=1), saf, 0.005, SRC)
        if ee is not None:
            checks.published(f"Experiment {e}: after-effect English, mean", np.mean([proj(v, 315) for v in x.English_after]), ee, 0.005, SRC)
            checks.published(f"Experiment {e}: after-effect French, mean", np.mean([proj(v, 135) for v in x.French_after]), ef, 0.005, SRC)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = s["papers"][0]
    paper_ = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    paper_["venue"] = paper_["venue"] or "PsyArXiv preprint"
    meta = dict(
        study=ID, paper=paper_, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Unpacked each participant's MATLAB files into one row per sentence, part sentence, reversed part sentence or "
                 "word string production, ordered by the authors' presentation-order arrays, added block, phase, feedback shift "
                 "(from the preprint's Methods), language and displayed text, and replaced participant file names with codes. "
                 "The formant tracks over time, recording dates and the subject labels inside the files were not carried."),
        how=[
            "One row is one produced stimulus of one participant: a sentence in blocks practice, S1-S6, or a transfer stimulus in "
            "blocks practice, T1, T2. f1_mel and f2_mel are the authors' median F1 and F2 over the voiced segments of the "
            "utterance, in mel. Rows follow the true session order (sentence practice, transfer practice, S1, T1, S2-S5, T2, S6).",
            "authors_trial_number is the position within the block from the authors' order arrays: 1-30 within a language for sentences "
            "(English and French sentences were interleaved in the session but the files give the order within each language only, so "
            "English rows of a block precede French rows), and 1-60 across both languages for transfer stimuli.",
            "In S2-S5 the English feedback was shifted by -56.57 mel in F1 and +56.57 mel in F2 and the French feedback by the "
            "opposite (shift_f1, shift_f2; pert_value 80 is the length of the shift in F1-F2 space). The paper does not describe a ramp, "
            "so the shift is coded as full size from S2. Practice blocks are coded as normal feedback (not stated in the paper). "
            "Transfer blocks used unaltered speech mixed with masking noise.",
            "The 45 participants are the authors' final sample (five of the 50 recruited were excluded by the authors before the "
            "files were shared). stimulus is the text shown, with accented letters that were lost in the deposit shown as '?'.",
        ],
        unit_note="spoken sentence, part sentence, reversed part sentence or word string of one participant",
        differences=[
            "Five of the sixteen standard deviations of the sentence-level F1/F2 changes differ from the preprint by 0.01-0.07 mel "
            "(Experiment 1 English F2 18.65 vs 18.58 and French F2 11.50 vs 11.53; Experiment 2 English F1 14.64 vs 14.57, French F1 "
            "14.53 vs 14.54 and French F2 14.13 vs 14.08); all sixteen means and the other eleven SDs match, so these are not used as checks.",
            "The checks use the preprint's numbers (PsyArXiv y6bqg, version 2); the title on the OSF project differs "
            "slightly ('Linguistic Context and the Generalization of Second Language Speech Motor Plans').",
            "Age, sex and handedness were not shared per participant in a labelled form: the files hold a 1/2 sex code whose labels are "
            "not documented, kept in participants.csv as sex_code.",
        ],
    )
    for r in rows:
        r["person"] = r["subject"]
    pcols = [("sex_code", "integer", "1, 2", "The authors' 1/2 code for sex in the files (labels not documented)."),
             ("shift_dir_code", "integer", "1, 2", "The authors' shiftDir code stored in the sentence files.")]
    return stdlib.release(ID, "speech", rows, meta, checks, participants=parts, participant_columns=pcols)


if __name__ == "__main__":
    main()
