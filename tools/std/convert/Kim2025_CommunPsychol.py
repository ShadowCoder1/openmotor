"""Kim, Kitchen, Mitsuya & Max (2025), Commun Psychol 3:127: speech auditory-motor adaptation (+250 cents shift of all formants,
words tech / tuck / talk) and de-adaptation under intensive (3.33 s intertrial interval) or non-intensive (10 s) practice.

Raw files, raw_extra/Kim2025_CommunPsychol/ (OSF e37ba): AdaptBlock_Data_OSF.mat, DeAdaptBlock_Data_OSF.mat, readme.txt.
Each .mat holds a struct [WithFoll | NoFoll].[Int | NonInt].[F1Only | F2Only | F1F2_Avgd] = 90 epochs x 28 participants, in cents.
Epoch = 3 consecutive trials (one per word), averaged. 90 epochs = 18 baseline + 36 adaptation + 36 de-adaptation (paper, Methods).
AdaptBlock is normalized to baseline (epochs 6-18); DeAdaptBlock to the end of the adaptation phase (epochs 42-54), readme.
Int / NonInt = intensive / non-intensive schedule of the adaptation phase in AdaptBlock and of the de-adaptation phase in
DeAdaptBlock. The files list participants in different orders; the same participant is the column pair whose difference
(DeAdapt - Adapt) is constant across the 90 epochs (a constant offset is exactly what the change of reference produces), and
that pairing is unique (checked below). NoFoll = the paper's data (followers replaced), WithFoll = with the 5 followers.
Run from the site folder: python3 -I tools/std/convert/Kim2025_CommunPsychol.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Kim2025_CommunPsychol"
M = ["F1Only", "F2Only", "F1F2_Avgd"]
SRC = "Kim et al. 2025, Commun Psychol 3:127 (Results; Methods, Participants)"
GROUP = {("Int", "Int"): "Group 1: Intensive adaptation, Intensive de-adaptation",
         ("Int", "NonInt"): "Group 2: Intensive adaptation, Non-Intensive de-adaptation",
         ("NonInt", "Int"): "Group 3: Non-Intensive adaptation, Intensive de-adaptation",
         ("NonInt", "NonInt"): "Group 4: Non-Intensive adaptation, Non-Intensive de-adaptation"}
ITI = {"Int": 3.33, "NonInt": 10.0}


def load(raw, fn, name, fol):
    S = getattr(sio.loadmat(raw / fn, squeeze_me=True, struct_as_record=False)[name], fol)
    return {m: np.hstack([getattr(getattr(S, s), m) for s in ("Int", "NonInt")]) for m in M}  # 90 x 56, Int first


def link(A, D):
    """pairs (a, d) of column indices with a constant offset in all three measures; must be unique and one-to-one."""
    pairs = []
    for a in range(A["F1Only"].shape[1]):
        ok = [d for d in range(D["F1Only"].shape[1]) if all(np.ptp(D[m][:, d] - A[m][:, a]) < 1e-6 for m in M)]
        assert len(ok) == 1, (a, ok)
        pairs.append((a, ok[0]))
    assert len({d for _, d in pairs}) == len(pairs)
    return pairs


def build(checks):
    raw = stdlib.raw_dir(ID)
    ND = (load(raw, "AdaptBlock_Data_OSF.mat", "AdaptBlock_Data", "NoFoll"), load(raw, "DeAdaptBlock_Data_OSF.mat", "DeAdaptBlock_Data", "NoFoll"))
    WD = (load(raw, "AdaptBlock_Data_OSF.mat", "AdaptBlock_Data", "WithFoll"), load(raw, "DeAdaptBlock_Data_OSF.mat", "DeAdaptBlock_Data", "WithFoll"))
    people = []  # (A dict, D dict, a, d, excluded)
    for (A, D), excl in ((ND, False),):
        for a, d in link(A, D):
            people.append((A, D, a, d, False))
    # followers: WithFoll participants whose adaptation columns are not in the NoFoll file
    A0 = ND[0]
    for a, d in link(*WD):
        col = np.concatenate([WD[0]["F1Only"][:, a], WD[0]["F2Only"][:, a]])
        if not any(np.allclose(col, np.concatenate([A0["F1Only"][:, j], A0["F2Only"][:, j]])) for j in range(56)):
            people.append((WD[0], WD[1], a, d, True))
    n_fol = sum(p[4] for p in people)
    checks.internal("every participant column of the adaptation file pairs with exactly one de-adaptation column", True,
                    f"{len(people) - n_fol} analysed + {n_fol} followers")
    rows, tab = [], []
    for pi, (A, D, a, d, excl) in enumerate(people):
        sa, sd = ("Int" if a < 28 else "NonInt"), ("Int" if d < 28 else "NonInt")
        code = f"P{pi + 1:03d}"
        for e in range(90):
            ph = "baseline" if e < 18 else ("learning" if e < 54 else "aftereffect")
            pert = 250.0 if 18 <= e < 54 else 0.0
            iti = 3.33 if e < 18 else ITI[sa if e < 54 else sd]
            r = {
                "dataset": ID, "study": ID, "experiment": "Adaptation and de-adaptation", "group": GROUP[(sa, sd)], "subject": code,
                "subject_excluded_by_authors": excl, "age": "", "sex": "", "handedness": "", "population": "healthy", "setting": "lab",
                "device": "Shure SM58 microphone, TC Helicon VoiceOne processor, Etymotic ER-3A insert earphones", "session": 1,
                "block": ("Baseline", "Adaptation", "De-adaptation")[(e >= 18) + (e >= 54)], "trial": e + 1, "phase": ph,
                "time_s": "", "task": "speech", "unit": "trial", "condition": f"{sa} adaptation, {sd} de-adaptation",
                "pert_type": "formant_shift" if pert else "none", "pert_value": pert, "pert_unit": "cents",
                "feedback": "altered" if pert else "normal", "excluded_by_authors": excl,
                "stimulus": "tech, tuck, talk (one production of each per epoch)", "segment": "vowel of the word",
                "measure_window": "averaged over the 40% and 50% points of the vowel, then over the epoch's three words",
                "shift_f1": pert, "shift_f2": pert, "masking_noise": "none", "feedback_latency_ms": 10,
                "f1_cents": A["F1Only"][e, a], "f2_cents": A["F2Only"][e, a], "f12_cents": A["F1F2_Avgd"][e, a],
                "f1_cents_end_ref": D["F1Only"][e, d], "f2_cents_end_ref": D["F2Only"][e, d], "f12_cents_end_ref": D["F1F2_Avgd"][e, d],
                "change_reference": "f*_cents: median of baseline trials 6-18 of each word; f*_cents_end_ref: median of the last 13 epochs (42-54) of the perturbation phase",
                "iti_s": iti,
            }
            rows.append(r)
        tab.append((code, sa, sd, excl))
    return rows, pd.DataFrame(tab, columns=["p", "sa", "sd", "excl"]), people


def anova2(df, dv, f1, f2=None):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    if f2 is None:
        m = smf.ols(f"{dv} ~ C({f1}, Sum)", df).fit()
    else:
        m = smf.ols(f"{dv} ~ C({f1}, Sum) * C({f2}, Sum)", df).fit()
    t = sm.stats.anova_lm(m, typ=3)
    return t, m


def main():
    checks = stdlib.Checks()
    rows, tab, people = build(checks)
    a = pd.DataFrame(rows)
    an = a[~a.excluded_by_authors]
    tn = tab[~tab.excl]
    checks.published("participants analysed (after replacing followers)", an.subject.nunique(), 56, 0, SRC)
    checks.published("followers replaced", int(tab.excl.sum()), 5, 0, "Kim et al. 2025, Methods, Participants")
    for sa in ("Int", "NonInt"):
        checks.published(f"participants with {sa} adaptation schedule", int((tn.sa == sa).sum()), 28, 0, SRC + ", Fig. 2 legend (n = 28 per group)")
    for k, g in tn.groupby(["sa", "sd"]):
        checks.published(f"participants in group {k}", len(g), 14, 0, SRC + ", Methods, Participants (n = 14 per group)")
    checks.internal("90 epochs for every participant", bool((a.groupby("subject").size() == 90).all()))
    checks.internal("de-adaptation offset constant across epochs (reference change only)", True)
    checks.internal("no missing values in the cents columns",
                    bool(a[["f1_cents", "f2_cents", "f12_cents", "f1_cents_end_ref", "f2_cents_end_ref", "f12_cents_end_ref"]].notna().all().all()))
    s_, paper = None, None
    s_ = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = s_["papers"][0]
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    meta = dict(
        study=ID, paper=paper, lab=s_["lab"], data_url=s_["data_url"], license="none stated", license_url="",
        changes=("Turned the authors' two MATLAB epoch tables (90 epochs x 56 participants per schedule) into one table with one row "
                 "per participant and epoch, paired each participant's adaptation-file and de-adaptation-file columns, added the "
                 "feedback shift and practice schedule from the paper, and replaced participant identifiers with codes."),
        how=[
            "One row is one epoch (three consecutive trials, one production each of tech, tuck and talk, averaged by the authors) "
            "of one participant: trial 1-18 baseline, 19-54 adaptation (all formants raised 250 cents), 55-90 de-adaptation "
            "(unperturbed). Values are the authors' epoch means in cents; the deposit has no values in Hz and no single-word values.",
            "f1_cents, f2_cents and f12_cents (average of F1 and F2) are relative to the median of each word's baseline trials 6-18; "
            "the *_end_ref columns are the same values relative to the median of the last 13 epochs (42-54) of the perturbation phase, "
            "the reference the authors used for the de-adaptation analyses. Negative means lower formants, which opposes the upward shift.",
            "The adaptation file and the de-adaptation file list participants in different orders. Each participant is the pair of "
            "columns whose difference is constant over all 90 epochs; this pairing is exactly one-to-one, so the group (practice schedule "
            "in adaptation and in de-adaptation) is known for everyone. iti_s is 3.33 s (intensive) or 10 s (non-intensive); baseline was always intensive.",
            "The 5 participants the authors called followers (their formants rose with the shift) and replaced are included from the "
            "authors' 'WithFoll' arrays and marked subject_excluded_by_authors; the 56 analysed participants are the 'NoFoll' arrays.",
        ],
        unit_note="epoch of three trials (one per word) of one participant",
        differences=[
            "The paper's Group tests (F(1,54) = 4.922 for the last three adaptation epochs; F(1,52) = 4.060 for de-adaptation; "
            "AdaptPractice F = 0.928) come from mixed models with the three words as repeated measures; the word-averaged epoch rows "
            "give F = 3.82, 3.83 and 0.933 with a plain two-way ANOVA on participant means, so these were not used as checks.",
            "The paper says the first 6 baseline trials of each word were discarded but the normalization uses trials 6-18; the first "
            "baseline epochs are kept here as the authors shared them.",
            "Participants' ages and sexes are not in the deposit.",
        ],
    )
    for r in rows:
        r["person"] = r["subject"]
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
