"""Hantzsch, Parrell & Niziolek (2022), eLife 11:e73694: compensation for an unpredictable F1 shift and the "one-shot"
adaptation on the next, unaltered trial, pooled from six earlier studies (131 participant datasets from 91 people).

Raw files, raw_extra/Hantzsch2022_eLife/ (GitHub blab-lab/postMan, commit 1358d36, linked from OSF 2gm76; MIT License):
  byTrial_4_23_21.csv        oneShot, OSTrial, cond, shiftMag, exp, participant, group, compTrial, compensation
                             one row per altered trial (compTrial) that was followed by an unaltered trial (OSTrial)
  oneShot_4_23_21.csv        oneShot, cond, participant, exp, shiftMag, group        one row per participant dataset and
  compensation_04_23_21.csv  compensation, cond, participant, exp, shiftMag, group   shift direction (the paper's
                                                                                     participant-level values)
What the values are (paper, Methods; the authors' code in the same repository, read as text: gen_fmtMatrix_postMan.m,
gen_fmtMatrixByCond_postMan.m, get_time_window_postMan.m, gen_byTrial_table.m, lmer_byTrial.Rmd, lmer_byParticipant.Rmd,
correlation_Test.Rmd):
  compensation  on the altered trial, F1 (mels) minus the mean F1 track of the participant's unshifted trials of the same
                word ('noShift'), averaged over 150-250 ms after vowel onset (make_params_postMan(0.15, 0.25), 'diff1')
  oneShot       on the next trial (unaltered), F1 (mels) minus the mean F1 track of the participant's unshifted trials of
                the same word that followed an unshifted trial ('postNo'), averaged over the first 100 ms
  shiftMag      the F1 part of the shift in mels (expt.shifts.mels{1}(1) and {2}(1); 125 / -125 for noisyfb and
                reachAndSpeech); positive = F1 raised in the feedback ('up'; 'shiftAE' in fmtAlt)
  exp / group   source study: cereb = Study 1, reachAndSpeech = Study 2, noisyfb normal / noisy = Studies 3 and 4 (two
                sessions of the same people), cat = Study 5, fmtAlt = Study 6 (matched on participants, trials and shift
                sizes of Table 1); the two outlier participants (lmer_*.Rmd remove OA14 and HOC8) are already absent.
Run from the site folder: python3 -I tools/std/convert/Hantzsch2022_eLife.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hantzsch2022_eLife"
SRC = "Hantzsch, Parrell & Niziolek 2022, eLife"
REPO = "https://github.com/blab-lab/postMan"
# (exp, group) -> experiment label, perturbation system, whether the shift was F1 only, words (Table 1), included n
STUDIES = {
    ("cereb", "na"): ("Study 1: Parrell et al. 2017", "FUSP", True, "beck, bet, deck, debt, pet, tech", 13),
    ("reachAndSpeech", "na"): ("Study 2: Parrell et al. 2021", "Audapter", True, "dead, fed, said, shed", 12),
    ("noisyfb", "normal"): ("Studies 3 and 4: Niziolek & Parrell 2021, normal-feedback session", "Audapter", True,
                            "bed, dead, head", 40),
    ("noisyfb", "noisy"): ("Studies 3 and 4: Niziolek & Parrell 2021, noisy-feedback session", "Audapter", True,
                           "bed, dead, head", 40),
    ("cat", "na"): ("Study 5: Niziolek & Guenther 2013", "Audapter", False, "bed, bet, dead, deb, debt, ped, tech, ted", 11),
    ("fmtAlt", "na"): ("Study 6: Niziolek et al. 2014", "FUSP", False, "head", 15),
}
DEVICE = {"FUSP": "microphone and headphones, Feedback Utility for Speech Production (FUSP)",
          "Audapter": "microphone and headphones, Audapter"}
WINDOW = ("f1_change_mel: mean over 150-250 ms after vowel onset of this trial; f1_change_later_mel: mean over the first "
          "100 ms after vowel onset of the next trial")
REFERENCE = ("f1_change_mel: this altered trial minus the mean F1 track of the participant's unshifted trials of the same "
             "word; f1_change_later_mel: the next trial (unaltered) minus the mean F1 track of the participant's unshifted "
             "trials of the same word that followed an unshifted trial")
MIT = """MIT License

Copyright (c) 2021 BLAB Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def main():
    raw = stdlib.raw_dir(ID)
    ck = stdlib.Checks()
    lic = (raw / "LICENSE").read_text()
    ck.internal("the repository's LICENSE is the MIT License text reproduced in LICENSE.txt",
                " ".join(lic.split()) == " ".join(MIT.split()))
    b = pd.read_csv(raw / "byTrial_4_23_21.csv")
    os_t = pd.read_csv(raw / "oneShot_4_23_21.csv")
    cp_t = pd.read_csv(raw / "compensation_04_23_21.csv")
    b["grp"] = b.group.fillna("na")
    keys = list(dict.fromkeys(zip(b.participant, b.exp, b.grp)))
    scode = {k: f"{ID}_S{i + 1:04d}" for i, k in enumerate(keys)}  # the codes release() assigns (first appearance)

    rows = []
    for r in b.itertuples(index=False):
        exp_label, system, f1_only, _, _ = STUDIES[(r.exp, r.grp)]
        sh = float(r.shiftMag)
        rows.append({
            "dataset": ID, "study": ID, "experiment": exp_label, "group": "",
            "subject": scode[(r.participant, r.exp, r.grp)], "person": r.participant,
            "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
            "setting": "lab", "device": DEVICE[system], "session": "" if r.exp == "noisyfb" else 1, "block": "",
            "trial": int(r.compTrial), "phase": "learning", "time_s": "", "task": "speech", "unit": "utterance",
            "condition": r.cond, "pert_type": "formant_shift", "pert_value": sh if f1_only else "", "pert_unit": "mel",
            "feedback": "altered", "excluded_by_authors": False,
            "stimulus": "", "segment": "vowel /ɛ/", "measure_window": WINDOW,
            "f1_change_mel": r.compensation, "f1_change_later_mel": r.oneShot, "change_reference": REFERENCE,
            "shift_f1": sh, "shift_f2": 0.0 if f1_only else "",
        })
    a = pd.DataFrame(rows)
    a["exp"], a["grp"], a["code"], a["next_trial"] = b.exp.values, b.grp.values, b.participant.values, b.OSTrial.values
    a["f1c"] = pd.to_numeric(a.f1_change_mel)
    a["f1l"] = pd.to_numeric(a.f1_change_later_mel)

    # ---- the authors' participant-level table, written alongside with the same subject codes
    pm = os_t.merge(cp_t, on=["participant", "cond", "exp", "shiftMag", "group"], how="outer", validate="one_to_one")
    pm["grp"] = pm.group.fillna("na")
    # the participant tables call Study 1 'random' (the trial table calls it 'cereb'); same people, same shift sizes
    pm["exp"] = pm.exp.replace({"random": "cereb"})
    # and Study 6 participants are 's<N>' in the trial table but 'f<N>' in the participant tables (same 15 numbers; the
    # shift-size check below confirms the match, because Study 6 shifts are participant-specific)
    pm["participant"] = np.where(pm.exp == "fmtAlt", pm.participant.str.replace(r"^f", "s", regex=True), pm.participant)
    pkeys = set(zip(pm.participant, pm.exp, pm.grp))
    ck.internal("the participant-level tables hold the same 131 participant datasets as the trial table, two rows each "
                "(up, down), with the same shift sizes", pkeys == set(keys) and len(pm) == 262
                and pm.groupby(["participant", "exp", "grp"]).cond.apply(lambda s: sorted(s) == ["down", "up"]).all()
                and all(np.isclose(pm.set_index(["participant", "exp", "grp", "cond"]).shiftMag.loc[k],
                                   b.groupby(["participant", "exp", "grp", "cond"]).shiftMag.first().loc[k])
                        for k in pm.set_index(["participant", "exp", "grp", "cond"]).index), f"{len(pm)} rows")
    pm["subject"] = [scode[k] for k in zip(pm.participant, pm.exp, pm.grp)]
    pm = pm.sort_values(["subject", "cond"], ascending=[True, False])
    means_csv = "subject,condition,shift_f1_mel,f1_change_mel,f1_change_later_mel\r\n" + "".join(
        f"{r.subject},{r.cond},{stdlib._fmt(float(r.shiftMag), 'number')},{stdlib._fmt(float(r.compensation), 'number')},"
        f"{stdlib._fmt(float(r.oneShot), 'number')}\r\n" for r in pm.itertuples())
    for v in ("compensation", "oneShot"):
        pm[v] = pm[v].map(lambda x: float(stdlib._fmt(float(x), "number")))  # the values as written

    # ---- internal checks
    ck.internal("one row per row of the authors' trial table, in its order", len(a) == len(b) == 10410, f"{len(a)} rows")
    ck.internal("the next trial is always the altered trial + 1 (OSTrial = compTrial + 1)", bool((a.next_trial == a.trial + 1).all()))
    per = a.groupby(["exp", "grp"]).subject.nunique().to_dict()
    ck.internal("participant datasets per study: 13, 12, 40, 40, 11, 15", all(per[k] == STUDIES[k][4] for k in STUDIES), str(per))
    ck.internal("131 subject sequences and 92 participant codes; 39 codes take part in both noisyfb sessions",
                a.subject.nunique() == 131 and a.code.nunique() == 92
                and a[a.exp == "noisyfb"].groupby("code").grp.nunique().eq(2).sum() == 39)
    one = a.groupby(["subject", "condition"]).shift_f1.nunique()
    sg = a.groupby(["subject", "condition"]).shift_f1.first().unstack()
    ck.internal("one shift size per subject and direction, positive for up and negative for down, 61-164 mel",
                bool((one == 1).all() and (sg.up > 0).all() and (sg.down < 0).all()
                     and sg.abs().stack().between(61, 164).all()), f"{sg.abs().stack().min():.1f}-{sg.abs().stack().max():.1f}")
    ck.internal("F1-only studies (1-4) shift by 125 mel or by a per-participant 105-141 mel; 2-D studies (5, 6) have no "
                "pert_value", bool((a[a.exp.isin(["noisyfb", "reachAndSpeech"])].shift_f1.abs() == 125).all()
                                   and a[a.exp == "cereb"].shift_f1.abs().between(104, 142).all()
                                   and (a[a.exp.isin(["cat", "fmtAlt"])].pert_value == "").all()))
    ck.internal("rows of each subject come as all up rows, then all down rows (the authors' order)",
                all((np.diff((g.condition == "down").to_numpy().astype(int)) >= 0).all() for _, g in a.groupby("subject", sort=False)))
    ck.internal("no altered trial appears twice within a subject", not a.duplicated(["subject", "trial"]).any())
    ck.internal("trial numbers within each study's session length (Table 1: 160, 120, 240, 240, 400, 800 trials; "
                "one normal-session dataset is numbered up to 689)",
                bool(a.trial.between(2, 800).all()), f"{a.trial.min()}-{a.trial.max()}")
    ck.internal("f1_change_later_mel present on every row; f1_change_mel missing on 65 rows (as shared)",
                bool(a.f1l.notna().all() and a.f1c.isna().sum() == 65), f"{int(a.f1c.isna().sum())} missing")
    ck.internal("F1 changes within -320 and 290 mel", bool(a.f1l.abs().max() < 320 and a.f1c.abs().max() < 300),
                f"{a.f1c.min():.1f} to {a.f1c.max():.1f}; {a.f1l.min():.1f} to {a.f1l.max():.1f}")

    # ---- published numbers, trial level (lmer_byTrial.Rmd; Results)
    P = f"{SRC}, Results"
    up, dn = a[a.condition == "up"], a[a.condition == "down"]
    ck.published("participant datasets analysed (t(130))", a.subject.nunique(), 131, 0, f"{P}, participant-level tests; Table 1")
    for k, (lab, _, _, _, n) in STUDIES.items():
        ck.published(f"participant datasets included, {lab} (included minus outliers)", per[k], n, 0, f"{SRC}, Table 1")
    ck.published("compensation, up-shift trials, trial-level mean (mel)", up.f1c.mean(), -3.99, 0.005, f"{P}, Compensation")
    ck.published("one-shot adaptation, post-up trials, trial-level mean (mel)", up.f1l.mean(), -1.55, 0.005, f"{P}, One-shot adaptation")
    ck.published("one-shot adaptation, post-down trials, trial-level mean (mel)", dn.f1l.mean(), 0.59, 0.005, f"{P}, One-shot adaptation")
    import statsmodels.formula.api as smf
    t = a.copy()
    t["exp2"] = np.where(t.grp == "normal", "normalfb", t.exp)  # as lmer_byTrial.Rmd
    t["one"] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = {}
        for v in ("f1l", "f1c"):
            d = t.dropna(subset=[v])
            fits[v, "pe"] = smf.mixedlm(f"{v} ~ C(condition, Treatment('down'))", d, groups=d["one"], re_formula="0",
                                        vc_formula={"p": "0 + C(code)", "e": "0 + C(exp2)"}).fit(reml=True)
            fits[v, "p"] = smf.mixedlm(f"{v} ~ C(condition, Treatment('down'))", d, groups=d["code"]).fit(reml=True)
    m = fits["f1l", "pe"]
    ck.published("one-shot ~ condition, trial level, LMM (1|participant) + (1|study): beta up vs down (mel)", m.params.iloc[1],
                 -2.14, 0.005, f"{P}, One-shot adaptation")
    ck.published("one-shot ~ condition, trial level, LMM: SE of beta", m.bse.iloc[1], 0.53, 0.005, f"{P}, One-shot adaptation")
    ck.published("one-shot ~ condition, trial level, effect size 2.14 / residual SD (as lmer_byTrial.Rmd)", 2.14 / np.sqrt(m.scale),
                 0.079, 0.0005, f"{P}, One-shot adaptation")
    m = fits["f1c", "p"]
    ck.published("compensation ~ condition, trial level, LMM (1|participant): beta up vs down (mel)", m.params.iloc[1], -6.93,
                 0.005, f"{P}, Compensation")
    ck.published("compensation ~ condition, trial level, LMM: SE of beta", m.bse.iloc[1], 0.66, 0.005, f"{P}, Compensation")
    ck.published("compensation ~ condition, trial level, effect size 6.93 / residual SD (as lmer_byTrial.Rmd)", 6.93 / np.sqrt(m.scale),
                 0.21, 0.005, f"{P}, Compensation")
    comp_pe = fits["f1c", "pe"].params.iloc[1]
    # relationship model on flipped signs (up-shift responses times -1; shift size over its mean), lmer_byTrial.Rmd
    f = t.copy()
    f.loc[f.condition == "up", ["f1c", "f1l"]] *= -1
    f["mag"] = f.shift_f1.abs() / f.shift_f1.abs().mean()
    fd = f.dropna(subset=["f1c"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mr = smf.mixedlm("f1l ~ f1c + mag + f1c:mag", fd, groups=fd["code"]).fit(reml=True)
    R = f"{P}, Relationship between compensation and one-shot adaptation"
    ck.published("trial-level relationship model: compensation x shift magnitude beta", mr.params["f1c:mag"], 0.16, 0.005, R)
    ck.published("trial-level relationship model: compensation x shift magnitude SE", mr.bse["f1c:mag"], 0.052, 0.0005, R)
    ck.published("trial-level relationship model: compensation main effect beta", mr.params["f1c"], -0.033, 0.0005, R)
    ck.published("trial-level relationship model: compensation main effect SE", mr.bse["f1c"], 0.053, 0.0005, R)
    ck.published("trial-level relationship model: shift magnitude beta", mr.params["mag"], 7.46, 0.01, R)
    # within-participant correlations by participant code (correlation_Test.Rmd: 92 codes, Fisher z, t-test against 0)
    rs = np.array([stats.pearsonr(g.f1l, g.f1c)[0] for _, g in fd.groupby("code")])
    z = np.arctanh(rs)
    tz = stats.ttest_1samp(z, 0).statistic
    ck.published("participants in the trial-wise correlation analysis", len(rs), 92, 0, R)
    ck.published("participants with a positive trial-wise correlation", int((rs > 0).sum()), 78, 0, R)
    ck.published("mean trial-wise correlation r (back-transformed mean Fisher z)", np.tanh(z.mean()), 0.21, 0.005, R)
    ck.published("t-test of Fisher z against 0", tz, 11.07, 0.005, R)
    ck.published("Cohen's d of Fisher z against 0 (pooled-SD formula of effsize::cohen.d with a zero vector)",
                 z.mean() / np.sqrt(z.var(ddof=1) / 2), 1.6, 0.05, R)

    # ---- published numbers, participant level (lmer_byParticipant.Rmd), from subject_means.csv
    pu = pm[pm.cond == "up"].set_index("subject").sort_index()
    pdn = pm[pm.cond == "down"].set_index("subject").sort_index()
    ck.internal("subject_means.csv: one up and one down row per subject", (pu.index == pdn.index).all() and len(pu) == 131)
    Q = f"{SRC}, Results, participant level"
    for v, nm, sec, means in (("compensation", "compensation", "Compensation", (-2.75, 4.40)),
                              ("oneShot", "one-shot adaptation", "One-shot adaptation", (-2.08, 0.82))):
        ck.published(f"{nm}, up shifts, mean of participant means (mel)", pu[v].mean(), means[0], 0.005, f"{Q} ({sec})")
        ck.published(f"{nm}, down shifts, mean of participant means (mel)", pdn[v].mean(), means[1], 0.005, f"{Q} ({sec})")
    ck.published("compensation, paired t up vs down, t(130)", stats.ttest_rel(pu.compensation, pdn.compensation).statistic,
                 -7.00, 0.005, f"{Q} (Compensation)")
    ck.published("compensation, up shifts vs 0, t(130)", stats.ttest_1samp(pu.compensation, 0).statistic, -3.63, 0.005,
                 f"{Q} (Compensation)")
    ck.published("compensation, down shifts vs 0, t(130)", stats.ttest_1samp(pdn.compensation, 0).statistic, 7.24, 0.005,
                 f"{Q} (Compensation)")
    ck.published("one-shot adaptation, paired t up vs down, t(130)", stats.ttest_rel(pu.oneShot, pdn.oneShot).statistic,
                 -2.98, 0.005, f"{Q} (One-shot adaptation)")
    ck.published("one-shot adaptation, paired t up vs down, p", stats.ttest_rel(pu.oneShot, pdn.oneShot).pvalue, 0.0034,
                 0.00005, f"{Q} (One-shot adaptation)")
    ck.published("one-shot adaptation, post-up vs 0, t(130)", stats.ttest_1samp(pu.oneShot, 0).statistic, -3.2, 0.05,
                 f"{Q} (One-shot adaptation)")
    ck.published("one-shot adaptation, post-up vs 0, p", stats.ttest_1samp(pu.oneShot, 0).pvalue, 0.0016, 0.00005,
                 f"{Q} (One-shot adaptation)")
    ck.published("one-shot adaptation, post-down vs 0, t(130)", stats.ttest_1samp(pdn.oneShot, 0).statistic, 1.2, 0.05,
                 f"{Q} (One-shot adaptation)")
    ck.published("one-shot adaptation, post-down vs 0, p", stats.ttest_1samp(pdn.oneShot, 0).pvalue, 0.23, 0.005,
                 f"{Q} (One-shot adaptation)")

    def d_pooled(x, y):
        return (x.mean() - y.mean()) / np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)

    zero = pd.Series(np.zeros(131))
    ck.published("compensation, Cohen's d up vs down (pooled SD)", abs(d_pooled(pu.compensation, pdn.compensation)), 0.91,
                 0.005, f"{Q} (Compensation)")
    ck.published("compensation, down shifts vs 0, Cohen's d (pooled SD with a zero vector)", d_pooled(pdn.compensation, zero),
                 0.89, 0.005, f"{Q} (Compensation)")
    ck.published("one-shot adaptation, Cohen's d up vs down (pooled SD)", abs(d_pooled(pu.oneShot, pdn.oneShot)), 0.38, 0.005,
                 f"{Q} (One-shot adaptation)")
    ck.published("one-shot adaptation, post-down vs 0, Cohen's d (pooled SD with a zero vector)", d_pooled(pdn.oneShot, zero),
                 0.15, 0.005, f"{Q} (One-shot adaptation)")
    fl = pd.concat([pu.assign(cond="up"), pdn.assign(cond="down")]).reset_index()
    fl["code"] = fl.subject.map({v: k[0] for k, v in scode.items()})
    fl.loc[fl.cond == "up", ["oneShot", "compensation"]] *= -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp = smf.mixedlm("oneShot ~ compensation", fl, groups=fl["code"]).fit(reml=True)
    ck.published("participant-level relationship model (one-shot ~ compensation + (1|participant), flipped signs): beta",
                 mp.params["compensation"], 0.14, 0.005, f"{R}, participant level")
    near = dict(
        comp_down_trial_mean=dn.f1c.mean(), comp_pe=comp_pe, mag_se=mr.bse["mag"], part_se=mp.bse["compensation"],
        d_comp_up0=abs(d_pooled(pu.compensation, zero)), d_os_up0=abs(d_pooled(pu.oneShot, zero)))

    for c in ck.items:
        if not c["ok"]:
            print("FAILED:", c)
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="MIT License",
        license_url=f"{REPO}/blob/main/LICENSE", license_text=MIT,
        changes=("Re-encoded the authors' per-trial table (byTrial_4_23_21.csv) with one row per row, added the perturbation "
                 "and study columns from the authors' shift sizes and Table 1, wrote the authors' per-participant tables "
                 "(oneShot_4_23_21.csv, compensation_04_23_21.csv) as subject_means.csv, and replaced participant "
                 "identifiers with codes."),
        how=[
            "One row is one trial with an unpredictable F1 shift in the feedback that was followed by an unaltered trial, in "
            "the order of the authors' table (each subject's up-shift rows, then its down-shift rows; trial is the "
            "authors' trial number in the session, and the next trial is trial + 1); the word is not in the shared table "
            "(Study 1 used beck, bet, deck, debt, pet, tech; Study 2 dead, fed, said, shed; Studies 3-4 bed, dead, head; "
            "Study 5 bed, bet, dead, deb, debt, ped, tech, ted; Study 6 head).",
            "f1_change_mel is the authors' compensation on this trial and f1_change_later_mel their one-shot adaptation on "
            "the next trial: F1 in mels minus the participant's mean F1 track for the same word over the reference "
            "trials in change_reference, averaged over the window in measure_window; 65 trials have no compensation "
            "value in the shared table.",
            "pert_value and shift_f1 are the F1 shift in mels from the authors' table (positive raises F1 in what the "
            "participant heard); Studies 5 and 6 shifted F1 and F2 together by participant-specific vectors and the table "
            "gives only the F1 part, so there pert_value and shift_f2 are empty.",
            "subject is one participant in one study and person is the participant code, so the people of Niziolek & "
            "Parrell 2021 (normal and noisy sessions; the noisy session began with an exposure phase of small random "
            "formant shifts, not in the table; session order is not given) have two subject codes; the authors removed "
            "their two outlier participants and vowels shorter than 100 ms before sharing, and subject_means.csv holds "
            "their participant-level values (one row per subject and shift direction, same codes, taken from the "
            "averaged F1 track, so they differ slightly from means of the rows here); the authors' participant tables label "
            "Study 1 'random' (the trial table: 'cereb') and give Study 6 participants an f instead of an s before the number, "
            "so they were matched to the trial table by study, participant number and shift size, which agree for all 131.",
        ],
        unit_note="altered-feedback trial (one spoken word) of one participant, with the F1 change on the trial after it",
        differences=[
            f"Trial-level mean compensation on down-shift trials is {near['comp_down_trial_mean']:.2f} mel in the shared "
            "table; the paper reports 2.69 (probably transposed digits; the up-shift mean, -3.99, matches).",
            f"The paper's compensation model with random intercepts for participant and study gives {near['comp_pe']:.2f} "
            "here (statsmodels); the reported -6.93 is matched by the participant-only model used as the check, as if lme4 "
            "estimated the study variance at zero. The one-shot model with both intercepts matches (-2.14).",
            f"SEs not reproduced: shift magnitude in the trial-level relationship model {near['mag_se']:.2f} (paper 3.34); "
            f"participant-level relationship {near['part_se']:.3f} (paper 0.058). The trial-level SEs of the condition means "
            "(0.26-0.33) are not plain SEs of the rows and were not reproduced.",
            f"Cohen's d against 0 for up shifts: {near['d_comp_up0']:.2f} for compensation (paper 0.49) and "
            f"{near['d_os_up0']:.2f} for one-shot adaptation (paper 0.38), with the formula that reproduces the paper's "
            "other d values.",
            "The paper counts 91 people, 40 of them in both Niziolek & Parrell 2021 sessions; the shared table has 92 "
            "participant codes, 39 in both sessions and one only in each, so one person probably carries two codes.",
        ],
    )
    rec = stdlib.release(ID, "speech", rows, meta, ck, extra_files=[("subject_means.csv", means_csv.encode())])
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.5g} vs {c['reported']}")
    print(near)
    return rec


if __name__ == "__main__":
    main()
