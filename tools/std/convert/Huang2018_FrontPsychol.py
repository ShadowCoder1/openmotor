"""Huang, Hegele & Billino (2018) Front Psychol: motivation and age effects on reaching (visuomotor rotation) adaptation.
Raw: raw_extra/Huang2018_FrontPsychol/MotReachAdapt.zip (Zenodo 1461957): MotReachAdapt_total (67 participants, exclusion flag)
and MotReachAdapt_final (55 included participants, more measures), tab-separated, no header; column labels from the
description files. One row per participant; released in long form (summary module). Nothing is executed."""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Huang2018_FrontPsychol"
COND = {1: "reward", 2: "punishment", 3: "neutral"}
AGE = {1: "younger", 2: "older"}
# (column index in total, name, description, unit)
TOT = [(5, "exclusion_rate", "Proportion of excluded trials", "percent"),
       (6, "baseline_time", "Overall movement time in the baseline block", "ms"),
       (7, "learning_rate_adaptation", "Learning rate B of the single-state model, adaptation block", "ratio"),
       (8, "final_direction_adaptation", "Final reaching direction in the adaptation block (last third of trials)", "deg"),
       (9, "final_direction_retention", "Final reaching direction in the retention block (last third of trials)", "deg"),
       (10, "savings", "Learning rate in readaptation minus learning rate in adaptation", "ratio")]
FIN = [(5, "points_per_adaptation_trial", "Points per trial in the adaptation block (reward and punishment conditions)", "points"),
       (8, "retention_rate", "Retention rate A of the single-state model, retention block", "ratio"),
       (10, "final_direction_washout", "Final reaching direction in the washout block", "deg"),
       (11, "learning_rate_readaptation", "Learning rate B, readaptation block", "ratio")]
for i, (blk) in enumerate(["baseline", "adaptation", "retention", "readaptation"]):
    FIN.append((13 + 2 * i, f"RT_{blk}", f"Reaction time (trial start to leaving the start square), {blk} block", "ms"))
    FIN.append((14 + 2 * i, f"MT_{blk}", f"Movement time (leaving the start square to reaching the target circle), {blk} block", "ms"))


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(DS) / "MotReachAdapt.zip")
    t = pd.read_csv(io.BytesIO(z.read("MotReachAdapt_total/MotReachAdapt_total.txt")), sep="\t", header=None, skip_blank_lines=True)
    ls = [l for l in z.read("MotReachAdapt_final/MotReachAdapt_final.txt").decode().splitlines() if l.strip()]
    f = pd.read_csv(io.StringIO("\n".join(ls)), sep="\t", header=None, usecols=range(21)).set_index(0)
    ck = stdlib.Checks()
    ck.internal("final participants are the non-excluded ones of the total file",
                set(f.index) == set(t[t[11] == 0][0]) and len(f) == 55)
    tt = t.set_index(0)
    ix = f.index
    d = lambda fc, tc: float((f[fc] - tt.loc[ix, tc]).abs().max())
    ck.internal("shared measures agree between the two files (learning rate, final direction, retention dir, savings)",
                d(6, 7) < 0.006 and d(7, 8) < 0.006 and d(9, 9) < 0.006 and d(12, 10) < 0.006,
                f"{d(6,7):.4f} {d(7,8):.4f} {d(9,9):.4f} {d(12,10):.4f}")
    rows = []
    for pid, r in tt.iterrows():
        exc = bool(r[11])
        base = dict(dataset=DS, study="Huang 2018 Front Psychol", experiment="reaching-adaptation",
                    group=f"{AGE[r[3]]}_{COND[r[4]]}", subject=f"p{pid}", person=f"p{pid}", subject_excluded_by_authors=exc,
                    age=r[1], sex="M" if r[2] == 1 else "F", handedness=None, population="healthy", setting="lab",
                    device="robot handle", session=None, block=None, trial=None, phase="other", time_s=None,
                    task="reaching", unit="participant summary", condition=COND[r[4]], pert_type="rotation",
                    pert_value=30, pert_unit="deg", feedback="altered", excluded_by_authors=exc)
        meas = [(r[c], n, d, u) for c, n, d, u in TOT]
        if pid in f.index:
            meas += [(f.loc[pid, c], n, d, u) for c, n, d, u in FIN]
        for v, n, d, u in meas:
            if pd.isna(v):
                continue
            rows.append({**base, "measure": n, "measure_description": d, "statistic": "value", "value": float(v),
                         "value_unit": u, "n_trials": None})
    ck.internal("rows = 67x6 + 55x12 minus 29 empty point cells", len(rows) == 67 * 6 + 55 * 12 - int(f[5].isna().sum()), str(len(rows)))
    yg = t[3] == 1
    ck.published("young adults recruited", int(yg.sum()), 35, 0, "Methods, Participants")
    ck.published("older adults recruited", int((~yg).sum()), 32, 0, "Methods, Participants")
    ck.published("mean age young (years)", t[yg][1].mean(), 25.3, 0.05, "Methods, Participants")
    ck.published("SD age young (years)", t[yg][1].std(), 4.3, 0.05, "Methods, Participants")
    ck.published("mean age older (years)", t[~yg][1].mean(), 68.4, 0.05, "Methods, Participants")
    ck.published("SD age older (years)", t[~yg][1].std(), 5.0, 0.05, "Methods, Participants")
    ck.published("final sample young", int((f[3] == 1).sum()), 29, 0, "Data Analysis")
    ck.published("final sample older", int((f[3] == 2).sum()), 26, 0, "Data Analysis")
    for g, c, n in ((1, 3, 13), (1, 1, 10), (1, 2, 12), (2, 3, 10), (2, 1, 10), (2, 2, 12)):
        ck.published(f"recruited {AGE[g]} {COND[c]}", int(((t[3] == g) & (t[4] == c)).sum()), n, 0, "Procedure")
    ck.published("young reward points/trial", f[(f[3] == 1) & (f[4] == 1)][5].mean(), 2.72, 0.005, "Results, Visuomotor Learning")
    ck.published("older reward points/trial", f[(f[3] == 2) & (f[4] == 1)][5].mean(), 1.90, 0.005, "Results, Visuomotor Learning")
    for (g, c), (rt, mt) in {(1, 3): (403, 123), (1, 1): (413, 131), (1, 2): (424, 144),
                             (2, 3): (481, 144), (2, 1): (514, 137), (2, 2): (529, 140)}.items():
        s = f[(f[3] == g) & (f[4] == c)]
        ck.published(f"baseline RT {AGE[g]} {COND[c]}", s[13].mean(), rt, 0.5, "Table 1")
        ck.published(f"baseline MT {AGE[g]} {COND[c]}", s[14].mean(), mt, 0.5, "Table 1")
    meta = dict(
        study=DS,
        paper=dict(authors="Huang J, Hegele M, Billino J", year=2018,
                   title="Motivational Modulation of Age-Related Effects on Reaching Adaptation",
                   venue="Front Psychol 9:2285", doi="10.3389/fpsyg.2018.02285"),
        lab="Hegele & Billino (University of Giessen)", data_url="https://doi.org/10.5281/zenodo.1461957",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Gave the two tab-separated tables column names from the description files, merged them by participant, "
                 "turned them into one long table (a row per participant and measure) and replaced participant numbers with codes."),
        how=["One row is one measure of one participant: young or older adults reaching with a handle under a 30 deg "
             "counter-clockwise visuomotor rotation in a reward, punishment or neutral motivation condition (group gives age group "
             "and condition). Directions are in deg, reaction and movement times in ms, learning and retention rates are model "
             "parameters (ratios), savings is the readaptation minus adaptation learning rate.",
             "Participants flagged exclusion = 1 in the authors' total table (12) are kept and marked excluded; the extra measures "
             "of the final table exist only for the 55 included participants. Points per trial exist only in the reward and "
             "punishment conditions; the unit sign is as in the file.",
             "The perturbation columns describe the adaptation block (30 deg); the baseline measures were taken without it."],
        unit_note="measure of one participant",
        differences=["Points per adaptation trial in the punishment condition are positive in the file (young about 2.45, older about 1.71), "
                     "whereas the paper reports mean losses of -1.55 and -2.30 per trial; the definition of the file's column is not "
                     "documented, so these were not used as a check.",
                     "The paper's Table 1 gives means with SEM; only the means were compared."])
    stdlib.release(DS, "summary", rows, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.3f} vs {c['reported']} ok={c['ok']}")


if __name__ == "__main__":
    main()
