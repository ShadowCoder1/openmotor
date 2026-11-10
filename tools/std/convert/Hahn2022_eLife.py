"""Hahn et al. 2022, eLife: juggling performance of 28 adolescents and 41 adults (figure source data of the Dryad/Zenodo deposit).
Run on the PC: python3 -I tools/std/convert/Hahn2022_eLife.py. Raw: raw_extra/Hahn2022_eLife/hahn.zip (read only)."""
import io, sys, zipfile
from pathlib import Path
import numpy as np, scipy.io as sio
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib

DS = "Hahn2022_eLife"
Z = zipfile.ZipFile(stdlib.raw_dir(DS) / "hahn.zip")
BASE = [n for n in Z.namelist() if n.endswith("PlotFigure2.m")][0][:-len("PlotFigure2.m")]


def mat(n):
    return sio.loadmat(io.BytesIO(Z.read(BASE + "Data/" + n + ".mat")))


def main():
    ck = stdlib.Checks()
    f2a, f2b, f2d, f2e = mat("Figure2A"), mat("Figure2B"), mat("Figure2D"), mat("Figure2E")
    rows = []
    sizes = {}

    def base(pop, grp, pid, trial):
        return dict(dataset=DS, study="Hahn 2022 eLife", experiment="juggling learning, sleep-first vs wake-first",
                    group=f"{pop}, {grp}", subject=pid, person=pid, subject_excluded_by_authors=False, age=None, sex=None,
                    handedness=None, population="healthy adolescents" if pop == "adol" else "healthy adults", setting="lab",
                    device="three-ball cascade juggling, counted cascades", session=None, block=None, trial=trial, phase="other",
                    time_s=None, task="sports and games", unit="participant summary", condition=grp, pert_type="none",
                    pert_value=0, pert_unit=None, feedback="normal", excluded_by_authors=False)

    def add(b, measure, desc, v, unit, n=None):
        rows.append({**b, "measure": measure, "measure_description": desc, "statistic": "value", "value": float(v),
                     "value_unit": unit, "n_trials": n, "session_context": ""})

    for pop in ("adol", "adult"):
        for grp in ("sleepfirst", "wakefirst"):
            P = f"{pop}_{grp}"
            blocks = (f2a if pop == "adol" else f2b)[f"{P}_blocks"]
            prof = {k: f2e[f"{P}_taskproficiency_{k}"].ravel() for k in ("pre", "post")}
            lc = {k: f2d[f"{P}_learningcurve_{k}"].ravel() for k in ("pre", "post")}
            n = len(prof["pre"])
            sizes[P] = n
            same = blocks.shape[0] == n
            for i in range(n):
                pid = f"{P}_{i + 1:02d}"
                b = base(pop, grp, pid, 1)
                for k in ("pre", "post"):
                    add(b, f"taskproficiency_{k}", f"Figure2E variable {P}_taskproficiency_{k} (authors' task proficiency estimate, "
                        "from 3-ball cascades per juggling block)", prof[k][i], "cascades")
                    add(b, f"learningcurve_{k}", f"Figure2D variable {P}_learningcurve_{k} (slope of the learning curve)",
                        lc[k][i], "cascades per block")
                if same:
                    for j, v in enumerate(blocks[i]):
                        add({**b, "block": str(j + 1)}, "cascades", "Number of 3-ball cascades in the juggling block (Figure2A/B)",
                            v, "count")
            if not same:
                for i in range(blocks.shape[0]):  # 15 rows against 17 people: cannot be matched to the proficiency rows
                    b = base(pop, grp, f"{P}_blocks_{i + 1:02d}", 1)
                    for j, v in enumerate(blocks[i]):
                        add({**b, "block": str(j + 1)}, "cascades", "Number of 3-ball cascades in the juggling block (Figure2A)",
                            v, "count")
    ck.internal("block matrices have 9 (adolescent) or 15 (adult) blocks", f2a["adol_sleepfirst_blocks"].shape[1] == 9
                and f2b["adult_wakefirst_blocks"].shape[1] == 15)
    ck.internal("no missing values in rows", all(not np.isnan(r["value"]) for r in rows))
    src = "Materials and methods, Participants"
    ck.published("adolescents", sizes["adol_sleepfirst"] + sizes["adol_wakefirst"], 28, 0, "Results, first paragraph (n = 28)")
    ck.published("adults", sizes["adult_sleepfirst"] + sizes["adult_wakefirst"], 41, 0, "Results, first paragraph (n = 41)")
    ck.published("adolescent sleep-first", sizes["adol_sleepfirst"], 17, 0, src)
    ck.published("adolescent wake-first", sizes["adol_wakefirst"], 11, 0, src)
    ck.published("adult sleep-first", sizes["adult_sleepfirst"], 25, 0, src)
    ck.published("adult wake-first", sizes["adult_wakefirst"], 16, 0, src)
    meta = dict(
        study=DS,
        paper=dict(authors="Hahn MA, Bothe K, Heib D, Schabus M, Helfrich RF, Hoedlmoser K", year=2022,
                   title="Slow oscillation-spindle coupling strength predicts real-life gross-motor learning in adolescents and adults",
                   venue="eLife 11:e66761", doi="10.7554/eLife.66761"),
        lab="Hoedlmoser (University of Salzburg)", data_url="https://doi.org/10.5061/dryad.qfttdz0gh (copy: https://zenodo.org/records/6115961)",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Read the figure source-data MATLAB files (Figure2A, 2B, 2D, 2E) and wrote one row per participant, measure and "
                 "juggling block. EEG coupling files (Figure 3) and the Figure 2F change scores were not included."),
        how=["One row is one measure of one participant: cascades (number of 3-ball cascades) in one of 9 (adolescents) or 15 (adults) juggling blocks, "
             "or the authors' task proficiency and learning-curve estimates before (pre) and after (post) a retention interval, as named in their Figure 2D/E files.",
             "Participants are identified only by their row within each group file. Adult and adolescent wake-first block rows are assumed to be the same people as the "
             "proficiency rows because the counts agree; for adolescent sleep-first the block file has 15 rows against 17 people, so those rows are coded as separate persons "
             "(the same people may appear under both codes).",
             "Group is population and whether the first retention interval was sleep or wake. No demographics are in the deposit."],
        unit_note="measure of one participant",
        differences=["The paper text gives no performance means (only regression statistics), so published checks are the sample sizes. "
                     "The meaning of 'pre' and 'post' is the authors' variable naming; the deposit readme does not define them further."])
    stdlib.release(DS, "summary", rows, meta, ck)


if __name__ == "__main__":
    main()
