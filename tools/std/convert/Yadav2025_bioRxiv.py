"""Yadav, Chauvaux & Duque 2025 (bioRxiv): per-participant mean Total Error per bin of trials, skill training (right arm),
transfer (left arm) and retention. Run on the PC: python3 -I tools/std/convert/Yadav2025_bioRxiv.py
Raw (read only): raw_extra/Yadav2025_bioRxiv/fig2.xlsx from https://osf.io/5fxqj/ (Fig 2 source file)."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib

DS = "Yadav2025_bioRxiv"
PHASE = {"1_Training": "learning", "2_Transfer": "transfer", "3_Retention": "test"}


def main():
    d = pd.read_excel(stdlib.raw_dir(DS) / "fig2.xlsx")
    ck = stdlib.Checks()
    d["trial"] = d.groupby("Subject ID").cumcount() + 1
    rows = []
    for r in d.itertuples(index=False):
        rec = r._asdict()
        rows.append(dict(
            dataset=DS, study="Yadav 2025 bioRxiv", experiment="intermanual transfer with rTMS", group=r[3],
            subject=f"S{r[0]}", person=f"S{r[0]}", subject_excluded_by_authors=False, age=None, sex=None, handedness="right",
            population="healthy", setting="lab", device="skill reaching task (arm), rTMS over M1", session=None, block=str(r[2]),
            trial=int(rec["trial"]), phase=PHASE[r[1]], time_s=None, task="sports and games", unit="participant summary",
            condition=f"{r[1]}, {r[4]} hand", pert_type="none", pert_value=0, pert_unit=None, feedback="normal",
            excluded_by_authors=False, measure="Mean(Total Error)",
            measure_description="Mean of the authors' Total Error over the trials of the bin (column 'Mean(Total Error)')",
            statistic="mean", value=r[6], value_unit="unspecified (authors' Total Error)", n_trials=int(r[5]),
            session_context=""))
    ck.internal("1302 rows, none missing", len(d) == 1302 and not d.isna().any().any())
    ck.internal("20 training and 20 transfer bins per participant, 2 retention bins",
                bool((d.groupby(["Subject ID", "Condition"]).size().unstack().fillna(0).values == [20, 20, 2]).all()))
    g = d.groupby("Group")["Subject ID"].nunique()
    ck.published("participants", d["Subject ID"].nunique(), 31, 0, "bioRxiv abstract/Methods, Participants (31 right-handed adults)")
    ck.published("cM1 group size (Trained M1)", g["Trained M1"], 15, 0, "Methods, Participants (cM1 n = 15)")
    ck.published("iM1 group size (Untrained M1)", g["Untrained M1"], 16, 0, "Methods, Participants (iM1 n = 16)")
    meta = dict(
        study=DS,
        paper=dict(authors="Yadav G, Chauvaux M, Duque J", year=2025,
                   title="Non-invasive Stimulation of Contralateral Primary Motor Cortex Reduces the Amount of Skill Generalization to the Untrained Arm",
                   venue="bioRxiv (preprint; no published version found in Europe PMC)", doi="10.1101/2025.07.22.666199"),
        lab="Duque (UCLouvain)", data_url="https://osf.io/5fxqj/", license="none stated", license_url="",
        changes=("Read the Fig 2 source spreadsheet (one row per participant and bin) and wrote it unchanged as long-form summary rows. "
                 "Trial is the order of the bin within the participant; block is the authors' bin number. The Fig 3-6 spreadsheets (first/last blocks, "
                 "MEP data) were not included."),
        how=["One row is the mean Total Error of one participant over one bin of consecutive trials (n_trials says how many) in 20 bins of right-arm training, "
             "20 bins of left-arm transfer and 2 bins of right-arm retention.",
             "The deposit does not state the unit or definition of Total Error, so value_unit is left unspecified. Group labels are the authors': 'Trained M1' "
             "and 'Untrained M1'; they are matched to the paper's cM1 (n = 15) and iM1 (n = 16) groups by group size, an inference."],
        unit_note="mean of the bin of trials",
        differences=["Total Error and the bin size are not described in the deposit or the abstract; no mean or SD could be checked, so published checks are counts."])
    stdlib.release(DS, "summary", rows, meta, ck)


if __name__ == "__main__":
    main()
