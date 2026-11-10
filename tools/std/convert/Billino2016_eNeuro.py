"""Billino, Hennig & Gegenfurtner (2016) eNeuro: dopamine genotypes and anticipatory pursuit.

Raw file: raw_extra/Billino2016_eNeuro/PursuitDopamine.zip (Zenodo 197059): anticippursuit.txt (110 rows x 13 columns),
visualpursuit.txt (110 rows x 8 columns), columndescription.txt. One row per participant (the description file says
'single trial' for the visual task, but the file has the same 110 participants once each). Nothing is executed.
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Billino2016_eNeuro"
MODULE = "summary"
ANT = [("peakvel", "Peak eye velocity during the blank period", "deg/s", "masked"),
       ("anticipvel", "Anticipatory eye velocity before visible target onset", "deg/s", "masked"),
       ("anticippos", "Anticipatory eye position at the end of the blank period", "deg", "masked"),
       ("latency", "Pursuit latency after visible target onset", "ms", "normal"),
       ("Csampl", "Amplitude of the first catch-up saccade after visible target onset", "deg", "normal"),
       ("Cslag", "Amplitude lag of the first catch-up saccade after visible target onset", "deg", "normal"),
       ("gain", "Velocity gain during steady-state pursuit", "ratio", "normal"),
       ("pursuitratio", "Pursuit ratio during steady-state pursuit", "ratio", "normal")]
VIS = [("gain", "Velocity gain during steady-state pursuit", "ratio"),
       ("latency", "Pursuit latency after target onset", "ms"),
       ("pursuitratio", "Pursuit ratio during steady-state pursuit", "ratio")]


def main():
    zf = zipfile.ZipFile(stdlib.raw_dir(DS) / "PursuitDopamine.zip")
    a = pd.read_csv(io.BytesIO(zf.read("anticippursuit.txt")), sep=r"\s+", header=None)
    v = pd.read_csv(io.BytesIO(zf.read("visualpursuit.txt")), sep=r"\s+", header=None)
    ck = stdlib.Checks()
    ck.internal("both files hold the same 110 participants with the same age, sex and genotypes in the same order",
                bool((a[[0, 1, 2, 3, 4]].values == v[[0, 1, 2, 3, 4]].values).all()) and len(a) == 110)
    ck.internal("DAT code 99 (missing) only for 4 participants", int((a[4] == 99).sum()) == 4)

    rows, parts = [], []
    for r_a, (_, r_v) in zip(a.itertuples(index=False), v.iterrows()):
        pid = f"p{int(r_a[0])}"
        dat = int(r_a[4])
        base = dict(dataset=DS, study="Billino 2016 eNeuro", group=None, subject=pid, person=pid,
                    subject_excluded_by_authors=(dat == 99), age=r_a[2], sex="M" if r_a[1] == 1 else "F",
                    handedness=None, population="healthy", setting="lab", device=None, session=None, block=None,
                    trial=None, phase="other", time_s=None, task="eye movements", unit="participant summary",
                    condition=None, pert_type="none", pert_value=0, pert_unit=None, excluded_by_authors=(dat == 99))
        # wait: the age/sex columns of the file are 2 and 3 in the description (1-based); here 0-based 1 = age
        parts.append(dict(person=pid, comt_genotype={1: "Val/Val", 2: "Val/Met", 3: "Met/Met"}[int(r_a[3])],
                          dat9_genotype="missing (11-repeat carrier, excluded by the authors)" if dat == 99
                          else ("DAT9+" if dat == 1 else "DAT9-")))
        for i, (k, desc, unit, fb) in enumerate(ANT):
            rows.append({**base, "experiment": "anticipatory-pursuit", "feedback": fb, "measure": k,
                         "measure_description": desc, "statistic": "value", "value": r_a[5 + i], "value_unit": unit})
        for i, (k, desc, unit) in enumerate(VIS):
            rows.append({**base, "experiment": "visually-guided-pursuit", "feedback": "normal", "measure": k,
                         "measure_description": desc, "statistic": "value", "value": r_v[5 + i], "value_unit": unit})
    E_ = None
    ck.internal("110 participants x (8 + 3) measures kept", len(rows) == 110 * 11, str(len(rows)))
    sex_ok = None
    # published numbers (Methods, Participants; Table 1; Table 2)
    ck.published("participants", len(a), 110, 0, "Methods, Participants")
    ck.published("males", int((a[1] == 1).sum()), 18, 0, "Methods, Participants")
    ck.published("mean age (years)", a[2].mean(), 23.7, 0.05, "Methods, Participants")
    ck.published("SD of age (years)", a[2].std(), 5.1, 0.05, "Methods, Participants")
    ck.published("minimum age", a[2].min(), 18, 0, "Methods, Participants (range 18-45)")
    ck.published("maximum age", a[2].max(), 45, 0, "Methods, Participants (range 18-45)")
    for g, n in ((1, 24), (2, 54), (3, 32)):
        ck.published(f"COMT genotype {['Val/Val', 'Val/Met', 'Met/Met'][g - 1]} count", int((a[3] == g).sum()), n, 0, "Table 1")
    ck.published("DAT9+ participants", int((a[4] == 1).sum()), 50, 0, "Table 2")
    ck.published("DAT9- participants", int((a[4] == 0).sum()), 56, 0, "Table 2")
    ck.published("participants in the analysis (110 minus 4 excluded)", int((a[4] != 99).sum()), 106, 0, "Methods, Genotyping")

    meta = dict(
        study=DS,
        paper=dict(authors="Billino J, Hennig J, Gegenfurtner KR", year=2016,
                   title="The Role of Dopamine in Anticipatory Pursuit Eye Movements: Insights from Genetic Polymorphisms in Healthy Adults",
                   venue="eNeuro 3(6):ENEURO.0190-16.2016", doi="10.1523/ENEURO.0190-16.2016"),
        lab="Billino / Gegenfurtner (University of Giessen)", data_url="https://doi.org/10.5281/zenodo.197059",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Gave the two tab-separated files (no header) named columns from the description file, turned them into one "
                 "long table with a row per participant and measure, replaced the participant numbers with codes, and moved "
                 "the two genotypes to participants.csv."),
        how=["One row is one measure of one participant in one task: 8 measures in the anticipatory pursuit task (target blanked "
             "for 500 ms after motion onset on half of 40 trials) and 3 in the visually guided task. Velocities are in deg/s, "
             "positions and saccade amplitudes in deg, latencies in ms; gain and pursuit ratio are ratios.",
             "The deposit gives one value per participant and does not say how trials were averaged, so statistic is 'value' and "
             "the definitions are the authors' one-line descriptions. Signs are as in the file: a negative catch-up saccade lag "
             "and a few negative latencies occur and are kept.",
             "feedback is 'masked' for the three measures of the blank period of the anticipatory task, 'normal' otherwise "
             "(our assignment). COMT (Val/Val, Val/Met, Met/Met) and DAT9 (DAT9+ / DAT9-) genotypes are in participants.csv. "
             "Four DAT 11-repeat carriers (code 99 in the file) were excluded by the authors; they are kept and flagged."],
        unit_note="measure of one participant in one task",
        differences=["The deposit's description file says the visual-pursuit file has 'a single trial' per row; it has one row per participant."],
    )
    pc = [("comt_genotype", "category", "Val/Val, Val/Met, Met/Met", "COMT Val158Met genotype."),
          ("dat9_genotype", "text", "DAT9+, DAT9-", "DAT (SLC6A3) 3' UTR VNTR group; DAT9+ carries a 9-repeat allele.")]
    stdlib.release(DS, MODULE, rows, meta, ck, participants=parts, participant_columns=pc)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
