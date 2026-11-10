"""Bennett, Ni Chiosain, Padgett & McGuire (2018), J Int Phon Assoc 48(3):261-304 (online 16 Nov 2017): an ultrasound study of
Connemara Irish palatalization and velarization (Phase 1, Connacht dialect).

Raw files, raw_extra/Bennett2018_JIPA/ (Zenodo copy of the Dryad deposit doi:10.7291/D1WC74, record 4934835):
  Phase_1_results.zip   Phase_1_results/Connacht/Connacht_phase1_tracing_coordinates.csv: the authors' tongue-contour table, one line for
                        each of the 100 points (pt.num) of each tracing at each of 3 landmarks (frm.pos 1-3) of each token (subj, word,
                        rep), with X, Y, the consonant (c), the vowel context (v), the word code (word) and its Irish spelling (irish);
                        the 1,140 PDF plots (SSANOVA) are not converted
  readme_phase1_connacht.txt   the deposit's readme
Not converted: Connacht_Audio_and_TextGrids.zip (audio and the authors' Praat annotations), Connacht_Ultrasound.zip (ultrasound video,
frames and the EdgeTrak .con tracings; not downloaded).
What the paper says (Method): 5 native speakers (2 women, 3 men, 35-60 years); a 24-word list read 6-8 times, 4-8 useable repetitions;
tongue surface traced with EdgeTrak as 100 points at three landmarks (onset, midpoint and offset of the consonant); 20 tracings of the
/a/ of the reference word scairt per speaker; average repetitions 6.1 overall and 6.7, 8.0, 5.9, 6.0, 4.0 by speaker.
Run from the site folder: python3 -I tools/std/convert/Bennett2018_JIPA.py
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Bennett2018_JIPA"
SRC = "Bennett et al. 2018, J Int Phon Assoc"
LANDMARKS = {1: "onset", 2: "mid", 3: "offset"}  # frm.pos; the authors' plot folders are onset, mdpt, offset (paper: onset, midpoint, offset)
DEVICE = ("Terason T3000 ultrasound system with an 8MC3 probe in an Articulate Instruments stabilization headset (57-60 frames per "
          "second); Shure WH20 microphone")

raw = stdlib.raw_dir(ID)
z = zipfile.ZipFile(raw / "Phase_1_results.zip")
df = pd.read_csv(io.BytesIO(z.read("Phase_1_results/Connacht/Connacht_phase1_tracing_coordinates.csv")))
assert list(df.columns) == ["Unnamed: 0", "subj", "c", "v", "rep", "frm.pos", "X", "Y", "pt.num", "word", "irish", "dialect"], list(df.columns)
assert df["dialect"].nunique() == 1 and df["dialect"].iloc[0] == "Connacht"
assert not df.isna().any().any()
df = df.rename(columns={"frm.pos": "frm", "pt.num": "pt"})
key = ["subj", "word", "irish", "rep"]
# tokens in the order of the authors' table (rows are in blocks of one token)
blk = (df[key] != df[key].shift()).any(axis=1).cumsum()
assert blk.nunique() == df.groupby(key).ngroups, "a token is split into several blocks"


def f(v):
    return format(float(v), ".8g")


rows = []
trial = {}
n_pts_all = set()
for _, g in df.groupby(blk, sort=True):
    first = g.iloc[0]
    assert len(g) == 300 and g.groupby("frm").size().eq(100).all()
    row = dict(
        dataset=ID, study=SRC, experiment="Phase 1, Connacht dialect", group="", subject=first.subj, person=first.subj,
        subject_excluded_by_authors=False, age=None, sex=None, handedness=None, population="healthy", setting="lab", device=DEVICE,
        session=1, block="", phase="other", time_s=None, task="speech", unit="utterance", condition=first.word, pert_type="none",
        pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
        stimulus=first.irish, target_consonant=first.c, vowel_context=first.v, repetition=int(first.rep), n_contour_points=100)
    trial[first.subj] = trial.get(first.subj, 0) + 1
    row["trial"] = trial[first.subj]
    for fr, name in LANDMARKS.items():
        h = g[g.frm == fr].sort_values("pt")
        assert h.pt.tolist() == list(range(1, 101))
        row[f"contour_{name}_x"] = ";".join(f(v) for v in h.X)
        row[f"contour_{name}_y"] = ";".join(f(v) for v in h.Y)
    rows.append(row)
out = pd.DataFrame(rows)

# ---- checks -------------------------------------------------------------------------------------------------------------
tgt = out[out.target_consonant != "ref"]
cells = tgt.groupby(["subject", "stimulus"]).size()  # repetitions kept per speaker and word (words by spelling)
per_spk = cells.groupby("subject").mean()
ref_n = out[out.target_consonant == "ref"].groupby("subject").size()
ck = stdlib.Checks()
ck.published("speakers", out["person"].nunique(), 5, 0, "Method, Participants (five speakers)")
ck.published("words in the list", tgt["stimulus"].nunique(), 24, 0, "Method, Stimuli (24-word list)")
ck.published("average repetitions per word and speaker, overall", cells.mean(), 6.1, 0.05, "Table 2 caption (average number of repetitions overall = 6.1)")
for i, (spk, rep) in enumerate(zip(sorted(per_spk.index), [6.7, 8.0, 5.9, 6.0, 4.0]), start=1):
    ck.published(f"average repetitions, speaker {i}", per_spk[spk], rep, 0.05, "Method (6.7, 8.0, 5.9, 6.0 and 4.0 repetitions by speaker)")
ck.published("fewest useable repetitions of a word", cells.min(), 4, 0, "Method (between 4 and 8 useable repetitions)")
ck.published("most useable repetitions of a word", cells.max(), 8, 0, "Method (between 4 and 8 useable repetitions)")
ck.published("points per tracing", out["n_contour_points"].iloc[0], 100, 0, "Method, Ultrasound tracing (100 points)")
ck.published("landmarks traced per token", len(LANDMARKS), 3, 0, "Method (onset, midpoint and offset of the consonant)")
ck.published("tracings of the reference word per speaker", ref_n.min(), 20, 0, "Method (20 tracings of the /a/ of scairt per speaker)")
ck.published("tracings of the reference word per speaker (largest)", ref_n.max(), 20, 0, "Method")
ck.internal("833 tokens: 733 target words and 100 reference words", len(out) == 833 and len(tgt) == 733, f"{len(out)} tokens, {len(tgt)} target")
ck.internal("249,900 table lines are all used (833 tokens x 3 landmarks x 100 points)", len(df) == 833 * 300, f"{len(df)} lines")
xs = np.array([[float(v) for v in s.split(";")] for s in out["contour_offset_x"]])
ck.internal("x rises with the point number in every offset contour", bool(np.all(xs[:, -1] > xs[:, 0])), f"mean x from {xs[:, 0].mean():.1f} to {xs[:, -1].mean():.1f}")
print("tokens", len(out), "| per speaker", out.groupby("subject").size().to_dict(), "| mean repetitions", per_spk.round(3).to_dict(), "overall", round(cells.mean(), 3))
for c in ck.items:
    if not c["ok"]:
        print("FAILING:", c)

odd = out[(out.condition == "p_i")]
meta = dict(
    paper=dict(authors="Bennett R, Ni Chiosain M, Padgett J, McGuire G", year=2018,
               title="An ultrasound study of Connemara Irish palatalization and velarization",
               venue="J Int Phon Assoc 48(3):261-304", doi="10.1017/S0025100317000494"),
    lab="Bennett / Padgett (UC Santa Cruz)",
    data_url="https://doi.org/10.7291/D1WC74 (the files were read from the deposit's Zenodo copy, https://zenodo.org/records/4934835)",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Only the tongue-contour table of the deposit (Phase_1_results) was converted: its 249,900 lines (one per contour point) "
             "became 833 rows, one per spoken token, with the 100 points of each of the 3 landmark contours as lists in six columns. "
             "The audio, the Praat TextGrids, the ultrasound video and frames, the EdgeTrak tracing files and the SSANOVA plots were not "
             "converted. The table's row-number column (which runs to 2,498,910 for 249,900 lines) and its constant dialect column were "
             "dropped; numbers are written to 8 significant digits; speaker labels are replaced by codes."),
    how=["One row is one spoken token: one of five Connemara Irish speakers saying one of the 24 target words, or the reference word "
         "scairt, in the carrier phrase 'Scairt Aoife ___ Dé Céadaoin', with the authors' repetition number. For each of three landmarks "
         "(onset, midpoint and offset of the consonant) the row has the 100 points of the tongue-surface contour that the authors "
         "traced with EdgeTrak, as an x list and a y list (numbers separated by semicolons, in the authors' point order).",
         "The authors' table numbers the landmarks 1, 2 and 3 and does not name them; the paper names three landmarks (onset, midpoint, "
         "offset) and the folders of the authors' plots are called onset, mdpt and offset, so 1, 2 and 3 are taken in that order.",
         "x runs from the rear of the tongue (point 1) to the front (point 100). y behaves like an image row counted downward, so a "
         "larger y is lower in the mouth (inferred: in every speaker the front of the tongue has smaller y for palatalized than for "
         "velarized consonants). Neither the paper nor the readme states the unit of x and y; tongue contours are about 85 units "
         "long, so millimetres are likely.",
         "trial follows the order of the authors' table (grouped by word, then repetition), not the order of speaking; the speaking "
         "order and the closure times are in the deposit's TextGrid files, which were not converted. Age and sex are not given per "
         "speaker (the paper reports 35-60 years, two women and three men)."],
    unit_note="spoken token (one word said once, with its tongue contours at three landmarks in six columns)",
    differences=[
        f"{len(odd)} token of speaker P0004 (repetition {int(odd.repetition.iloc[0]) if len(odd) else 0}) is coded p_i (p before i) in the authors' "
        "table but carries the Irish spelling ciunas, which is the spelling of the word coded kj_u; it is kept as shared and counted "
        "under ciunas for the repetition counts.",
        "The paper's 24-word list was read 6-8 times (144-192 productions per speaker); the table holds only the useable repetitions "
        "(96-191 tokens per speaker), as the paper's 4-8 useable repetitions say.",
    ],
)
if not ck.passed():
    sys.exit("checks failed")
rec = stdlib.release(ID, "tongue_ultrasound", out, meta, ck)
