"""Wenk, Jordi, Buetler & Marchal-Crespo (2022), IEEE TNSRE 30:390-399: 28 healthy adults followed seven paths in virtual reality with an
end-effector robot, with the robot visible or hidden in the virtual scene and with or without path-following assistance.
One row per lap (two laps per path visit). Raw files (Zenodo 10.5281/zenodo.5558770, CC BY 4.0): DfLapBased_Metrics.csv (all laps) and
DfLapBased_MetricsFiltered.csv (the laps the authors analysed); DataFrameDescription.rtf was read as text.
Run from the site folder: python3 -I tools/std/convert/Wenk2022_TNSRE.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Wenk2022_TNSRE"
raw = stdlib.raw_dir(DS)
a = pd.read_csv(raw / "DfLapBased_Metrics.csv")
f = pd.read_csv(raw / "DfLapBased_MetricsFiltered.csv")
KEY = ["id", "controller", "visibility", "levelId", "pathOrder", "lap"]
ck = stdlib.Checks()
ck.internal("lap key is unique in both files", not a.duplicated(KEY).any() and not f.duplicated(KEY).any())
m = a[KEY].merge(f[KEY].assign(_kept=True), how="left", on=KEY)
kept = m["_kept"].fillna(False).to_numpy(dtype=bool)
ck.internal("filtered file is a subset of the full file (all 1281 filtered laps found)", int(kept.sum()) == len(f) == 1281, f"{kept.sum()} of {len(f)}")
ck.internal("filtered rows carry identical values to the full file",
            bool(a[kept].reset_index(drop=True).equals(f.reset_index(drop=True))))
ck.internal("no missing values", int(a.isna().sum().sum()) == 0)
ck.internal("28 participants x 4 modalities x 7 paths x 2 laps = 1568 laps expected, 1564 present (four laps are missing in the authors' file)",
            len(a) == 1564 and a.groupby("id").size().between(54, 56).all())
ck.internal("rows are grouped by participant in ascending order", bool(a.id.is_monotonic_increasing))
ck.internal("error and duration positive", bool((a["EE.mse"] > 0).all() and (a.duration > 0).all()))

assist = a.controller.eq("PathController").to_numpy()
vis = a.visibility.eq("Visible").to_numpy()
out = pd.DataFrame({
    "dataset": DS, "study": DS, "experiment": DS + "_Exp1", "group": "", "subject": "S" + a.id.astype(str).str.zfill(2),
    "subject_excluded_by_authors": False, "population": "healthy", "setting": "lab",
    "device": "End-effector rehabilitation robot with a head-mounted display and eye tracking",
    "session": 1,
    "block": (a.controller.map({"Transparent": "no assistance", "PathController": "assistance"}) + ", robot "
              + a.visibility.str.lower()).to_numpy(),
    "trial": a.groupby("id").cumcount() + 1, "phase": "learning", "task": "tracking", "unit": "trial",
    "condition": ("robot " + a.visibility.str.lower() + ", " + a.controller.map({"Transparent": "without", "PathController": "with"})
                  + " path assistance").to_numpy(),
    "pert_type": "path_assistance", "pert_value": assist.astype(int), "pert_unit": "off (0) / on (1)", "feedback": "normal",
    "excluded_by_authors": ~kept,
    "game": "path following in virtual reality", "attempt": a.lap, "score_unit": "", "outcome_error": a["EE.mse"],
    "outcome_error_unit": "m^2 (mean squared distance between end-effector and closest path point)",
    "environment": "path " + a.levelId.astype(str), "duration_s": a.duration, "path_order": a.pathOrder,
    "gaze_path_ratio": a["gaze.timeRatio.occlusion.Path_tolerance"], "gaze_robot_ratio": a["gaze.timeRatio.occlusion.Robot_tolerance"],
    "gaze_avatar_ratio": a["gaze.timeRatio.occlusion.Avatar_tolerance"],
    "gaze_end_effector_ratio": a["gaze.timeRatio.occlusion.EndEffector_tolerance"],
})
ck.published("participants", out.subject.nunique(), 28, 0, "Abstract (28 healthy participants; TU Delft repository record of the paper)")
ck.internal("each modality has 4 x 7 x 2 laps per participant at most", bool(out.groupby(["subject", "block"]).size().le(14).all()))
meta = dict(
    paper=dict(authors="Wenk N, Jordi MV, Buetler KA, Marchal-Crespo L", year=2022,
               title="Hiding Assistive Robots During Training in Immersive VR Does Not Affect Users' Motivation, Presence, Embodiment, Performance, Nor Visual Attention",
               venue="IEEE Trans Neural Syst Rehabil Eng 30:390-399", doi="10.1109/tnsre.2022.3147260"),
    lab="Marchal-Crespo (University of Bern)", data_url="https://zenodo.org/records/5558770",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Merged the deposit's two lap tables (all laps, and the laps the authors analysed) into one table with the analysed laps "
             "unmarked and the others marked excluded_by_authors; renamed columns; participant numbers replaced by codes. The "
             "questionnaire table of the deposit (one row per participant and modality) is not included."),
    how=["One row is one lap of one path by one participant in one of four modalities (robot visible or invisible in the virtual scene, "
         "crossed with path-following assistance by the robot on or off). Each path was shown for two laps. outcome_error is the "
         "authors' error (EE.mse, mean squared distance in m^2 between the robot end-effector and the closest point of the path), "
         "duration_s the lap time and the gaze columns the proportion of the lap spent looking at the path, the robot's space, the avatar "
         "and the end-effector.",
         "Rows keep the authors' order (participant, then modality, path and lap as listed). excluded_by_authors marks the 283 laps that the "
         "authors' filtered file leaves out (laps with 'unnatural behaviour'). Participants all have 54 to 56 of the 56 laps they should "
         "have (seven paths x two laps x four modalities); the deposit lists no lap for the others. The authors' pathOrder is kept as "
         "path_order; their levelId (path shape) is in environment."],
    unit_note="lap",
    differences=["The full text of the paper could not be read (IEEE and the institutional repository blocked access), so only the "
                 "participant count from the abstract is compared; the paper's error, time and gaze statistics were not checked. "
                 "The Zenodo record's title omits 'Nor Visual Attention' and the deposit calls the path assistance 'PathController' "
                 "and no assistance 'Transparent'; the mapping to assisted/unassisted follows the deposit's description."])
out = out.reindex(columns=[c[1] for c in stdlib.columns_for("game") if c[1] != "person"])
stdlib.release(DS, "game", out, meta, ck)
for c in ck.items:
    print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""))
