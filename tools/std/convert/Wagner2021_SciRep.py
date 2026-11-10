"""Wagner, Wolf & Schutz (2021) Sci Rep: motor learning by selection in visual working memory.

Raw files: raw_extra/Wagner2021_SciRep/1_memory-task.zip and 2_saccade-only.zip (Zenodo 4618049). Members used:
<exp>_saccades.csv (one row per trial, 20 unnamed columns, described in columnDescription_saccades.xlsx) and
<exp>_gazeShifts.csv (one row per detected gaze shift, 11 unnamed columns, columnDescription_gazeShifts.xlsx).
The average gaze-trace files are not converted. Nothing from the deposit is executed.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Wagner2021_SciRep"
MODULE = "saccade"
COND = {1: "delayed-selection", 2: "immediate-error", 3: "delayed-error"}
EXPS = [("memory-task", "1_memory-task", "memoryTask", [9]),
        ("saccade-only", "2_saccade-only", "saccadeOnly", [5, 6, 8, 10, 12])]
TCOLS = ['pid', 'cond', 'trial', 'ecc', 'rad', 'upy', 'lowy', 'contr', 'sf', 'cued', 'oru', 'orl', 'orc', 'orr',
         'resp', 'correct', 'lat', 'hamp', 'vamp', 'excl']
GCOLS = ['pid', 'cond', 'hamp', 'vamp', 'trial', 'cued', 't_cue', 't_ref', 'f_cue', 'f_ref', 'excl']


def nz(x):
    return None if x != x else float(x)


def main():
    ck = stdlib.Checks()
    rows = []
    stats = {}
    for exp, zn, pre, excl_p in EXPS:
        zf = zipfile.ZipFile(stdlib.raw_dir(DS) / f"{zn}.zip")
        t = pd.read_csv(io.BytesIO(zf.read(f"{zn}/{pre}_saccades.csv")), header=None, names=TCOLS, na_values=["NaN"])
        g = pd.read_csv(io.BytesIO(zf.read(f"{zn}/{pre}_gazeShifts.csv")), header=None, names=GCOLS, na_values=["NaN"])
        t["run"] = t.groupby("pid").cumcount() + 1
        runmap = {(a, b, c): r for a, b, c, r in zip(t.pid, t.cond, t.trial, t.run)}
        ck.internal(f"{exp}: 200 trials per participant and condition",
                    bool((t.groupby(["pid", "cond"]).size() == 200).all()))
        ck.internal(f"{exp}: every gaze shift belongs to a trial row",
                    all((a, b, c) in runmap for a, b, c in zip(g.pid, g.cond, g.trial)))
        stats[exp] = (t, g, excl_p)
        for pid, tp in t.groupby("pid", sort=False):
            sub = f"{exp}_{int(pid)}"
            sx = pid in excl_p
            base = dict(dataset=DS, study="Wagner 2021 Sci Rep", experiment=exp, group=None, subject=sub, person=sub,
                        subject_excluded_by_authors=sx, age=None, sex=None, handedness=None, population="healthy",
                        setting="lab", device="EyeLink 1000 Plus, right eye, 1000 Hz", session=None,
                        phase="learning", time_s=None, task="eye movements", feedback="altered")
            for r in tp.itertuples(index=False):
                cued_y = r.upy if r.cued == 1 else r.lowy
                oth_y = r.lowy if r.cued == 1 else r.upy
                info = (f"trial_in_condition={int(r.trial)};ring_radius_deg={r.rad:g};contrast={r.contr:g};"
                        f"spatial_frequency_cpd={r.sf:g};cued_gabor={'upper' if r.cued == 1 else 'lower'};"
                        f"orientation_upper_deg={r.oru:g};orientation_lower_deg={r.orl:g};orientation_cued_deg={r.orc:g};"
                        f"orientation_reference_deg={r.orr:g}")
                rows.append({**base, "block": COND[r.cond], "trial": int(r.run), "unit": "trial", "condition": COND[r.cond],
                             "pert_type": "post_saccadic_stimulus_offset", "pert_value": cued_y, "pert_unit": "deg",
                             "excluded_by_authors": bool(r.excl), "target_amplitude_deg": r.ecc, "target_direction_deg": 0,
                             "step_x_deg": 0, "step_y_deg": cued_y, "distractor_x_deg": 0, "distractor_y_deg": oth_y,
                             "saccade_amplitude_x_deg": nz(r.hamp), "saccade_amplitude_y_deg": nz(r.vamp),
                             "saccade_latency_ms": nz(r.lat), "eye": "right",
                             "response": None if r.resp != r.resp else str(int(r.resp)),
                             "response_correct": (r.correct == 1) if exp == "memory-task" else None,
                             "trial_info": info})
            gp = g[g.pid == pid]
            for r in gp.itertuples(index=False):
                info = (f"row_kind=gaze_shift;trial_in_condition={int(r.trial)};cued_gabor={'upper' if r.cued == 1 else 'lower'};"
                        f"onset_minus_retrocue_ms={r.t_cue:g};onset_minus_reference_ms={r.t_ref:g};"
                        f"after_saccade_before_retrocue={int(r.f_cue)};after_saccade_before_reference={int(r.f_ref)}")
                rows.append({**base, "block": COND[r.cond], "trial": runmap[(pid, r.cond, r.trial)], "unit": "saccade",
                             "condition": COND[r.cond], "pert_type": "none", "pert_value": 0, "pert_unit": "deg",
                             "excluded_by_authors": bool(r.excl),
                             "saccade_amplitude_x_deg": nz(r.hamp), "saccade_amplitude_y_deg": nz(r.vamp),
                             "eye": "right", "trial_info": info})
    ck.internal("all trial rows and gaze-shift rows kept", len(rows) == 7800 + 9200 + 75145 + 87604, str(len(rows)))

    # ---- published numbers (Methods 'Eye movement and data analysis'; Results 'Trial-by-trial changes')
    rep_valid = {"memory-task": {1: 90.00, 3: 89.79, 2: 91.04}, "saccade-only": {1: 86.54, 3: 89.29, 2: 88.29}}
    rep_acc = {1: 0.75, 3: 0.88, 2: 0.81}
    for exp, (t, g, ex) in stats.items():
        a = t[~t.pid.isin(ex)]
        ck.published(f"{exp}: participants analysed", a.pid.nunique(), 12, 0, "Methods, Participants")
        for c in (1, 2, 3):
            pct = a[a.cond == c].groupby("pid").excl.apply(lambda s: 100 * (1 - s.mean()))
            ck.published(f"{exp}, {COND[c]}: mean % of valid trials", pct.mean(), rep_valid[exp][c], 0.005,
                         "Methods, Eye movement and data analysis")
        if exp == "memory-task":
            for c in (1, 2, 3):
                v = a[(a.cond == c) & (a.excl == 0)]
                acc = v.groupby("pid").correct.apply(lambda s: (s == 1).mean()).mean()
                ck.published(f"memory-task, {COND[c]}: proportion correct", acc, rep_acc[c], 0.005,
                             "Results, Trial-by-trial changes")
    ck.published("saccade-only: participants recorded with data", stats["saccade-only"][0].pid.nunique() + 2, 19, 0,
                 "Methods, Participants (19 recruited; 2 opted out after session 1 and have no data)")

    meta = dict(
        study=DS,
        paper=dict(authors="Wagner I, Wolf C, Schutz AC", year=2021, title="Motor learning by selection in visual working memory",
                   venue="Sci Rep 11:9331", doi="10.1038/s41598-021-87572-6"),
        lab="Schutz (University of Marburg)", data_url="https://zenodo.org/records/4618049",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Converted the trial tables and gaze-shift tables of both experiments to the OpenMotor format with participant "
                 "codes and named columns. The average vertical gaze traces (files memoryTask_gazeTraces_*, "
                 "saccadeOnly_gazeTraces_*) stay only in the original deposit."),
        how=["Rows with unit trial are trials: the primary saccade to a ring 10 deg to the right of fixation, with latency and "
             "the horizontal and vertical amplitude in deg (vertical not recoded, positive as in the authors' file). Each "
             "participant did three experimental sessions of 200 trials, one per condition; the order of sessions is not in the "
             "deposit, so session is empty and trial counts rows within the participant.",
             "Rows with unit saccade are the gaze shifts the authors detected (Engbert-Kliegl algorithm) in the same trials: "
             "amplitude x and y in deg, and trial_info gives their onset relative to the auditory retro-cue and the reference "
             "stimulus. trial links them to the trial row; excluded_by_authors is the trial's exclusion flag.",
             "Two Gabor patches (2 deg above and below the ring centre) appear after the saccade; pert_value and step_y_deg are "
             "the vertical position of the cued, task-relevant one and distractor_y_deg the other (this mapping of the authors' "
             "columns is ours). In the memory task, response is -1 (reference and cued Gabor tilted differently) or 1 (same) and "
             "response_correct the authors' correct flag; the saccade-only experiment had no response.",
             "Excluded participants are flagged: participant 9 (memory task) and participants 5, 6, 8, 10, 12 (saccade-only). "
             "Two further saccade-only participants who opted out after the first session have no data in the deposit."],
        unit_note="trial or detected gaze shift",
    )
    stdlib.release(DS, MODULE, rows, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
