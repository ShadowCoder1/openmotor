"""Modchalingam, Vachon, 't Hart & Henriques (2019) PLoS ONE: awareness of a visuomotor rotation and hand localization.
Raw (OSF mx5u2, data folder, read as text tables): participantAwareness.csv (85 participants: group, age, sex, task order,
questionnaire points), <grp>_curves.csv (90 rotated-training trials x participants), <grp>_locAwareness.csv, <grp>_reachAEs_pp.csv,
<grp>_sessionAEs_pp.csv, <grp>_{active,passive}TapData[_aligned|_rotated].csv with grp in exp30, exp60, imp30, imp60.
File-name participant codes contain initials and dates and are never written out. Nothing is executed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Modchalingam2019_PLoSOne"
RAW = stdlib.raw_dir(DS)
GRP = {"exp30": "30explicit", "exp60": "60explicit", "imp30": "30implicit", "imp60": "60implicit"}
LOC = [("exclusive", "Reach aftereffect without a strategy (no-cursor reaches, participants told to not use the strategy)", "deg"),
       ("inclusive", "Reach aftereffect with the strategy (no-cursor reaches, participants told to reach as in training)", "deg"),
       ("awarenessRatio", "Ratio of the exclusive to the inclusive reach aftereffect (authors' awareness ratio)", "ratio"),
       ("active_means", "Mean hand-localization error after active (self-generated) movement, rotated session", "deg"),
       ("prop_means", "Mean hand-localization error after passive movement (proprioceptive estimate), rotated session", "deg"),
       ("pred_means", "Active minus passive localization error (predicted-consequence component)", "deg"),
       ("awareness_score", "Awareness score from the questionnaire (0 none, 1 low, 3 high)", "score")]
SES = [("aligned", "Reach aftereffect (no-cursor reaches) in the aligned session", "deg"),
       ("rotatedWithout", "Reach aftereffect (no-cursor reaches) in the rotated session without strategy", "deg")]


def main():
    pa = pd.read_csv(RAW / "participantAwareness.csv")
    info = pa.set_index("filename")
    ck = stdlib.Checks()
    rows = []
    ids = {}
    n_pp = {g: 0 for g in GRP}
    lc = {}

    def base(code, g):
        r = info.loc[code]
        pid = f"p{len(ids) + 1}" if code not in ids else ids[code]
        ids[code] = pid
        rot = int(g[3:])
        return dict(dataset=DS, study="Modchalingam 2019 PLoS ONE", experiment="awareness-localization", group=GRP[g],
                    subject=pid, person=pid, subject_excluded_by_authors=False, age=float(r.Age), sex=r.Sex, handedness="right",
                    population="healthy", setting="lab", device="tablet", session=None, block=None, trial=None, phase="other",
                    time_s=None, task="reaching", unit="participant summary", condition=None, pert_type="rotation",
                    pert_value=rot, pert_unit="deg", feedback="none", excluded_by_authors=False,
                    session_context=f"task order {r.version}")

    def add(b, name, desc, v, unit, trial=None, fb=None):
        if pd.isna(v):
            return
        rows.append({**b, "measure": name, "measure_description": desc, "statistic": "value", "value": float(v),
                     "value_unit": unit, "trial": trial, **({"feedback": fb} if fb else {})})

    for g in GRP:
        ppl = pd.read_csv(RAW / f"{g}_locAwareness.csv").set_index("participant")
        ses = pd.read_csv(RAW / f"{g}_sessionAEs_pp.csv").set_index("participant")
        rae = pd.read_csv(RAW / f"{g}_reachAEs_pp.csv").set_index("participant")
        cur = pd.read_csv(RAW / f"{g}_curves.csv")
        ck.internal(f"{g}: reachAEs file repeats exclusive/inclusive of locAwareness",
                    bool(np.allclose(rae.loc[ppl.index, ["exclusive", "inclusive"]], ppl[["exclusive", "inclusive"]])))
        ck.internal(f"{g}: curve columns are the same participants", set(cur.columns) == set(ppl.index) and len(cur) == 90)
        n_pp[g] = len(ppl)
        taps = {}
        for kind in ("active", "passive"):
            for suf in ("", "_aligned", "_rotated"):
                if (RAW / f"{g}_{kind}TapData{suf}.csv").exists():
                    taps[(kind, suf)] = pd.read_csv(RAW / f"{g}_{kind}TapData{suf}.csv").set_index("participant")
        for code in ppl.index:
            b = base(code, g)
            for k, d, u in LOC:
                add(b, k, d, ppl.loc[code, k], u)
            for k, d, u in SES:
                add(b, "session_AE_" + k, d, ses.loc[code, k], u)
            add(b, "questionnaire_points", "Points column of the authors' awareness table", info.loc[code, "Points"], "points")
            for (kind, suf), t in taps.items():
                if code not in t.index:
                    continue
                for ang in (50, 90, 130):
                    nm = f"{kind}_localization_{ang}{suf}"
                    add(b, nm, f"Localization error of the {kind} movement at the {ang} deg arc position"
                        + (f", {suf[1:]} session" if suf else ", rotated minus aligned session (authors' unlabelled file)"),
                        t.loc[code, f"{ang} deviation"], "deg")
            for i, v in enumerate(cur[code].to_numpy()):
                add(b, "reach_deviation", "Reach deviation on rotated-training trial (positive = countering the rotation, our reading of the authors' curves)",
                    v, "deg", trial=i + 1, fb="altered")
            lc[(g, "last9")] = lc.get((g, "last9"), []) + [float(np.nanmean(cur[code].to_numpy()[-9:]))]
    sizes = pa.groupby("group").size()
    ck.internal("participantAwareness has the 85 participants of the four curve files", len(ids) == 85 == len(pa))
    ck.published("non-instructed 30 deg: n", sizes["30implicit"], 20, 0, "Methods, Participants")
    ck.published("instructed 30 deg: n", sizes["30explicit"], 21, 0, "Methods, Participants")
    ck.published("non-instructed 60 deg: n", sizes["60implicit"], 20, 0, "Methods, Participants")
    ck.published("instructed 60 deg: n", sizes["60explicit"], 24, 0, "Methods, Participants")
    fem = pa[pa.Sex == "F"].groupby("group").size()
    for gname, n in (("30implicit", 14), ("30explicit", 13), ("60implicit", 14), ("60explicit", 18)):
        ck.published(f"{gname} females", fem[gname], n, 0, "Methods, Participants")
    r = pd.read_csv(RAW / "imp30_reachAEs_pp.csv")
    ck.published("non-instructed 30: mean reach aftereffect with strategy (deg)", r.inclusive.mean(), 16.437, 0.0005, "Results, reach aftereffects")
    ck.published("non-instructed 30: SD with strategy", r.inclusive.std(), 7.578, 0.0005, "Results, reach aftereffects")
    ck.published("non-instructed 30: mean without strategy", r.exclusive.mean(), 14.425, 0.0005, "Results, reach aftereffects")
    ck.published("non-instructed 30: SD without strategy", r.exclusive.std(), 4.166, 0.0005, "Results, reach aftereffects")
    ck.internal("rows written", len(rows) > 9000, str(len(rows)))
    meta = dict(
        study=DS,
        paper=dict(authors="Modchalingam S, Vachon CM, 't Hart BM, Henriques DYP", year=2019,
                   title="The effects of awareness of the perturbation during motor adaptation on hand localization",
                   venue="PLoS ONE 14(8):e0220884", doi="10.1371/journal.pone.0220884"),
        lab="Henriques & 't Hart (York University)", data_url="https://osf.io/mx5u2/",
        license="none stated", license_url="",
        changes=("Merged the authors' per-group CSV tables into one long table (a row per participant and measure, and per trial "
                 "for the training curves) and replaced the file-name participant codes with codes."),
        how=["Participants made reaches with a hand-controlled cursor under a 30 or 60 deg visuomotor rotation, with (explicit) or without "
             "(implicit) instructions about the rotation (group). One row is one measure of one participant; reach_deviation rows "
             "are single rotated-training trials (trial 1-90, empty where the authors' curve has a gap), all other rows are "
             "participant values. Angles are in deg.",
             "Reach deviations are positive when they counter the rotation; this is inferred from the values (they rise from about 0 to "
             "the rotation size). exclusive/inclusive are the authors' file names for the no-strategy and with-strategy reach aftereffects "
             "(matched to the paper's means). The tap-file measures are localization errors at three arc positions; the unlabelled "
             "active/passive files are, by their values, rotated minus aligned session, which is our inference.",
             "session_context gives the task order (ei or ie) of the authors' participant table."],
        unit_note="measure of one participant, or one training trial",
        differences=["The paper's Methods text says 84 participants (41 at 30 deg, 43 at 60 deg) but its group sizes sum to 85, as in the "
                     "deposit; the deposit's 85 were released. The 8 participants the paper excluded are not in the deposit."])
    stdlib.release(DS, "summary", rows, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']} ok={c['ok']}")


if __name__ == "__main__":
    main()
