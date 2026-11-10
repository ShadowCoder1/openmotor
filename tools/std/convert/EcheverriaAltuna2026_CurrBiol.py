"""Echeverria-Altuna, Demirel, Boettcher, Watkins & Nobre (2026), Curr Biol 36(15):3983-3989: dynamic motor selection in
stuttering. 20 adults who stutter (PWS) and 20 typically fluent speakers did a visuomotor working-memory task with a retro-cue and
a short or long delay during MEG; one row per trial, module retrocue_hand.

Run from the site folder: python3 -I tools/std/convert/EcheverriaAltuna2026_CurrBiol.py
Raw file (untrusted data, read only): raw_extra/EcheverriaAltuna2026_CurrBiol/behav_data.csv from https://osf.io/gp84x/ (data/behav).
The authors' analysis script 00_behaviour.Rmd and its rendered output were read as text (never run): RT = RT x 1000 (ms), trials
with badtrials == 1 are removed, subType HC is renamed TFS (typically fluent speakers), trialType inf/noninf = informative or
uninformative retro-cue, targTime short/long = delay2_time 1 or 3 s; error = absolute report error on correct-hand trials.
sid (a participant code with letters) is not read into any output."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "EcheverriaAltuna2026_CurrBiol"
RAW = stdlib.raw_dir(ID)
ck = stdlib.Checks()
d = pd.read_csv(RAW / "behav_data.csv", usecols=lambda c: c not in ("sid", "ssi4"))
ck.internal("7680 rows = 40 participants x 4 blocks x 48 trials, rows of a participant contiguous",
            len(d) == 7680 and d.subnum.nunique() == 40 and (d.groupby("subnum").size() == 192).all()
            and d.subnum.ne(d.subnum.shift()).sum() == 40)
ck.internal("20 HC and 20 PWS; delay2_time 1 <-> short and 3 <-> long; wm_TargDev = target tilt minus reported tilt; error = |wm_TargDev| on correct-hand trials",
            d.groupby("subType").subnum.nunique().to_dict() == {"HC": 20, "PWS": 20}
            and ((d.delay2_time == 1) == (d.targTime == "short")).all()
            and (d.targTilt - d.wm_Resp == d.wm_TargDev)[d.noResp == 0].all()
            and np.allclose(d.error.dropna(), d.wm_TargDev.abs()[d.error.notna()]))
ck.internal("wrongHand = 1 - correctResp; 20 trials have no key press (wm_respKey 0, wm_DT 0), all among the 30 noResp trials (response trigger 20); the other 10 noResp trials have a late key press (2.1-2.9 s) and a report",
            ((d.wrongHand == 1 - d.correctResp).all()) and d.noResp.sum() == 30 and (d.wm_respKey == "0").sum() == 20
            and (d.noResp[d.wm_respKey == "0"] == 1).all() and (d.wm_DT[d.wm_respKey == "0"] == 0).all())

names = [c[1] for c in stdlib.columns_for("retrocue_hand")]
POP = {"HC": ("TFS", "typically fluent speakers"), "PWS": ("PWS", "adults who stutter")}
rows = []
for sn, g in d.groupby("subnum", sort=False):
    for k, r in enumerate(g.itertuples(index=False), 1):
        resp = r.wm_respKey != "0"
        row = dict.fromkeys(names, "")
        grp, pop = POP[r.subType]
        row.update(
            dataset=ID, study=ID, experiment=ID, group=grp, subject=f"S{sn}", person=f"S{sn}", subject_excluded_by_authors="False",
            population=pop, setting="lab", device="MEG session, two-handed response box (stimulus on a Propixx projector)",
            session=1, block=f"block{r.block}", trial=k, phase="test", task="visuomotor working memory with hand selection (retro-cue)",
            unit="trial", condition=f"{'informative' if r.trialType == 'inf' else 'uninformative'} cue, {r.targTime} delay",
            pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors="True" if r.badtrials == 1 else "False",
            trial_in_block=r.trial, cue_informative="True" if r.trialType == "inf" else "False", delay_s=r.delay2_time,
            target_tilt_deg=r.targTilt, nontarget_tilt_deg=r.nonTargTilt, target_side=r.targPos, required_hand=r.targTiltDir,
            response_hand=r.wm_respKey if resp else "", responded="True" if resp else "False",
            correct_hand="True" if r.correctResp == 1 else "False", rt_ms=r.wm_DT * 1000 if resp else "",
            reported_tilt_deg=r.wm_Resp if resp else "", signed_error_deg=r.wm_TargDev if resp else "",
            nontarget_deviation_deg=r.wm_NTDev if resp else "", abs_error_deg=r.error if r.error == r.error else "",
            target_colour=r.targCol, cue_colour=r.retroCueCol)
        rows.append(row)
ck.internal("every row kept, in order; 488 rows flagged as the authors' bad trials", len(rows) == 7680
            and sum(r["excluded_by_authors"] == "True" for r in rows) == 488)

# published numbers: Supplementary Tables (cell means of participant means), reproduced by the authors' script
m = d[d.badtrials == 0].copy()
m["RT"] = m.RT * 1000
m["subType"] = m.subType.replace({"HC": "TFS"})
rtm = m.dropna(subset=["RT"]).groupby(["subType", "trialType", "targTime", "subnum"]).RT.mean().groupby(["subType", "trialType", "targTime"]).mean()
erm = m.dropna(subset=["error"]).groupby(["subType", "trialType", "targTime", "subnum"]).error.mean().groupby(["subType", "trialType", "targTime"]).mean()
RT_REP = {("PWS", "inf", "short"): 665.9325, ("PWS", "noninf", "short"): 778.7089, ("PWS", "inf", "long"): 527.9009, ("PWS", "noninf", "long"): 678.6793,
          ("TFS", "inf", "short"): 640.0051, ("TFS", "noninf", "short"): 829.5200, ("TFS", "inf", "long"): 538.3884, ("TFS", "noninf", "long"): 734.9829}
ER_REP = {("PWS", "inf", "short"): 12.10894, ("PWS", "noninf", "short"): 12.55080, ("PWS", "inf", "long"): 13.56889, ("PWS", "noninf", "long"): 13.40282,
          ("TFS", "inf", "short"): 10.40837, ("TFS", "noninf", "short"): 10.10602, ("TFS", "inf", "long"): 11.32307, ("TFS", "noninf", "long"): 11.93542}
for k, v in RT_REP.items():
    ck.published(f"mean RT (ms), {k[0]}, {k[1]} cue, {k[2]} delay", rtm[k], v, 0.001, "Supplementary Table 2 (OSF 00_behaviour output of the authors' script)")
for k, v in ER_REP.items():
    ck.published(f"mean report error (deg), {k[0]}, {k[1]} cue, {k[2]} delay", erm[k], v, 0.0001, "Supplementary Table 4 (OSF 00_behaviour output)")
per = d.drop_duplicates("subnum")
ck.published("participants who stutter", (per.subType == "PWS").sum(), 20, 0, "Abstract (twenty adults who stutter)")
ck.published("typically fluent participants", (per.subType == "HC").sum(), 20, 0, "Abstract (twenty typically fluent speakers)")

parts = []
for sn, g in d.groupby("subnum", sort=False):
    parts.append(dict(person=f"S{sn}", oases=g.oases.iloc[0] if g.oases.notna().any() else "", pais=g.pais.iloc[0] if g.pais.notna().any() else ""))
pcols = [("oases", "number", "score", "OASES (Overall Assessment of the Speaker's Experience of Stuttering) score as in the deposit's behavioural file, adults who stutter only."),
         ("pais", "number", "score", "PAIS score as in the deposit (adults who stutter only; the file does not name the scale).")]

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Nobre / Watkins (University of Oxford)", data_url="https://osf.io/gp84x/",
    license="none stated", license_url="",
    changes="converted to the OpenMotor format, participant codes replaced, the authors' RT converted to ms (x 1000) as in their script, subType HC written TFS (typically fluent speakers) as in the paper; the original participant code column (sid) and the analysis-only columns (stimulus trigger codes, onset clock times) were not carried over.",
    unit_note="trial of one participant",
    how=["One row is one trial of a visuomotor working-memory task during MEG: the authors' table has the target and non-target tilts and colours, an informative or uninformative retro-cue, a short (1 s) or long (3 s) delay, the hand required by the target's tilt direction, the hand used, the reaction time and the reported tilt and its error. 40 participants x 192 trials (4 blocks of 48): 20 adults who stutter (PWS) and 20 typically fluent speakers (TFS).",
         "excluded_by_authors marks the 488 trials the authors' script removes (badtrials = 1: 443 wrong-hand trials, all 30 trials with a late or missing response, and 15 others); abs_error_deg is the authors' error (absolute report error on correct-hand trials only). The task instructions, the response device and the exact report procedure are in the paper's Methods, not in the deposit; the response was made with the left or right hand as wm_respKey records.",
         "The deposit's MEG epochs and time-frequency files (large .fif and .npy files) and the questionnaire files are not part of this conversion; participants.csv carries the two questionnaire scores from the behavioural file (OASES and PAIS, adults who stutter only). Age and sex are not in the deposit."],
    differences=["The deposit and its analysis notebooks contain no statement about participant consent limits; none are recorded in the files."],
)
stdlib.release(ID, "retrocue_hand", rows, meta, ck, participants=parts, participant_columns=pcols)
