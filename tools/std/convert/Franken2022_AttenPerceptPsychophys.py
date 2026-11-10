"""Franken, Hartsuiker, Johansson, Hall & Lind (2022), Atten Percept Psychophys 84:2027-2039: drifting pitch awareness after
exposure to altered auditory feedback. Experiments 1 and 2 (vowel production with +500 / -500 cents pitch-shifted feedback)
are standardized; Experiments 3 and 4 (passive listening, nobody spoke) are not motor data and have no production file.

Raw files, raw_extra/Franken2022_AttenPerceptPsychophys/ (OSF 3ksq2, data folder, fetched with tools/std/fetch.py):
  exp<e>_production.csv  trial, phase, pitch_feedback (Hz heard), pitch_out (Hz produced), duration (s), participant, vwl,
                         pitchShift (units of 50 cents), pitch_out_z, avgpitch, trialbin, trialbin_30, exp   (trials 11-340)
  exp<e>_responses.csv   pitchShift (cents), participant, pretest.thisN / posttest.thisN (0-based within the block, the
                         pretest starting at 10), response (1 = 'higher'), Stimulus   (pretest and posttest trials only)
  questionnaire.csv      subject, exp, age, gender, ..., include (1 = analysed)  (';' separated, ',' decimals)
Facts used (accepted manuscript, OSF qzf2g; authors' code OSF_Analysis.Rmd, read as text):
  pretest 100 trials, exposure 120, posttest 120; first 10 trials are not in the files; shifts of 0, 30, 60, 100, 150 cents up or
  down in pre- and posttest (pitchShift x 50 reproduces them), a constant +500 (Exp 1) or -500 (Exp 2) in the exposure block;
  pitch of the vowel centre by autocorrelation in Praat; the authors' code drops rows with |pitch_out_z| >= 2.5.
Run from the site folder: python3 -I tools/std/convert/Franken2022_AttenPerceptPsychophys.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Franken2022_AttenPerceptPsychophys"
PAPER_SRC = "Franken et al. 2022 (accepted manuscript, OSF qzf2g)"
PHASE = {"pretest": "baseline", "train": "learning", "posttest": "test"}
BLOCK = {"pretest": "pretest", "train": "exposure", "posttest": "posttest"}
DEVICE = ("DPA 4088-B microphone, custom headphones with passive sound attenuation, Eventide Eclipse v4 pitch shifter, "
          "Behringer Xenyx 802 mixer")
MASKING = "Brownian noise, 85 dBA, on throughout"
VOWEL = {"ie": "/i/ (ie)", "ee": "/e/ (ee)", "aa": "/a/ (aa)", "oo": "/o/ (oo)", "oe": "/u/ (oe)"}


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def build(checks):
    raw = stdlib.raw_dir(ID)
    q = pd.read_csv(raw / "questionnaire.csv", sep=";", decimal=",")
    rows, frames = [], {}
    for e, exposure in ((1, 500), (2, -500)):
        d = pd.read_csv(raw / f"exp{e}_production.csv")
        r = pd.read_csv(raw / f"exp{e}_responses.csv")
        r["trial"] = np.where(r.phase == "pretest", r["pretest.thisN"] + 1, r["posttest.thisN"] + 221).astype(int)
        m = d.merge(r[["participant", "trial", "pitchShift", "Stimulus", "response"]], on=["participant", "trial"], how="left",
                    suffixes=("", "_r"))
        jm = m[m.phase != "train"]
        checks.internal(f"Experiment {e}: every pretest and posttest production row has its response, with the same shift and vowel",
                        bool(jm.response.notna().all() and ((jm.pitchShift * 50 - jm.pitchShift_r).abs() < 1e-6).all()
                             and (jm.vwl == jm.Stimulus).all() and m[m.phase == "train"].response.isna().all()))
        frames[e] = m
        qq = q[(q.exp == e) & (q.include == 1)].set_index("subject")
        for pid, g in m.groupby("participant", sort=False):
            n = int(pid[1:])
            age = qq.age.get(n)
            sex = {"f": "F", "m": "M"}.get(qq.gender.get(n), "")
            person = f"E{e}_{pid}"
            first = True
            for i, x in enumerate(g.itertuples(index=False), start=1):
                shift = float(x.pitchShift) * 50
                out_z = x.pitch_out_z
                excl = bool(pd.isna(x.pitch_out) or abs(out_z) >= 2.5)
                rows.append({
                    "dataset": ID, "study": ID, "experiment": f"Experiment {e}", "group": "", "subject": person,
                    "person": person, "subject_excluded_by_authors": False,
                    "age": "" if pd.isna(age) else float(age), "sex": sex, "handedness": "", "population": "healthy",
                    "setting": "lab", "device": DEVICE, "session": 1, "block": BLOCK[x.phase], "trial": i,
                    "phase": PHASE[x.phase], "time_s": "", "task": "speech", "unit": "utterance",
                    "condition": f"{exposure:+d} cents exposure", "pert_type": "pitch_shift" if shift else "none",
                    "pert_value": shift, "pert_unit": "cents", "feedback": "altered" if shift else "normal",
                    "excluded_by_authors": excl, "stimulus": VOWEL[x.vwl], "segment": "vowel",
                    "measure_window": "pitch at the vowel centre", "duration_ms": float(round(x.duration * 1000, 3)),
                    "f0_hz": "" if pd.isna(x.pitch_out) else float(x.pitch_out),
                    "f0_heard_hz": "" if pd.isna(x.pitch_feedback) else float(x.pitch_feedback),
                    "f0_zscore": "" if pd.isna(out_z) else float(out_z),
                    "masking_noise": MASKING, "authors_trial_number": int(x.trial),
                    "judged_feedback_higher": "" if pd.isna(x.response) else bool(x.response == 1),
                })
    return rows, frames, q


def main():
    checks = stdlib.Checks()
    rows, frames, q = build(checks)
    a = pd.DataFrame(rows)
    a["exp"] = a.experiment.str[-1].astype(int)
    per = a.groupby("exp").person.nunique().to_dict()
    checks.internal("330 rows per participant (trials 11-340), 14 participants per experiment",
                    per == {1: 14, 2: 14} and (a.groupby("person").size() == 330).all(), str(per))
    checks.internal("blocks hold 90, 120 and 120 trials per participant (the paper's 100, 120, 120 less the first 10)",
                    (a.groupby(["person", "block"]).size().unstack()[["pretest", "exposure", "posttest"]] == [90, 120, 120]).all().all())
    ex = a[a.block == "exposure"]
    checks.internal("exposure shift is the constant +500 (Experiment 1) / -500 (Experiment 2) cents on every exposure trial",
                    bool((ex.pert_value[ex.exp == 1] == 500).all() and (ex.pert_value[ex.exp == 2] == -500).all()))
    nex = a[a.block != "exposure"]
    checks.internal("pretest and posttest shifts are 0, 30, 60, 100, 150 cents up or down",
                    set(nex.pert_value.abs()) == {0, 30, 60, 100, 150})
    # checks against the paper (Participants, Procedure)
    src = f"{PAPER_SRC}, Methods"
    for e, (nf, nm) in ((1, (13, 1)), (2, (11, 3))):
        x = a[a.exp == e].drop_duplicates("person")
        checks.published(f"Experiment {e}: participants", len(x), 14, 0, f"{src}, Participants")
        checks.published(f"Experiment {e}: female participants", int((x.sex == "F").sum()), nf, 0, f"{src}, Participants")
        checks.published(f"Experiment {e}: male participants", int((x.sex == "M").sum()), nm, 0, f"{src}, Participants")
    blk = a.groupby(["person", "block"]).size().unstack()
    checks.published("exposure block: trials per participant", int(blk.exposure.iloc[0]), 120, 0, f"{src}, Procedure")
    checks.published("posttest block: trials per participant", int(blk.posttest.iloc[0]), 120, 0, f"{src}, Procedure")
    x = a[(a.exp == 1)].drop_duplicates("person")
    checks.published("Experiment 1: SD of age (years; one age missing)", float(pd.to_numeric(x.age).std(ddof=1)), 1.9, 0.05,
                     f"{src}, Participants")
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Merged each participant's production file with the pitch-awareness response of the same trial, converted the "
                 "authors' shift unit (50 cents) to cents and duration to milliseconds, added age and sex from the questionnaire "
                 "(no free text), and replaced participant numbers with codes. Experiments 3 and 4, in which nobody spoke, and "
                 "the per-block perceptual thresholds are not included."),
        how=[
            "One row is one vowel production (one of five Dutch vowels) by one participant in Experiment 1 (exposure block "
            "+500 cents) or Experiment 2 (-500 cents), in the authors' order; trial counts the rows shared (authors_trial_number is "
            "the trial in the 340-trial run, starting at 11 because the first 10 trials are not in the files).",
            "f0_hz is the pitch the participant produced at the vowel centre (Praat autocorrelation), f0_heard_hz the pitch of the "
            "feedback they heard (empty where tracking failed), f0_zscore the authors' z-score within participant. "
            "pert_value is the pitch shift of the feedback in cents (the authors' pitchShift column times 50), positive raising "
            "the feedback pitch: 0, 30, 60, 100 or 150 cents up or down in pretest and posttest, a constant +500 or -500 in exposure.",
            "judged_feedback_higher is the participant's answer to the question asked after each pretest and posttest vowel, whether "
            "the feedback was higher (true) or lower (false) than their actual pitch; no question was asked during exposure.",
            "excluded_by_authors marks rows with no pitch estimate or with an absolute z-score of 2.5 or more, the rows the "
            "authors' analysis code drops; the paper states a 3 SD limit.",
        ],
        unit_note="vowel production (utterance) of one participant",
        differences=[
            "Age: the questionnaire gives a mean age of 20.6 years (SD 1.9, one age missing) in Experiment 1 and 21.2 (SD 1.3) in "
            "Experiment 2; the paper reports 20.7 (SD 1.9) and 21 (SD 1.46).",
            "Outlier rule: the paper states that pitch values more than 3 SD from the participant's mean were removed; the authors' "
            "analysis code removes rows with |z| >= 2.5, which is what excluded_by_authors marks.",
            "Table 4 (pitch change between experiments, model estimates): plain participant means from these rows give 55.4 "
            "(exposure minus pretest) and -70.0 (posttest minus exposure) cents against the reported 55.72 and -69.82; the paper's "
            "values come from a mixed model with random slopes, so they are not used as checks.",
        ],
    )
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
