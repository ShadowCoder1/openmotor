"""Acosta, Kearney, Nieto-Castanon & Guenther (2026, PsyArXiv c56ka): reflexive responses to upward and downward f0 (100 cents) and
F1 (30 percent) perturbations of auditory feedback in 23 adults over four sessions at different times of day.

Raw files, raw_extra/Acosta2026_PsyArXiv/ (OSF awkc8, fetched with tools/std/fetch.py):
  README.txt                                   (read as text) says how the files are laid out
  PTP_<type>-Session<n>-<F0|F1>-hz-Response-First-Level.csv   24 files = 6 types x 4 sessions; no header; 951 rows (one per ms);
                   column 0 = the perturbation at each time point as a decimal fraction (SimpleDIVA format), columns 1-23 = the
                   response in Hz of participants 1-23, each averaged over that participant's trials of the condition.
                   types: D0 / U0 = f0 shifted down / up (f0 measured), N0 = no shift (f0 processing), D1 / U1 / N1 the same for F1
  baseline_trials_measures_table.csv           per-trial f0 and F1 (Hz) of unperturbed productions: Hz, Feature, Subjects,
                   SessionNumber, MorningOrAfternoon, Gender
The participant order of the response columns is not stated in the README; it is checked here against the baseline table (each
column's unperturbed mean must follow the participants sorted by id; the ids skip number 016).
Run from the site folder: python3 -I tools/std/convert/Acosta2026_PsyArXiv.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Acosta2026_PsyArXiv"
PAPER_SRC = "Acosta et al. 2026 (PsyArXiv preprint c56ka)"
TYPES = {  # code: (measure, direction, label)
    "N0": ("f0", 0, "no shift (f0 processing)"), "U0": ("f0", +1, "f0 shifted up"), "D0": ("f0", -1, "f0 shifted down"),
    "N1": ("F1", 0, "no shift (F1 processing)"), "U1": ("F1", +1, "F1 shifted up"), "D1": ("F1", -1, "F1 shifted down")}
ORDER = ["N0", "U0", "D0", "N1", "U1", "D1"]
DEVICE = ("Audio-Technica BPHS1 headset, Audapter real-time processing, Focusrite Scarlett 212 soundcard, Behringer Xenyx 802 mixer "
          "(headphone signal 5 dB above the microphone signal)")


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["venue"] = "PsyArXiv preprint"
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def main():
    checks = stdlib.Checks()
    raw = stdlib.raw_dir(ID)
    base = pd.read_csv(raw / "baseline_trials_measures_table.csv")
    pn = {s: i + 1 for i, s in enumerate(sorted(base.Subjects.unique()))}  # ids skip 016: columns follow the sorted ids
    base["p"] = base.Subjects.map(pn)
    sex = base.groupby("p").Gender.first().map({"female": "F", "male": "M"}).to_dict()
    files = {}
    for f in sorted(raw.glob("PTP_*.csv")):
        m = re.fullmatch(r"PTP_([DNU][01])-Session(\d)-F[01]-hz-Response-First-Level\.csv", f.name)
        files[(m.group(1), int(m.group(2)))] = pd.read_csv(f, header=None).to_numpy()
    checks.internal("24 files = 6 types x 4 sessions, each 951 samples x (1 + 23 participants)",
                    len(files) == 24 and all(a.shape == (951, 24) for a in files.values()))
    # participant order check: unperturbed mean of each column vs the baseline table of the same participant, per session
    bmean = base.groupby(["p", "SessionNumber", "Feature"]).Hz.mean()
    ok = 0
    for meas, t in (("f0", "N0"), ("F1", "N1")):
        for s in range(1, 5):
            col_means = files[(t, s)][:, 1:].mean(axis=0)
            ref = np.array([bmean.get((p, s, meas), np.nan) for p in range(1, 24)])
            keep = ~np.isnan(ref)
            col_means, ref = col_means[keep], ref[keep]
            # does the participant of column j have the closest baseline mean to column j? (rank correlation)
            ok += int(np.corrcoef(col_means, ref)[0, 1] > 0.95)
    checks.internal("response columns follow the sorted participant numbers (correlation of column mean with that participant's "
                    "baseline mean above 0.95 in all 8 type x session sets)", ok == 8, f"{ok}/8")
    # perturbation columns
    last = {t: files[(t, 1)][-1, 0] for t in TYPES}
    first = {t: files[(t, 1)][0, 0] for t in TYPES}
    rows, extra = [], []
    rec_no = {}
    for p in range(1, 24):
        for s in range(1, 5):
            for t in ORDER:
                meas, d, label = TYPES[t]
                a = files[(t, s)]
                pert = a[:, 0]
                resp = a[:, p]
                rec_no[p] = rec_no.get(p, 0) + 1
                n = rec_no[p]
                fn = f"samples/P{p:02d}_session{s}_{t}.csv"
                out = pd.DataFrame({"time_s": np.arange(len(a)) / 1000.0, "response_hz": resp, "perturbation_fraction": pert})
                extra.append((fn, out.to_csv(index=False, float_format="%.9g", lineterminator="\r\n")))
                if d == 0:
                    pt, pv, pu, fb = "none", 0.0, "cents" if meas == "f0" else "percent", "normal"
                elif meas == "f0":
                    pt, pv, pu, fb = "pitch_shift", 100.0 * d, "cents", "altered"
                else:
                    pt, pv, pu, fb = "formant_shift", 30.0 * d, "percent", "altered"
                rows.append(dict(
                    dataset=ID, study=ID, experiment="Reflexive perturbation", group="", subject=f"P{p:03d}", person=f"P{p:03d}",
                    subject_excluded_by_authors=False, age="", sex=sex[p], handedness="", population="healthy", setting="lab",
                    device=DEVICE, session=s, block=f"session {s}", trial=n, phase="test", time_s="", task="speech",
                    unit="recording", condition=label, pert_type=pt, pert_value=pv, pert_unit=pu, feedback=fb,
                    excluded_by_authors=False, recording=n, samples_file=fn,
                    signal_kind=f"{meas} of the unaltered microphone signal, average over the participant's trials of this condition",
                    channels="response_hz;perturbation_fraction", channel_units="Hz;fraction of the signal frequency",
                    sample_rate_hz=1000, n_samples=len(a), duration_s=len(a) / 1000.0, events="",
                    processing=("average of the participant's trials of this condition in this session (the authors excluded 3.96% "
                                "of trials for tracking or recording problems), formatted for SimpleDIVA"),
                    trial_info=f"measure={meas};perturbation_code={t};trials_per_condition_designed=30"))
    df = pd.DataFrame(rows)
    # baseline table: recode to the final person codes (order of first appearance = participant 1..23)
    bt = base.rename(columns={"SessionNumber": "session", "MorningOrAfternoon": "time_of_day", "Feature": "measure"})
    bt["person"] = bt.p.map(lambda p: f"{ID}_P{p:04d}")
    bt["time_of_day"] = bt.time_of_day.map({"M": "morning", "A": "afternoon"})
    bt["sex"] = bt.p.map(sex)
    bt = bt[["person", "sex", "session", "time_of_day", "measure", "Hz"]].rename(columns={"Hz": "hz"})
    extra.append(("baseline_trials.csv", bt.to_csv(index=False, float_format="%.9g", lineterminator="\r\n")))
    checks.internal("person codes of baseline_trials.csv follow the release's order of first appearance",
                    list(dict.fromkeys(df.person)) == [f"P{p:03d}" for p in range(1, 24)])
    checks.internal("one row per participant x session x condition = 23 x 4 x 6", len(df) == 552 and df.person.nunique() == 23)
    checks.internal("no shift in the N conditions, perturbation present in the others",
                    all((files[(t, s)][:, 0] == 0).all() for t in ("N0", "N1") for s in range(1, 5))
                    and all(abs(files[(t, 1)][:, 0]).max() > 0.05 for t in ("U0", "D0", "U1", "D1")))
    checks.internal("perturbation column identical in all four sessions of a type",
                    all(np.allclose(files[(t, 1)][:, 0], files[(t, s)][:, 0]) for t in TYPES for s in range(2, 5)))
    # published checks
    src = f"{PAPER_SRC}, Methods"
    persons = df.drop_duplicates("person")
    checks.published("participants", len(persons), 23, 0, f"{src}, Participants")
    checks.published("women", int((persons.sex == "F").sum()), 18, 0, f"{src}, Participants")
    checks.published("men", int((persons.sex == "M").sum()), 5, 0, f"{src}, Participants")
    checks.published("sessions per participant", int(df.groupby("person").session.nunique().iloc[0]), 4, 0, f"{src}, Experimental Design")
    checks.published("F1 shift size, from the perturbation column (percent)", abs(first["D1"]) * 100, 30, 0.05, f"{src}, Experimental Design and Procedures")
    # Table 1 session baselines (mean over participants of each participant's session mean)
    pm = base.groupby(["Feature", "SessionNumber", "p"]).Hz.mean().groupby(["Feature", "SessionNumber"]).mean()
    pt_ = base.groupby(["Feature", "MorningOrAfternoon", "p"]).Hz.mean().groupby(["Feature", "MorningOrAfternoon"]).mean()
    t1 = {("f0", 1): 182.09, ("f0", 3): 189.39, ("f0", 4): 193.28, ("F1", 1): 640.13, ("F1", 2): 645.02, ("F1", 3): 637.17, ("F1", 4): 631.56}
    for (ft, s), r in t1.items():
        checks.published(f"Table 1: mean baseline {ft} in session {s} (Hz)", pm[(ft, s)], r, 0.005, f"{PAPER_SRC}, Results, Table 1")
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Converted the 24 SimpleDIVA-format CSV files (6 perturbation types x 4 sessions, participants in columns) into one "
                 "samples file per participant x session x type, with time in seconds; the response columns were assigned to "
                 "the participants in order (checked against the baseline table); the per-trial baseline table is added as "
                 "baseline_trials.csv with the participant codes of this release."),
        how=[
            "One row is one participant's average response in one session to one of six conditions: f0 shifted up or down by 100 cents, "
            "F1 shifted up or down by 30 percent, or no shift (separate no-shift conditions processed with the f0 and the F1 settings). "
            "Its samples (samples/P<participant>_session<n>_<type>.csv) have time_s (one sample per ms from the first row of the "
            "authors' file, the perturbation onset in their SimpleDIVA format), response_hz (the produced f0 or F1 in Hz, averaged over the "
            "participant's trials of that condition), and perturbation_fraction (the shift applied at that time as a decimal fraction of "
            "the signal: the f0 shift ramps in, the F1 shift is constant at -0.3 or +0.3).",
            "pert_value is the designed shift (+-100 cents for f0, +-30 percent for F1), positive raising the feedback. The deposit "
            "does not give single-trial traces or the number of trials averaged; the authors excluded 3.96% of trials. "
            "trial counts as the order of the participant's recordings.",
            "baseline_trials.csv holds the authors' per-trial unperturbed productions: person, sex, session, time of day (the four "
            "sessions were two mornings and two afternoons), measure (f0 or F1) and hz (the production in Hz).",
            "The response columns of the deposit carry no participant labels; they were assigned to the 23 participants in the order of the "
            "participant numbers (which skip one number), which the baseline table confirms (each column's mean follows that participant's unperturbed level in all 8 type x "
            "session sets). Age was not shared per participant (paper: mean 20.0 years, SD 1.70).",
        ],
        unit_note="recording (participant x session x condition average response)",
        differences=[
            "Table 1 baselines: the session means of f0 in sessions 1, 3 and 4 and of F1 in all four sessions reproduce, but the f0 mean "
            "of session 2 is 186.83 Hz here (paper 186.93), and the time-of-day means are 186.05 / 189.78 Hz (f0, morning / afternoon) and "
            "632.90 / 644.10 Hz (F1) against the reported 186.05 / 189.75 and 632.87 / 644.07; the SDs are not reproduced either "
            "(the table's SD definition is not stated).",
            "The size of the f0 shift (100 cents) is not used as a check: the deposit stores the perturbation as a 4-decimal fraction, "
            "which gives 100.0 and 100.06 cents for the two directions.",
        ],
    )
    return stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)


if __name__ == "__main__":
    main()
