"""Li, Yang, Fuchs & Aussems (2026), J Acoust Soc Am 159(6):5491-5503 (first posted as PsyArXiv preprint epz2u): "Perturbing
postural stability affects acoustic properties but not fluency of adult speech production during face-to-face interaction".
118 university students (59 pairs) played Taboo (a word-guessing game) in four rounds of 10 trials; the clue-giver described a target
word for up to a minute while standing on the ground or on a wobble board. One row is one clue-giving trial.

Raw files, raw_extra/Li2026_PsyArXiv_posture/ (OSF 68uxv, node licence CC BY-NC-SA 4.0; the deposit's 00_ReadMe.txt describes them):
  trial_exclusions_AE.csv ...  the authors' per-trial table of all 2640 clue-giving trials of the 66 pairs tested (the deposit has six
                          identical copies of it, one per analysis folder; the copies have the same sha256, see SOURCES.json): the target
                          word, which participant was clue-giver, the analysis variable postural_stability (ground, board), the key-press
                          time that ended the trial, the exclusion flags and reasons at participant and trial level. Its own gyro_* columns
                          are not used (see below).
  fundamental_frequency.csv  per-trial f0 and amplitude-envelope means, minima, maxima and SDs (amplitude_envelope.csv is byte-identical)
  speech_rate.csv, pause_duration.csv, filler_particle_frequency.csv  per-trial speech rate, mean and median pause duration, filler counts
  pause_frequency.csv     the annotated intervals (start, end, label) of every trial's TextGrid; only the <pause> labels are used, to
                          count pauses per trial exactly as the authors' Pause_frequency.Rmd does (start_time <= 60 s). The transcribed
                          words in this table are not copied.
  board_trial_metrics.csv, board_speech_trial_metrics.csv  gyroscope metrics of the wobble-board trials (whole trial; speaking versus silent stretches)
  demographic_information.csv  age, gender, native language, hand, leg, friend or stranger pairing (118 participants)
Not converted: the lab walk-through video, the Praat coding manual, the R scripts' output, and the transcribed words.
Run from the site folder: python3 -I tools/std/convert/Li2026_PsyArXiv_posture.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Li2026_PsyArXiv_posture"
SRC = "Li et al. 2026, J Acoust Soc Am"
DEVICE = ("wooden wobble board (Strongtek Professional Wobbleboard, 44.5 x 34.3 x 7.6 cm, two cylindrical supports so that it tilts only "
          "from side to side) with a WitMotion BWT901CL gyroscope under it; DJI Mic Mini wireless lavalier microphones")
SEG = "all of the clue-giver's speech segments in the trial"
WIN = ("mean over the speech segments of the trial, leaving out stretches that overlap the guesser's speech or marked noise; the trial "
       "lasted up to 60 s")
SEX = {"Female": "F", "Male": "M", "Non-binary / third gender": "other"}
HAND = {"Right hand": "right", "Left hand": "left"}
SHEET_ROUND = {"p1_b1": 1, "p2_b1": 2, "p1_b2": 3, "p2_b2": 4}  # filler Rmd: participant1 is clue-giver in rounds 1 and 3, participant2 in 2 and 4

R = stdlib.raw_dir(ID)
ck = stdlib.Checks()


def csv(name, **kw):
    return pd.read_csv(R / name, **kw)


def sha(name):
    return next(x["sha256"] for x in json.load(open(R / "SOURCES.json")) if x["file"] == name)


# ---- read -----------------------------------------------------------------------------------------------------------------
ex = csv("trial_exclusions_AE.csv")
same = {sha(n) for n in ("trial_exclusions_AE.csv", "trial_exclusions_F0.csv", "trial_exclusions_PD.csv", "trial_exclusions_PF.csv",
                         "trial_exclusions_SR.csv", "trial_exclusions_FP.csv")}
ck.internal("the six trial_exclusions.csv copies of the deposit are identical (same sha256)", len(same) == 1)
ck.internal("amplitude_envelope.csv and fundamental_frequency.csv are identical (same sha256)",
            sha("amplitude_envelope.csv") == sha("fundamental_frequency.csv"))
f0 = csv("fundamental_frequency.csv")
sr = csv("speech_rate.csv")
pdur = csv("pause_duration.csv")
fil = csv("filler_particle_frequency.csv")
pf = csv("pause_frequency.csv")
bt = csv("board_trial_metrics.csv")
bs = csv("board_speech_trial_metrics.csv")
demo = csv("demographic_information.csv", encoding="utf-8-sig")

ex["tg"] = ex["textgrid_file_name"]
ck.internal("one exclusion-file row per pair and trial (2640 rows, 66 pairs)",
            len(ex) == 2640 and ex.pair.nunique() == 66 and not ex.duplicated(["pair", "trial_number"]).any())
ck.internal("the clue-giver of each row is participant 1 in rounds 1 and 3 (sheets p1_*) and participant 2 in rounds 2 and 4",
            bool(((ex.Sheet_Name.str.startswith("p1")) & (ex.clue_giver == ex.participant_1)
                  | (ex.Sheet_Name.str.startswith("p2")) & (ex.clue_giver == ex.participant_2)).all()))


def attach(base, tab, cols, key="tg", name=""):
    """Left-join tab[cols] onto base by the recording file name; every row of tab must find its trial."""
    assert tab[key].is_unique, name
    lost = set(tab[key]) - set(base["tg"])
    assert not lost, (name, len(lost))
    return base.merge(tab[[key] + cols].rename(columns={key: "tg"}), on="tg", how="left")


m = ex.copy()
for df in (f0, sr, pdur):
    df["tg"] = df["textgrid_file_name"]
m = attach(m, f0, ["clue_giver_condition", "f0min_mean", "f0max_mean", "f0mean_mean", "f0sd_mean", "envmin_mean", "envmax_mean",
                   "envmean_mean", "envsd_mean"], name="f0")
m = attach(m, sr, ["speech_rate_mean"], name="speech_rate")
m = attach(m, pdur, ["mean_pause", "median_pause"], name="pause_duration")
ck.internal("the condition column of the f0 table agrees with the exclusion file's postural_stability on every trial it covers",
            bool((m.dropna(subset=["clue_giver_condition"]).clue_giver_condition.map({"Stable (ground)": "ground", "Unstable (board)": "board"})
                  == m.dropna(subset=["clue_giver_condition"]).postural_stability).all()))
pdur_cond = pdur.merge(ex[["tg", "postural_stability", "clue_giver"]], on="tg")
pdur_cond["tag"] = pdur_cond.tg.str.extract(r"_(ground|board)_p[12]\.TextGrid")[0]
ck.internal("pause_duration.csv: clue-giver agrees with the exclusion file on every trial; its stability_condition column copies the "
            "ground or board tag of the recording file name (not the analysis variable), so it is not used",
            bool((pdur_cond.clue_giver_x == pdur_cond.clue_giver_y).all() and (pdur_cond.stability_condition == pdur_cond.tag).all()),
            f"{int((pdur_cond.stability_condition != pdur_cond.postural_stability).sum())} of {len(pdur_cond)} rows differ from the analysis variable")

# pause counts, as in the authors' Pause_frequency.Rmd: keep start_time <= 60 s, keep label == "<pause>", count per trial
pf["tg"] = pf["textgrid_file_name"]
pf_pause = pf[(pf.start_time <= 60) & (pf.label == "<pause>")]
npause = pf_pause.groupby("tg").size().rename("n_pauses")
has_grid = set(pf["tg"])
m = m.merge(npause, on="tg", how="left")
m["n_pauses"] = np.where(m.tg.isin(has_grid) | (m.trial_exclusion == 0), m.n_pauses.fillna(0), np.nan)  # authors: complete(...) gives 0 to valid trials without a pause
lost = set(pf["tg"]) - set(ex["tg"])
ck.internal("every TextGrid of the interval table belongs to a trial of the exclusion file", not lost, f"{len(lost)} unmatched")

# fillers (keyed by the pair and trial number)
fil["pair"] = fil.participant1.astype(str) + "_" + fil.participant2.astype(str)
fil = fil.rename(columns={"uh_freq": "filler_uh_n", "uhm_freq": "filler_uhm_n", "um_freq": "filler_um_n", "total_count": "filler_total_n"})
ck.internal("filler table: one row per pair and trial, total = uh + uhm + um", bool(not fil.duplicated(["pair", "trial_number"]).any()
            and (fil.filler_total_n == fil.filler_uh_n + fil.filler_uhm_n + fil.filler_um_n).all()))
m = m.merge(fil[["pair", "trial_number", "filler_uh_n", "filler_uhm_n", "filler_um_n", "filler_total_n"]], on=["pair", "trial_number"], how="left")
ck.internal("every filler row belongs to a trial of the exclusion file", len(fil) == m.filler_total_n.notna().sum())

# wobble-board metrics
bt = bt.rename(columns={"group_name": "pair"})
ck.internal("board_trial_metrics: one row per pair and trial, clue-giver as in the exclusion file, only wobble-board trials",
            bool(not bt.duplicated(["pair", "trial_number"]).any()))
m = m.merge(bt[["pair", "trial_number", "clue_giver_pid", "angular_vel_rms_x", "tilt_sd_x"]], on=["pair", "trial_number"], how="left")
ck.internal("board metrics sit on wobble-board trials of the right clue-giver",
            bool((m.dropna(subset=["angular_vel_rms_x"]).postural_stability == "board").all()
                 and (m.dropna(subset=["angular_vel_rms_x"]).clue_giver == m.dropna(subset=["angular_vel_rms_x"]).clue_giver_pid).all()))
bs = bs.rename(columns={"group_name": "pair"})
bsw = bs.pivot_table(index=["pair", "trial_number"], columns="speech_production", values=["angular_vel_rms_x", "tilt_sd_x"], aggfunc="first")
bsw.columns = [f"{a}|{b}" for a, b in bsw.columns]
bsw = bsw.reset_index()
ck.internal("board_speech_trial_metrics: one speaking and one non-speaking row per trial at most",
            bool(not bs.duplicated(["pair", "trial_number", "speech_production"]).any()))
m = m.merge(bsw, on=["pair", "trial_number"], how="left")
ck.internal("rows kept in order and number after all merges", len(m) == len(ex) and (m.tg.values == ex.tg.values).all())

# ---- participants ---------------------------------------------------------------------------------------------------------
demo = demo.set_index("Participant_ID")
ck.internal("demographics: 118 participants, all of them clue-givers of the exclusion file",
            len(demo) == 118 and set(demo.index) <= set(ex.clue_giver))
pair_no = {p: i for i, p in enumerate(pd.unique(ex.pair), 1)}
pers_pair = ex.drop_duplicates("clue_giver").set_index("clue_giver").pair.map(pair_no)
rel = demo.Friend_Stranger.str.startswith("Yes").map({True: "friend", False: "stranger"})
pair_rel = pd.DataFrame({"pair": pers_pair.reindex(demo.index), "rel": rel}).groupby("pair").rel.nunique()
n_disagree = int((pair_rel > 1).sum())

# ---- rows -----------------------------------------------------------------------------------------------------------------
ctr = {}
rows = []
for r in m.itertuples(index=False):
    cg = int(r.clue_giver)
    ctr[cg] = ctr.get(cg, 0) + 1
    board = r.postural_stability == "board"
    d = demo.loc[cg] if cg in demo.index else None
    reason = (r.reason_participant_exclusion if r.participant_exclusion == 1 else r.reason_trial_exclusion)
    rows.append(dict(
        dataset=ID, study=SRC, experiment="", group="", subject=str(cg), person=str(cg),
        subject_excluded_by_authors=bool(r.participant_exclusion == 1),
        age=(d.Age_in_years if d is not None else None), sex=(SEX.get(d.Gender, "") if d is not None else ""),
        handedness=(HAND[d.Dominant_Hand] if d is not None else ""), population="healthy", setting="lab", device=DEVICE,
        session=1, block=r.Sheet_Name, trial=ctr[cg], phase="test", time_s="", task="speech", unit="trial",
        condition="Unstable (board)" if board else "Stable (ground)",
        pert_type="wobble_board" if board else "none", pert_value="" if board else 0, pert_unit="", feedback="normal",
        excluded_by_authors=bool(r.trial_exclusion == 1),
        stimulus=r.target_word, segment=SEG, measure_window=WIN, f0_hz=r.f0mean_mean, authors_trial_number=int(r.trial_number),
        f0_min_hz=r.f0min_mean, f0_max_hz=r.f0max_mean, f0_sd_hz=r.f0sd_mean,
        amp_env_mean=r.envmean_mean, amp_env_min=r.envmin_mean, amp_env_max=r.envmax_mean, amp_env_sd=r.envsd_mean,
        speech_rate_syll_s=r.speech_rate_mean, n_pauses=r.n_pauses, pause_mean_s=r.mean_pause, pause_median_s=r.median_pause,
        filler_uh_n=r.filler_uh_n, filler_uhm_n=r.filler_uhm_n, filler_um_n=r.filler_um_n, filler_total_n=r.filler_total_n,
        guess_time_s=r.response_time, board_angvel_rms_dps=r.angular_vel_rms_x, board_tilt_sd_deg=r.tilt_sd_x,
        exclusion_reason=(str(reason).strip() if isinstance(reason, str) else ""),
    ))
a = pd.DataFrame(rows)
# the pivoted board columns have '|' in their names, so they are filled by column access, not itertuples attributes
a["board_angvel_rms_speaking_dps"] = m["angular_vel_rms_x|Speaking"].values
a["board_angvel_rms_pause_dps"] = m["angular_vel_rms_x|Non-speaking"].values
a["board_tilt_sd_speaking_deg"] = m["tilt_sd_x|Speaking"].values
a["board_tilt_sd_pause_deg"] = m["tilt_sd_x|Non-speaking"].values

people = []
for cg, pr in pers_pair.items():
    d = demo.loc[cg] if cg in demo.index else None
    people.append(dict(person=str(int(cg)), pair=int(pr), pair_relationship=(rel.loc[cg] if d is not None else ""),
                       english_native=(d.English_native == "Yes") if d is not None else "",
                       other_languages=(d.Other_languages.startswith("Yes")) if d is not None else "",
                       dominant_leg=({"Right leg": "right", "Left leg": "left", "I do not know": "unknown",
                                      "I have no dominant leg or preference": "none"}[d.Dominant_Leg] if d is not None else "")))
PCOLS = [("pair", "integer", "", "Number of the pair (dyad) the participant played Taboo with, 1-66, in order of first appearance."),
         ("pair_relationship", "category", "friend, stranger",
          "Whether the participant said the partner was a friend or someone they had never interacted with before (the authors' Friend_Stranger answer); each partner answered for themselves, and the two partners disagree in 2 of the 59 pairs. Empty for participants whose data the authors excluded (no questionnaire shared)."),
         ("english_native", "true/false", "", "The participant reported English as a first language."),
         ("other_languages", "true/false", "", "The participant reported speaking at least one other language (the languages themselves are not copied)."),
         ("dominant_leg", "category", "right, left, unknown, none",
          "The participant's dominant leg as they reported it; unknown = 'I do not know', none = no dominant leg or preference.")]

# ---- checks against the paper ---------------------------------------------------------------------------------------------
P = "Li et al. 2026 (J Acoust Soc Am 159:5491; preprint v2 identical)"
kept = a[~a.subject_excluded_by_authors]
ppl_kept = kept.drop_duplicates("person")
ck.published("final sample: participants", ppl_kept.person.nunique(), 118, 0, f"{P}, Abstract and Methods, Participants")
ck.published("final sample: pairs", len({p for p in m[m.participant_exclusion == 0].pair}), 59, 0, f"{P}, Abstract")
ck.published("tested but excluded: participants", a[a.subject_excluded_by_authors].person.nunique(), 14, 0, f"{P}, Methods, Participants (14 participants, 7 pairs)")
ck.published("tested but excluded: pairs", len({p for p in m[m.participant_exclusion == 1].pair}), 7, 0, f"{P}, Methods, Participants")
rs = m[m.participant_exclusion == 1].drop_duplicates("clue_giver").reason_participant_exclusion.str.strip()
ck.published("excluded for microphone popping or rustling", rs.str.startswith("continuous microphone").sum(), 8, 0, f"{P}, Methods, Participants (1) n = 8")
ck.published("excluded for low English proficiency only", (rs == "low English language proficiency and speaking a foreign language during the task").sum(), 2, 0,
             f"{P}, Methods, Participants (2) n = 2")
ck.published("excluded for not adhering to the balance manipulation", rs.str.startswith("not adhering").sum(), 2, 0, f"{P}, Methods, Participants (3) n = 2")
ck.published("excluded for several reasons", rs.str.contains("as well as").sum(), 2, 0, f"{P}, Methods, Participants (4) n = 2")
dm = demo.copy()
ck.published("participants: females", (dm.Gender == "Female").sum(), 95, 0, f"{P}, Methods, Participants")
ck.published("participants: males", (dm.Gender == "Male").sum(), 19, 0, f"{P}, Methods, Participants")
ck.published("participants: non-binary", (dm.Gender == "Non-binary / third gender").sum(), 2, 0, f"{P}, Methods, Participants")
ck.published("participants: prefer not to disclose", (dm.Gender == "Prefer not to disclose").sum(), 2, 0, f"{P}, Methods, Participants")
ck.published("age: youngest", dm.Age_in_years.min(), 18, 0, f"{P}, Methods, Participants (18 to 27 years)")
ck.published("age: oldest", dm.Age_in_years.max(), 27, 0, f"{P}, Methods, Participants (18 to 27 years)")
ck.published("age: mean (years; 117 of 118 gave an age)", dm.Age_in_years.mean(), 18.93, 0.005, f"{P}, Methods, Participants (M = 18.93)")
ck.published("age: SD (years)", dm.Age_in_years.std(), 1.36, 0.005, f"{P}, Methods, Participants (SD = 1.36)")
ck.published("English as first language", (dm.English_native == "Yes").sum(), 98, 0, f"{P}, Methods, Participants")
ck.published("trials of the 59 included pairs", len(kept), 2360, 0, f"{P}, Methods, Trial exclusions (25 of 2360 trials)")
ck.published("trials excluded", kept.excluded_by_authors.sum(), 25, 0, f"{P}, Methods, Trial exclusions")
ck.published("valid trials", (~kept.excluded_by_authors).sum(), 2335, 0, f"{P}, Methods, Trial exclusions")
rt = kept[kept.excluded_by_authors].exclusion_reason.value_counts()
for key, n, label in (("persistent mic popping and rustling noise", 6, "microphone popping or rustling"), ("persistent external noise", 2, "external noise"),
                      ("skipped by the experimenter", 8, "skipped by the experimenter"), ("skipped by the participant", 1, "skipped by the participant"),
                      ("participant asked the experimenter a question mid-trial", 4, "participant asked the experimenter a question"),
                      ("participant said the target word", 3, "participant said the target word"),
                      ("participants spoke a different language", 1, "participants spoke another language")):
    ck.published(f"trials excluded: {label}", rt.get(key, 0), n, 0, f"{P}, Methods, Trial exclusions")
npp = kept.groupby(kept.person.map(dict(zip(m.clue_giver.astype(str), m.pair)))).excluded_by_authors.sum()
vc = npp.value_counts()
for k, n in ((0, 39), (1, 15), (2, 5)):
    ck.published(f"pairs with {k} trial exclusions", vc.get(k, 0), n, 0, f"{P}, Methods, Trial exclusions")
ok = kept[~kept.excluded_by_authors]
for col, n, label, sec in (("f0_hz", 2295, "mean f0", "Operationalization"),
                           ("speech_rate_syll_s", 2303, "mean speech rate", "Operationalization"), ("pause_mean_s", 2165, "mean pause duration", "Operationalization"),
                           ("n_pauses", 2335, "pause frequency", "Operationalization"), ("filler_total_n", 2335, "filler particle frequency", "Operationalization")):
    ck.published(f"valid trials with a {label}", ok[col].notna().sum(), n, 0, f"{P}, Methods, {sec}")
ck.internal("valid trials with a mean amplitude envelope: 2303, the same as for speech rate (the paper's text says 2305)",
            int(ok.amp_env_mean.notna().sum()) == 2303 and int(ok.speech_rate_syll_s.notna().sum()) == 2303, f"{int(ok.amp_env_mean.notna().sum())}")
bd = ok[ok.condition == "Unstable (board)"]
ck.published("valid wobble-board trials with balance metrics", bd.board_angvel_rms_dps.notna().sum(), 1153, 0, f"{P}, Methods, Operationalization (1153 usable board trials)")
bpp = bd.groupby("person")[["board_angvel_rms_dps", "board_tilt_sd_deg"]].mean()
ck.published("clue-givers with balance metrics", bpp.board_angvel_rms_dps.notna().sum(), 117, 0, f"{P}, Results, Manipulation checks (N = 117)")
ck.published("angular velocity RMS, mean over clue-givers (deg/s)", bpp.board_angvel_rms_dps.mean(), 12.84, 0.005, f"{P}, Results, Manipulation checks (M = 12.84)")
ck.published("angular velocity RMS, SD over clue-givers (deg/s)", bpp.board_angvel_rms_dps.std(), 3.73, 0.005, f"{P}, Results, Manipulation checks (SD = 3.73)")
ck.published("tilt SD, mean over clue-givers (deg)", bpp.board_tilt_sd_deg.mean(), 4.20, 0.005, f"{P}, Results, Manipulation checks (M = 4.20)")
ck.published("tilt SD, SD over clue-givers (deg)", bpp.board_tilt_sd_deg.std(), 1.23, 0.005, f"{P}, Results, Manipulation checks (SD = 1.23)")

# condition means: average of the participants' own condition means over the 118 participants; the paper's values are estimated
# marginal means of mixed models, so they can differ slightly from these raw averages
pm = ok.groupby(["person", "condition"])[["f0_hz", "amp_env_mean", "speech_rate_syll_s", "pause_mean_s", "n_pauses", "filler_total_n"]].mean().reset_index()
cm = pm.groupby("condition").mean(numeric_only=True)
dm_ = pm.pivot(index="person", columns="condition")
for col, board, ground, tol, label in (("f0_hz", 198, 195, 1.0, "f0 (Hz)"), ("amp_env_mean", 0.00468, 0.00439, 0.0003, "amplitude envelope"),
                                       ("speech_rate_syll_s", 3.15, 3.12, 0.05, "speech rate (syllables/s)"), ("pause_mean_s", 0.97, 1.02, 0.05, "pause duration (s)")):
    ck.published(f"{label}: unstable (board) mean over participants", cm.loc["Unstable (board)", col], board, tol, f"{P}, Results, estimated marginal mean")
    ck.published(f"{label}: stable (ground) mean over participants", cm.loc["Stable (ground)", col], ground, tol, f"{P}, Results, estimated marginal mean")
# pause and filler counts: the paper's means come from negative binomial mixed models (log link), so they sit below the raw means;
# the board-to-ground ratio is the comparable quantity (paper: rate ratio 0.97 for pauses, 0.99 for fillers)
ck.published("pause frequency: rate ratio board / ground (raw participant means)", cm.loc["Unstable (board)", "n_pauses"] / cm.loc["Stable (ground)", "n_pauses"], 0.97, 0.01,
             f"{P}, Results, Pause frequency (rate ratio = 0.97)")
ck.published("pause frequency: percent lower on the board", 100 * (1 - cm.loc["Unstable (board)", "n_pauses"] / cm.loc["Stable (ground)", "n_pauses"]), 2.6, 0.2,
             f"{P}, Results, Pause frequency (estimated to be 2.6% lower)")
ck.published("filler particle frequency: rate ratio board / ground (raw participant means)", cm.loc["Unstable (board)", "filler_total_n"] / cm.loc["Stable (ground)", "filler_total_n"], 0.99, 0.02,
             f"{P}, Results, Filler particle frequency (rate ratio = 0.99)")
diff_f0 = (dm_[("f0_hz", "Unstable (board)")] - dm_[("f0_hz", "Stable (ground)")]).dropna()
ck.published("f0: mean within-participant difference board minus ground (Hz)", diff_f0.mean(), 3.34, 0.3, f"{P}, Results, f0 (beta = 3.34 Hz)")
diff_ae = (dm_[("amp_env_mean", "Unstable (board)")] - dm_[("amp_env_mean", "Stable (ground)")]).dropna()
ck.published("amplitude envelope: mean within-participant difference board minus ground", diff_ae.mean(), 0.000294, 0.00005, f"{P}, Results, Amplitude envelope (beta = 0.000294)")
for col, beta, tol, label in (("speech_rate_syll_s", 0.025, 0.005, "speech rate (syllables/s)"), ("pause_mean_s", -0.052, 0.005, "pause duration (s)")):
    dd = (dm_[(col, "Unstable (board)")] - dm_[(col, "Stable (ground)")]).dropna()
    ck.published(f"{label}: mean within-participant difference board minus ground", dd.mean(), beta, tol, f"{P}, Results ({label.split(' (')[0]}: estimated difference {abs(beta)})")
ck.internal("value ranges: f0 80-335 Hz, speech rate under 7 syllables/s, pauses and fillers non-negative integers",
            bool(a.f0_hz.dropna().between(80, 335).all() and (a.speech_rate_syll_s.dropna() < 7).all()
                 and (a.n_pauses.dropna() >= 0).all() and (a.filler_total_n.dropna() >= 0).all()))
ck.internal("each participant's rows run in the authors' trial order", bool(a.groupby("subject").authors_trial_number.apply(lambda s: s.is_monotonic_increasing).all()))
ck.internal("20 clue-giving trials per participant", bool((a.groupby("subject").size() == 20).all()))

print("participant means by condition:\n", cm.round(5).T)
print("diff f0", round(diff_f0.mean(), 3), "n", len(diff_f0), "| diff AE", round(diff_ae.mean(), 6))
for c in ck.items:
    if not c["ok"]:
        print("FAILED:", c)
    elif c["kind"] == "published":
        print(f"  ok  {c['what']}: {c['computed']:.6g} vs {c['reported']:.6g}")

mism = int((m.tg.str.extract(r"_(ground|board)_p[12]\.TextGrid")[0] != m.postural_stability).sum())
meta = dict(
    paper=dict(authors="Li M, Yang JE, Fuchs S, Aussems S", year=2026,
               title="Perturbing postural stability affects acoustic properties but not fluency of adult speech production during face-to-face interaction",
               venue="J Acoust Soc Am 159(6):5491-5503 (first posted as PsyArXiv preprint epz2u)", doi="10.1121/10.0043950"),
    lab="Aussems (University of Warwick) / Fuchs (ZAS Berlin)", data_url="https://osf.io/68uxv/",
    license="CC BY-NC-SA 4.0", license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    changes=("Built one table of the 2640 clue-giving trials of the exclusion file, in its order (by pair number, then by the "
             "trial number of the session), and added to each trial the authors' trial-level values from their other tables, matched by recording file name: f0 and "
             "amplitude-envelope statistics, speech rate, mean and median pause duration, filler counts, and the wobble-board gyroscope "
             "metrics. Pauses per trial were counted from the authors' table of annotated intervals with the filter of their analysis script "
             "(only <pause> intervals that start within 60 s); the transcribed words of that table were not copied. The authors' exclusions are "
             "kept and flagged (14 participants, 25 trials), with their stated reasons. Left out: the deposit's video, coding manual and "
             "analysis output, heights and weights, the participants' other-language text answers and health screening answers (all 'No'), "
             "recording dates and times, taboo words, the exclusion file's own gyro_* columns, and file names. Participant identifiers are "
             "replaced by codes."),
    how=["One row is one clue-giving trial: one participant, standing on the ground or on a wobble board, described a target word to "
         "the partner for up to 60 s without saying the target or five related taboo words (the partner is the guesser; "
         "only the clue-giver's speech was analysed). Each pair played four rounds of 10 trials, the two partners taking turns as "
         "clue-giver, so a clue-giver has 20 trials (10 on the ground, 10 on the board). block is the authors' sheet label (p1_b1 = "
         "round 1 with the first partner as clue-giver, p2_b1 = round 2, p1_b2 = round 3, p2_b2 = round 4); trial counts the "
         "participant's own clue-giving trials from 1 and authors_trial_number is the authors' trial counter of the session (12 to 54).",
         "condition is the authors' postural stability variable; pert_type is wobble_board on board trials (the size of the "
         "perturbation is not a number, so pert_value is empty) and none on ground trials. Auditory feedback was normal. "
         "f0_hz, the amplitude envelope, speech rate and pause durations are the authors' per-trial means over the trial's speech "
         "segments in Hz, uncalibrated Hilbert-envelope units, syllables per second and seconds; the amplitude envelope has no "
         "calibration (compare within a participant). n_pauses and the filler counts are counts per trial.",
         "board_* columns are the gyroscope metrics of the board under the participant (angular velocity RMS in degrees per second, "
         "SD of the unwrapped tilt angle in degrees), whole trial and for the speaking and the silent stretches of the trial; they "
         "exist on board trials only. Rows of the 14 participants the authors left out of the analysis have only the trial "
         "information (word, condition, key-press time, reason): the authors shared no acoustic values for them."],
    unit_note="clue-giving trial (one participant describing a target word for up to 60 s, with acoustic, fluency and wobble-board measures)",
    differences=[
        f"The recording file names carry a ground or board tag that disagrees with the authors' postural stability variable on {mism} of 2640 "
        "rows, all of them in participant 2's rounds (rounds 2 and 4); this tag is not used. The condition here is the authors' "
        "postural_stability variable, which is also the condition column of their f0 and pause-duration tables. The gyroscope backs it: the "
        "board moves (angular velocity RMS about 13 deg/s) on exactly the rows labelled board and is at rest (about 0) on the ground rows "
        "of participant 1, whereas the file name tags of participant 2's rounds would put board trials on rows where the board did "
        "not move. In the authors' analysis variable, therefore, participant 2 of every pair stands on the ground in round 2 and on the board in "
        "round 4, and only participant 1's order is counterbalanced across pairs (the paper says the order was counterbalanced over pairs "
        "without saying for which partner).",
        "The exclusion file's own gyro_tilt_sd_x column (values near 140 to 180) looks like the SD of an angle that wraps at 180 degrees "
        "(the paper uses the unwrapped angle) and is not used; the board_trial_metrics table (tilt SD 0.7 to 14.6 degrees), whose "
        "clue-giver means reproduce the paper's manipulation check, is used for the board metrics. "
        "The exclusion file also holds gyro values on ground trials (board at rest, near zero); they are not copied.",
        "The paper's text gives 2305 trials with a mean amplitude envelope; the shared table has 2303 valid trials with one, the same as for "
        "speech rate (2303), while the f0 count (2295) and all other counts match, so the 2305 looks like an error in the text. "
        "The stability_condition column of the authors' pause_duration table copies the ground or board tag of the recording file name, so it "
        "disagrees with the analysis variable on 578 of its 2183 rows (their R script takes the condition from the exclusion file); it is not used.",
        "The paper's condition means are estimated marginal means of mixed models (negative binomial for the counts); the values here "
        "are averages of the participants' own condition means over the valid trials, so they can differ slightly (the checks use a tolerance).",
    ],
)
if not ck.passed():
    sys.exit("checks failed")
rec = stdlib.release(ID, "speech", a.to_dict("records"), meta, ck, participants=people, participant_columns=PCOLS)
