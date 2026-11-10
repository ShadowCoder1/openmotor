"""Gabriel, Mushtaq & Morehead (2024), PLoS Comput Biol: de novo learning with surface-EMG cursor control (retry of a 'too large' skip).
Raw files (untrusted data, read only) in raw_extra/Gabriel2024_PLoSCompBiol/ from https://osf.io/79s38/ (CC BY 4.0):
all_participant_performance_features.hdf5 (2.3 GB; one group per participant with 3612 trials: cursor_trajectory, controller_output and the
scaled hand and shin EMG power profiles (3612 x 2000 samples at 1 kHz, the 2 s window starting 0.5 s into the trial) plus per-trial feature
columns) and participants_anonymised.csv (sex, condition A/B). The authors' notebook main_analyses.ipynb and utilities/*.py (read as text)
define the derived measures; this converter re-implements compute_peak_aligned_trajectory, find_peak_time, compute_peak_aligned_rms and
count_target_hits exactly. Condition A = incongruent, B = congruent (utilities/data_load.py load_participant_table).
One row per trial; samples/ has 100 Hz cursor, controller and EMG-power time courses per participant and session."""
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Gabriel2024_PLoSCompBiol"
RAW = stdlib.raw_dir(ID)
TMP = stdlib.HERE / "tmp_samples" / ID
TMP.mkdir(parents=True, exist_ok=True)
CACHE = TMP / "rows.pkl"
checks = stdlib.Checks()
names = [c[1] for c in stdlib.columns_for("emg_path_following")]
SCREEN_WIDTH, BOX = 1080, 450


def path_function(ts, s):
    f = s * (0.5 + 0.5 * np.cos((ts / 0.84 + 0.5) * 2 * np.pi))
    f[ts / 0.84 > 1] = 0
    return f


def ideal_path(s):
    f = SCREEN_WIDTH // 2 + path_function(np.linspace(0, 1, 500), s) * BOX
    return np.concatenate([np.ones(500) * f[0], f, np.ones(1000) * f[-1]])


TARGET = {s: ideal_path(s) for s in (-0.6, -0.4, -0.2, 0.2, 0.4, 0.6)}
T_START, T_END = 500 + 6 * 500 / BOX, 1000 - 6 * 500 / BOX
T_COORDS = np.linspace(T_START, T_END, 25)
TS_MS = np.arange(0, 2000)


def hits(cursor, target):
    tx = interpolate.interp1d(TS_MS, target)(T_COORDS)
    cc = np.stack([cursor[500:1000], TS_MS[500:1000]]).T
    tc = np.stack([tx, T_COORDS]).T
    return int(np.sum(np.any(cdist(cc, tc) <= 6, axis=0)))


part = pd.read_csv(RAW / "participants_anonymised.csv")
assert part.dropped.isna().all()
order = list(part.id)
sex = dict(zip(part.id, part.sex))
group = dict(zip(part.id, part.condition.map({"A": "incongruent", "B": "congruent"})))

if CACHE.exists():
    rows = pickle.load(open(CACHE, "rb"))
else:
    rows = []
    f = h5py.File(RAW / "all_participant_performance_features.hdf5", "r")
    for pi, pid in enumerate(order):
        g = f[pid]
        d = pd.DataFrame({k: np.array(g[k]) for k in g.keys() if g[k].ndim == 1})
        d["direction"] = d.direction.astype(str)
        d["_i"] = np.arange(len(d))
        d = d.sort_values(["session_number", "block_number", "trial_number"], kind="stable").reset_index(drop=True)
        CUR = np.array(g["cursor_trajectory"])
        CTL = np.array(g["controller_output"])
        HP = np.array(g["scaled_hand_ch_profile"])
        SP = np.array(g["scaled_shin_ch_profile"])
        blocksize = d.groupby(["session_number", "block_number"]).trial_number.transform("size")
        for sess in range(1, 6):
            sel = d[d.session_number == sess]
            rows_s = []
            for j, (_, r) in enumerate(sel.iterrows()):
                trial_idx = len(rows) + 1
                cur = CUR[int(r._i)].astype(np.float64)
                signed = -r.scale if r.direction == "left" else r.scale
                signed = round(signed, 1)
                tgt = TARGET[signed]
                hi, si = r.hand_peak_index, r.shin_peak_index
                aligned = cur
                ptime = np.nan
                ok = not (np.isnan(hi) or np.isnan(si))
                if ok:
                    i1, i2 = int(min(hi, si)), int(max(hi, si))
                    if hi != si and not np.isnan(cur[i1:i2]).any() and i1 < i2:
                        i_mx = int(np.argmax(np.abs(cur[i1:i2] - SCREEN_WIDTH / 2)))
                        ts = np.linspace(0.5, 2.5, len(cur))
                        lag = (0.5 + (i1 + i_mx) / 1000) - 1.21
                        aligned = np.interp(ts, ts - lag, cur)
                        ptime = 0.5 + (i1 + i_mx) / 1000
                nanc = np.isnan(cur).any()
                h_al = hits(aligned, tgt) if not np.isnan(aligned).all() else None
                h_un = hits(cur, tgt) if not np.isnan(cur).all() else None
                rms = float(np.sqrt(np.mean((tgt[500:1000] - aligned[500:1000]) ** 2)))
                im = int(np.nanargmax(np.abs(cur - SCREEN_WIDTH / 2)))
                n_in_block = int(blocksize[sel.index[j]])
                fb = bool(r.post_trial_feedback_visible)
                kind = "practice" if (sess == 1 and n_in_block == 12) else ("training" if fb else "test")
                t = lambda x: None if np.isnan(x) else 0.5 + float(x) / 1000  # noqa: E731
                rows.append(dict(
                    dataset=ID, study="Gabriel, Mushtaq & Morehead 2024, PLoS Comput Biol",
                    experiment="de novo EMG path following", group=group[pid], subject=f"P{pi + 1:02d}",
                    subject_excluded_by_authors=False, age=None, sex={"Male": "M", "Female": "F"}[sex[pid]], handedness="",
                    population="healthy", setting="lab",
                    device="two bipolar surface EMG channels (abductor digiti minimi of the right hand, left tibialis anterior), cursor on a monitor",
                    session=sess, block=f"block {int(r.block_number)}", trial=trial_idx,
                    phase="practice" if kind == "practice" else ("learning" if kind == "training" else "test"),
                    time_s=None, task="tracking", unit="trial",
                    condition=f"{r.direction} {r.scale:g}, preview {r.preview_time:g} s",
                    pert_type="none", pert_value=0, pert_unit="", feedback="normal" if fb else "none",
                    excluded_by_authors=bool(r.wrong_order_peak) or np.isnan(hi) or np.isnan(si),
                    path_direction=r.direction, path_scale=r.scale, preview_time_s=r.preview_time,
                    post_trial_feedback_visible=fb, block_kind=kind, trial_in_block=int(r.trial_number),
                    trial_in_condition=int(r.trial_number_per_condition), trial_is_valid=bool(r.is_valid),
                    wrong_order_peak=bool(r.wrong_order_peak), target_hits_peak_aligned=h_al, target_hits=h_un,
                    rms_error_peak_aligned_px=rms, trajectory_peak_time_s=None if np.isnan(ptime) else ptime,
                    cursor_peak_amplitude_px=float(cur[im] - SCREEN_WIDTH / 2),
                    hand_peak_time_s=t(hi), shin_peak_time_s=t(si),
                    hand_peak_amplitude=r.hand_peak_amplitude, shin_peak_amplitude=r.shin_peak_amplitude,
                    inter_peak_interval_s=None if not ok else (hi - si) / 1000,
                    hand_start_time_s=t(r.hand_start_index_by_jerk), hand_end_time_s=t(r.hand_end_index_by_grad),
                    shin_start_time_s=t(r.shin_start_index_by_jerk), shin_end_time_s=t(r.shin_end_index_by_grad),
                    pos_start_px=r.pos_start, pos_peak_px=r.pos_peak, pos_outgoing_px=r.pos_outgoing, pos_return_px=r.pos_return,
                    pos_tail_px=r.pos_tail, pos_end_px=r.pos_end, error_start_px=r.error_start, error_peak_px=r.error_peak,
                    error_outgoing_px=r.error_outgoing, error_return_px=r.error_return, error_tail_px=r.error_tail,
                    error_end_px=r.error_end, _scale=signed, _cur_nan=bool(nanc)))
                rows_s.append((trial_idx, int(r._i)))
            # samples: 100 Hz (every tenth sample) of the 2 s window
            fname = f"samples/{ID}_S{pi + 1:04d}_session{sess}.csv"
            p = TMP / Path(fname).name
            with open(p, "w", newline="") as fh:
                fh.write("trial,time_s,cursor_x_px,controller_output,hand_power,shin_power\r\n")
                tvec = 0.5 + np.arange(0, 2000, 10) / 1000
                for trial_idx, i in rows_s:
                    arr = np.column_stack([np.full(200, trial_idx), tvec, CUR[i, ::10], CTL[i, ::10], HP[i, ::10], SP[i, ::10]])
                    fh.write("\r\n".join(",".join("" if np.isnan(v) else (("%d" % v) if k == 0 else ("%.3f" % v if k == 1 else "%.4g" % v))
                                                  for k, v in enumerate(row)) for row in arr) + "\r\n")
            rows[-1]["_sample_file"] = (fname, str(p))
        print("done participant", pi + 1, flush=True)
    pickle.dump(rows, open(CACHE, "wb"))

sample_files = [r.pop("_sample_file") for r in rows if "_sample_file" in r]
for r in rows:
    r.pop("_scale", None)
    r.pop("_cur_nan", None)
df = pd.DataFrame(rows)

# ------------------------------------------------------------------ checks
S = "Methods, Participants"
subj = df.drop_duplicates("subject")
checks.published("participants", len(subj), 20, 0, S)
checks.published("men", (subj.sex == "M").sum(), 12, 0, S)
for gname in ("congruent", "incongruent"):
    checks.published(f"participants in the {gname} group", (subj.group == gname).sum(), 10, 0, S)
    checks.published(f"men in the {gname} group", ((subj.group == gname) & (subj.sex == "M")).sum(), 6, 0, S)
checks.published("sessions", df.session.nunique(), 5, 0, "Methods, Task")
checks.published("practice trials in session 1 (one of each of the 12 conditions)", (df[df.subject == "P01"].block_kind == "practice").sum(), 12, 0, "Methods, Task")
test = df[df.block_kind == "test"]
cnt = test.groupby(["subject", "session", "block", "path_direction", "path_scale", "preview_time_s"]).size()
checks.published("test trials per condition per block (five per each of 12 conditions)", cnt.median(), 5, 0, "Methods, Task")
checks.published("conditions in a test block (2 directions x 3 magnitudes x 2 preview times)",
                 test.groupby(["subject", "session", "block"]).apply(lambda x: x.groupby(["path_direction", "path_scale", "preview_time_s"]).ngroups).median(), 12, 0, "Methods, Task")
tr = df[df.block_kind == "training"]
checks.internal("training blocks use only the two trained paths (right, magnitude 1 and 3) with 0.8 s preview",
                set(zip(tr.path_direction, tr.path_scale)) == {("right", 0.2), ("right", 0.6)} and set(tr.preview_time_s) == {0.8},
                str(sorted(set(zip(tr.path_direction, tr.path_scale)))))
checks.internal("72240 rows = 20 x 3612", len(df) == 72240, str(len(df)))
checks.internal("hits between 0 and 25", df.target_hits_peak_aligned.dropna().between(0, 25).all())
checks.internal("channel peak times inside the record", df.hand_peak_time_s.dropna().between(0.5, 2.5).all())
# paper: R1 = right, magnitude 1 (0.2), no-feedback trials, hit percentage of the peak-aligned path
ok = df[(df.block_kind == "test") & (~df.wrong_order_peak) & df.hand_peak_time_s.notna() & df.shin_peak_time_s.notna()]
for gname, rep_mean, rep_sd in (("congruent", 6.8, 7.6), ("incongruent", 4.4, 5.0)):
    x = ok[(ok.group == gname) & (ok.path_direction == "right") & (ok.path_scale == 0.2)]
    med = x.assign(h=x.target_hits_peak_aligned / 25 * 100).groupby(["subject", "session"]).h.median().unstack()
    inc = med[5] - med[1]
    print("EXPLORE R1", gname, "mean increase", inc.mean(), "sd", inc.std(), "median", inc.median(), "(paper", rep_mean, rep_sd, ")", flush=True)
    print("   per participant", inc.round(2).tolist(), flush=True)

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json", encoding="utf-8")) if e["id"] == ID)["papers"][0]
meta = dict(
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Morehead (University of Leeds)", data_url="https://osf.io/79s38/", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="the 2.3 GB HDF5 of 2000-sample cursor, controller and EMG-power records was reduced to one row of per-trial measures "
            "(the authors' stored features and the four measures that their notebook computes, re-implemented exactly) plus 100 Hz "
            "time courses in samples/; participant codes replaced; channel indices converted to seconds; the Bayesian model outputs and analysis notebook are not included.",
    unit_note="trial (72240 trials: 20 participants x 3612 trials)",
    how=["One row is one trial of one participant. Twenty participants (ten with the congruent and ten with the incongruent mapping "
         "of muscles to cursor direction) controlled the sideways velocity of a cursor with the power of two surface-EMG channels "
         "(a hand muscle and a shin muscle) to touch as many as possible of 25 target circles on a path that descended the screen, "
         "over five daily sessions. Training blocks (feedback after each trial) used two paths, test blocks all twelve conditions (no feedback).",
         "The deposit's raw time courses (cursor path, controller output and scaled EMG power of the two channels, 2000 samples at 1 kHz per "
         "trial, 2.3 GB) are given in samples/ at 100 Hz (every tenth sample, no extra filtering): one file per participant and session, "
         "trial = the table's trial column, time_s = 0.5 + sample/1000 (the authors' window starts 0.5 s into the trial). cursor_x_px is "
         "the cursor position (screen centre 540 px, positive right), controller_output the stored controller signal, hand_power and "
         "shin_power the scaled EMG power (1 = 35% of maximal power).",
         "Trials are ordered by session, block and the authors' trial number (stable order for the repeats of abandoned trials). "
         "group is congruent or incongruent mapping (participant file condition B = congruent, A = incongruent, as in the authors' code). "
         "excluded_by_authors marks trials whose channel peaks are missing or in the wrong order, which the authors' notebook drops "
         "from the test-trial analyses (those analyses also use only test trials, post_trial_feedback_visible False).",
         "The 20 participants of the deposit are all kept; the paper's analyses use 18 (one man from each group was excluded for extreme "
         "cursor amplitude or late mean peak time), and the deposit does not say which."],
    differences=["The paper analyses 18 of the 20 participants; the deposit's participant list has no excluded participants and "
                 "its notebook's exclusion list is empty, so all 20 are released.",
                 "The paper's Bayesian regression outputs and session-to-session changes in hit percentage were not reproduced from the rows."])
stdlib.release(ID, "emg_path_following", df, meta, checks, extra_files=sample_files)
