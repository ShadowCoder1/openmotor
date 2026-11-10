"""Jin, Isa, Burden & Ingraham (2026) Sci Rep: adaptation to a novel split-belt ratio after guided exploration of step length
asymmetries with visual biofeedback. 15 adults, 11 six-minute treadmill trials each.

Raw file (Zenodo 16970717, Processed_data.zip): per participant folder, 'SLA mats processed from QTM' holds one MAT-file per trial
with the authors' per-stride step lengths (mm), step times (s), SLA, heel-strike and toe-off times. Processed_data/Subject_data.m
(read as text, never run) holds body mass, leg length, the trial order and the target SLA of each guided trial. Force-plate
samples, biofeedback scores (layout undocumented), real-time recordings and COSMED metabolic files in the same ZIP are not
converted.

Run from the site folder: python3 -I tools/std/convert/Jin2026_SciRep.py
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Jin2026_SciRep"
MODULE = "walking"
DEVICE = ("Bertec instrumented split-belt treadmill (left belt fast); Qualisys motion capture of both lateral malleoli "
          "(100 Hz); iPad showing target step lengths in the feedback trials; COSMED K5 metabolic system")


def mat_numbers(s, name):
    """The numbers of a MATLAB assignment 'name = [...];' in the deposit's Subject_data.m, read as text."""
    m = re.search(name + r"\s*=\s*\[(.*?)\];\s*(?=[ \t]*(?:%|\r?\n|$))", s, re.S)
    body = m.group(1).replace("...", " ").replace("[", " ").replace("]", " ")
    rows = [r for r in body.split(";") if r.strip()]
    return [[float(x.replace("+", "")) if x.strip().lower() != "nan" else np.nan
             for x in re.split(r"[,\s]+", r.strip()) if x] for r in rows]


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(DS) / "Processed_data.zip")
    sd = z.read("Processed_data/Subject_data.m").decode("utf8", "replace")
    mass = mat_numbers(sd, "subject_mass_all")[0]
    leg = mat_numbers(sd, "leg_length_all")[0]
    sla_base = mat_numbers(sd, "SLA_baseline_all")[0]
    targets = np.array(mat_numbers(sd, "cond_numeric_target_all"))
    cond_lists = [re.findall(r"'([^']*)'", c) for c in re.findall(r"\{('baseline tied w/o GUI'.*?)\}", sd, re.S)]
    assert len(mass) == len(leg) == len(sla_base) == targets.shape[0] == len(cond_lists) == 15, (len(cond_lists),)
    ck = stdlib.Checks()
    frames, people = [], []
    mism_slow = mism_sla = 0
    conflicts = []
    stance_bad = stance_n = 0
    for k in range(15):
        d = f"{k + 1:02d}"
        files = sorted([n for n in z.namelist() if n.startswith(f"Processed_data/{d}/SLA mats") and n.endswith(".mat")],
                       key=lambda n: (len(Path(n).name.split(" ")[0]), Path(n).name))
        assert len(files) == 11, (d, len(files))
        cl = cond_lists[k]
        assert len(cl) == 11
        code = f"{DS}_S{k + 1:03d}"
        people.append(dict(person=code, body_mass_kg=mass[k], leg_length_m=leg[k], baseline_sla=sla_base[k]))
        sub = []
        for j, n in enumerate(files):
            m = sio.loadmat(io.BytesIO(z.read(n)), squeeze_me=True)
            name = Path(n).name.lower()
            # trial kind from the authors' own order list; file names must agree with it
            lab = cl[j]
            if j == 0:
                assert "without gui" in name
            elif j == 1:
                assert "with gui" in name
            elif j <= 8:
                mt = re.search(r"(pos|neg)(\d\d)|zero", name)
                flab = mt.group(0)
                ftgt = 0.0 if flab == "zero" else (1 if mt.group(1) == "pos" else -1) * int(mt.group(2)) / 100
                if flab != lab or abs(ftgt - targets[k, j]) > 1e-9:
                    conflicts.append(f"participant {k + 1}, trial {j + 1}: file says {flab}, list says {lab}, "
                                     f"target array says {targets[k, j]:+.2f}")
                lab, targets[k, j] = flab, ftgt
            else:
                assert "free" in name
            sbr = 1.0 if j <= 1 else (2.0 if j == 10 else 3.0)
            fast, slow = {1.0: (1.0, 1.0), 2.0: (1.33, 0.67), 3.0: (1.5, 0.5)}[sbr]
            nst = len(m["SLA"])
            slf, sls = np.asarray(m["SL_fast"], float)[:nst] / 1000.0, np.asarray(m["SL_slow"], float)[:nst] / 1000.0
            assert len(m["SL_fast"]) >= nst and len(m["SL_slow"]) >= nst
            hsf, hss = np.asarray(m["t_HS_fast"], float), np.asarray(m["t_HS_slow"], float)
            tof, tos = np.asarray(m["t_toeoff_fast"], float), np.asarray(m["t_toeoff_slow"], float)
            stf, sts = np.asarray(m["ST_fast"], float), np.asarray(m["ST_slow"], float)
            sla = np.asarray(m["SLA"], float)
            mism_sla += int(np.nanmax(np.abs(sla - (slf - sls) / (slf + sls))) > 1e-6)
            # step times: fast step = slow heel strike -> next fast heel strike (ST_fast[i]); slow step = fast heel strike
            # -> slow heel strike, stored one position later (ST_slow[i-1]); checked against the heel-strike times
            stepf = np.full(nst, np.nan)
            stepf[:min(nst, len(stf))] = stf[:nst]
            steps = np.full(nst, np.nan)
            steps[1:min(nst, len(sts) + 1)] = sts[:nst - 1]
            ok = np.isfinite(steps[:len(hss)]) & (np.arange(nst)[:len(hss)] < len(hss))
            ii = np.arange(1, min(nst, len(hss), len(hsf)))
            mism_slow += int(np.nanmax(np.abs(steps[ii] - (hss[ii] - hsf[ii]))) > 0.0101)
            # stance: heel strike to the first later toe-off of the same leg
            def stance(hs, to, n):
                out = np.full(n, np.nan)
                for i in range(min(n, len(hs))):
                    a = to[to > hs[i]]
                    nxt = hs[i + 1] if i + 1 < len(hs) else np.inf
                    if a.size and a[0] < nxt:  # toe-off must fall before the leg's next heel strike, else left empty
                        out[i] = a[0] - hs[i]
                return out
            stf_, sts_ = stance(hsf, tof, nst), stance(hss, tos, nst)
            stance_bad += int(np.isnan(np.r_[stf_, sts_]).sum())
            stance_n += 2 * nst
            g = lambda a: np.pad(np.asarray(a, float)[:nst], (0, max(0, nst - len(a))), constant_values=np.nan)  # noqa: E731
            tgt = targets[k, j]
            kind = ("baseline tied belt, no feedback" if j == 0 else "baseline tied belt, feedback" if j == 1
                    else "free exploration, split-belt ratio 3" if j == 9 else "free exploration, split-belt ratio 2" if j == 10
                    else f"guided exploration, target SLA {tgt:+.2f}")
            sub.append(pd.DataFrame(dict(
                dataset=DS, study=DS, experiment="split-belt biofeedback exploration", group="", subject=code,
                subject_excluded_by_authors=False, age=np.nan, sex="", handedness="", population="healthy", setting="lab",
                device=DEVICE, session=1, block=lab, phase=["baseline", "practice", *["learning"] * 7, "test", "transfer"][j],
                time_s=g(hsf), task="walking", unit="stride", condition=kind, pert_type="split_belt" if sbr > 1 else "none",
                pert_value=0.0 if sbr == 1 else sbr, pert_unit="ratio" if sbr > 1 else "",
                feedback="altered" if (1 <= j <= 8) else "none", excluded_by_authors=False,
                fast_leg="left", belt_speed_fast_mps=fast, belt_speed_slow_mps=slow, walking_trial=j + 1,
                stride_start_s=g(hsf), step_length_fast_m=slf, step_length_slow_m=sls, step_time_fast_s=stepf,
                step_time_slow_s=steps, stance_time_fast_s=stf_, stance_time_slow_s=sts_, step_length_asymmetry=sla,
                target_step_length_asymmetry=tgt if 2 <= j <= 8 else np.nan,
            )))
        s = pd.concat(sub, ignore_index=True)
        s["trial"] = np.arange(1, len(s) + 1)
        frames.append(s)
    df = pd.concat(frames, ignore_index=True)
    df["time_s"] = df["stride_start_s"]

    # ---- internal checks
    ck.internal("SLA equals (fast - slow) / (fast + slow) of the shared step lengths in every file", mism_sla == 0, f"{mism_sla} files differ")
    ck.internal("file names, order list and target array agree except in one participant's last two guided trials",
                len(conflicts) == 2 and all(c.startswith("participant 7") for c in conflicts), "; ".join(conflicts))
    ck.internal("slow step times stored one position later match slow minus fast heel-strike times (within 0.01 s)", mism_slow == 0, f"{mism_slow} files differ")
    ck.internal("stance times found for at least 99% of strides (toe-off before the leg's next heel strike)", stance_bad / stance_n < 0.01, f"{stance_bad} of {stance_n} empty")
    ck.internal("15 participants x 11 trials", df.groupby("subject").walking_trial.nunique().eq(11).all() and df.subject.nunique() == 15)
    ck.internal("slow-belt stance is longer than fast-belt stance in split-belt trials",
                df[df.pert_value > 0].stance_time_slow_s.mean() > df[df.pert_value > 0].stance_time_fast_s.mean(),
                f"{df[df.pert_value > 0].stance_time_slow_s.mean():.3f} vs {df[df.pert_value > 0].stance_time_fast_s.mean():.3f}")

    # ---- published numbers
    P = pd.DataFrame(people)

    def ss(block_mask):
        out = []
        for sub, q in df[block_mask].groupby("subject", sort=False):
            out.append(q.step_length_asymmetry.to_numpy()[-100:].mean())
        return np.array(out)

    base, f1, f2 = ss(df.walking_trial == 1), ss(df.walking_trial == 10), ss(df.walking_trial == 11)
    ck.published("participants", df.subject.nunique(), 15, 0, "Abstract and Methods, Participants")
    ck.published("walking trials per participant", df.groupby("subject").walking_trial.nunique().mean(), 11, 0, "Fig. 1 caption")
    ck.published("guided exploration trials per participant", df[df.phase == "learning"].groupby("subject").walking_trial.nunique().mean(),
                 7, 0, "Methods, Study design")
    ck.published("mean steady-state SLA, free exploration 1 (SBR 3; last 100 strides)", f1.mean(), 0.010, 0.0005, "Results, free exploration")
    ck.published("SD across participants, same", f1.std(ddof=1), 0.059, 0.0005, "Results, free exploration")
    ck.published("mean steady-state SLA, free exploration 2 (SBR 2)", f2.mean(), 0.013, 0.0005, "Results, free exploration")
    ck.published("mean tied-belt baseline SLA (last 100 strides)", base.mean(), -0.006, 0.0005, "Results, free exploration")
    ck.published("SD across participants, same", base.std(ddof=1), 0.026, 0.0005, "Results, free exploration")
    ck.published("mean body mass (kg)", np.mean(mass), 76.1, 0.05, "Methods, Participants")
    ck.published("mean leg length (cm)", np.mean(leg) * 100, 83.3, 0.05, "Methods, Participants")

    meta = dict(
        study=DS,
        paper=dict(authors="Jin Z, Isa J, Burden SA, Ingraham KA", year=2026,
                   title="Quantifying human adaptation to a novel split-belt walking condition after broad experience at different belt speeds",
                   venue="Sci Rep 16(1):15627", doi="10.1038/s41598-026-46937-5"),
        lab="Ingraham (University of Washington)",
        data_url="https://doi.org/10.5281/zenodo.16970717",
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Read the authors' per-stride MATLAB files and kept their step lengths (mm to m), step times, step length "
                 "asymmetry and heel-strike times; added stance times (heel strike to the next toe-off of the same leg, from the "
                 "shared times), the belt speeds and the target SLA from the protocol and the deposit's Subject_data.m, and "
                 "participant codes. Force-plate samples, real-time biofeedback scores and metabolic files were not converted."),
        how=["One row is one stride of one participant (fast-belt heel strike to the next one), in the order walked: 11 six-minute "
             "trials per person (two tied-belt baselines at 1 m/s, seven guided explorations at split-belt ratio 3, then free "
             "explorations at ratio 3 and ratio 2). The left belt was the fast belt. Step lengths are the ankle-to-ankle distance at "
             "heel strike in metres (the deposit gives millimetres), step length asymmetry is (fast - slow) / (fast + slow) as the "
             "authors computed it, and times are in seconds from the start of the trial.",
             "In the deposit the fast step time at position i runs from the slow heel strike i to the fast heel strike i+1, and the "
             "slow step time is stored one position later than the stride it belongs to; the converter shifts it so that both step "
             "times sit on the stride that contains them (the first stride of a trial has no slow step time). Stance time is "
             "computed from the deposited heel-strike and toe-off times; for the slow leg the stance of a stride ends at the "
             "following toe-off. block keeps the authors' trial labels (target SLA in hundredths: pos10 is +0.10).",
             "Guided trials had visual biofeedback toward a target SLA (target_step_length_asymmetry), and so did the second tied "
             "baseline (target = the participant's own baseline SLA, not shared per stride); both are marked feedback = altered. "
             "The free explorations had no feedback. The second free exploration used belts at 1.33 and 0.67 m/s (ratio 2). "
             "Age, sex and handedness are not in the deposit; body mass and leg length are in participants.csv.",
             "The biofeedback scores in the deposit (step_scores) are not documented well enough to convert and were left out."],
        unit_note="stride (fast-belt heel strike to the next one)",
        differences=["For participant 7 the deposit's Subject_data.m disagrees with the file names: its order list gives neg15 as "
                     "the last guided trial where the file is named neg10 (so neg10 appears twice and neg15 never), and its target array "
                     "gives -0.10 for the trial named zero. The rows follow the file names, which label the data directly (" + "; ".join(conflicts) + ").",
                     "Three numbers of the paper do not reproduce and are not used as checks: the SD of steady-state SLA in the second "
                     f"free exploration ({f2.std(ddof=1):.4f} here, 0.052 in the paper), and the SDs of body mass and leg length "
                     f"({np.std(mass, ddof=1):.1f} kg and {np.std(leg, ddof=1) * 100:.2f} cm here, 14.5 kg and 5.31 cm in the paper; "
                     f"the paper's values match the population SD, {np.std(mass):.1f} and {np.std(leg) * 100:.2f})."],
    )
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in df.columns and c != "person":
            df[c] = np.nan
    df = df[[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"]]
    pcols = [("body_mass_kg", "number", "kg", "Body mass, as in the deposit's Subject_data."),
             ("leg_length_m", "number", "m", "Leg length (lateral malleolus to trochanter), as in the deposit's Subject_data."),
             ("baseline_sla", "number", "unitless", "Step length asymmetry in tied-belt walking recorded by the authors, used as the target of the second baseline.")]
    stdlib.release(DS, MODULE, df, meta, ck, participants=P, participant_columns=pcols)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.5g} vs {c['reported']}  ({c['source']})")
        elif c.get("detail"):
            print(f"  [internal] {c['what'][:60]}: {c['detail'][:300]}")


if __name__ == "__main__":
    main()
