"""Orschiedt & Franklin (2023) PLoS Comput Biol: bimanual force-field learning, Evident and Ambiguous groups (8 participants each).

Raw files: raw_extra/Orschiedt2023_PLoSCompBiol/*.csv (Figshare 22895378), figure-level data: one row per participant. Used here:
Fig_3A (MPE per 12-trial bin), Fig_3B and Fig_3D (phase bar charts), Fig_3C (force compensation per bin, bimanual and unimanual),
Fig_2_Peak_Velocity_Data and Fig_8A_Omega. Not used: velocity-difference profiles (Fig 2), force profiles (Fig 4, 9) and model
predictions and fit statistics (Figs 5-7, 8B, 9 predicted), which are curves and model outputs. Nothing is executed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Orschiedt2023_PLoSCompBiol"
R = stdlib.raw_dir(DS)
GROUPS = ["Evident", "Ambiguous"]


def rd(name):
    return pd.read_csv(R / name, index_col=0)


def phase_of_bin(b):  # 12-trial bins: 288 baseline + 960 exposure + 960 generalization = 2208 trials = 184 bins
    return ("baseline", "none") if b <= 24 else (("learning", "force_field") if b <= 104 else ("transfer", "force_field"))


ck = stdlib.Checks()
rows = []


def base(group, p, phase="other", pert="none", cond=None, block=None, trial=None):
    return dict(dataset=DS, study="Orschiedt 2023 PLoS Comput Biol", experiment=f"{group.lower()} group", group=group,
                subject=f"{group[0]}{p}", person=f"{group[0]}{p}", subject_excluded_by_authors=False, age=None, sex=None,
                handedness="right", population="healthy", setting="lab", device="vBOT planar robotic manipulandum, bimanual",
                session=1, block=block, trial=trial, phase=phase, time_s=None, task="reaching", unit="participant summary",
                condition=cond, pert_type=pert, pert_value=None, pert_unit=None, feedback="normal", excluded_by_authors=False)


def add(b, measure, desc, stat, value, unit, n_trials=None):
    if pd.isna(value):
        return
    rows.append({**b, "measure": measure, "measure_description": desc, "statistic": stat, "value": value, "value_unit": unit, "session_context": None,
                 "n_trials": n_trials})


# MPE per bin: the Ambiguous file in the deposit is byte-identical to the Evident one, so only Evident is used
a, b_ = rd("Fig_3A_MaximumPerpendicularError_Evident.csv"), rd("Fig_3A_MaximumPerpendicularError_Ambiguous.csv")
ck.internal("Fig 3A Ambiguous file equals the Evident file (deposit duplicate), so it is left out", a.equals(b_))
ck.internal("184 bins x 8 participants", a.shape == (8, 184))
for p, r in a.iterrows():
    for j, v in enumerate(r.values, 1):
        ph, pt = phase_of_bin(j)
        add(base("Evident", p, ph, pt, "bimanual training conditions", f"bin {j}", j), "mpe_bin",
            "Maximum perpendicular error, mean of a 12-trial bin of bimanual non-channel trials", "mean", v, "cm", 12)
for g in GROUPS:
    for kind, cond in (("", "bimanual training and generalization conditions"), ("_Uni", "unimanual left-hand conditions")):
        d = rd(f"Fig_3C_ForceCompensation{kind}_{g}.csv")
        ck.internal(f"Fig 3C {kind or 'bimanual'} {g}: 8 participants x 184 bins", d.shape == (8, 184))
        for p, r in d.iterrows():
            for j, v in enumerate(r.values, 1):
                ph, pt = phase_of_bin(j)
                add(base(g, p, ph, pt, cond, f"bin {j}", j), "force_compensation_bin",
                    "Force compensation on error-clamp trials: regression coefficient between measured lateral force and the force "
                    "that would compensate the field, mean of a 12-trial bin", "mean", v, "fraction (1 = 100 percent)", None)
# phase-level bar-chart values per participant
PH = {"Late beaseline": ("baseline", "none"), "Late exposure": ("learning", "force_field"),
      "Late generalization": ("transfer", "force_field"), "Early exposure": ("learning", "force_field")}
bars = {}
for g in GROUPS:
    for fn, meas, desc, unit in (("3D_FC_BarChart", "force_compensation_phase", "Force compensation in the named phase (Fig 3D)", "percent"),
                                 ("3B_MPE_BarChart", "mpe_phase", "Maximum perpendicular error in the named phase (Fig 3B)", "cm")):
        d = rd(f"Fig_{fn}_{g}.csv")
        bars[(g, fn)] = d
        for p, r in d.iterrows():
            for col, v in r.items():
                lab = col.replace("beaseline", "baseline")
                ph, pt = PH[col]
                add(base(g, p, ph, pt, "bimanual training conditions", lab), meas, f"{desc}: {lab.lower()}", "mean", v, unit)
pv = rd("Fig_2_Peak_Velocity_Data.csv")
for p, r in pv.iterrows():
    for col, v in r.items():
        _, cond, hand, exp = col.split("_")
        g = "Ambiguous" if exp == "A" else "Evident"
        add(base(g, p, "learning", "force_field", f"b{cond} ({hand} hand)", "exposure"), "peak_velocity",
            f"Peak hand velocity of the {hand} hand in bimanual condition b{cond} during exposure (Fig 2G)", "mean", v, "cm/s")
om = pd.read_csv(R / "Fig_8A_Omega.csv")
for g, lab in (("Ambiguous", "A"), ("Evident", "B")):
    sub = om[om.Experiment == lab].reset_index(drop=True)
    for p, r in sub.iterrows():
        add(base(g, p, "transfer", "force_field", "bimanual training conditions", "generalization"), "encoding_weight_omega",
            "Weight omega of the weighted force-encoding model fitted to the participant's generalization data (1 = right-hand velocity only, "
            "0 = left-hand velocity only), the authors' model output", "value", r.Popt_shift_w, "weight")

df = pd.DataFrame(rows)
ck.internal("16 participants", df.person.nunique() == 16, str(df.person.nunique()))


def se(x):
    return x.std() / len(x.dropna()) ** 0.5


E_, A_ = bars[("Evident", "3D_FC_BarChart")], bars[("Ambiguous", "3D_FC_BarChart")]
# the paper's +- values are standard errors (checked: SD / sqrt(n) of the same rows)
ck.published("participants per group, Evident", df[df.group == "Evident"].person.nunique(), 8, 0, "Methods, Participants (16 in two groups of 8)")
ck.published("participants per group, Ambiguous", df[df.group == "Ambiguous"].person.nunique(), 8, 0, "Methods, Participants (16 in two groups of 8)")
ck.published("late baseline force compensation, Evident (percent)", E_["Late beaseline"].mean(), 7.62, 0.005, "Results, force compensation")
ck.published("late baseline force compensation, Ambiguous (percent)", A_["Late beaseline"].mean(), 6.88, 0.005, "Results, force compensation")
ck.published("late exposure force compensation, Evident (percent)", E_["Late exposure"].mean(), 79.52, 0.005, "Results, force compensation")
ck.published("SE of late baseline FC, Evident", se(E_["Late beaseline"]), 2.27, 0.005, "Results, force compensation (the paper's +- is a standard error)")
ck.published("SE of late baseline FC, Ambiguous", se(A_["Late beaseline"]), 2.77, 0.005, "Results, force compensation (the paper's +- is a standard error)")
ck.published("SE of late exposure FC, Evident", se(E_["Late exposure"]), 3.83, 0.005, "Results, force compensation (the paper's +- is a standard error)")
ck.published("SE of late exposure FC, Ambiguous", se(A_["Late exposure"]), 2.85, 0.005, "Results, force compensation (the paper's +- is a standard error)")
w = om.groupby("Experiment").Popt_shift_w
ck.published("omega, Evident group mean", w.mean()["B"], 0.935, 0.0005, "Results, encoding parameters")
ck.published("omega, Ambiguous group mean", w.mean()["A"], 0.374, 0.0005, "Results, encoding parameters")
ck.published("SE of omega, Evident", w.std()["B"] / 8 ** 0.5, 0.021, 0.0005, "Results, encoding parameters (standard error)")
ck.published("SE of omega, Ambiguous", w.std()["A"] / 8 ** 0.5, 0.045, 0.0005, "Results, encoding parameters (standard error)")
mpe_drop = bars[("Evident", "3B_MPE_BarChart")]["Early exposure"].mean() - bars[("Evident", "3B_MPE_BarChart")]["Late exposure"].mean()
ambi = bars[("Ambiguous", "3B_MPE_BarChart")]
both = (bars[("Evident", "3B_MPE_BarChart")]["Early exposure"].mean() + ambi["Early exposure"].mean() - bars[("Evident", "3B_MPE_BarChart")]["Late exposure"].mean() - ambi["Late exposure"].mean()) / 2
ck.published("MPE decrease from early to late exposure, both groups (cm)", both, 2.18, 0.005, "Results, kinematic error")

meta = dict(
    paper=dict(authors="Orschiedt J, Franklin DW", year=2023, title="Learning context shapes bimanual control strategy and generalization of novel dynamics",
               venue="PLoS Comput Biol 19(12):e1011189", doi="10.1371/journal.pcbi.1011189"),
    lab="Franklin (TU Munich)", data_url="https://doi.org/10.6084/m9.figshare.22895378",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("The deposit has figure-level CSVs with one row per participant; they were turned into one long table (a row per participant, "
             "measure and bin or phase), empty cells were not written, and participant rows were coded by group and row number. Velocity-difference "
             "profiles (Fig 2), force profiles (Figs 4 and 9) and model predictions and fit statistics (Figs 5-8B) are not included."),
    how=["One row is one value of one participant: the maximum perpendicular error (MPE, cm) or force compensation (fraction, 1 = 100 percent) "
         "of one 12-trial bin (the 2208-trial experiment is 184 bins: 24 baseline, 80 exposure and 80 generalization; trial is the bin number), "
         "a phase value from the authors' bar charts (late baseline, early or late exposure, late generalization; force compensation in "
         "percent), a peak hand velocity per condition and hand (cm/s), or the weight omega of the authors' encoding model (a model output). "
         "The force field acted on the left hand and depended on right-hand velocity; the Evident group trained on conditions with clearly "
         "different hand speeds, the Ambiguous group on matched speeds.",
         "Units for the bins and the peak velocities are inferred from the paper (MPE in cm, peak speeds 45-75 cm/s, force compensation a "
         "fraction in the series and a percent in the bar charts); the deposit has no readme. The deposit does not say which participants "
         "are the same across files: row i of each group's files is taken as the same participant, and the model-weight file's participants "
         "(Experiments A = Ambiguous, B = Evident, labels inferred from the omega values) are matched to rows in order of appearance. "
         "pert_value is empty (the field strength is given inconsistently in the paper).",
         "Participants have no age or sex in the deposit (the paper reports 25.9 +- 3.0 years overall, 7 women)."],
    unit_note="value (bin, phase or condition) of one participant",
    differences=["The Ambiguous group's MPE file (Fig_3A) is byte-identical to the Evident group's file, so it is a duplicate by mistake in "
                 "the deposit; only the Evident one is released and the Ambiguous group's MPE per bin is missing (its early, late and "
                 "generalization values are in the phase bar chart).",
                 "The paper's +- values for force compensation and omega are standard errors (SD / sqrt(n) of the shared rows), not SDs.",
                 "Two reported numbers do not reproduce and are not used as checks: late-exposure force compensation of the Ambiguous group (paper 86.88 percent; the shared rows give 86.87) and the change in MPE from late exposure to late generalization (paper 0.042 cm; the bar-chart rows give 0.085 cm averaged over both groups).",
                 "Late baseline force compensation exists for only 4 Evident and 7 Ambiguous participants in the deposit."],
)
stdlib.release(DS, "summary", df, meta, ck)
