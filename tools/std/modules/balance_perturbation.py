"""Task module: responses to a brief balance perturbation during walking (created for Reimann2018_FrontPhysiol, 2026-11-08).
One row is the response of one participant to one perturbation, as the authors summarised it for their statistical model:
the first step after the perturbation, each measure minus the mean of the two control steps before it. Values are in the
authors' own deposit units (metres, radians, integrated EMG as a fraction of the participant's control activation, times
seconds), signed so that positive means toward the side of the triggering foot."""

DESCRIPTION = ("Walking with brief balance perturbations (for example a visual scene rotation that evokes the feeling of a "
               "sideways fall). One row is one perturbation of one participant, with the foot placement, centre-of-pressure, "
               "joint-angle and muscle responses the authors computed for the first step after it, relative to control steps.")

# (column, type, values or units, description)
COLUMNS = [
    ("walking_trial", "integer", "", "The authors' number of the walking trial (recording) of this participant that holds the perturbation."),
    ("stretch_start_s", "number", "seconds", "Start time of the analysed step, from the start of its walking trial."),
    ("stretch_end_s", "number", "seconds", "End time of the analysed step, from the start of its walking trial."),
    ("stance_foot", "category", "left, right", "Foot on the ground during the analysed step."),
    ("trigger_foot", "category", "left, right", "Foot whose heel strike triggered the perturbation."),
    ("perturbation_side", "category", "left, right", "Side named in the authors' perturbation label (ILLUSION_LEFT, ILLUSION_RIGHT): the side toward which the visual scene was rolled to evoke a fall."),
    ("step_after", "integer", "", "Which step after the trigger the row describes (1 = the first step after the trigger)."),
    ("direction", "category", "toward, away", "Whether the perceived fall was toward the triggering foot's side (toward) or away from it (away)."),
    ("step_placement_x_m", "number", "m", "Change in mediolateral foot placement of the swing foot relative to the stance foot at heel strike, minus the mean of the two control steps; positive toward the triggering foot's side (deposit column step_placement_x_inverted). The paper reports it in mm."),
    ("stimulus_response_x_m", "number", "m", "Stimulus-induced foot placement change: the foot placement change minus the foot placement predicted from the centre of mass state by a linear model fitted to each participant's control steps (deposit column stimulus_response_x_inverted). The paper reports it in mm."),
    ("cop_from_com_x_integrated_m_s", "number", "m*s", "Mediolateral centre of pressure relative to the centre of mass, minus the control mean, integrated over the step (deposit column cop_from_com_x_integrated_inverted; the authors multiply by 1000 to report mm*s)."),
    ("trigger_leg_ankle_eversion_rad", "number", "rad", "Ankle eversion of the stance (triggering) leg at the end of the step, minus the control mean (deposit column trigger_leg_ankle_eversion_step_end_inverted; the authors convert to degrees)."),
    ("contra_leg_hip_abduction_rad", "number", "rad", "Hip abduction of the swing (contralateral) leg at the end of the step, minus the control mean (deposit column contra_leg_hip_abduction_step_end_inverted; the authors convert to degrees)."),
    ("trigger_leg_peroneus_longus_emg", "number", "fraction of control activation * s", "Peroneus longus EMG of the stance (triggering) leg, minus the control mean, integrated over the step, where the participant's average control activation is 1 (deposit column trigger_leg_pero_lng_integrated_inverted; the authors multiply by 100 to report % s)."),
    ("contra_leg_gluteus_medius_emg", "number", "fraction of control activation * s", "Gluteus medius EMG of the swing (contralateral) leg, minus the control mean, integrated over the step, where the participant's average control activation is 1 (deposit column contra_leg_glut_med_integrated_inverted; the authors multiply by 100 to report % s)."),
]
