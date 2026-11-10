"""Task module: a first movement on every trial followed, on a minority of trials, by a cued second movement with either hand
(button press with the thumb after a wrist flexion; center-out joystick movement after a trigger press), with reaction
times, movement onset and end, peak velocity and EMG timings. First defined with Muralidharan2021_JCognNeurosci."""

DESCRIPTION = ("Two-movement trials: every trial starts with a first (primary) movement, and on a minority of trials a cue "
               "asks for a second (secondary) movement with the right or the left hand; control blocks have the secondary "
               "movement on every trial. One row is one trial, with the secondary movement's reaction time, onset, "
               "end and peak velocity and, where recorded, EMG event times.")

# (column, type, values or units, description)
COLUMNS = [
    ("block_number", "integer", "", "The authors' block number (block_no)."),
    ("trial_kind", "category", "primary only, primary and secondary, secondary only",
     "Whether the trial held only the first movement, the first movement followed by a cued second movement, or (control "
     "task) only the cued second movement."),
    ("secondary_hand", "category", "right, left, none",
     "Hand that made the cued secondary movement (the authors' movement code 1 = right, 2 = left, 0 = none)."),
    ("secondary_outcome_code", "integer", "",
     "The authors' correct code for the secondary movement: 1 performed with the correct hand, 0 on trials with a cued "
     "secondary movement that was not, empty on trials without one. One participant also has 3 on three trials; the "
     "deposit does not say what 3 means."),
    ("secondary_rt_ms", "number", "ms",
     "Reaction time of the secondary movement: from the cue (arrow, or enlarged joystick) to the thumb press in "
     "Experiment 1, or to the joystick's maximum deflection in Experiment 2 (the authors' RT). Empty where the authors' "
     "outlier rule removed it (Experiment 2) or no secondary movement was cued."),
    ("secondary_onset_ms", "number", "ms", "Experiment 2: time from the cue to the start of the center-out movement, as estimated by the authors from the joystick trajectory (outliers removed). Empty otherwise."),
    ("secondary_peak_velocity", "number", "joystick units per second (a.u./s)", "Experiment 2: peak velocity of the center-out movement (outliers removed)."),
    ("primary_done", "true/false", "",
     "The first movement was detected: wrist flexion found in the EMG (Experiment 1, the authors' ifflexed) or trigger "
     "press registered (Experiment 2, ifpressed). False where the authors' flag is missing."),
    ("primary_rt_ms", "number", "ms", "Experiment 2: time from the 'Press' cue to the trigger press of the first movement (the authors' RTPress)."),
    ("emg_secondary_on_ms", "number", "ms", "Experiment 1: EMG onset in the thumb that made the secondary press (emgON_Move). The deposit does not state the time origin; values lie below the button-press RT, so the arrow is taken as origin."),
    ("emg_secondary_peak_ms", "number", "ms", "Experiment 1: time of the peak EMG of the secondary press (emgPK_Move), same origin."),
    ("emg_secondary_dip_ms", "number", "ms", "Experiment 1: time of the EMG 'dip' (decline) of the secondary press (emgDP_Move), same origin."),
    ("emg_secondary_off_ms", "number", "ms", "Experiment 1: EMG offset of the secondary press (emgOF_Move), same origin."),
    ("emg_primary_on_ms", "number", "ms", "Experiment 1: EMG onset of the first movement, the right wrist flexion (emgON_Flex). The deposit does not state the origin; the cue square is taken as origin."),
    ("emg_primary_peak_ms", "number", "ms", "Experiment 1: time of the peak EMG of the wrist flexion (emgPK_Flex)."),
    ("emg_primary_dip_ms", "number", "ms", "Experiment 1: time of the EMG dip of the wrist flexion (emgDP_Flex)."),
    ("emg_primary_off_ms", "number", "ms", "Experiment 1: EMG offset of the wrist flexion (emgOF_Flex)."),
]
