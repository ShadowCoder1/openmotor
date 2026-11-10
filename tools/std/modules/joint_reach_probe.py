"""Task module: two people reach together on linked robots and the centre cursor between their hands is what each tries to put
into their own target, with some trials in which a visual jump is applied while the hands are held in a force channel
(Sullivan2026_eLife, defined 2026-11-10). One row is one participant's reach on one trial: each trial file of a pair gives two
rows, one per participant."""

DESCRIPTION = ("Joint (dyadic) reaching with a shared centre cursor. One row is one participant's reach on one trial of a pair; "
               "the participant's own and partner's target relevance, the trial type (regular, perturbation, probe), the "
               "outcome of the reach and, on probe trials, the lateral force against the force channel in three epochs after "
               "the visual jump.")

# (column, type, values or units, description)
COLUMNS = [
    ("pair", "text", "", "Code of the pair (the two participants who reached together); both partners' rows share it."),
    ("role", "category", "p1, p2", "Which robot of the pair the participant used: p1 (the authors' Right hand channels) or p2 (Left)."),
    ("trial_type", "category", "regular, perturbation, probe, null_probe, washout_regular",
     "regular (no visual change), perturbation (the centre cursor or both targets jumped and stayed, free reach), probe (a brief "
     "jump while both hands were in a force channel), null_probe (force channel, no jump)."),
    ("jump_direction", "category", "left, right, none",
     "Direction of the visual jump in this participant's own display, taken from the sign of the change of this participant's own "
     "centre cursor x (the partner's display is mirrored). none for regular and null_probe trials."),
    ("jump_onset_ms", "number", "ms", "Time of the jump command from the trial start signal (the go tone); empty without a jump."),
    ("self_target", "category", "relevant, irrelevant, washout", "Width of this participant's own target: relevant (1.05 cm), irrelevant (20 cm)."),
    ("partner_target", "category", "relevant, irrelevant, washout", "Width of the partner's target."),
    ("block_number", "integer", "", "The authors' block number."),
    ("block_trial", "integer", "", "Trial number within the block."),
    ("trial_protocol", "integer", "", "The authors' trial protocol (TP) number of the trial."),
    ("movement_onset_ms", "number", "ms", "First time after the start signal at which the participant's smoothed hand speed exceeded 5 cm/s."),
    ("reach_plus_hold_ms", "number", "ms",
     "Time from the start signal at which this participant's centre cursor had stayed in their own target for 500 ms (the "
     "authors' definition); empty if it never did."),
    ("reach_time_ms", "number", "ms", "reach_plus_hold_ms minus the 500 ms hold."),
    ("timing_result", "category", "good, early, late, none",
     "Timing message by the authors' rule: good if reach_plus_hold_ms is between 1400 and 1600 ms, early below, late above, "
     "none if the target was never held."),
    ("start_x_m", "number", "m", "Lateral position of the participant's hand at the start signal, in the participant's own robot frame."),
    ("final_x_m", "number", "m", "Lateral position of the participant's hand when the target had been held (end of the reach) in the own robot frame."),
    ("final_y_m", "number", "m", "Forward position of the hand at the same moment."),
    ("force_180_230_n", "number", "N",
     "Probe and null_probe trials: mean lateral force (robot frame) applied to the force channel 180 to 230 ms after the jump was "
     "displayed (jump command + 42 ms screen delay), after the authors' 14 Hz low-pass filter; the involuntary epoch."),
    ("force_230_300_n", "number", "N", "Same for 230 to 300 ms (semi-involuntary epoch)."),
    ("force_300_400_n", "number", "N", "Same for 300 to 400 ms (voluntary epoch)."),
    ("samples_file", "text", "path inside the ZIP",
     "CSV of the participant's trial from the start signal to trial completion, one row per 10 ms (every 10th raw 1000 Hz sample, "
     "not filtered): t_ms, hand_x_m, hand_y_m, force_x_n, center_cursor_x_m, center_cursor_y_m. Empty if not included."),
]
