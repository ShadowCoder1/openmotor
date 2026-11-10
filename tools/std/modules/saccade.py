"""OpenMotor task module: saccade adaptation (one row per trial, holding that trial's primary saccade)."""

DESCRIPTION = (
    "Saccade adaptation. One row is one trial: the primary saccade a participant made to a target that could be "
    "displaced during or after the saccade (an intrasaccadic target step, or a task-relevant stimulus shown after the "
    "saccade), plus the first secondary saccade where the authors give one. Positions and amplitudes are in degrees of "
    "visual angle relative to the fixation point, x horizontal (positive rightward unless the README states a recoded "
    "axis) and y vertical (positive upward). The step itself is also in pert_type, pert_value and pert_unit."
)

# (column, type, values or units, description)
COLUMNS = [
    ("target_amplitude_deg", "number", "deg",
     "Distance from the fixation point to the pre-saccadic target, that is the target step the saccade is made to."),
    ("target_direction_deg", "number", "deg; 0 = rightward, 90 = upward, 180 = leftward, 270 = downward",
     "Direction of the pre-saccadic target from the fixation point."),
    ("step_x_deg", "number", "deg",
     "Horizontal displacement, from the pre-saccadic target position, of the target shown after saccade onset (the "
     "stepped target, or the task-relevant stimulus). 0 when the target is not displaced; empty when no target is "
     "shown after the saccade."),
    ("step_y_deg", "number", "deg", "Vertical displacement of the same target, positive upward."),
    ("step_delay_ms", "number", "ms",
     "Time from online saccade detection to the appearance of the displaced target; 0 for a step made during the "
     "saccade."),
    ("distractor_x_deg", "number", "deg",
     "Horizontal displacement, from the pre-saccadic target position, of a second, task-irrelevant stimulus shown "
     "after saccade onset. Empty when none is shown."),
    ("distractor_y_deg", "number", "deg", "Vertical displacement of that task-irrelevant stimulus."),
    ("distractor_delay_ms", "number", "ms",
     "Time from online saccade detection to the appearance of that task-irrelevant stimulus."),
    ("saccade_start_x_deg", "number", "deg", "Horizontal eye position at primary saccade onset."),
    ("saccade_start_y_deg", "number", "deg", "Vertical eye position at primary saccade onset."),
    ("saccade_end_x_deg", "number", "deg", "Horizontal eye position at primary saccade offset (landing position)."),
    ("saccade_end_y_deg", "number", "deg", "Vertical eye position at primary saccade offset (landing position)."),
    ("saccade_amplitude_deg", "number", "deg", "Amplitude of the primary saccade, start to end."),
    ("saccade_amplitude_x_deg", "number", "deg", "Horizontal component of the primary saccade amplitude (end minus start)."),
    ("saccade_amplitude_y_deg", "number", "deg", "Vertical component of the primary saccade amplitude (end minus start)."),
    ("saccade_gain", "number", "ratio", "Primary saccade amplitude divided by target amplitude, where the authors give it."),
    ("saccade_latency_ms", "number", "ms", "Time from target onset to primary saccade onset."),
    ("saccade_duration_ms", "number", "ms", "Duration of the primary saccade."),
    ("saccade_peak_velocity_deg_s", "number", "deg/s", "Peak velocity of the primary saccade."),
    ("postsaccadic_error_deg", "number", "deg",
     "Post-saccadic error as the authors computed it: primary saccade landing position minus the position of the "
     "post-saccadic target, along the axis the README states."),
    ("blink", "true/false", "", "The authors flagged a blink in place of, or during, the primary saccade."),
    ("secondary_saccade_end_x_deg", "number", "deg", "Horizontal landing position of the first saccade after the primary saccade."),
    ("secondary_saccade_end_y_deg", "number", "deg", "Vertical landing position of the first saccade after the primary saccade."),
    ("secondary_saccade_amplitude_x_deg", "number", "deg", "Horizontal amplitude of that secondary saccade."),
    ("secondary_saccade_amplitude_y_deg", "number", "deg", "Vertical amplitude of that secondary saccade."),
    ("secondary_saccade_latency_ms", "number", "ms", "Time from primary saccade offset to secondary saccade onset."),
    ("eye", "category", "left, right, both", "Eye whose movements were recorded."),
    ("response", "text", "",
     "Participant's perceptual report on the trial (for example a discrimination or localization judgement), in the "
     "authors' coding. Empty when no report was given or required."),
    ("response_correct", "true/false", "", "The perceptual report was correct, for tasks with a correct answer."),
    ('localization_report_deg', 'number', 'deg', "Reported position of a briefly flashed target in a localization task (distance from fixation along the saccade axis), in the authors' report coding; empty when no localization was asked on the row."),
    ('localization_kind', 'category', 'pre-saccadic, post-saccadic', 'When the flash was localized: pre-saccadic = participant fixates and reports a flash shown before any saccade; post-saccadic = participant makes the saccade and then reports where the earlier flash was.'),
    ('vertical_endpoint_recoded_deg', 'number', 'deg', "Vertical saccade endpoint as the authors recoded it for their analysis (sign set so that positive means toward a named target) and corrected by subtracting the participant's baseline mean; empty for trials the authors excluded. See the README for the dataset's sign convention."),
    ('n_saccades', 'integer', '', 'Number of saccades detected during target presentation on the trial, where the authors give it.'),
    ('trial_info', 'text', 'name=value pairs separated by ;', 'Stimulus or protocol values for the trial that have no column of their own (for example contrast, spatial frequency, orientation), as name=value pairs; empty when none.'),
    ('localization_report_y_deg', 'number', 'deg', 'Vertical component of the reported position in a localization task (relative to the fixation point, positive upward); localization_report_deg then holds the horizontal component (positive rightward) for rows where the authors report a two-dimensional position.'),
    ('time_from_fixation_offset_ms', 'number', 'ms', 'Time from the offset of the fixation stimulus to the onset of the saccade this row describes, in sequential-saccade tasks.'),
    ('intersaccadic_interval_ms', 'number', 'ms', 'Time from the end of the previous saccade to the onset of the saccade this row describes, in sequences of saccades.'),
    ('initial_deviation_deg', 'number', 'deg', 'Initial curvature of the saccade as the authors computed it: perpendicular distance of the trajectory from the straight line from start to end, measured a fixed time after saccade onset; the README states the sign convention and the measurement time.'),
    ('pursuit_target_speed_deg_s', 'number', 'deg/s', 'Speed of the moving target that participants pursued, in smooth pursuit tasks.'),
    ('pursuit_target_direction', 'category', 'leftward, rightward', 'Horizontal direction in which the pursued target moved.'),
    ('saccade_onset_ms', 'number', 'ms', 'Time of onset of this saccade from a reference event in the trial that the README names (for example the start of the stimulus interval); empty when the row has no saccade.'),
    ('saccade_offset_ms', 'number', 'ms', 'Time of offset of this saccade from the same reference event as saccade_onset_ms.'),
    ('saccade_index_in_trial', 'integer', '', 'Running number of this saccade among the saccades the authors detected in the trial (1 = first); 0 or empty when the trial has none. n_saccades then holds the trial total.'),
    ('flash_seen_report', 'true/false', '', 'Participant reported seeing a stimulus flash on the trial (yes/no question).'),
    ('saccade_made_report', 'true/false', '', 'Participant reported having made a catch-up saccade on the trial (yes/no question).'),
    ('report_confidence', 'integer', '1-4', 'Confidence rating the participant gave for the saccade report, on a four-step scale (1 = not sure ... 4 = very sure), where asked.'),
    ('first_ramp_velocity_deg_s', 'number', 'deg/s; positive = rightward', 'Velocity of the first target ramp (the steady-state pursuit stage) in double step-ramp pursuit tasks.'),
    ('second_step_deg', 'number', 'deg; positive = rightward', 'Second target position step (the one at which the target jumps during pursuit) in double step-ramp tasks.'),
    ('second_ramp_velocity_deg_s', 'number', 'deg/s; positive = rightward', 'Velocity of the second target ramp that starts with the second step.'),
    ('velocity_step_deg_s', 'number', 'deg/s', 'Second ramp velocity minus first ramp velocity.'),
    ('saccades_after_second_step', 'integer', '', 'Number of saccades the authors detected after the second target step in the trial.'),
    ('saccades_before_second_step', 'integer', '', 'Number of saccades detected before the second target step.'),
    ('eye_x_100ms_before_saccade_deg', 'number', 'deg', 'Horizontal eye position 100 ms before the onset of the saccade this row describes (the saccade made after the second step); empty when the trial has no such saccade.'),
    ('position_error_100ms_before_saccade_deg', 'number', 'deg; target minus eye', 'Horizontal position error (target minus eye) 100 ms before saccade onset.'),
    ('retinal_slip_100ms_before_saccade_deg_s', 'number', 'deg/s', 'Retinal slip (second ramp velocity minus mean eye velocity) over 50 ms centred 100 ms before saccade onset.'),
    ('position_error_at_saccade_start_deg', 'number', 'deg; target minus eye', 'Horizontal position error at saccade onset.'),
    ('position_error_at_saccade_end_deg', 'number', 'deg; target minus eye', 'Horizontal position error at saccade offset (accuracy of the saccade).'),
    ('position_error_at_second_step_deg', 'number', 'deg; target minus eye', 'Horizontal position error at the time of the second target step.'),
]
