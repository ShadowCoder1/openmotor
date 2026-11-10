"""Task module: reaching to report a sensed position (visual, proprioceptive or both), one row per trial, with the reported
position, the ideal position and the authors' bias measures. First defined with Wong2024_eNeuro (state estimation while
stationary or moving, KINARM exoskeleton)."""

DESCRIPTION = ("Reporting the position of a visual cue, a felt hand position or both with a reaching movement: one row per "
               "trial with the movement endpoint or mid-movement position, the reported and ideal position, movement time "
               "and peak velocity, and the authors' bias toward the other modality.")

COLUMNS = [
    ("task_variant", "category", "static, dynamic, hstatic, dynamic_dmc", "The authors' task: static (report the endpoint of a movement along the reach axis), dynamic (report the position of an arc at its midpoint), hstatic (static task with the cues offset orthogonally to the reach axis; Experiment 2) and dynamic-dmc (dynamic task with both arcs curving to the same side; Experiment 3)."),
    ("task_protocol", "text", "", "The authors' 'task_protocol' (where the stimuli were placed)."),
    ("first_task", "text", "static, dynamic", "The task the person did first (Experiment 1, authors' EXP1_TASKORDER); empty where not given."),
    ("trial_orig", "integer", "", "The authors' trial number within the task ('trialNumber'), from 1."),
    ("block_orig", "integer", "1, 2", "Block of the protocol: 1 unimodal trials, 2 multimodal (bimodal) trials mixed with unimodal ones."),
    ("trial_type", "category", "V, P, VPv, VPp", "Cue condition: V visual only, P proprioceptive only (felt hand), VPv bimodal trial probing vision, VPp bimodal trial probing proprioception."),
    ("probed_modality", "category", "V, P", "Modality the person was asked to report."),
    ("target_index", "integer", "", "The authors' 'tpidx': index of the stimulus configuration in their trial table."),
    ("x_m", "number", "m", "Horizontal hand position at the reported moment: movement endpoint in static tasks ('x_endpoint'), the mid-movement position in dynamic tasks ('x_midpoint'). Metres in the apparatus frame as stored."),
    ("y_m", "number", "m", "Position along the reach axis at the reported moment ('y_endpoint' or 'y_midpoint')."),
    ("x_max_deviation_m", "number", "m", "Dynamic tasks: horizontal position at the point of maximum deviation ('x_max_deviation'); the authors' report measure for the dynamic task."),
    ("y_max_deviation_m", "number", "m", "Dynamic tasks: position along the reach axis at the point of maximum deviation ('y_max_deviation')."),
    ("report_position_m", "number", "m", "The reported position the analysis uses ('report_position': y_endpoint in the static task, x_max_deviation in the dynamic task, as the authors' data labels say; in Experiments 2 and 3 as stored)."),
    ("ideal_report_m", "number", "m", "The position a perfect report would have ('ideal_report')."),
    ("average_unimodal_m", "number", "m", "Average report of the unimodal block for the probed modality ('average_unimodal')."),
    ("normalization_factor_m", "number", "m", "Absolute difference between the two unimodal average reports ('normalization_factor')."),
    ("bias_m", "number", "m", "Report minus the unimodal average, positive toward the other modality's cue; 0 on unimodal trials ('bias')."),
    ("bias_normalized", "number", "", "bias divided by the normalization factor ('bias_normalized')."),
    ("movement_time_ms", "number", "ms", "Movement duration ('movement_time')."),
    ("peak_velocity_m_s", "number", "m/s", "Peak velocity during the movement ('movement_vpeak')."),
    ("velocity_midpoint_m_s", "number", "m/s", "Velocity at the midpoint ('v_midpoint') in dynamic tasks."),
    ("velocity_max_deviation_m_s", "number", "m/s", "Velocity at maximum deviation ('v_maxdev'), Experiment 3 only."),
    ("offset_1_m", "number", "m", "The authors' 'offset_1' (stimulus offset, first coordinate), Experiments 2 and 3."),
    ("offset_2_m", "number", "m", "The authors' 'offset_2' (stimulus offset, second coordinate), Experiments 2 and 3."),
    ("movement_err_x_m", "number", "m", "The authors' 'movement_errX' (Experiment 2)."),
    ("movement_err_y_m", "number", "m", "The authors' 'movement_errY' (Experiment 2)."),
    ("err_flag", "true/false", "", "The authors' 'err' flag (Experiment 2); meaning not given in the deposit."),
    ("isbad", "true/false", "", "The authors' 'isbad' flag (Experiments 2 and 3); the deposit does not define it."),
]
