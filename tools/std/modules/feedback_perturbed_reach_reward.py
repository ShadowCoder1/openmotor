"""Task module: arm reaching on a KINARM exoskeleton with rewarded and non-rewarded trials and a perturbation (cursor jump, target
jump, torque pulse) that tests fast feedback responses, one row per trial. First defined with Codol2023_eLife; the per-trial
scalars of the authors' MATLAB files (the EMG and the kinematic traces are not released)."""

DESCRIPTION = ("Arm reaching with and without monetary reward while the cursor, the target or the arm is perturbed to probe fast "
               "feedback responses: one row per trial, with the authors' trial codes (control, displacement, reward), the target "
               "position and the movement time.")

# (column, type, values or units, description)
COLUMNS = [
    ("authors_trial", "integer", "", "The authors' trial number in the participant's file (starts at 5, 6 or 7: the first trials are not in the file, and some numbers are missing)."),
    ("authors_block", "integer", "", "The authors' block number b (2 and up)."),
    ("control_flag", "integer", "0, 1", "The authors' control: 1 on control trials as the authors' LABELS use it (cursor and target jump tasks: trials without a jump; In-Out and target-selection tasks: the control condition of their 2 x 2 x 2 design)."),
    ("displacement_code", "integer", "-1, 0, 1", "The authors' displ: direction of the cursor or target jump (-1, 0 = none, 1) in the jump tasks; 0 or 1 (perturbation condition) in the torque-pulse tasks. Its meaning in the torque tasks is not spelled out in the deposit."),
    ("n_targets", "integer", "1, 2", "Number of targets shown (target-selection task only)."),
    ("reward_max_cents", "number", "cents", "The authors' reward: 0 on non-rewarded trials, otherwise the maximum return g in cents (10 or 15)."),
    ("reward_tau", "number", "", "The authors' tau for the trial: the time-pressure parameter of the return function (1 to about 6), as recorded."),
    ("movement_time_s", "number", "seconds", "The authors' MT: movement time, the performance measure on which the return is based."),
    ("target_x_m", "number", "m", "End target position relative to the start position, horizontal (the authors' targ.end.x)."),
    ("target_y_m", "number", "m", "End target position relative to the start, vertical."),
    ("target_angle_deg", "number", "degrees", "Direction of the end target from the start, as the authors store it (targ.end.angle; the frame is the KINARM workspace, counter-clockwise positive is inferred)."),
    ("target_radius_m", "number", "m", "Visual radius of the end target."),
    ("jump_x_m", "number", "m", "Position, relative to the start, of the jumped target or cursor (the authors' targ.jump.x; jump tasks only; equal to the end target on trials without a jump)."),
    ("jump_y_m", "number", "m", "Vertical position of the jumped target or cursor."),
    ("jump_angle_deg", "number", "degrees", "Direction of the jumped target or cursor from the start."),
]
