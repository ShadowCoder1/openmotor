"""Task module: reach-grasp-and-displace trials with an object of light or heavy weight that the child knows or does not know in
advance, one row per trial, with the authors' wrist and grip kinematic peaks of the Reach phase (before contact) and the Displace
phase (lifting and moving the object). First defined with Martel2023_Autism (children with autism, developmental coordination
disorder and typically developing children, Optotrak)."""

DESCRIPTION = ("Reach-to-displace trials: one row per trial in which the participant reaches for a bottle, grasps and displaces it. "
               "The bottle is light or heavy and its weight is known or unknown before contact. Columns hold the authors' latency and "
               "amplitude of the wrist's peak acceleration, velocity and deceleration in the Reach and Displace phases, the maximum "
               "grip aperture, the lift height and the movement times.")

# (column, type, values or units, description)
COLUMNS = [
    ("object_weight", "category", "light, heavy", "Weight of the bottle in this trial (light = 50 g, heavy = 500 g, as in the paper)."),
    ("weight_knowledge", "category", "known, unknown",
     "Known: the same bottle for a block of trials, so the weight can be anticipated before contact. Unknown: light and heavy "
     "bottles in pseudo-random order."),
    ("known_first_weight", "category", "light, heavy", "Which weight came first in the known-weight blocks for this child."),
    ("condition_code", "integer", "1-4", "The authors' condition code: 1 light known, 2 heavy known, 3 light unknown, 4 heavy unknown."),
    ("session_trial", "integer", "", "The authors' trial number within the session (the deposit keeps only 8 trials per cell, so numbers skip)."),
    ("reach_acc_latency_ms", "number", "ms", "Reach phase: latency of the maximum wrist acceleration (authors' TPACC; unit and origin inferred)."),
    ("reach_acc_amplitude_mm_s2", "number", "mm/s^2", "Reach phase: amplitude of the maximum wrist acceleration (ACC; unit inferred from the paper's m/s^2 and the values)."),
    ("reach_vel_latency_ms", "number", "ms", "Reach phase: latency of the maximum wrist velocity (TPV1)."),
    ("reach_vel_amplitude_mm_s", "number", "mm/s", "Reach phase: amplitude of the maximum wrist velocity (VEL1)."),
    ("reach_dec_latency_ms", "number", "ms", "Reach phase: latency of the maximum wrist deceleration (TPD1)."),
    ("reach_dec_amplitude_mm_s2", "number", "mm/s^2", "Reach phase: amplitude of the maximum wrist deceleration (DEC1; negative)."),
    ("grip_aperture_latency_ms", "number", "ms", "Latency of the maximum finger aperture (TPMGA)."),
    ("grip_aperture_max_mm", "number", "mm", "Maximum finger aperture during the reach (MGA)."),
    ("displace_acc_latency_ms", "number", "ms", "Displace phase: latency of the maximum wrist acceleration (TPA3)."),
    ("displace_acc_amplitude_mm_s2", "number", "mm/s^2", "Displace phase: amplitude of the maximum wrist acceleration (ACC3)."),
    ("displace_vel_latency_ms", "number", "ms", "Displace phase: latency of the maximum wrist velocity (TPV2)."),
    ("displace_vel_amplitude_mm_s", "number", "mm/s", "Displace phase: amplitude of the maximum wrist velocity (VEL2)."),
    ("displace_dec_latency_ms", "number", "ms", "Displace phase: latency of the maximum wrist deceleration (TPD2)."),
    ("displace_dec_amplitude_mm_s2", "number", "mm/s^2", "Displace phase: amplitude of the maximum wrist deceleration (DEC2; negative)."),
    ("lift_height_mm", "number", "mm", "Maximal height to which the bottle was lifted (Z)."),
    ("movement_time_reach_ms", "number", "ms", "Duration of the Reach phase (MT1; meaning inferred, MT1 + MT2 = MTTOT in every row)."),
    ("movement_time_displace_ms", "number", "ms", "Duration of the Displace phase (MT2; inferred)."),
    ("movement_time_total_ms", "number", "ms", "Total movement time (MTTOT)."),
]
