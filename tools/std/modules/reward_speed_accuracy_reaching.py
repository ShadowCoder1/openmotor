"""Task module: point-to-point reaching for reward with a speed-accuracy trade-off, one row per trial, with the per-trial
measures a deposit shares when it has no signed reach direction: radial and unsigned angular error of the endpoint, timing,
peak velocity and the reward given. First defined with Codol2020_JNeurosci (reward and punishment, KINARM)."""

DESCRIPTION = ("Reaching to targets for monetary reward or to avoid punishment, with a reward function tied to speed: one row per "
               "trial with the authors' timing, peak velocity, radial error and unsigned angular error, the potential and "
               "obtained reward, and the distractor flags.")

COLUMNS = [
    ("target_angle_deg", "number", "degrees", "Direction of the target from the start position (authors' 'targ_angle'; 0 right, counter-clockwise positive)."),
    ("trial_orig", "integer", "", "The authors' trial number ('trialno'); starts at 49 because 48 training trials are not in the deposit."),
    ("block_orig", "integer", "", "The authors' block number ('b')."),
    ("reward_context", "category", "reward, punishment", "Whether the block paid rewards or took punishments (from the sign of the potential reward; in the reward-magnitude experiment always reward)."),
    ("potential_reward_pence", "number", "pence", "Potential reward of the trial as the authors store it ('potrwd'): 1, 10 or 50 in the reward-magnitude experiment (1 is the paper's 0 p condition, inferred), and -50, -1, 1, 50 in the reward-punishment experiment (negative values are punishment blocks; the paper's -0 p condition is -1, 0 p is 1, inferred)."),
    ("score_pence", "number", "pence", "The authors' 'score': the amount obtained on the trial (in the punishment blocks negative, a loss; the paper says missing the target or moving to a distractor multiplied the loss by 1.2)."),
    ("tau1_ms", "number", "ms", "Lower threshold of the authors' adaptive reward function ('tau1', ms): movements faster than this earned the full potential reward (inferred from the paper's description; the readme only says 'tau1 value')."),
    ("tau2_ms", "number", "ms", "Upper threshold of the adaptive reward function ('tau2', ms), beyond which nothing was earned (inferred)."),
    ("reaction_time_s", "number", "seconds", "Reaction time ('RT'). Negative or very long values are kept; the authors drop trials under 0.2 s or over 1 s."),
    ("movement_time_s", "number", "seconds", "Movement time ('MT')."),
    ("peak_velocity_m_s", "number", "m/s", "Peak velocity ('peakVel')."),
    ("radial_error_cm", "number", "cm", "Radial error of the endpoint ('err'), always non-negative."),
    ("angular_error_deg_abs", "number", "degrees", "Angular error of the endpoint ('errangle'). Unsigned: the minimum in the file is above 0 and the authors' script removes trials above 20 with a one-sided test, so the sign of the error is not recoverable."),
    ("distractor_present", "true/false", "", "A distractor target was displayed on the trial ('DistOn')."),
    ("not_distracted", "true/false", "", "The authors' manual sort ('D'): True for non-distracted trials (the trial is analysed as a selection of the real target), False when the person went to the distractor."),
    ("distractor_delay_s", "number", "seconds", "Difference between the distractor target and normal target onset ('targjitter', s); empty on trials without a distractor."),
]
