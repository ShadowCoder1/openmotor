"""Task module: robot-based effort and reward tasks (isometric force matching, reach-to-choose utility task, effort-reward
accept/reject), one row per trial, with the task named in task_name and unused columns left empty. First defined with
Wood2026_npjParkinsonsDis (people with Parkinson's disease on and off medication and controls, in a Kinarm exoskeleton)."""

DESCRIPTION = ("Three robot-based tasks on effort and reward, in one long table: isometric force matching, a reaching task "
               "choosing between a sure and a gamble reward, and an effort-reward accept/reject task. One row per trial.")

# (column, type, values or units, description)
COLUMNS = [
    ("task_name", "category", "force_matching, reward_utility, effort_reward", "Which of the three tasks the row belongs to. Columns of the other tasks are empty."),
    ("med_status", "category", "on, off, control", "Medication state of the session for people with Parkinson's disease; control for controls."),
    ("both_sessions_done", "true/false", "", "The authors' complete flag: the participant with Parkinson's disease has data for both the on and the off session."),
    ("active_arm", "category", "right, left", "Arm used in the task."),
    ("fm_arm", "category", "same, opp", "Force matching: whether the match was made with the same arm as the initial force or the opposite arm."),
    ("fm_mvc", "number", "authors' units (see README)", "Force matching: the participant's maximum voluntary contraction, as in the deposit."),
    ("fm_goal_pct_mvc", "number", "% of MVC", "Force matching: the goal force."),
    ("fm_initial_force_pct_mvc", "number", "% of MVC", "Force matching: the force produced in the initial (reference) contraction."),
    ("fm_match_force_pct_mvc", "number", "% of MVC", "Force matching: the force produced when matching the felt effort."),
    ("fm_survey_report", "number", "", "Force matching: the participant's survey response on the trial, as in the deposit."),
    ("fm_initial_force_sd", "number", "authors' units", "Force matching: variability (SD) of the initial-force plateau."),
    ("fm_initial_stabilize_time", "number", "authors' units", "Force matching: time from getting into the goal range to the plateau of the initial force."),
    ("fm_initial_hold_time", "number", "authors' units", "Force matching: length of the initial-force plateau."),
    ("fm_match_force_sd", "number", "authors' units", "Force matching: variability (SD) during the match-force plateau."),
    ("fm_match_rise_time", "number", "authors' units", "Force matching: time from the initial hold to the end of the match hold."),
    ("fm_match_hold_time", "number", "authors' units", "Force matching: length of the match-force plateau."),
    ("ru_fixed_value", "number", "reward units as in the deposit", "Reward utility: value of the sure (fixed) option, in percent of the maximum reward."),
    ("ru_gamble_expected_value", "number", "same units", "Reward utility: expected value of the gamble option."),
    ("ru_gamble_amount", "number", "same units", "Reward utility: the reward if the gamble wins (twice the expected value)."),
    ("ru_orig_choice", "integer", "0, 1, 2", "Reward utility: the target the hand ended in (1 sure option, 2 gamble, 0 neither), the authors' orig_choice."),
    ("ru_final_choice_gamble", "true/false", "", "Reward utility: the final choice was the gamble (authors' choices_end)."),
    ("ru_initial_choice_gamble", "true/false", "", "Reward utility: the initial movement was toward the gamble (authors' init_choices)."),
    ("ru_mt_ms", "number", "ms (assumed from magnitude)", "Reward utility: movement time."),
    ("ru_rt_ms", "number", "ms (assumed from magnitude)", "Reward utility: reaction time relative to the go signal; the authors note it may be inaccurate, and it can be negative (movement before the go signal)."),
    ("ru_peak_velocity", "number", "m/s (assumed)", "Reward utility: peak hand velocity of the reach."),
    ("ru_amplitude", "number", "m (assumed)", "Reward utility: straight-line distance from reach start to end."),
    ("er_force_level", "number", "% (of the maximum force, as in the deposit)", "Effort-reward: force level of the offer."),
    ("er_reward_level", "number", "% of the maximum reward", "Effort-reward: reward level of the offer."),
    ("er_accept", "number", "1 or 0", "Effort-reward: the offer was accepted."),
]
