"""Task module: incentivized joystick vigor task, one row per trial. The person pushes a spring-loaded joystick out of a start
circle to a bounding circle in any direction within a response window while a reward is at stake; the authors record the time
the cursor took to reach the bound and the straightness of the path. First defined with Brissenden2023_MovDisord (de novo
Parkinson's disease before and after levodopa)."""

DESCRIPTION = ("Incentivized joystick task: one row per trial with the reward at stake, the size of the bounding circle, "
               "the time from the go signal to the bound, the direction in which the bound was crossed and the path error.")

# (column, type, values or units, description)
COLUMNS = [
    ("reward_usd", "number", "US dollars",
     "Reward at stake on the trial (5, 10 or 20). Earned in full if the cursor reached the bound, otherwise in proportion to the "
     "farthest radius reached."),
    ("bound_diameter_norm", "number", "screen units (full screen width = 2)",
     "Diameter of the bounding circle (1.2, 1.5 or 1.8) in the task's normalized screen units."),
    ("bound_radius_cm", "number", "cm",
     "Radius of the bounding circle on the screen: half the diameter times 15.215 cm per unit (a 30.43 cm wide display spans 2 "
     "units); matches the 9.13, 11.41 and 13.69 cm the paper gives."),
    ("bound_reached", "true/false", "", "The cursor reached the bounding circle within the response window."),
    ("time_to_bound_ms", "number", "ms",
     "Time from the go signal until the cursor crossed the bound, as stored by the authors (timeToBound; the paper's movement "
     "time also subtracts the initiation time, which needs the trajectory files and is not computed here). Empty if the bound "
     "was not reached."),
    ("cross_angle_deg", "number", "degrees",
     "Direction in which the cursor crossed the bound, counter-clockwise from the right on the screen (the authors' polar angle "
     "of the crossing point; there is no instructed direction). Empty if the bound was not reached."),
    ("amount_earned_usd", "number", "US dollars", "Amount earned on the trial, as stored by the authors."),
    ("path_rms_norm", "number", "screen units",
     "Root mean square distance of the cursor from the straight line from the origin to the crossing point, until the crossing "
     "(the paper's movement linearity). Empty if the bound was not reached."),
    ("pre_go_rms_norm", "number", "screen units", "Root mean square distance of the cursor from the origin before the go signal."),
    ("start_delay_s", "number", "seconds", "Random hold time at the origin before the go signal (uniform 1 to 2 s)."),
    ("medication_state", "category", "baseline, sdr, ldr, sdr_ldr",
     "Levodopa state of the visit: baseline (untreated), sdr (acute dose, short-duration response), ldr (after 2-3 months of "
     "treatment with levodopa withheld 10-12 h, long-duration response) or sdr_ldr (usual dose, both responses)."),
]
